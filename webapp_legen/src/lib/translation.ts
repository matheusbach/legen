import type { SubtitleEntry } from './srt'

export type TranslationEngine = 'google' | 'gemini'

type TranslateOptions = {
  entries: SubtitleEntry[]
  targetLanguage: string
  engine: TranslationEngine
  geminiApiKey?: string
  geminiModel?: string
  geminiAdditionalPrompt?: string
  geminiTemperature?: number
  geminiTopP?: number
  geminiTopK?: number
  geminiMaxOutputTokens?: number
  geminiThinkingEnabled?: boolean
  geminiThinkingBudget?: number
  onPrompt?: (prompt: string) => void
  onDebug?: (info: string) => void
  onProgress?: (completed: number, total: number) => void
  onPartialSrt?: (text: string, percent: number) => void
  signal?: AbortSignal
}

type GeminiResponse = {
  candidates?: Array<{
    content?: { parts?: Array<{ text?: string }> }
    finishReason?: string
  }>
  usageMetadata?: {
    promptTokenCount?: number
    candidatesTokenCount?: number
    totalTokenCount?: number
  }
  error?: { message?: string }
}

const sentenceEndings = ['.', '!', '?', ')', 'よ', 'ね', 'の', 'さ', 'ぞ', 'な', 'か', '！', '。', '」', '…']
const separator = ' ◌ '
const separatorUnjoin = separator.replace(' ', '')
const chunkMaxChars = 4999
const googleConcurrency = 7
const googleTimeoutMs = 30000
const geminiBatchSize = 500
const geminiMaxChars = 45000
const geminiTemperature = 0.3
const geminiTopP = 0.9
const geminiTopK = 50
const maxRetries = 3
const hardSeparators = ['⟧⟦', '⟬⟭', '⟪⟫']
let hardSeparatorIndex = 0

export async function translateCaptions({
  entries,
  targetLanguage,
  engine,
  geminiApiKey,
  geminiModel = 'gemini-2.5-flash',
  geminiAdditionalPrompt,
  geminiTemperature,
  geminiTopP,
  geminiTopK,
  geminiMaxOutputTokens,
  geminiThinkingEnabled,
  geminiThinkingBudget,
  onPrompt,
  onDebug,
  onProgress,
  onPartialSrt,
  signal,
}: TranslateOptions): Promise<SubtitleEntry[]> {
  if (!entries.length) {
    throw new Error('Carregue um arquivo SRT antes de traduzir.')
  }

  const normalizedEntries = entries.map((entry, idx) => ({
    ...entry,
    index: entry.index || idx + 1,
  }))

  if (engine === 'gemini') {
    const translated = await translateWithGemini({
      entries: normalizedEntries,
      targetLanguage,
      apiKeys: normalizeApiKeys(geminiApiKey),
      model: geminiModel,
      additionalPrompt: geminiAdditionalPrompt,
      temperature: geminiTemperature,
      topP: geminiTopP,
      topK: geminiTopK,
      maxOutputTokens: geminiMaxOutputTokens,
      thinkingEnabled: geminiThinkingEnabled,
      thinkingBudget: geminiThinkingBudget,
      onPrompt,
      onDebug,
      onProgress,
      onPartialSrt,
      signal,
    })
    return translated
  }

  const subContent = normalizedEntries.map((entry) => joinLines(entry.text))
  const lineCounts = normalizedEntries.map((entry) => Math.max(1, entry.text.split(/\r?\n/).length))
  const chunks = joinSentences(subContent, chunkMaxChars)
  const translatedChunks = await translateWithGoogle({
    chunks,
    targetLanguage,
    onProgress,
    onPartialSrt,
    entries: normalizedEntries,
    lineCounts,
    signal,
  })

  return rebuildEntriesFromChunks({
    entries: normalizedEntries,
    lineCounts,
    chunks,
    translatedChunks,
  })
}

async function translateWithGoogle({
  chunks,
  targetLanguage,
  onProgress,
  onPartialSrt,
  entries,
  lineCounts,
  signal,
}: {
  chunks: string[]
  targetLanguage: string
  onProgress?: (completed: number, total: number) => void
  onPartialSrt?: (text: string, percent: number) => void
  entries: SubtitleEntry[]
  lineCounts: number[]
  signal?: AbortSignal
}): Promise<string[]> {
  const translatedChunks: string[] = new Array(chunks.length)
  let completed = 0

  const pool = createPromisePool(googleConcurrency)
  const tasks = chunks.map((chunk, index) =>
    pool.run(async () => {
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')
      const expectedSeparators = countSeparators(chunk)
      const translated = await retryWithBackoff(() =>
        translateChunkGoogle(chunk, targetLanguage, expectedSeparators, signal),
      )
      translatedChunks[index] = translated
      completed += 1
      onProgress?.(completed, chunks.length)

      if (onPartialSrt) {
        const preview = rebuildEntriesFromChunks({
          entries,
          lineCounts,
          chunks,
          translatedChunks,
          allowFallback: true,
        })
        onPartialSrt(formatEntries(preview), completed / chunks.length)
      }
    }),
  )

  await Promise.all(tasks)
  return translatedChunks
}

async function translateWithGemini({
  entries,
  targetLanguage,
  apiKeys,
  model,
  additionalPrompt,
  temperature,
  topP,
  topK,
  maxOutputTokens,
  thinkingEnabled,
  thinkingBudget,
  onPrompt,
  onDebug,
  onProgress,
  onPartialSrt,
  signal,
}: {
  entries: SubtitleEntry[]
  targetLanguage: string
  apiKeys: string[]
  model: string
  additionalPrompt?: string
  temperature?: number
  topP?: number
  topK?: number
  maxOutputTokens?: number
  thinkingEnabled?: boolean
  thinkingBudget?: number
  onPrompt?: (prompt: string) => void
  onDebug?: (info: string) => void
  onProgress?: (completed: number, total: number) => void
  onPartialSrt?: (text: string, percent: number) => void
  signal?: AbortSignal
}): Promise<SubtitleEntry[]> {
  if (!apiKeys.length) {
    throw new Error('Gemini API key é obrigatória para traduzir com Gemini.')
  }

  const batches = buildGeminiBatches(entries)
  const translatedTextById: Map<number, string> = new Map()

  for (let i = 0; i < batches.length; i += 1) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')
    const batch = batches[i]
    const translated = await translateGeminiBatch({
      batch,
      targetLanguage,
      apiKeys,
      model,
      additionalPrompt,
      temperature,
      topP,
      topK,
      maxOutputTokens,
      thinkingEnabled,
      thinkingBudget,
      onPrompt,
      onDebug,
      signal,
    })

    translated.forEach((item) => translatedTextById.set(item.id, item.text))
    onProgress?.(i + 1, batches.length)

    if (onPartialSrt) {
      const preview = entries.map((entry, idx) => {
        const text = translatedTextById.get(idx) ?? ''
        return {
          ...entry,
          text: text ? enforceLineCount(text, entry.text) : entry.text,
        }
      })
      onPartialSrt(formatEntries(preview), (i + 1) / batches.length)
    }
  }

  return entries.map((entry, idx) => {
    const translated = translatedTextById.get(idx) ?? ''
    return {
      ...entry,
      text: translated ? enforceLineCount(translated, entry.text) : entry.text,
    }
  })
}

async function translateChunkGoogle(
  chunk: string,
  targetLanguage: string,
  expectedSeparators: number,
  signal?: AbortSignal,
): Promise<string> {
  const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=${encodeURIComponent(
    targetLanguage,
  )}&dt=t&q=${encodeURIComponent(chunk)}`
  const originalClean = stripSeparators(chunk)
  const checkUnchanged = originalClean.length > 20
  const maxAttempts = checkUnchanged ? 3 : 1

  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), googleTimeoutMs)
  try {
    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      const text = await fetchGoogleText(url, signal ?? controller.signal)
      const normalized = normalizeSeparators(text)

      if (!text || text.replace(separator.trim(), '').trim().length === 0) {
        continue
      }

      const missingSeparators = !hasEnoughSeparators(normalized, expectedSeparators)
      const unchanged = checkUnchanged && isLikelyUnchanged(normalized, chunk)
      if (!missingSeparators && !unchanged) {
        return normalized
      }
    }

    // Retry with hard separator tokens, rotating between them.
    const startIndex = reserveHardSeparators(3)
    for (let i = 0; i < 3; i += 1) {
      const token = hardSeparators[(startIndex + i) % hardSeparators.length]
      const tokenChunk = chunk.replaceAll(separator, `${token} `)
      const tokenUrl = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=${encodeURIComponent(
        targetLanguage,
      )}&dt=t&q=${encodeURIComponent(tokenChunk)}`
      const tokenText = await fetchGoogleText(tokenUrl, signal ?? controller.signal)
      const tokenNormalized = normalizeSeparators(tokenText.replaceAll(token, separator))

      if (hasEnoughSeparators(tokenNormalized, expectedSeparators) && !(checkUnchanged && isLikelyUnchanged(tokenNormalized, chunk))) {
        return tokenNormalized
      }
    }

    // Last resort: translate per-line within this chunk only.
    const perLine = await translateChunkPerLine(chunk, targetLanguage, signal)
    return normalizeSeparators(perLine)
  } finally {
    window.clearTimeout(timeout)
  }
}

async function translateGeminiBatch({
  batch,
  targetLanguage,
  apiKeys,
  model,
  additionalPrompt,
  temperature,
  topP,
  topK,
  maxOutputTokens,
  thinkingEnabled,
  thinkingBudget,
  onPrompt,
  onDebug,
  depth = 0,
  signal,
}: {
  batch: Array<{ id: number; text: string }>
  targetLanguage: string
  apiKeys: string[]
  model: string
  additionalPrompt?: string
  temperature?: number
  topP?: number
  topK?: number
  maxOutputTokens?: number
  thinkingEnabled?: boolean
  thinkingBudget?: number
  onPrompt?: (prompt: string) => void
  onDebug?: (info: string) => void
  depth?: number
  signal?: AbortSignal
}): Promise<Array<{ id: number; text: string }>> {
  const payload = JSON.stringify(batch.map((item) => ({ id: item.id, text: item.text })))
  const prompt = buildGeminiPrompt(targetLanguage, payload, additionalPrompt)
  const strictPrompt = buildGeminiPromptStrict(targetLanguage, payload, additionalPrompt)
  const resolvedMaxOutputTokens =
    typeof maxOutputTokens === 'number' && maxOutputTokens > 0 ? maxOutputTokens : undefined

  let lastError: Error | null = null
  for (const key of apiKeys) {
    try {
      onPrompt?.(prompt)
      const firstResponse = await retryWithBackoff(() =>
        requestGemini({
          prompt,
          apiKey: key,
          model,
          maxOutputTokens: resolvedMaxOutputTokens,
          temperature,
          topP,
          topK,
          thinkingEnabled,
          thinkingBudget,
          signal,
        }),
      )
      const text = firstResponse.text
      let strictResponse: { text: string; usage?: GeminiResponse['usageMetadata'] } | null = null
      let parsed = parseGeminiBatch(text, batch)

      if (!parsed) {
        onDebug?.(
          buildGeminiDebug(
            'parse_failed',
            prompt,
            text,
            batch.length,
            thinkingEnabled,
            thinkingBudget,
            firstResponse.usage,
          ),
        )
        onPrompt?.(`STRICT RETRY PROMPT\n${strictPrompt}`)
        strictResponse = await retryWithBackoff(() =>
          requestGemini({
            prompt: strictPrompt,
            apiKey: key,
            model,
            maxOutputTokens: resolvedMaxOutputTokens,
            temperature,
            topP,
            topK,
            thinkingEnabled,
            thinkingBudget,
            signal,
          }),
        )
        parsed = parseGeminiBatch(strictResponse.text, batch)
      }

      if (!parsed) {
        onDebug?.(
          buildGeminiDebug(
            'strict_parse_failed',
            strictPrompt,
            strictResponse?.text ?? '',
            batch.length,
            thinkingEnabled,
            thinkingBudget,
            strictResponse?.usage,
          ),
        )
        if (batch.length > 1 && depth < 4) {
          const mid = Math.ceil(batch.length / 2)
          const left = await translateGeminiBatch({
            batch: batch.slice(0, mid),
            targetLanguage,
            apiKeys,
            model,
            additionalPrompt,
            temperature,
            topP,
            topK,
            maxOutputTokens,
            thinkingEnabled,
            thinkingBudget,
            onPrompt,
            onDebug,
            depth: depth + 1,
            signal,
          })
          const right = await translateGeminiBatch({
            batch: batch.slice(mid),
            targetLanguage,
            apiKeys,
            model,
            additionalPrompt,
            temperature,
            topP,
            topK,
            maxOutputTokens,
            thinkingEnabled,
            thinkingBudget,
            onPrompt,
            onDebug,
            depth: depth + 1,
            signal,
          })
          return [...left, ...right]
        }
        return await fallbackBatchWithGoogle(batch, targetLanguage, signal)
      }

      onDebug?.(
        buildGeminiResultDebug(
          'success',
          batch.length,
          parsed.length,
          thinkingEnabled,
          thinkingBudget,
          firstResponse.usage,
        ),
      )
      return parsed
    } catch (err) {
      lastError = err instanceof Error ? err : new Error('Falha no Gemini')
      if (lastError.message.includes('API key inválida')) {
        continue
      }
    }
  }

  throw lastError ?? new Error('Gemini falhou em todas as chaves fornecidas.')
}

function buildGeminiDebug(
  reason: string,
  prompt: string,
  responseText: string,
  expectedCount: number,
  thinkingEnabled?: boolean,
  thinkingBudget?: number,
  usage?: GeminiResponse['usageMetadata'],
): string {
  const trimmed = responseText.trim()
  const full = trimmed ? trimmed : '(empty response)'
  return [
    `reason=${reason}`,
    `expectedItems=${expectedCount}`,
    `thinkingEnabled=${Boolean(thinkingEnabled)}`,
    `thinkingBudget=${typeof thinkingBudget === 'number' ? thinkingBudget : 'default'}`,
    `totalOutputTokens=${usage?.totalTokenCount ?? 'unknown'}`,
    '--- PROMPT ---',
    prompt,
    '--- RESPONSE (full) ---',
    full,
  ].join('\n')
}

function buildGeminiResultDebug(
  reason: string,
  expectedCount: number,
  receivedCount: number,
  thinkingEnabled?: boolean,
  thinkingBudget?: number,
  usage?: GeminiResponse['usageMetadata'],
): string {
  return [
    `reason=${reason}`,
    `expectedItems=${expectedCount}`,
    `receivedItems=${receivedCount}`,
    `thinkingEnabled=${Boolean(thinkingEnabled)}`,
    `thinkingBudget=${typeof thinkingBudget === 'number' ? thinkingBudget : 'default'}`,
    `totalOutputTokens=${usage?.totalTokenCount ?? 'unknown'}`,
  ].join('\n')
}

async function requestGemini({
  prompt,
  apiKey,
  model,
  maxOutputTokens,
  temperature,
  topP,
  topK,
  thinkingEnabled,
  thinkingBudget,
  signal,
}: {
  prompt: string
  apiKey: string
  model: string
  maxOutputTokens?: number
  temperature?: number
  topP?: number
  topK?: number
  thinkingEnabled?: boolean
  thinkingBudget?: number
  signal?: AbortSignal
}): Promise<{ text: string; usage?: GeminiResponse['usageMetadata'] }> {
  const endpoint = `https://generativelanguage.googleapis.com/v1/models/${model}:generateContent?key=${apiKey}`
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), 40000)
  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        ...(thinkingEnabled
          ? {
              thinkingConfig: {
                ...(typeof thinkingBudget === 'number' ? { thinkingBudget } : {}),
              },
            }
          : {}),
        generationConfig: {
          temperature: typeof temperature === 'number' ? temperature : geminiTemperature,
          top_p: typeof topP === 'number' ? topP : geminiTopP,
          top_k: typeof topK === 'number' ? topK : geminiTopK,
          ...(typeof maxOutputTokens === 'number' ? { maxOutputTokens } : {}),
        },
      }),
      signal: signal ?? controller.signal,
    })

    if (!res.ok) {
      const payload = await res.text().catch(() => '')
      const friendly = formatGeminiError(payload)
      throw new Error(friendly || `Gemini falhou: ${res.status} ${payload}`)
    }

    const json: GeminiResponse = await res.json()
    const text = extractGeminiText(json)
    if (!text) {
      throw new Error(json.error?.message || 'Resposta vazia do Gemini.')
    }
    return { text, usage: json.usageMetadata }
  } finally {
    window.clearTimeout(timeout)
  }
}

function buildGeminiPrompt(targetLanguage: string, payload: string, additionalPrompt?: string): string {
  return [
    'You are a subtitle translator. Follow the instructions strictly.',
    `Target language: ${targetLanguage}.`,
    'Input is a JSON array of objects: {"id": number, "text": string}.',
    'CRITICAL INSTRUCTIONS:',
    '1. STRICT 1:1 MAPPING: You MUST return exactly the same number of objects as the input batch. If you receive 500 items, you MUST output 500 items.',
    '2. DO NOT MERGE IDs: Even if a sentence starts in one ID and ends in the next, translate only the exact fragment belonging to that specific ID. Do NOT combine different IDs to make the translation flow better. This breaks subtitle sync.',
    '3. PRESERVE FRAGMENTS: If an input text is just a broken phrase like "and then the", the output must be the exact translation of "and then the". Do not complete the sentence.',
    '4. DO NOT SPLIT: Preserve line breaks (\n) inside each text. Do not create new objects for new lines. If input text is empty, return empty text. If input has content, never return empty.',
    '5. Do NOT add extra keys, metadata, explanations, or Markdown.',
    '6. Do NOT wrap the response in code fences(no ```json).',
    '7. Return ONLY valid JSON array of objects: [{"id": number, "text": string}, ...].',
    ...buildAdditionalPromptBlock(additionalPrompt),
    'Input JSON:',
    payload,
  ].join('\n')
}

function buildGeminiPromptStrict(targetLanguage: string, payload: string, additionalPrompt?: string): string {
  const basePrompt = buildGeminiPrompt(targetLanguage, payload, additionalPrompt)
  return [
    basePrompt,
    '',
    'FOCUS: Return ONLY valid JSON. No explanations, no extra text, no Markdown.',
  ].join('\n')
}

function normalizeAdditionalPrompt(additionalPrompt?: string): string | null {
  if (!additionalPrompt) return null
  const trimmed = additionalPrompt.trim()
  return trimmed ? trimmed : null
}

function buildAdditionalPromptBlock(additionalPrompt?: string): string[] {
  const extra = normalizeAdditionalPrompt(additionalPrompt)
  if (!extra) return []
  return [
    '---',
    'BEGIN USER INSTRUCTIONS (MANDATORY)',
    'These instructions MUST be followed and take precedence over default behavior.',
    'Do not ignore them unless they conflict with the required JSON output format.',
    extra,
    'END USER INSTRUCTIONS',
    '---',
  ]
}

function buildGeminiBatches(entries: SubtitleEntry[]): Array<Array<{ id: number; text: string }>> {
  const batches: Array<Array<{ id: number; text: string }>> = []
  let current: Array<{ id: number; text: string }> = []
  let currentChars = 0

  entries.forEach((entry, idx) => {
    const item = { id: idx, text: entry.text }
    const itemChars = JSON.stringify(item).length

    const wouldOverflow =
      current.length >= geminiBatchSize ||
      (currentChars + itemChars > geminiMaxChars && current.length > 0)

    if (wouldOverflow) {
      batches.push(current)
      current = []
      currentChars = 0
    }

    current.push(item)
    currentChars += itemChars
  })

  if (current.length) batches.push(current)
  return batches
}

function joinSentences(lines: string[], maxChars: number): string[] {
  const joinedLines: string[] = []
  let currentChunk = ''

  lines.forEach((line, index) => {
    const safeLine = line || '\u3164'
    const addition = safeLine + separator

    if (currentChunk && currentChunk.length + safeLine.length + separator.length > maxChars) {
      joinedLines.push(currentChunk)
      currentChunk = ''
    }

    if (addition.length > maxChars) {
      const endIndex = safeLine.lastIndexOf(' ', maxChars - (1 + separator.length))
      const cutIndex = endIndex === -(1 + separator.length) ? maxChars - (1 + separator.length) : endIndex
      joinedLines.push((safeLine.slice(0, cutIndex) + '\u2026' + separator).slice(0, maxChars))
      return
    }

    currentChunk += addition

    const isLastLine = index === lines.length - 1
    const endsSentence = sentenceEndings.some((ending) => safeLine.endsWith(ending))

    if (!endsSentence && !isLastLine) return

    if (isLastLine) {
      joinedLines.push(currentChunk)
      currentChunk = ''
      return
    }

    const nextLine = lines[index + 1] || '\u3164'
    const nextAdditionLength = nextLine.length + separator.length

    if (currentChunk.length + nextAdditionLength > maxChars) {
      joinedLines.push(currentChunk)
      currentChunk = ''
    }
  })

  if (currentChunk) joinedLines.push(currentChunk)
  return joinedLines
}

function unjoinSentences(originalSentence: string, modifiedSentence: string, sep: string): string[] {
  if (originalSentence == null) return []

  const originalLines = splitClean(originalSentence, separator)
  const cleanedModified = modifiedSentence
    ? modifiedSentence
        .replace(`${separatorUnjoin} `, separatorUnjoin)
        .replace(` ${separatorUnjoin}`, separatorUnjoin)
        .replace(`${separatorUnjoin}.`, `.${separatorUnjoin}`)
        .replace(`${separatorUnjoin},`, `,${separatorUnjoin}`)
    : ''

  const modifiedLines = splitClean(cleanedModified, sep)

  if (originalLines.length === 1 && (originalLines[0] === '...' || originalLines[0] === '…')) {
    return originalLines
  }

  if (originalLines.length === modifiedLines.length) return modifiedLines

  const originalWordCount = originalLines.reduce((acc, line) => acc + line.split(/\s+/).length, 0)
  const modifiedWordCount = modifiedLines.join(' ').split(/\s+/).length

  if (!originalWordCount || !modifiedWordCount) {
    return [originalSentence.replace(separator, ' ').replace(/\s{2,}/g, ' ').trim()]
  }

  const proportion = modifiedWordCount / originalWordCount
  const modifiedWords = modifiedLines.join(' ').replace(separator, '').replace(separatorUnjoin, '').trim().split(/\s+/)

  const rebuilt: string[] = []
  let cursor = 0

  originalLines.forEach((line, idx) => {
    let numWords = Math.round(line.split(/\s+/).length * proportion)
    if (numWords < 1) numWords = 1
    let generated = modifiedWords.slice(cursor, cursor + numWords).join(' ')
    cursor += numWords

    if (idx === originalLines.length - 1 && cursor < modifiedWords.length) {
      const tail = modifiedWords.slice(cursor).join(' ')
      generated = `${generated} ${tail}`.trim()
      cursor = modifiedWords.length
    }

    rebuilt.push(generated.replace(/\s{2,}/g, ' ').trim())
  })

  while (rebuilt.length < originalLines.length) {
    rebuilt.push(rebuilt[rebuilt.length - 1])
  }

  return rebuilt || originalLines || [' ']
}

function rebuildEntriesFromChunks({
  entries,
  lineCounts,
  chunks,
  translatedChunks,
  allowFallback = false,
}: {
  entries: SubtitleEntry[]
  lineCounts: number[]
  chunks: string[]
  translatedChunks: string[]
  allowFallback?: boolean
}): SubtitleEntry[] {
  const rebuiltLines: string[] = []

  chunks.forEach((chunk, idx) => {
    const translated = translatedChunks[idx]
    if (!translated && !allowFallback) {
      throw new Error('Faltou bloco traduzido para remontar o SRT.')
    }

    const chunkLines = translated ? unjoinSentences(normalizeSeparators(chunk), normalizeSeparators(translated), separatorUnjoin) : []
    const normalizedLines = (chunkLines.length ? chunkLines : Array.from({ length: 1 }, () => '')).map(stripSeparators)
    rebuiltLines.push(...normalizedLines)
  })

  while (rebuiltLines.length < entries.length) {
    rebuiltLines.push('')
  }
  if (rebuiltLines.length > entries.length) rebuiltLines.length = entries.length

  const rebuiltEntries = entries.map((entry, idx) => {
    const lineCount = lineCounts[idx]
    let text = rebuiltLines[idx] || ''
    if (!text) text = entry.text
    const enforced = splitStringToMaxLines(text, lineCount)
    return { ...entry, text: enforced.join('\n').trim() }
  })

  return rebuiltEntries
}

function splitStringToMaxLines(text: string, maxLines: number): string[] {
  if (maxLines <= 1) return [text.trim()]
  const words = text.split(/\s+/).filter(Boolean)
  if (!words.length) return Array.from({ length: maxLines }, () => '')

  const lines: string[] = []
  const targetWords = Math.max(1, Math.ceil(words.length / maxLines))

  for (let i = 0; i < maxLines; i += 1) {
    const slice = words.slice(i * targetWords, (i + 1) * targetWords)
    if (i === maxLines - 1 && (i + 1) * targetWords < words.length) {
      lines.push(words.slice(i * targetWords).join(' '))
      break
    }
    lines.push(slice.join(' '))
  }

  while (lines.length < maxLines) lines.push('')
  if (lines.length > maxLines) lines.length = maxLines
  return lines
}

function enforceLineCount(translated: string, original: string): string {
  const count = Math.max(1, original.split(/\r?\n/).length)
  const lines = splitStringToMaxLines(translated, count)
  return lines.join('\n').trim()
}

function splitClean(text: string, sep: string): string[] {
  return text
    .split(sep)
    .map((s) => s.replace(/\s{2,}/g, ' ').trim().replace(/^[,.:;)\s]+/, ''))
    .filter((s) => s && s.trim())
}

function normalizeSeparators(text: string): string {
  if (!text) return ''
  let normalized = text.replace(/\s*◌\s*/g, `${separator}`)
  normalized = normalized.replace(/\n+/g, separator)
  normalized = normalized.replace(/\s{2,}/g, ' ').trim()
  return normalized
}

function stripSeparators(text: string): string {
  if (!text) return ''
  let cleaned = text
  cleaned = cleaned.replaceAll(separator, ' ')
  cleaned = cleaned.replaceAll(separatorUnjoin, ' ')
  cleaned = cleaned.replace(/[◌○]/gu, ' ')
  cleaned = cleaned.replace(/\s{2,}/g, ' ').trim()
  return cleaned
}

function joinLines(text: string): string {
  return text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).join(' ')
}

function formatEntries(entries: SubtitleEntry[]): string {
  return entries
    .map((entry, i) => `${entry.index || i + 1}\n${entry.start} --> ${entry.end}\n${entry.text.trim()}\n`)
    .join('\n')
    .trimEnd()
}

function createPromisePool(concurrency: number) {
  let active = 0
  const queue: Array<() => void> = []

  const runNext = () => {
    if (active >= concurrency) return
    const next = queue.shift()
    if (!next) return
    active += 1
    next()
  }

  const run = <T>(task: () => Promise<T>) =>
    new Promise<T>((resolve, reject) => {
      const exec = () => {
        task()
          .then(resolve)
          .catch(reject)
          .finally(() => {
            active -= 1
            runNext()
          })
      }
      queue.push(exec)
      runNext()
    })

  return { run }
}

async function retryWithBackoff<T>(fn: () => Promise<T>, attempts = maxRetries): Promise<T> {
  let lastErr: unknown
  for (let i = 0; i < attempts; i += 1) {
    try {
      return await fn()
    } catch (err) {
      lastErr = err
      if (i === attempts - 1) break
      const delay = 500 * (i + 1)
      await new Promise((r) => setTimeout(r, delay))
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error('Falha após retentativas')
}

function decodeHtmlEntities(input: string): string {
  if (typeof document === 'undefined') return input
  const el = document.createElement('textarea')
  el.innerHTML = input
  return el.value
}

function reserveHardSeparators(count: number): number {
  const start = hardSeparatorIndex
  hardSeparatorIndex = (hardSeparatorIndex + count) % hardSeparators.length
  return start
}

async function fetchGoogleText(url: string, signal?: AbortSignal): Promise<string> {
  const res = await fetch(url, { signal })
  if (!res.ok) {
    const payload = await res.text().catch(() => '')
    throw new Error(`Google Translate falhou: ${res.status} ${payload}`)
  }

  type GoogleResponse = [Array<[string, string, string?, string?]>, ...unknown[]]
  const data = (await res.json()) as GoogleResponse
  const translations = Array.isArray(data?.[0]) ? data[0] : []
  const raw = translations.map((item) => item?.[0] ?? '').join('')
  return decodeHtmlEntities(raw)
}

function countSeparators(text: string): number {
  return text.split(separator).length - 1
}

function hasEnoughSeparators(text: string, expected: number): boolean {
  if (expected <= 0) return true
  const count = countSeparators(text)
  return count === expected
}

function isLikelyUnchanged(translated: string, original: string): boolean {
  const cleanTranslated = stripSeparators(translated).toLowerCase()
  const cleanOriginal = stripSeparators(original).toLowerCase()
  if (!cleanTranslated || !cleanOriginal) return false
  if (cleanTranslated === cleanOriginal) return true
  const overlap = commonPrefixRatio(cleanTranslated, cleanOriginal)
  return overlap > 0.9
}

function commonPrefixRatio(a: string, b: string): number {
  const len = Math.min(a.length, b.length)
  let i = 0
  while (i < len && a[i] === b[i]) i += 1
  return len ? i / len : 0
}

async function translateChunkPerLine(
  chunk: string,
  targetLanguage: string,
  signal?: AbortSignal,
): Promise<string> {
  const lines = chunk.split(separator).map((line) => line.trim())
  const pool = createPromisePool(googleConcurrency)
  const output: string[] = new Array(lines.length).fill('')

  const tasks = lines.map((line, idx) =>
    pool.run(async () => {
      if (!line) {
        output[idx] = ''
        return
      }
      const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=${encodeURIComponent(
        targetLanguage,
      )}&dt=t&q=${encodeURIComponent(line)}`
      const translated = await retryWithBackoff(() => fetchGoogleText(url, signal))
      output[idx] = translated || line
    }),
  )

  await Promise.all(tasks)
  return output.join(separator)
}

function extractGeminiText(resp: GeminiResponse): string | undefined {
  const candidate = resp.candidates?.find((c) => c.content?.parts?.length)
  const text = candidate?.content?.parts?.map((part) => part.text ?? '').join('').trim()
  return text
}

function normalizeApiKeys(raw: string | undefined): string[] {
  if (!raw) return []
  const parts = raw
    .split(/[\n,]/)
    .map((part) => part.trim())
    .filter(Boolean)

  const unique: string[] = []
  const seen = new Set<string>()
  parts.forEach((key) => {
    if (!seen.has(key)) {
      unique.push(key)
      seen.add(key)
    }
  })
  return unique
}

function extractJsonArray(raw: string): string {
  const cleaned = stripCodeFences(raw)
  const span = findJsonArraySpan(cleaned)
  if (!span) {
    throw new Error('Resposta do Gemini não contém JSON válido.')
  }
  const candidate = cleaned.slice(span.start, span.end + 1)
  return sanitizeJson(candidate)
}

function parseGeminiBatch(text: string, batch: Array<{ id: number; text: string }>): Array<{ id: number; text: string }> | null {
  try {
    const jsonText = extractJsonArray(text)
    let safe = sanitizeJson(jsonText)
    let data: Array<{ id: number; text: string }>
    try {
      data = JSON.parse(safe) as Array<{ id: number; text: string }>
    } catch {
      safe = repairUnescapedQuotesInTextFields(safe)
      data = JSON.parse(safe) as Array<{ id: number; text: string }>
    }
    return normalizeParsedBatch(data, batch)
  } catch {
    const fallback = extractJsonObjects(text)
    if (!fallback.length) return null
    return normalizeParsedBatch(fallback, batch)
  }
}

function normalizeParsedBatch(
  data: Array<{ id: number | string; text: string }>,
  batch: Array<{ id: number; text: string }>,
): Array<{ id: number; text: string }> | null {
  if (!Array.isArray(data) || data.length !== batch.length) {
    return null
  }

  const mapped = data.map((item, idx) => {
    const idValue = typeof item.id === 'string' ? Number(item.id) : item.id
    return {
      id: Number.isFinite(idValue) ? Number(idValue) : batch[idx].id,
      text: typeof item.text === 'string' ? item.text : '',
      idx,
    }
  })

  return mapped.map((item, idx) => {
    const source = batch[idx]
    const textValue = item.text.trim()
    return {
      id: item.id,
      text: textValue || source.text,
    }
  })
}

function extractJsonObjects(raw: string): Array<{ id: number | string; text: string }> {
  const cleaned = stripCodeFences(raw)
  const matches = cleaned.match(/\{[^\}]*"id"[^\}]*"text"[^\}]*\}/g)
  if (!matches) return []
  const results: Array<{ id: number | string; text: string }> = []
  for (const candidate of matches) {
    try {
      let safe = sanitizeJson(candidate)
      try {
        const parsed = JSON.parse(safe) as { id: number | string; text: string }
        if (parsed && typeof parsed === 'object' && 'text' in parsed) {
          results.push(parsed)
          continue
        }
      } catch {
        // retry after repairing unescaped quotes in text field
      }

      safe = repairUnescapedQuotesInTextFields(safe)
      const parsed = JSON.parse(safe) as { id: number | string; text: string }
      if (parsed && typeof parsed === 'object' && 'text' in parsed) {
        results.push(parsed)
      }
    } catch {
      continue
    }
  }
  return results
}

async function fallbackBatchWithGoogle(
  batch: Array<{ id: number; text: string }>,
  targetLanguage: string,
  signal?: AbortSignal,
): Promise<Array<{ id: number; text: string }>> {
  const pool = createPromisePool(googleConcurrency)
  const output: Array<{ id: number; text: string }> = new Array(batch.length)
  const tasks = batch.map((item, idx) =>
    pool.run(async () => {
      const translated = await retryWithBackoff(() =>
        translateChunkGoogle(item.text, targetLanguage, countSeparators(item.text), signal),
      )
      output[idx] = { id: item.id, text: translated || item.text }
    }),
  )
  await Promise.all(tasks)
  return output
}

function stripCodeFences(text: string): string {
  return text.replace(/```[a-zA-Z]*\n?/g, '').replace(/```/g, '').trim()
}

function findJsonArraySpan(text: string): { start: number; end: number } | null {
  const start = text.indexOf('[')
  if (start === -1) return null

  let depth = 0
  let inString = false
  let escaped = false

  for (let i = start; i < text.length; i += 1) {
    const ch = text[i]

    if (escaped) {
      escaped = false
      continue
    }

    if (ch === '\\') {
      escaped = true
      continue
    }

    if (ch === '"') {
      inString = !inString
      continue
    }

    if (inString) continue

    if (ch === '[') depth += 1
    if (ch === ']') {
      depth -= 1
      if (depth === 0) {
        return { start, end: i }
      }
    }
  }

  return null
}

function sanitizeJson(text: string): string {
  return text
    .replace(/\u201c|\u201d/g, '"')
    .replace(/\u2018|\u2019/g, "'")
    .replace(/,\s*([}\]])/g, '$1')
}

function repairUnescapedQuotesInTextFields(text: string): string {
  // Attempts to escape stray double quotes inside "text" fields.
  const key = '"text"'
  let idx = text.indexOf(key)
  let output = text
  while (idx !== -1) {
    const startQuote = output.indexOf('"', idx + key.length)
    if (startQuote === -1) break
    const lastQuote = output.lastIndexOf('"')
    const closingQuote = output.lastIndexOf('"', output.indexOf('}', startQuote + 1))

    // Choose the last quote before the next object close if possible; fallback to the global last quote.
    const effectiveEnd = closingQuote > startQuote ? closingQuote : lastQuote
    if (effectiveEnd === -1 || effectiveEnd <= startQuote) break

    const before = output.slice(0, startQuote + 1)
    const middle = output.slice(startQuote + 1, effectiveEnd)
    const after = output.slice(effectiveEnd)

    const repaired = middle.replace(/(?<!\\)"/g, '\\"')
    output = `${before}${repaired}${after}`

    idx = output.indexOf(key, effectiveEnd + 1)
  }
  return output
}

function formatGeminiError(payload: string): string | null {
  if (!payload) return null
  try {
    const parsed = JSON.parse(payload) as {
      error?: {
        message?: string
        details?: Array<{ reason?: string; message?: string }>
      }
    }
    const message = parsed.error?.message || ''
    const reason = parsed.error?.details?.find((d) => d.reason)?.reason || ''
    if (reason === 'API_KEY_INVALID' || /api key not valid/i.test(message)) {
      return 'Gemini: API key inválida. Verifique a chave e tente novamente.'
    }
    if (message) return `Gemini falhou: ${message}`
  } catch {
    // ignore parse errors
  }
  if (/api key not valid/i.test(payload)) {
    return 'Gemini: API key inválida. Verifique a chave e tente novamente.'
  }
  return null
}
