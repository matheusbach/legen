import { clampContent, formatSrt, parseTimestampToMs } from './srt'
import type { SubtitleEntry } from './srt'

type GeminiResponse = {
  candidates?: Array<{
    content?: { parts?: Array<{ text?: string }> }
    finishReason?: string
  }>
  error?: { message?: string }
}

const maxOutputTokens = 16364
const maxRounds = 10
const continuationTailChars = 800
const requestTimeoutMs = 120000

export async function generateTltwSummary({
  entries,
  language,
  apiKey,
  model = 'gemini-2.5-flash',
  maxChars = 120000,
  additionalPrompt,
  onPrompt,
  onChunk,
  thinkingEnabled,
  thinkingBudget,
  temperature,
  topP,
  topK,
  maxOutputTokensOverride,
  signal,
}: {
  entries: SubtitleEntry[]
  language: string
  apiKey?: string
  model?: string
  maxChars?: number
  additionalPrompt?: string
  onPrompt?: (prompt: string) => void
  onChunk?: (text: string) => void
  thinkingEnabled?: boolean
  thinkingBudget?: number
  temperature?: number
  topP?: number
  topK?: number
  maxOutputTokensOverride?: number
  signal?: AbortSignal
}): Promise<string> {
  const apiKeys = normalizeApiKeys(apiKey)
  if (!apiKeys.length) {
    throw new Error('Informe uma API key do Gemini para gerar o TLTW.')
  }

  if (!entries.length) {
    throw new Error('Carregue um SRT antes de gerar o TLTW.')
  }

  const durationSeconds = estimateDurationSeconds(entries)
  const { maxKeyPoints, minKeyPoints } = computeKeyPointLimits(durationSeconds)
  const prompt = buildTltwPrompt(language, maxKeyPoints, minKeyPoints, additionalPrompt)
  const srtText = clampContent(formatSrt(entries), maxChars)

  let lastError: Error | null = null
  for (const key of apiKeys) {
    try {
      const summary = await requestTltw({
        apiKey: key,
        model,
        prompt,
        subtitleText: srtText,
        onPrompt,
        onChunk,
        thinkingEnabled,
        thinkingBudget,
        temperature,
        topP,
        topK,
        maxOutputTokensOverride,
        signal,
      })
      return stripEndMarker(summary)
    } catch (err) {
      lastError = err instanceof Error ? err : new Error('Falha ao gerar TLTW')
      continue
    }
  }

  throw lastError ?? new Error('Gemini falhou com todas as chaves fornecidas.')
}

async function requestTltw({
  apiKey,
  model,
  prompt,
  subtitleText,
  onPrompt,
  onChunk,
  thinkingEnabled,
  thinkingBudget,
  temperature,
  topP,
  topK,
  maxOutputTokensOverride,
  signal,
}: {
  apiKey: string
  model: string
  prompt: string
  subtitleText: string
  onPrompt?: (prompt: string) => void
  onChunk?: (text: string) => void
  thinkingEnabled?: boolean
  thinkingBudget?: number
  temperature?: number
  topP?: number
  topK?: number
  maxOutputTokensOverride?: number
  signal?: AbortSignal
}): Promise<string> {
  const basePrompt = prompt
  let step = 1
  let fullText = ''
  const firstPrompt = `${basePrompt}\n\n${subtitleText}`

  const first = await sendGeminiRequest({
    apiKey,
    model,
    prompt: firstPrompt,
    onChunk,
    thinkingEnabled,
    thinkingBudget,
    temperature,
    topP,
    topK,
    maxOutputTokensOverride,
    signal,
  })
  onPrompt?.(firstPrompt)
  fullText = first.text
  let truncated = first.truncated || !fullText.includes('<!-- END -->')

  let rounds = 1
  while (truncated && rounds < maxRounds) {
    const tail = fullText.slice(-continuationTailChars)
    const continuationPrompt = [
      basePrompt,
      '',
      'You already started writing the Markdown document.',
      'However, your previous output was cut off due to length limits.',
      'Remember all the original instructions.',
      'Continue it from exactly where it stopped. Do NOT repeat content.',
      'Maintain the same structure and formatting.',
      'Finish by writing the final line exactly equal to: <!-- END -->',
      'Here is the last part you wrote (for alignment):',
      '```',
      tail,
      '```',
      'Continue now:',
    ].join('\n')

    step += 1
    const continuationFullPrompt = `${continuationPrompt}\n\n${subtitleText}`
    const next = await sendGeminiRequest({
      apiKey,
      model,
      prompt: continuationFullPrompt,
      onChunk: (chunkText) => {
        const merged = mergeContinuation(fullText, chunkText, tail)
        onChunk?.(merged)
      },
      thinkingEnabled,
      thinkingBudget,
      temperature,
      topP,
      topK,
      maxOutputTokensOverride,
      signal,
    })
    onPrompt?.(continuationFullPrompt)

    const merged = mergeContinuation(fullText, next.text, tail)
    fullText = merged
    truncated = next.truncated || !fullText.includes('<!-- END -->')
    rounds += 1
  }

  if (!fullText) {
    throw new Error('Resposta vazia do Gemini para TLTW.')
  }
  if (!fullText.includes('<!-- END -->')) {
    throw new Error('Gemini TLTW não finalizou (faltou marcador END).')
  }
  return fullText
}

async function sendGeminiRequest({
  apiKey,
  model,
  prompt,
  onChunk,
  thinkingEnabled,
  thinkingBudget,
  temperature,
  topP,
  topK,
  maxOutputTokensOverride,
  signal,
}: {
  apiKey: string
  model: string
  prompt: string
  onChunk?: (text: string) => void
  thinkingEnabled?: boolean
  thinkingBudget?: number
  temperature?: number
  topP?: number
  topK?: number
  maxOutputTokensOverride?: number
  signal?: AbortSignal
}): Promise<{ text: string; truncated: boolean }> {
  const endpoint = `https://generativelanguage.googleapis.com/v1/models/${model}:streamGenerateContent?alt=sse&key=${apiKey}`
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), requestTimeoutMs)
  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: {
          temperature: typeof temperature === 'number' ? temperature : 0.15,
          top_p: typeof topP === 'number' ? topP : 0.9,
          top_k: typeof topK === 'number' ? topK : 40,
          maxOutputTokens:
            typeof maxOutputTokensOverride === 'number' ? maxOutputTokensOverride : maxOutputTokens,
          ...(thinkingEnabled
            ? {
                thinkingConfig: {
                  thinkingBudget: typeof thinkingBudget === 'number' && thinkingBudget > 0 ? thinkingBudget : undefined,
                },
              }
            : {}),
        },
      }),
      signal: signal ?? controller.signal,
    })

    if (!res.ok) {
      const payload = await res.text()
      const friendly = formatGeminiError(payload)
      throw new Error(friendly || `Gemini falhou: ${res.status} ${payload}`)
    }

    if (!res.body) {
      throw new Error('Resposta vazia do Gemini para TLTW.')
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let fullText = ''
    let truncated = false
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim()
          if (data === '[DONE]') continue
          try {
            const json = JSON.parse(data) as GeminiResponse
            const chunkText = extractGeminiText(json)
            if (chunkText) {
              fullText += chunkText
              onChunk?.(fullText)
            }
            const finishReason = json.candidates?.[0]?.finishReason || ''
            if (/max|token/i.test(String(finishReason))) {
              truncated = true
            }
          } catch (e) {
            // ignore parse error for partial chunks
          }
        }
      }
    }

    if (buffer.startsWith('data: ')) {
      const data = buffer.slice(6).trim()
      if (data !== '[DONE]') {
        try {
          const json = JSON.parse(data) as GeminiResponse
          const chunkText = extractGeminiText(json)
          if (chunkText) {
            fullText += chunkText
            onChunk?.(fullText)
          }
          const finishReason = json.candidates?.[0]?.finishReason || ''
          if (/max|token/i.test(String(finishReason))) {
            truncated = true
          }
        } catch (e) {}
      }
    }

    if (!fullText) {
      throw new Error('Resposta vazia do Gemini para TLTW.')
    }

    return { text: fullText, truncated }
  } finally {
    window.clearTimeout(timeout)
  }
}

function buildTltwPrompt(
  language: string,
  maxKeyPoints: number,
  minKeyPoints: number,
  additionalPrompt?: string,
): string {
  const extraBlock = buildAdditionalPromptBlock(additionalPrompt)

  return (
    "Generate a high-quality 'TLTW' (Too Long To Watch) summary in " +
    `${language} strictly based on the provided transcript or subtitles. ` +
    'The TLTW must be factual, concise, and faithful to the source.\n\n' +
    extraBlock +
    'Use Markdown and follow this exact structure:\n\n' +
    '# Title\n' +
    '- A clear, descriptive title reflecting the main subject of the content\n\n' +
    '*Tags: [tag 1, tag 2, ...]*\n' +
    '- 3–9 short tags, comma-separated, singular words when possible, multi-word-expressions only when significant, lowercase when possible (e.g., `ai, subtitles, whisperx, translation`)\n' +
    '- Tags must reflect topics actually present in the source\n\n' +
    '## Key Points\n' +
    `- Write at most ${maxKeyPoints} bullet points\n` +
    `- Prefer ${minKeyPoints}–${maxKeyPoints} bullet points; you may go up to 40% smaller to avoid redundancy\n` +
    '- Put the most important points first\n' +
    '- Preserve technical terminology used in the source when relevant\n' +
    '- Each bullet must briefly explain the point (topic + a short clarifying clause), not just label it\n' +
    '- Avoid generic takeaways like "the importance of", "the need of", "is crucial", unless tied to a specific situation described\n' +
    '- Avoid advice/recommendations here; put them only in Actions when explicitly present\n' +
    '- Do not end bullet lines with a period (no trailing \\.)\n\n' +
    '## Actions or Next Steps (only if applicable)\n' +
    '- Bullet points describing explicit recommendations, procedures, or follow-ups mentioned in the source\n' +
    '- Omit this section entirely if no actionable items are present\n' +
    '- Do not end bullet lines with a period (no trailing \\.)\n\n' +
    '## Summary\n' +
    `- Write at most ${maxKeyPoints} chapter titles\n` +
    `- Prefer ${minKeyPoints}–${maxKeyPoints} chapter titles; you may go up to 40% smaller to avoid redundancy\n` +
    '- Each title must be 1 short sentence\n' +
    '- Write chapter-like titles, not takeaways: describe the topic, not what the viewer should learn/do\n' +
    '- Prefer noun phrases and topic labels; avoid advice, conclusions, recommendations, or moral-of-the-story phrasing\n' +
    '- Suppress authorship; write each line like a book chapter title, not about who said/did it\n' +
    '- Each title must be a single line in the format: HH:MM:SS description\n' +
    '- Use HH:MM:SS (no milliseconds)\n' +
    '- Keep items in chronological order\n' +
    '- Use timestamps aligned to the subtitle timeline (best possible approximation based on nearby lines)\n' +
    '- Do not use bullet markers for these lines\n' +
    '- Do not end lines with a period (no trailing \\.)\n\n' +
    'Example format:\n' +
    '```\n' +
    '00:00:02 chapter title description\n' +
    '00:22:13 chapter title description\n' +
    '```\n\n' +
    'Writing style requirements:\n' +
    '- Use active voice; avoid passive constructions\n' +
    '- Be concise; remove filler words\n' +
    '- Use an assertive, direct tone\n' +
    '- Start lines with an article only when it naturally fits; prefer direct noun-phrase titles and topic labels\n\n' +
    'Rules:\n' +
    `- Write exclusively in ${language}\n` +
    '- Do not invent, extrapolate, or assume information\n' +
    '- Avoid redundancy and meta commentary\n' +
    '- Keep sentences short, direct, and information-dense\n' +
    '- Dont cause repetition of structures\n' +
    '- Do not apologize or reference missing context\n' +
    '- Do not mention the transcript or the act of summarization\n' +
    '- End the document with a final line exactly equal to: <!-- END -->\n'
  )
}

function normalizeAdditionalPrompt(additionalPrompt?: string): string | null {
  if (!additionalPrompt) return null
  const trimmed = additionalPrompt.trim()
  return trimmed ? trimmed : null
}

function buildAdditionalPromptBlock(additionalPrompt?: string): string {
  const extra = normalizeAdditionalPrompt(additionalPrompt)
  if (!extra) return ''
  return (
    '---\n' +
    'BEGIN USER INSTRUCTIONS (MANDATORY)\n' +
    'These instructions MUST be followed and take precedence over default behavior.\n' +
    'Do not ignore them unless they conflict with the required output structure.\n' +
    `${extra}\n` +
    'END USER INSTRUCTIONS\n' +
    '---\n\n'
  )
}

function estimateDurationSeconds(entries: SubtitleEntry[]): number {
  if (!entries.length) return 0
  const last = entries[entries.length - 1]
  const endMs = parseTimestampToMs(last.end)
  return Number.isFinite(endMs) ? Math.max(0, endMs / 1000) : 0
}

function computeKeyPointLimits(durationSeconds: number): { maxKeyPoints: number; minKeyPoints: number } {
  if (!durationSeconds || durationSeconds <= 0) {
    return { maxKeyPoints: 10, minKeyPoints: 6 }
  }
  const hours = durationSeconds / 3600
  const maxKeyPoints = Math.max(10, Math.ceil(hours * 10))
  const minKeyPoints = Math.max(1, Math.floor(maxKeyPoints * 0.6))
  return { maxKeyPoints, minKeyPoints }
}

function mergeContinuation(fullText: string, nextText: string, tail: string): string {
  if (!nextText) return fullText
  if (!tail) return `${fullText.trim()} ${nextText.trim()}`.trim()

  const probe = tail.slice(-200)
  const pos = nextText.indexOf(probe)
  const trimmed = pos !== -1 ? nextText.slice(pos + probe.length).trim() : nextText.trim()

  if (!trimmed) return fullText
  const separator = fullText && trimmed && /[\w\p{L}]$/u.test(fullText) && /^[\w\p{L}]/u.test(trimmed) ? ' ' : ''
  return `${fullText.trim()}${separator}${trimmed}`.trim()
}

function stripEndMarker(text: string): string {
  const marker = '<!-- END -->'
  if (!text.includes(marker)) return text.trim()
  return text
    .split('\n')
    .filter((line) => line.trim() !== marker)
    .join('\n')
    .trim()
}

function extractGeminiText(resp: GeminiResponse): string | undefined {
  const candidate = resp.candidates?.find((c) => c.content?.parts?.length)
  const text = candidate?.content?.parts?.map((part) => part.text ?? '').join('')
  return text
}

function normalizeApiKeys(raw?: string): string[] {
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
