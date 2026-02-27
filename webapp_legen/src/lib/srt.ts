export type SubtitleEntry = {
  index: number
  start: string
  end: string
  text: string
}

export function parseSrt(raw: string): SubtitleEntry[] {
  const normalized = raw.replace(/\r\n/g, '\n').trim()
  if (!normalized) {
    throw new Error('O arquivo SRT está vazio.')
  }

  const blocks = normalized.split(/\n{2,}/).map((block) => block.trim()).filter(Boolean)
  const entries: SubtitleEntry[] = []

  for (const block of blocks) {
    const lines = block.split('\n').map((line) => line.trim()).filter(Boolean)
    if (lines.length < 2) {
      continue
    }

    let timeLineIndex = 1
    let index = Number.parseInt(lines[0], 10)
    const hasLeadingIndex = Number.isInteger(index) && lines[1]?.includes('-->')

    if (!hasLeadingIndex && lines[0].includes('-->')) {
      timeLineIndex = 0
      index = entries.length + 1
    } else if (!hasLeadingIndex) {
      index = entries.length + 1
    }

    const timeLine = lines[timeLineIndex]
    if (!timeLine || !timeLine.includes('-->')) {
      continue
    }

    const [start, end] = timeLine.split('-->').map((part) => part.trim())
    const textLines =
      timeLineIndex === 0 ? lines.slice(1) : lines.slice(timeLineIndex + 1)
    const text = textLines.join('\n').trim()

    entries.push({ index, start, end, text })
  }

  if (!entries.length) {
    throw new Error('Não foi possível ler nenhum bloco SRT. Verifique o formato.')
  }

  return entries
}

export function formatSrt(entries: SubtitleEntry[]): string {
  const normalized = normalizeEntries(entries)
  return normalized
    .map((entry) => `${entry.index}\n${entry.start} --> ${entry.end}\n${entry.text.trim()}\n`)
    .join('\n')
    .trimEnd()
}

export function srtToPlainText(entries: SubtitleEntry[]): string {
  const texts = entries
    .map((entry) => entry.text.trim().split(/\s+/).join(' '))
    .filter(Boolean)
  return texts.join(' ').trim()
}

export function clampContent(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text
  return text.slice(0, maxChars)
}

export function parseTimestampToMs(value: string): number {
  // Accepts "HH:MM:SS,ms" or "HH:MM:SS.ms"
  const parts = value.split(/[:,]/)
  if (parts.length < 4) return Number.POSITIVE_INFINITY
  const [h, m, s, ms] = parts
  const hours = Number.parseInt(h, 10) || 0
  const minutes = Number.parseInt(m, 10) || 0
  const seconds = Number.parseInt(s, 10) || 0
  const millis = Number.parseInt(ms, 10) || 0
  return hours * 3600000 + minutes * 60000 + seconds * 1000 + millis
}

export function formatMsToTimestamp(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) ms = 0
  const hours = Math.floor(ms / 3600000)
  const minutes = Math.floor((ms % 3600000) / 60000)
  const seconds = Math.floor((ms % 60000) / 1000)
  const millis = Math.floor(ms % 1000)
  const pad = (n: number, size = 2) => String(n).padStart(size, '0')
  return `${pad(hours)}:${pad(minutes)}:${pad(seconds)},${pad(millis, 3)}`
}

function fixOverlapsInOrder(entries: SubtitleEntry[]): SubtitleEntry[] {
  let lastEnd = 0
  return entries.map((entry, idx) => {
    const startMs = Math.max(parseTimestampToMs(entry.start), lastEnd)
    const endMs = Math.max(parseTimestampToMs(entry.end), startMs + 1)
    lastEnd = endMs
    return {
      ...entry,
      index: idx + 1,
      start: formatMsToTimestamp(startMs),
      end: formatMsToTimestamp(endMs),
    }
  })
}

export function normalizeEntries(entries: SubtitleEntry[]): SubtitleEntry[] {
  const copy = [...entries].map((entry) => ({
    ...entry,
    index: entry.index || 0,
  }))

  return fixOverlapsInOrder(copy)
}

export function updateEntry(
  entries: SubtitleEntry[],
  targetIndex: number,
  update: Partial<SubtitleEntry>,
): SubtitleEntry[] {
  const updated = entries.map((entry, idx) =>
    idx === targetIndex
      ? {
          ...entry,
          ...update,
        }
      : entry,
  )
  return normalizeEntries(updated)
}

export function insertEntryAfter(entries: SubtitleEntry[], afterIndex: number): SubtitleEntry[] {
  const base = entries[afterIndex] ?? entries[entries.length - 1]
  const following = entries[afterIndex + 1]
  const baseEnd = base ? parseTimestampToMs(base.end) : 0
  const nextStart = following ? parseTimestampToMs(following.start) : baseEnd + 4000
  const pad = 200

  let startMs = baseEnd + pad
  if (nextStart - startMs < 400) {
    startMs = Math.max(baseEnd + 50, nextStart - 1200)
  }
  let endMs = Math.min(startMs + 1500, nextStart - 50)
  if (endMs <= startMs) {
    endMs = startMs + 300
  }
  if (endMs >= nextStart) {
    endMs = Math.max(startMs + 100, nextStart - 50)
  }

  const newEntry: SubtitleEntry = {
    index: afterIndex + 2,
    start: formatMsToTimestamp(startMs),
    end: formatMsToTimestamp(endMs),
    text: 'Novo segmento',
  }

  const result = [...entries.slice(0, afterIndex + 1), newEntry, ...entries.slice(afterIndex + 1)]
  return normalizeEntries(result)
}

export function insertEntryBefore(entries: SubtitleEntry[], beforeIndex: number): SubtitleEntry[] {
  const base = entries[beforeIndex] ?? entries[0]
  const prev = entries[beforeIndex - 1]
  const baseStart = base ? parseTimestampToMs(base.start) : 0
  const prevEnd = prev ? parseTimestampToMs(prev.end) : 0
  const nextStart = baseStart || prevEnd + 4000
  const pad = 200

  let endMs = Math.max(prevEnd + 500, nextStart - pad)
  let startMs = Math.max(prevEnd + 50, endMs - 1500)
  if (startMs < 0) startMs = 0
  if (endMs <= startMs) endMs = startMs + 300
  if (endMs >= nextStart) endMs = nextStart - 50

  const newEntry: SubtitleEntry = {
    index: beforeIndex + 1,
    start: formatMsToTimestamp(startMs),
    end: formatMsToTimestamp(endMs),
    text: 'Novo segmento',
  }

  const result = [...entries.slice(0, beforeIndex), newEntry, ...entries.slice(beforeIndex)]
  return normalizeEntries(result)
}

export function deleteEntry(entries: SubtitleEntry[], targetIndex: number): SubtitleEntry[] {
  const filtered = entries.filter((_, idx) => idx !== targetIndex)
  return normalizeEntries(filtered)
}
