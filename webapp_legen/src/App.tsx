import { useEffect, useMemo, useState } from 'react'
import './App.css'
import {
  deleteEntry,
  formatSrt,
  insertEntryAfter,
  insertEntryBefore,
  parseSrt,
  srtToPlainText,
  updateEntry,
} from './lib/srt'
import type { SubtitleEntry } from './lib/srt'
import { translateCaptions } from './lib/translation'
import { generateTltwSummary } from './lib/tltw'
import { useRef } from 'react'

type Engine = 'google' | 'gemini'

              <p className="label">Resultado</p>

type TranslateConfig = {
  engine: Engine
  language: string
  geminiKey: string
  geminiModel: string
  geminiAdditionalPrompt: string
  geminiThinkingEnabled: boolean
  geminiThinkingBudget: number
  geminiTemperature: number
  geminiTopP: number
  geminiTopK: number
  geminiMaxOutputTokens: number
}

type TltwConfig = {
  language: string
  geminiKey: string
  geminiModel: string
  maxChars: number
  geminiAdditionalPrompt: string
  geminiThinkingEnabled: boolean
  geminiThinkingBudget: number
  geminiTemperature: number
  geminiTopP: number
  geminiTopK: number
  geminiMaxOutputTokens: number
}

type Tool = 'translate' | 'converter' | 'tltw' | 'editor'
const PAGE_SIZE = 200

const GOOGLE_LANGUAGES: Array<{ code: string; label: string }> = [
  { code: 'af', label: 'Afrikaans' },
  { code: 'sq', label: 'Albanian' },
  { code: 'am', label: 'Amharic' },
  { code: 'ar', label: 'Arabic' },
  { code: 'hy', label: 'Armenian' },
  { code: 'as', label: 'Assamese' },
  { code: 'ay', label: 'Aymara' },
  { code: 'az', label: 'Azerbaijani' },
  { code: 'bm', label: 'Bambara' },
  { code: 'eu', label: 'Basque' },
  { code: 'be', label: 'Belarusian' },
  { code: 'bn', label: 'Bengali' },
  { code: 'bho', label: 'Bhojpuri' },
  { code: 'bs', label: 'Bosnian' },
  { code: 'bg', label: 'Bulgarian' },
  { code: 'ca', label: 'Catalan' },
  { code: 'ceb', label: 'Cebuano' },
  { code: 'ny', label: 'Chichewa' },
  { code: 'zh-CN', label: 'Chinese (Simplified)' },
  { code: 'zh-TW', label: 'Chinese (Traditional)' },
  { code: 'co', label: 'Corsican' },
  { code: 'hr', label: 'Croatian' },
  { code: 'cs', label: 'Czech' },
  { code: 'da', label: 'Danish' },
  { code: 'dv', label: 'Dhivehi' },
  { code: 'doi', label: 'Dogri' },
  { code: 'nl', label: 'Dutch' },
  { code: 'en', label: 'English' },
  { code: 'eo', label: 'Esperanto' },
  { code: 'et', label: 'Estonian' },
  { code: 'ee', label: 'Ewe' },
  { code: 'tl', label: 'Filipino' },
  { code: 'fi', label: 'Finnish' },
  { code: 'fr', label: 'French' },
  { code: 'fy', label: 'Frisian' },
  { code: 'gl', label: 'Galician' },
  { code: 'ka', label: 'Georgian' },
  { code: 'de', label: 'German' },
  { code: 'el', label: 'Greek' },
  { code: 'gn', label: 'Guarani' },
  { code: 'gu', label: 'Gujarati' },
  { code: 'ht', label: 'Haitian Creole' },
  { code: 'ha', label: 'Hausa' },
  { code: 'haw', label: 'Hawaiian' },
  { code: 'iw', label: 'Hebrew' },
  { code: 'hi', label: 'Hindi' },
  { code: 'hmn', label: 'Hmong' },
  { code: 'hu', label: 'Hungarian' },
  { code: 'is', label: 'Icelandic' },
  { code: 'ig', label: 'Igbo' },
  { code: 'id', label: 'Indonesian' },
  { code: 'ga', label: 'Irish' },
  { code: 'it', label: 'Italian' },
  { code: 'ja', label: 'Japanese' },
  { code: 'jw', label: 'Javanese' },
  { code: 'kn', label: 'Kannada' },
  { code: 'kk', label: 'Kazakh' },
  { code: 'km', label: 'Khmer' },
  { code: 'rw', label: 'Kinyarwanda' },
  { code: 'gom', label: 'Konkani' },
  { code: 'ko', label: 'Korean' },
  { code: 'kri', label: 'Krio' },
  { code: 'ku', label: 'Kurdish (Kurmanji)' },
  { code: 'ckb', label: 'Kurdish (Sorani)' },
  { code: 'ky', label: 'Kyrgyz' },
  { code: 'lo', label: 'Lao' },
  { code: 'la', label: 'Latin' },
  { code: 'lv', label: 'Latvian' },
  { code: 'ln', label: 'Lingala' },
  { code: 'lt', label: 'Lithuanian' },
  { code: 'lg', label: 'Luganda' },
  { code: 'lb', label: 'Luxembourgish' },
  { code: 'mk', label: 'Macedonian' },
  { code: 'mai', label: 'Maithili' },
  { code: 'mg', label: 'Malagasy' },
  { code: 'ms', label: 'Malay' },
  { code: 'ml', label: 'Malayalam' },
  { code: 'mt', label: 'Maltese' },
  { code: 'mi', label: 'Maori' },
  { code: 'mr', label: 'Marathi' },
  { code: 'mni-Mtei', label: 'Meiteilon (Manipuri)' },
  { code: 'lus', label: 'Mizo' },
  { code: 'mn', label: 'Mongolian' },
  { code: 'my', label: 'Myanmar (Burmese)' },
  { code: 'ne', label: 'Nepali' },
  { code: 'no', label: 'Norwegian' },
  { code: 'or', label: 'Odia (Oriya)' },
  { code: 'om', label: 'Oromo' },
  { code: 'ps', label: 'Pashto' },
  { code: 'fa', label: 'Persian' },
  { code: 'pl', label: 'Polish' },
  { code: 'pt', label: 'Portuguese' },
  { code: 'pa', label: 'Punjabi' },
  { code: 'qu', label: 'Quechua' },
  { code: 'ro', label: 'Romanian' },
  { code: 'ru', label: 'Russian' },
  { code: 'sm', label: 'Samoan' },
  { code: 'sa', label: 'Sanskrit' },
  { code: 'gd', label: 'Scots Gaelic' },
  { code: 'nso', label: 'Sepedi' },
  { code: 'sr', label: 'Serbian' },
  { code: 'st', label: 'Sesotho' },
  { code: 'sn', label: 'Shona' },
  { code: 'sd', label: 'Sindhi' },
  { code: 'si', label: 'Sinhala' },
  { code: 'sk', label: 'Slovak' },
  { code: 'sl', label: 'Slovenian' },
  { code: 'so', label: 'Somali' },
  { code: 'es', label: 'Spanish' },
  { code: 'su', label: 'Sundanese' },
  { code: 'sw', label: 'Swahili' },
  { code: 'sv', label: 'Swedish' },
  { code: 'tg', label: 'Tajik' },
  { code: 'ta', label: 'Tamil' },
  { code: 'tt', label: 'Tatar' },
  { code: 'te', label: 'Telugu' },
  { code: 'th', label: 'Thai' },
  { code: 'ti', label: 'Tigrinya' },
  { code: 'ts', label: 'Tsonga' },
  { code: 'tr', label: 'Turkish' },
  { code: 'tk', label: 'Turkmen' },
  { code: 'ak', label: 'Twi' },
  { code: 'uk', label: 'Ukrainian' },
  { code: 'ur', label: 'Urdu' },
  { code: 'ug', label: 'Uyghur' },
  { code: 'uz', label: 'Uzbek' },
  { code: 'vi', label: 'Vietnamese' },
  { code: 'cy', label: 'Welsh' },
  { code: 'xh', label: 'Xhosa' },
  { code: 'yi', label: 'Yiddish' },
  { code: 'yo', label: 'Yoruba' },
  { code: 'zu', label: 'Zulu' },
]

function resolveSystemLanguage(): string {
  if (typeof navigator === 'undefined') return 'en'

  const raw = (navigator.language || 'en').trim()
  if (!raw) return 'en'

  const supportedCodes = new Map(
    GOOGLE_LANGUAGES.map((lang) => [lang.code.toLowerCase(), lang.code]),
  )
  const normalizedRaw = raw.toLowerCase()
  const normalizedDash = normalizedRaw.replace('_', '-')

  if (supportedCodes.has(normalizedRaw)) return supportedCodes.get(normalizedRaw) || 'en'
  if (supportedCodes.has(normalizedDash)) return supportedCodes.get(normalizedDash) || 'en'

  const [base] = normalizedDash.split('-')
  if (base && supportedCodes.has(base)) return supportedCodes.get(base) || 'en'

  return 'en'
}

function App() {
  const defaultLanguage = useMemo(() => resolveSystemLanguage(), [])
  const [rawInput, setRawInput] = useState('')
  const [sourceName, setSourceName] = useState('Aguardando SRT')
  const [entries, setEntries] = useState<SubtitleEntry[]>([])
  const [translatedEntries, setTranslatedEntries] = useState<SubtitleEntry[]>([])
  const [translatedSrt, setTranslatedSrt] = useState('')
  const [tltw, setTltw] = useState('')
  const [status, setStatus] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isTranslating, setIsTranslating] = useState(false)
  const [isSummarizing, setIsSummarizing] = useState(false)
  const [isParsing, setIsParsing] = useState(false)
  const [isLoadingInput, setIsLoadingInput] = useState(false)
  const [isApplyingEdits, setIsApplyingEdits] = useState(false)
  const [translateProgress, setTranslateProgress] = useState(0)
  const [translateStep, setTranslateStep] = useState('')
  const [tltwProgress, setTltwProgress] = useState(0)
  const [tltwPhase, setTltwPhase] = useState('')
  const [promptDebug, setPromptDebug] = useState('')
  const [promptDebugTitle, setPromptDebugTitle] = useState('')
  const [geminiDebug, setGeminiDebug] = useState('')
  const [insertMenuIndex, setInsertMenuIndex] = useState<number | null>(null)
  const [translatePage, setTranslatePage] = useState(0)
  const [editorPage, setEditorPage] = useState(0)
  const [activeTool, setActiveTool] = useState<Tool>('translate')

  const readCache = <T,>(key: string): T | null => {
    try {
      const raw = localStorage.getItem(key)
      if (!raw) return null
      return JSON.parse(raw) as T
    } catch {
      return null
    }
  }

  const writeCache = (key: string, value: unknown) => {
    try {
      localStorage.setItem(key, JSON.stringify(value))
    } catch {
      // ignore cache errors (quota or private mode)
    }
  }

  const parseTimer = useRef<number | null>(null)
  const translateAbort = useRef<AbortController | null>(null)
  const tltwAbort = useRef<AbortController | null>(null)
  const translateInFlight = useRef(false)
  const tltwInFlight = useRef(false)

  const [translateConfig, setTranslateConfig] = useState<TranslateConfig>({
    engine: 'google',
    language: defaultLanguage,
    geminiKey: '',
    geminiModel: 'gemini-2.5-flash',
    geminiAdditionalPrompt: '',
    geminiThinkingEnabled: true,
    geminiThinkingBudget: -1,
    geminiTemperature: 0.3,
    geminiTopP: 0.9,
    geminiTopK: 50,
    geminiMaxOutputTokens: 0,
  })

  const [tltwConfig, setTltwConfig] = useState<TltwConfig>({
    language: defaultLanguage,
    geminiKey: '',
    geminiModel: 'gemini-2.5-flash',
    maxChars: 120000,
    geminiAdditionalPrompt: '',
    geminiThinkingEnabled: false,
    geminiThinkingBudget: -1,
    geminiTemperature: 0.15,
    geminiTopP: 0.9,
    geminiTopK: 40,
    geminiMaxOutputTokens: 16364,
  })

  useEffect(() => {
    const savedTranslate = readCache<TranslateConfig>('legen_translate_config')
    const savedTltw = readCache<TltwConfig>('legen_tltw_config')
    const savedTool = readCache<Tool>('legen_active_tool')
    if (savedTranslate) setTranslateConfig((prev) => ({ ...prev, ...savedTranslate }))
    if (savedTltw) setTltwConfig((prev) => ({ ...prev, ...savedTltw }))
    if (savedTool) setActiveTool(savedTool)
  }, [])

  useEffect(() => {
    writeCache('legen_translate_config', translateConfig)
  }, [translateConfig])

  useEffect(() => {
    writeCache('legen_tltw_config', tltwConfig)
  }, [tltwConfig])

  useEffect(() => {
    writeCache('legen_active_tool', activeTool)
  }, [activeTool])

  const plainText = useMemo(() => (entries.length ? srtToPlainText(entries) : ''), [entries])

  const handleSourceChange = async (content: string, name?: string) => {
    setRawInput(content)
    setSourceName(name || 'Conteudo colado')
    setError(null)
    setTranslatedSrt('')
    setTranslatedEntries([])
    setTranslatePage(0)
    setEditorPage(0)

    if (parseTimer.current) {
      window.clearTimeout(parseTimer.current)
    }

    setIsParsing(true)
    await new Promise<void>((resolve) => {
      parseTimer.current = window.setTimeout(() => {
        try {
          const parsed = parseSrt(content)
          setEntries(parsed)
          setStatus(`Legendas carregadas (${parsed.length} segmentos).`)
        } catch (err) {
          setEntries([])
          setStatus('')
          setError(err instanceof Error ? err.message : 'Nao foi possivel ler o SRT.')
        } finally {
          setIsParsing(false)
          resolve()
        }
      }, 80)
    })
  }

  const handleFile = async (file?: File) => {
    if (!file) return
    setIsLoadingInput(true)
    try {
      const text = await file.text()
      await handleSourceChange(text, file.name)
    } finally {
      setIsLoadingInput(false)
    }
  }

  const downloadText = (content: string, filename: string) => {
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
  }

  const copyToClipboard = async (content: string) => {
    await navigator.clipboard.writeText(content)
    setStatus('Copiado para a area de transferencia.')
  }

  const autosizeTextarea = (el: HTMLTextAreaElement | null) => {
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }

  const applyEditedCaptions = () => {
    const content = formatSrt(entries)
    const baseName = sourceName || 'edited.srt'
    const name = baseName.startsWith('[edited]') ? baseName : `[edited] ${baseName}`
    setIsApplyingEdits(true)
    handleSourceChange(content, name)
      .then(() => setStatus('Legenda editada aplicada.'))
      .finally(() => setIsApplyingEdits(false))
  }

  const runTranslation = async () => {
    if (translateInFlight.current) return
    translateInFlight.current = true
    translateAbort.current?.abort()
    translateAbort.current = new AbortController()
    setIsTranslating(true)
    setTranslateProgress(0)
    setTranslateStep('Preparando lotes...')
    setStatus('Traduzindo...')
    setError(null)
    setTranslatedSrt('')
    setTranslatedEntries([])
    setPromptDebug('')
    setPromptDebugTitle('')
    setGeminiDebug('')

    try {
      const translatedEntries = await translateCaptions({
        entries,
        targetLanguage: translateConfig.language,
        engine: translateConfig.engine,
        geminiApiKey: translateConfig.geminiKey,
        geminiModel: translateConfig.geminiModel,
        geminiAdditionalPrompt: translateConfig.geminiAdditionalPrompt,
        geminiThinkingEnabled: translateConfig.geminiThinkingEnabled,
        geminiThinkingBudget: translateConfig.geminiThinkingBudget,
        geminiTemperature: translateConfig.geminiTemperature,
        geminiTopP: translateConfig.geminiTopP,
        geminiTopK: translateConfig.geminiTopK,
        geminiMaxOutputTokens: translateConfig.geminiMaxOutputTokens,
        onPrompt: (prompt) => {
          setPromptDebugTitle('Prompt Gemini (Tradução)')
          setPromptDebug(prompt)
        },
        onDebug: (info) => setGeminiDebug(info),
        signal: translateAbort.current.signal,
        onProgress: (done, total) => {
          setTranslateProgress(done / total)
          setTranslateStep(`Processando lote ${done}/${total}`)
        },
        onPartialSrt: (text, pct) => {
          setTranslateProgress(pct)
          setTranslateStep(`Montando legenda ${Math.round(pct * 100)}%`)

          try {
            const parsed = parseSrt(text)
            setTranslatedEntries(parsed)
          } catch {
            // ignore preview parsing errors
          }
        },
      })
      const finalSrt = formatSrt(translatedEntries)
      setTranslatedEntries(translatedEntries)
      setTranslatedSrt(finalSrt)
      setStatus('Traducao concluida.')
      setTranslateStep('Concluido')
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        setStatus('Tradução cancelada.')
        setTranslateStep('Cancelado')
        setError(null)
        setTranslateProgress(0)
      } else {
        setError(err instanceof Error ? err.message : 'Erro desconhecido ao traduzir.')
        setStatus('')
      }
    } finally {
      setIsTranslating(false)
      setTimeout(() => setTranslateProgress(0), 600)
      translateAbort.current = null
      translateInFlight.current = false
    }
  }

  const runTltw = async () => {
    if (tltwInFlight.current) return
    tltwInFlight.current = true
    tltwAbort.current?.abort()
    tltwAbort.current = new AbortController()
    setIsSummarizing(true)
    setStatus('Gerando TLTW...')
    setTltwProgress(0.15)
    setTltwPhase('Preparando prompt...')
    setError(null)
    setTltw('')
    setPromptDebug('')
    setPromptDebugTitle('')

    try {
      setTltwPhase('Enviando para Gemini...')
      setTltwProgress(0.35)

      const summary = await generateTltwSummary({
        entries,
        language: tltwConfig.language,
        apiKey: tltwConfig.geminiKey,
        model: tltwConfig.geminiModel,
        maxChars: tltwConfig.maxChars,
        additionalPrompt: tltwConfig.geminiAdditionalPrompt,
        thinkingEnabled: tltwConfig.geminiThinkingEnabled,
        thinkingBudget: tltwConfig.geminiThinkingBudget,
        temperature: tltwConfig.geminiTemperature,
        topP: tltwConfig.geminiTopP,
        topK: tltwConfig.geminiTopK,
        maxOutputTokensOverride: tltwConfig.geminiMaxOutputTokens,
        onPrompt: (prompt) => {
          setPromptDebugTitle('Prompt Gemini (TLTW)')
          setPromptDebug(prompt)
        },
        signal: tltwAbort.current.signal,
      })
      setTltwProgress(1)
      setTltwPhase('Concluido')
      setTltw(summary)
      setStatus('TLTW gerado com sucesso.')
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        setStatus('TLTW cancelado.')
        setError(null)
        setTltwProgress(0)
        setTltwPhase('Cancelado')
      } else {
        setError(err instanceof Error ? err.message : 'Erro ao gerar TLTW.')
        setStatus('')
        setTltwProgress(0)
        setTltwPhase('Erro ao gerar')
      }
    } finally {
      setIsSummarizing(false)
      tltwAbort.current = null
      tltwInFlight.current = false
    }
  }

  const updateSegment = (idx: number, patch: Partial<SubtitleEntry>) => {
    const updated = updateEntry(entries, idx, patch)
    setEntries(updated)
    setRawInput(formatSrt(updated))
  }

  const updateTranslatedSegment = (idx: number, patch: Partial<SubtitleEntry>) => {
    const base = entries.map((entry, i) => {
      const existing = translatedEntries[i]
      return existing
        ? { ...existing, index: entry.index, start: entry.start, end: entry.end }
        : { ...entry, text: '' }
    })
    const updated = base.map((entry, i) => (i === idx ? { ...entry, ...patch } : entry))
    setTranslatedEntries(updated)
    setTranslatedSrt(formatSrt(updated))
  }

  const addSegment = (idx: number) => {
    const updated = insertEntryAfter(entries, idx)
    setEntries(updated)
    setRawInput(formatSrt(updated))
  }

  const addSegmentBefore = (idx: number) => {
    const updated = insertEntryBefore(entries, idx)
    setEntries(updated)
    setRawInput(formatSrt(updated))
  }

  const removeSegment = (idx: number) => {
    const updated = deleteEntry(entries, idx)
    setEntries(updated)
    setRawInput(formatSrt(updated))
  }

  const hasInput = entries.length > 0

  return (
    <div className="app-shell">
      <div className="bg-gradients" aria-hidden />
      <header className="hero">
        <div>
          <p className="eyebrow">LeGen no navegador</p>
          <h1>Workspace modular para legendas.</h1>
          <p className="lede">
            Fluxos isolados (traducao, conversao, TLTW, edicao) com configs salvas em cookies.
            Arquitetura em “blocos” para crescer com novas ferramentas.
          </p>
          <div className="hero-actions">
            <label className="file-button primary">
              Importar SRT
              <input type="file" accept=".srt" onChange={(e) => handleFile(e.target.files?.[0])} />
            </label>
            <span className="pill">{hasInput ? `${entries.length} segmentos` : 'Nenhum arquivo'}</span>
          </div>
        </div>
      </header>

      <section className="workspace">
        <aside className="dock">
          <p className="label">Modos</p>
          <div className="dock-buttons">
            <button
              className={activeTool === 'translate' ? 'dock-btn active' : 'dock-btn'}
              onClick={() => setActiveTool('translate')}
            >
              Tradução
              <span>Google / Gemini</span>
            </button>
            <button
              className={activeTool === 'converter' ? 'dock-btn active' : 'dock-btn'}
              onClick={() => setActiveTool('converter')}
            >
              Converter
              <span>SRT → TXT</span>
            </button>
            <button
              className={activeTool === 'tltw' ? 'dock-btn active' : 'dock-btn'}
              onClick={() => setActiveTool('tltw')}
            >
              TLTW
              <span>Gemini 2.5 Flash</span>
            </button>
            <button
              className={activeTool === 'editor' ? 'dock-btn active' : 'dock-btn'}
              onClick={() => setActiveTool('editor')}
            >
              Editor
              <span>Segmentos e tempos</span>
            </button>
          </div>

          <div className="panel upload compact">
            <div className="panel-top">
              <div>
                <p className="label">Entrada SRT</p>
                <h3>{sourceName}</h3>
                <p className="muted small">{hasInput ? `Arquivo: ${sourceName}` : 'Aguardando arquivo'}</p>
              </div>
            </div>
            <textarea
              className="input-area"
              placeholder="Cole o SRT aqui..."
              value={rawInput}
              onChange={(e) => handleSourceChange(e.target.value)}
            />
            {error ? <p className="error">{error}</p> : null}
            {!error && hasInput ? (
              <p className="muted small">{entries.length} segmentos detectados.</p>
            ) : null}
            <div className="status-line">
              <span className="pill subtle">{status || 'Pronto'}</span>
              {(isLoadingInput || isParsing || isApplyingEdits) && (
                <span className="pill subtle">{isLoadingInput ? 'Carregando arquivo...' : isParsing ? 'Processando SRT...' : 'Aplicando edição...'}</span>
              )}
              {(isTranslating || isSummarizing) && (
                <div className="progress-line compact">
                  <div className="progress">
                    <div
                      style={{
                        width: `${Math.round((isTranslating ? translateProgress : tltwProgress) * 100)}%`,
                      }}
                    />
                  </div>
                  <p className="muted small">
                    {isTranslating
                      ? translateStep || 'Traduzindo...'
                      : tltwPhase || 'Processando Gemini...'}
                  </p>
                </div>
              )}
            </div>
          </div>
        </aside>

        <main className="surface">
          {activeTool === 'translate' && (
            <article className="tool">
              <div className="tool-head">
                <div className="head-status">
                  <div>
                    <p className="label">Tradução</p>
                    <h3>Batching estilo CLI, Google free ou Gemini</h3>
                    <p className="muted">
                      Combina sentenças, traduz e remonta sem perder linha do tempo. Configurações
                      salvas em cookies.
                    </p>
                  </div>
                </div>
                <div className="tool-actions">
                  <button
                    className="primary"
                    onClick={runTranslation}
                    disabled={!hasInput || isTranslating}
                  >
                    {isTranslating ? 'Traduzindo...' : 'Traduzir'}
                  </button>
                  {isTranslating ? (
                    <button className="ghost" onClick={() => translateAbort.current?.abort()}>
                      Cancelar
                    </button>
                  ) : null}
                  <button
                    className="ghost"
                    onClick={() => translatedSrt && downloadText(translatedSrt, 'translated.srt')}
                    disabled={!translatedSrt}
                  >
                    Baixar SRT
                  </button>
                </div>
              </div>

              {(isTranslating || (translateProgress > 0 && translateProgress < 1)) && (
                <div className="progress-line">
                  <div className="progress">
                    <div style={{ width: `${Math.round(translateProgress * 100)}%` }} />
                  </div>
                  <p className="muted small">{translateStep || 'Traduzindo...'}</p>
                </div>
              )}

              <div className="grid two">
                <div>
                  <p className="muted small">Engine</p>
                  <div className="chips">
                    <button
                      className={translateConfig.engine === 'google' ? 'chip active' : 'chip'}
                      onClick={() => setTranslateConfig({ ...translateConfig, engine: 'google' })}
                    >
                      Google Translate (gratuito)
                    </button>
                    <button
                      className={translateConfig.engine === 'gemini' ? 'chip active' : 'chip'}
                      onClick={() => setTranslateConfig({ ...translateConfig, engine: 'gemini' })}
                    >
                      Gemini (preciso)
                    </button>
                  </div>
                </div>
                <div>
                  <p className="muted small">Idioma destino</p>
                  <input
                    className="text-input"
                    list="google-language-list"
                    value={translateConfig.language}
                    onChange={(e) =>
                      setTranslateConfig({ ...translateConfig, language: e.target.value.trim() })
                    }
                  />
                  <datalist id="google-language-list">
                    {GOOGLE_LANGUAGES.map((lang) => (
                      <option key={lang.code} value={lang.code}>
                        {lang.label}
                      </option>
                    ))}
                  </datalist>
                  <p className="muted small">Digite um código ISO (ex.: en, pt, es) ou selecione na lista.</p>
                </div>
              </div>

              {translateConfig.engine === 'gemini' ? (
                <>
                  <div className="grid two">
                    <div>
                      <p className="muted small">Gemini API key</p>
                      <input
                        className="text-input"
                        placeholder="AIza..."
                        value={translateConfig.geminiKey}
                        onChange={(e) =>
                          setTranslateConfig({ ...translateConfig, geminiKey: e.target.value })
                        }
                      />
                    </div>
                    <div>
                      <p className="muted small">Modelo Gemini</p>
                      <input
                        className="text-input"
                        value={translateConfig.geminiModel}
                        onChange={(e) =>
                          setTranslateConfig({ ...translateConfig, geminiModel: e.target.value })
                        }
                      />
                    </div>
                  </div>
                  <details className="advanced-details">
                    <summary>Configurações avançadas</summary>
                    <div className="grid two">
                      <label>
                        <span className="muted small">Temperatura</span>
                        <input
                          className="text-input"
                          type="number"
                          step="0.01"
                          min={0}
                          max={2}
                          value={translateConfig.geminiTemperature}
                          onChange={(e) =>
                            setTranslateConfig({
                              ...translateConfig,
                              geminiTemperature: Number(e.target.value) || 0,
                            })
                          }
                        />
                      </label>
                      <label>
                        <span className="muted small">Top P</span>
                        <input
                          className="text-input"
                          type="number"
                          step="0.01"
                          min={0}
                          max={1}
                          value={translateConfig.geminiTopP}
                          onChange={(e) =>
                            setTranslateConfig({
                              ...translateConfig,
                              geminiTopP: Number(e.target.value) || 0,
                            })
                          }
                        />
                      </label>
                    </div>
                    <div className="grid two">
                      <label>
                        <span className="muted small">Top K</span>
                        <input
                          className="text-input"
                          type="number"
                          min={0}
                          value={translateConfig.geminiTopK}
                          onChange={(e) =>
                            setTranslateConfig({
                              ...translateConfig,
                              geminiTopK: Number(e.target.value) || 0,
                            })
                          }
                        />
                      </label>
                      <label>
                        <span className="muted small">Max output tokens (0 = auto)</span>
                        <input
                          className="text-input"
                          type="number"
                          min={256}
                          value={translateConfig.geminiMaxOutputTokens}
                          onChange={(e) =>
                            setTranslateConfig({
                              ...translateConfig,
                              geminiMaxOutputTokens: Number(e.target.value) || 0,
                            })
                          }
                        />
                      </label>
                    </div>
                    <div className="grid two">
                      <label className="toggle">
                        <input
                          type="checkbox"
                          checked={translateConfig.geminiThinkingEnabled}
                          onChange={(e) =>
                            setTranslateConfig({
                              ...translateConfig,
                              geminiThinkingEnabled: e.target.checked,
                            })
                          }
                        />
                        <span>Thinking mode</span>
                      </label>
                      <label>
                        <span className="muted small">Thinking budget (-1 = auto)</span>
                        <input
                          className="text-input"
                          type="number"
                          min={-1}
                          value={translateConfig.geminiThinkingBudget}
                          onChange={(e) =>
                            setTranslateConfig({
                              ...translateConfig,
                              geminiThinkingBudget: Number(e.target.value) || 0,
                            })
                          }
                          disabled={!translateConfig.geminiThinkingEnabled}
                        />
                      </label>
                    </div>
                    <details className="prompt-details">
                      <summary>Instruções adicionais de prompt</summary>
                      <textarea
                        className="text-input"
                        rows={4}
                        placeholder="Opcional: regras extras, tom, terminologia específica..."
                        value={translateConfig.geminiAdditionalPrompt}
                        onChange={(e) =>
                          setTranslateConfig({
                            ...translateConfig,
                            geminiAdditionalPrompt: e.target.value,
                          })
                        }
                      />
                      <p className="muted small">
                        Essas instruções são adicionadas ao prompt do Gemini sem alterar o formato JSON.
                      </p>
                    </details>
                  </details>
                  {promptDebug && promptDebugTitle === 'Prompt Gemini (Tradução)' ? (
                    <details className="prompt-details">
                      <summary>Debug: prompt enviado</summary>
                      <pre className="prompt-debug">{promptDebug}</pre>
                    </details>
                  ) : null}
                  {geminiDebug ? (
                    <details className="prompt-details">
                      <summary>Debug: Gemini retry</summary>
                      <pre className="prompt-debug">{geminiDebug}</pre>
                    </details>
                  ) : null}
                </>
              ) : (
                <p className="muted small">
                  Usa endpoint público sem chave. Evite abusos para não ser limitado.
                </p>
              )}


              <div className="pager">
                <span className="muted small">
                  Página {translatePage + 1} / {Math.max(1, Math.ceil(Math.max(1, entries.length) / PAGE_SIZE))}
                </span>
                <div className="pager-actions">
                  <button
                    className="ghost"
                    onClick={() => setTranslatePage((p) => Math.max(0, p - 1))}
                    disabled={translatePage === 0}
                  >
                    Anterior
                  </button>
                  <button
                    className="ghost"
                    onClick={() =>
                      setTranslatePage((p) =>
                        p + 1 < Math.ceil(entries.length / PAGE_SIZE) ? p + 1 : p,
                      )
                    }
                    disabled={translatePage + 1 >= Math.ceil(entries.length / PAGE_SIZE)}
                  >
                    Próxima
                  </button>
                </div>
              </div>

              <div className="segment-list">
                {hasInput ? (
                  entries
                    .slice(translatePage * PAGE_SIZE, translatePage * PAGE_SIZE + PAGE_SIZE)
                    .map((entry, localIdx) => {
                      const idx = translatePage * PAGE_SIZE + localIdx
                      const translated = translatedEntries[idx]?.text || ''
                    return (
                      <div className="segment-row" key={entry.index}>
                        <div className="seg-header">
                          <span className="pill">#{entry.index}</span>
                          <span className="muted small">
                            {entry.start} → {entry.end}
                          </span>
                        </div>

                        <div className="seg-dual">
                          <div className="seg-dual-original">{entry.text}</div>
                          <textarea
                            ref={autosizeTextarea}
                            className="seg-dual-textarea"
                            value={translated}
                            onChange={(e) => {
                              updateTranslatedSegment(idx, { text: e.target.value })
                              autosizeTextarea(e.target)
                            }}
                            placeholder={translatedEntries.length ? '' : 'Tradução aparecerá aqui...'}
                          />
                        </div>
                      </div>
                    )
                  })
                ) : (
                  <p className="muted">Carregue um SRT para visualizar os segmentos.</p>
                )}
              </div>
            </article>
          )}

          {activeTool === 'converter' && (
            <article className="tool">
              <div className="tool-head">
                <div>
                  <p className="label">Conversão</p>
                  <h3>Extrair texto puro (SRT → TXT)</h3>
                  <p className="muted">Sem tempos, pronto para notas ou scripts.</p>
                </div>
                <div className="tool-actions">
                  <button
                    className="ghost"
                    onClick={() => plainText && downloadText(plainText, 'captions.txt')}
                    disabled={!plainText}
                  >
                    Baixar TXT
                  </button>
                  <button className="ghost" onClick={() => copyToClipboard(plainText)} disabled={!plainText}>
                    Copiar
                  </button>
                </div>
              </div>
              <div className="output">
                {plainText ? <pre>{plainText}</pre> : (
                  <p className="muted">Carregue um SRT para gerar o TXT.</p>
                )}
              </div>
            </article>
          )}

          {activeTool === 'tltw' && (
            <article className="tool">
              <div className="tool-head">
                <div className="head-status">
                  <div>
                    <p className="label">TLTW</p>
                    <h3>Resumo estruturado (Gemini 2.5 Flash)</h3>
                    <p className="muted">Configs independentes, persistidas em cookie.</p>
                  </div>
                </div>
                <div className="tool-actions">
                  <button className="primary" onClick={runTltw} disabled={!hasInput || isSummarizing}>
                    {isSummarizing ? 'Gerando...' : 'Gerar TLTW'}
                  </button>
                  {isSummarizing ? (
                    <button className="ghost" onClick={() => tltwAbort.current?.abort()}>
                      Cancelar
                    </button>
                  ) : null}
                  <button
                    className="ghost"
                    onClick={() => tltw && downloadText(tltw, 'tltw.md')}
                    disabled={!tltw}
                  >
                    Baixar MD
                  </button>
                </div>
              </div>

              {(isSummarizing || (tltwProgress > 0 && tltwProgress < 1)) && (
                <div className="progress-line">
                  <div className="progress">
                    <div style={{ width: `${Math.round(tltwProgress * 100)}%` }} />
                  </div>
                  <p className="muted small">{tltwPhase || 'Processando Gemini...'}</p>
                </div>
              )}

              <div className="grid three">
                <label>
                  <span className="muted small">Idioma</span>
                  <input
                    className="text-input"
                    value={tltwConfig.language}
                    onChange={(e) => setTltwConfig({ ...tltwConfig, language: e.target.value.trim() })}
                  />
                </label>
                <label>
                  <span className="muted small">Modelo Gemini</span>
                  <input
                    className="text-input"
                    value={tltwConfig.geminiModel}
                    onChange={(e) => setTltwConfig({ ...tltwConfig, geminiModel: e.target.value })}
                  />
                </label>
                <label>
                  <span className="muted small">Max chars do SRT</span>
                  <input
                    className="text-input"
                    type="number"
                    min={1000}
                    value={tltwConfig.maxChars}
                    onChange={(e) =>
                      setTltwConfig({ ...tltwConfig, maxChars: Number(e.target.value) || 0 })
                    }
                  />
                </label>
              </div>

              <div className="grid two">
                <label>
                  <span className="muted small">Gemini API key</span>
                  <input
                    className="text-input"
                    placeholder="AIza..."
                    value={tltwConfig.geminiKey}
                    onChange={(e) => setTltwConfig({ ...tltwConfig, geminiKey: e.target.value })}
                  />
                </label>
                <div />
              </div>

              <details className="advanced-details">
                <summary>Configurações avançadas</summary>
                <div className="grid two">
                  <label>
                    <span className="muted small">Temperatura</span>
                    <input
                      className="text-input"
                      type="number"
                      step="0.01"
                      min={0}
                      max={2}
                      value={tltwConfig.geminiTemperature}
                      onChange={(e) =>
                        setTltwConfig({
                          ...tltwConfig,
                          geminiTemperature: Number(e.target.value) || 0,
                        })
                      }
                    />
                  </label>
                  <label>
                    <span className="muted small">Top P</span>
                    <input
                      className="text-input"
                      type="number"
                      step="0.01"
                      min={0}
                      max={1}
                      value={tltwConfig.geminiTopP}
                      onChange={(e) =>
                        setTltwConfig({
                          ...tltwConfig,
                          geminiTopP: Number(e.target.value) || 0,
                        })
                      }
                    />
                  </label>
                </div>
                <div className="grid two">
                  <label>
                    <span className="muted small">Top K</span>
                    <input
                      className="text-input"
                      type="number"
                      min={0}
                      value={tltwConfig.geminiTopK}
                      onChange={(e) =>
                        setTltwConfig({
                          ...tltwConfig,
                          geminiTopK: Number(e.target.value) || 0,
                        })
                      }
                    />
                  </label>
                  <label>
                    <span className="muted small">Max output tokens</span>
                    <input
                      className="text-input"
                      type="number"
                      min={256}
                      value={tltwConfig.geminiMaxOutputTokens}
                      onChange={(e) =>
                        setTltwConfig({
                          ...tltwConfig,
                          geminiMaxOutputTokens: Number(e.target.value) || 0,
                        })
                      }
                    />
                  </label>
                </div>
                <div className="grid two">
                  <label className="toggle">
                    <input
                      type="checkbox"
                      checked={tltwConfig.geminiThinkingEnabled}
                      onChange={(e) =>
                        setTltwConfig({
                          ...tltwConfig,
                          geminiThinkingEnabled: e.target.checked,
                        })
                      }
                    />
                    <span>Thinking mode</span>
                  </label>
                  <label>
                    <span className="muted small">Thinking budget (-1 = auto)</span>
                    <input
                      className="text-input"
                      type="number"
                      min={-1}
                      value={tltwConfig.geminiThinkingBudget}
                      onChange={(e) =>
                        setTltwConfig({
                          ...tltwConfig,
                          geminiThinkingBudget: Number(e.target.value) || 0,
                        })
                      }
                      disabled={!tltwConfig.geminiThinkingEnabled}
                    />
                  </label>
                </div>
                <details className="prompt-details">
                  <summary>Instruções adicionais de prompt</summary>
                  <textarea
                    className="text-input"
                    rows={4}
                    placeholder="Opcional: regras extras, foco, estilo de escrita..."
                    value={tltwConfig.geminiAdditionalPrompt}
                    onChange={(e) =>
                      setTltwConfig({
                        ...tltwConfig,
                        geminiAdditionalPrompt: e.target.value,
                      })
                    }
                  />
                  <p className="muted small">
                    Essas instruções são anexadas ao prompt do TLTW e não alteram a estrutura exigida.
                  </p>
                </details>
              </details>
              {promptDebug && promptDebugTitle === 'Prompt Gemini (TLTW)' ? (
                <details className="prompt-details">
                  <summary>Debug: prompt enviado</summary>
                  <pre className="prompt-debug">{promptDebug}</pre>
                </details>
              ) : null}

              <p className="label">Resultado</p>
              <div className="output">
                {tltw ? (
                  <>
                    <pre>{tltw}</pre>
                    <div className="small-actions">
                      <button onClick={() => copyToClipboard(tltw)}>Copiar</button>
                      <button onClick={() => downloadText(tltw, 'tltw.md')}>Baixar</button>
                    </div>
                  </>
                ) : (
                  <p className="muted">
                    O TLTW segue a estrutura Title, Tags, Key Points, Actions e Summary.
                  </p>
                )}
              </div>
            </article>
          )}

          {activeTool === 'editor' && (
            <article className="tool">
              <div className="tool-head">
                <div>
                  <p className="label">Editor</p>
                  <h3>Segmentos, tempos e texto</h3>
                  <p className="muted">
                    Inserir/remover reindexa e evita sobreposição.
                  </p>
                </div>
                <div className="tool-actions">
                  <button className="ghost" onClick={() => downloadText(formatSrt(entries), 'edited.srt')}>
                    Baixar legenda
                  </button>
                  <button className="primary" onClick={applyEditedCaptions}>
                    Aplicar
                  </button>
                </div>
              </div>
              <div className="pager">
                <span className="muted small">
                  Página {editorPage + 1} / {Math.max(1, Math.ceil(Math.max(1, entries.length) / PAGE_SIZE))}
                </span>
                <div className="pager-actions">
                  <button
                    className="ghost"
                    onClick={() => setEditorPage((p) => Math.max(0, p - 1))}
                    disabled={editorPage === 0}
                  >
                    Anterior
                  </button>
                  <button
                    className="ghost"
                    onClick={() =>
                      setEditorPage((p) =>
                        p + 1 < Math.ceil(entries.length / PAGE_SIZE) ? p + 1 : p,
                      )
                    }
                    disabled={editorPage + 1 >= Math.ceil(entries.length / PAGE_SIZE)}
                  >
                    Próxima
                  </button>
                </div>
              </div>
              <div className="segment-list">
                {entries
                  .slice(editorPage * PAGE_SIZE, editorPage * PAGE_SIZE + PAGE_SIZE)
                  .map((entry, localIdx) => {
                    const idx = editorPage * PAGE_SIZE + localIdx
                    return (
                  <div className="segment-row" key={entry.index}>
                    <div className="seg-header">
                      <span className="pill">#{entry.index}</span>
                      <div className="seg-actions insert-group">
                        <button
                          className="small"
                          onClick={() =>
                            setInsertMenuIndex((prev) => (prev === idx ? null : idx))
                          }
                        >
                          Inserir Segmento
                        </button>
                        {insertMenuIndex === idx ? (
                          <div className="insert-menu">
                            <button onClick={() => { addSegment(idx); setInsertMenuIndex(null) }}>Abaixo</button>
                            <button onClick={() => { addSegmentBefore(idx); setInsertMenuIndex(null) }}>Acima</button>
                          </div>
                        ) : null}
                        <button className="small danger" onClick={() => removeSegment(idx)}>
                          Remover
                        </button>
                      </div>
                    </div>
                    <div className="seg-grid">
                      <label>
                        <span>Início</span>
                        <input
                          type="text"
                          value={entry.start}
                          onChange={(e) => updateSegment(idx, { start: e.target.value })}
                        />
                      </label>
                      <label>
                        <span>Fim</span>
                        <input
                          type="text"
                          value={entry.end}
                          onChange={(e) => updateSegment(idx, { end: e.target.value })}
                        />
                      </label>
                      <label className="full">
                        <span>Texto</span>
                        <textarea
                          value={entry.text}
                          onChange={(e) => {
                            updateSegment(idx, { text: e.target.value })
                            autosizeTextarea(e.target)
                          }}
                          ref={autosizeTextarea}
                        />
                      </label>
                    </div>
                  </div>
                    )
                  })}
              </div>
            </article>
          )}
        </main>
      </section>
    </div>
  )
}

export default App
