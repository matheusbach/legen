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
import CryptoJS from 'crypto-js'

type Engine = 'google' | 'gemini'

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
type UiLanguage = 'pt' | 'en' | 'es'
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

function resolveSystemUiLanguage(): UiLanguage {
  if (typeof navigator === 'undefined') return 'en'
  const raw = (navigator.language || 'en').toLowerCase().trim()
  if (!raw) return 'en'
  if (raw.startsWith('pt')) return 'pt'
  if (raw.startsWith('es')) return 'es'
  return 'en'
}

function App() {
  const defaultLanguage = useMemo(() => resolveSystemLanguage(), [])
  const defaultUiLanguage = useMemo(() => resolveSystemUiLanguage(), [])
  const [rawInput, setRawInput] = useState('')
  const [sourceName, setSourceName] = useState('SRT')
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
  const [uiLanguage, setUiLanguage] = useState<UiLanguage>(defaultUiLanguage)
  const [isMobileLayout, setIsMobileLayout] = useState(false)
  const [isSrtInputOpen, setIsSrtInputOpen] = useState(true)
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [passwordInput, setPasswordInput] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [authError, setAuthError] = useState('')

  // SHA-256 hash of the correct password.
  // To change the password, generate a new hash: CryptoJS.SHA256('your_new_password').toString()
  const EXPECTED_HASH = 'f215ff43b4cf7b87ec15602b499e6f1454d2d9c04566415b89389f25b8d13f4c'
  const AUTH_PASSWORD_CACHE_KEY = 'legen_auth_password'

  const i18n = useMemo(
    () => ({
      pt: {
        appEyebrow: 'LeGen Web',
        appTitle: 'Traduza, resuma e edite legendas em segundos.',
        appLead:
          'Importe seu SRT e use tradução com Google ou Gemini, gere TLTW com um clique e faça ajustes finos no editor.',
        importSrt: 'Importar SRT',
        segmentsCount: 'segmentos',
        segmentsDetected: 'segmentos detectados.',
        noFile: 'Nenhum arquivo',
        modes: 'Modos',
        modeTranslate: 'Tradução',
        modeTranslateDesc: 'Google / Gemini',
        modeTranslateInfo: 'Combina sentenças, traduz e remonta sem perder linha do tempo.',
        modeConverter: 'Converter',
        modeConverterDesc: 'SRT → TXT',
        modeTltw: 'TLTW',
        modeTltwDesc: 'Gemini 2.5 Flash',
        modeEditor: 'Editor',
        modeEditorDesc: 'Segmentos e tempos',
        srtInput: 'Entrada SRT',
        fileLabel: 'Arquivo',
        waitingFile: 'Aguardando arquivo',
        showSrtInput: 'Mostrar entrada',
        hideSrtInput: 'Recolher entrada',
        pasteSrt: 'Cole o SRT aqui...',
        ready: 'Pronto',
        loadingFile: 'Carregando arquivo...',
        processingSrt: 'Processando SRT...',
        applyingEdit: 'Aplicando edição...',
        translating: 'Traduzindo...',
        processingGemini: 'Processando Gemini...',
        copied: 'Copiado para a área de transferência.',
        editedApplied: 'Legenda editada aplicada.',
        preparingBatches: 'Preparando lotes...',
        processingBatch: 'Processando lote',
        assemblingSubtitle: 'Montando legenda',
        loadingCaptions: 'Legendas carregadas',
        cantReadSrt: 'Não foi possível ler o SRT.',
        translationDone: 'Tradução concluída.',
        completed: 'Concluído',
        canceled: 'Cancelar',
        canceledState: 'Cancelado',
        previous: 'Anterior',
        next: 'Próxima',
        page: 'Página',
        translationCanceled: 'Tradução cancelada.',
        unknownTranslateError: 'Erro desconhecido ao traduzir.',
        generatingTltw: 'Gerando TLTW...',
        preparingPrompt: 'Preparando prompt...',
        sendingGemini: 'Enviando para Gemini...',
        tltwSuccess: 'TLTW gerado com sucesso.',
        tltwCanceled: 'TLTW cancelado.',
        tltwError: 'Erro ao gerar TLTW.',
        tltwPhaseError: 'Erro ao gerar',
        result: 'Resultado',
        interfaceLanguage: 'Idioma da interface',
        outputLanguage: 'Idioma de saída',
        language: 'Idioma',
        downloadSrt: 'Baixar SRT',
        startTranslation: 'Traduzir',
        engineLabel: 'Engine',
        engineGoogle: 'Google Translate (gratuito)',
        engineGemini: 'Gemini (preciso)',
        isoHint: 'Digite um código ISO (ex.: en, pt, es) ou selecione na lista.',
        geminiApiKey: 'Gemini API key',
        geminiModel: 'Modelo Gemini',
        advancedSettings: 'Configurações avançadas',
        temperature: 'Temperatura',
        topP: 'Top P',
        topK: 'Top K',
        maxOutputTokensAuto: 'Max output tokens (0 = auto)',
        maxOutputTokens: 'Max output tokens',
        thinkingMode: 'Thinking mode',
        thinkingBudget: 'Thinking budget (-1 = auto)',
        additionalPrompt: 'Instruções adicionais de prompt',
        additionalPromptTranslatePlaceholder:
          'Opcional: regras extras, tom, terminologia específica...',
        additionalPromptTltwPlaceholder:
          'Opcional: regras extras, foco, estilo de escrita...',
        additionalPromptTranslateHint:
          'Essas instruções são adicionadas ao prompt do Gemini sem alterar o formato JSON.',
        additionalPromptTltwHint:
          'Essas instruções são anexadas ao prompt do TLTW e não alteram a estrutura exigida.',
        debugPromptSent: 'Debug: prompt enviado',
        debugGeminiRetry: 'Debug: Gemini retry',
        googlePublicHint: 'Usa endpoint público sem chave. Evite abusos para não ser limitado.',
        translationPlaceholder: 'Tradução aparecerá aqui...',
        loadSrtToViewSegments: 'Carregue um SRT para visualizar os segmentos.',
        converterLabel: 'Conversão',
        converterTitle: 'Extrair texto puro (SRT → TXT)',
        converterDesc: 'Sem tempos, pronto para notas ou scripts.',
        downloadTxt: 'Baixar TXT',
        copy: 'Copiar',
        loadSrtToGenerateTxt: 'Carregue um SRT para gerar o TXT.',
        tltwLabel: 'TLTW',
        tltwTitle: 'Resumo rápido para entender o vídeo sem assistir tudo',
        tltwDesc: 'Gera um TLTW claro com pontos principais, insights e próximos passos.',
        generateTltw: 'Gerar TLTW',
        downloadMd: 'Baixar MD',
        maxSrtChars: 'Max chars do SRT',
        tltwStructureHint:
          'O TLTW segue a estrutura Title, Tags, Key Points, Actions e Summary.',
        editorLabel: 'Editor',
        editorTitle: 'Segmentos, tempos e texto',
        editorDesc: 'Inserir/remover reindexa e evita sobreposição.',
        downloadSubtitle: 'Baixar legenda',
        apply: 'Aplicar',
        insertSegment: 'Inserir segmento',
        above: 'Acima',
        below: 'Abaixo',
        remove: 'Remover',
        start: 'Início',
        end: 'Fim',
        text: 'Texto',
        promptDebugTranslate: 'Prompt Gemini (Tradução)',
        promptDebugTltw: 'Prompt Gemini (TLTW)',
        authTitle: 'Acesso Restrito',
        authDesc: 'Esta ferramenta é exclusiva para apoiadores do projeto.',
        authPlaceholder: 'Digite a senha de acesso',
        authButton: 'Acessar',
        authError: 'Senha incorreta. Tente novamente.',
        authShowPassword: 'Mostrar senha',
        authHidePassword: 'Ocultar senha',
        authTelegramHint: 'A senha está divulgada na mensagem fixada no grupo de apoiadores do Telegram.',
        authGetPassword: 'Como obter a senha?',
        authGetPasswordDesc:
          'A senha é disponibilizada para apoiadores do projeto no grupo de apoiadores no Telegram.',
        authGetPasswordCta: 'Quero entrar no grupo de apoiadores',
      },
      en: {
        appEyebrow: 'LeGen Web',
        appTitle: 'Translate, summarize, and edit subtitles in seconds.',
        appLead:
          'Import your SRT and use Google or Gemini translation, generate a TLTW summary in one click, and fine-tune lines in the editor.',
        importSrt: 'Import SRT',
        segmentsCount: 'segments',
        segmentsDetected: 'segments detected.',
        noFile: 'No file',
        modes: 'Modes',
        modeTranslate: 'Translation',
        modeTranslateDesc: 'Google / Gemini',
        modeTranslateInfo: 'Combines sentences, translates, and rebuilds subtitles without losing timing.',
        modeConverter: 'Converter',
        modeConverterDesc: 'SRT → TXT',
        modeTltw: 'TLTW',
        modeTltwDesc: 'Gemini 2.5 Flash',
        modeEditor: 'Editor',
        modeEditorDesc: 'Segments and timings',
        srtInput: 'SRT input',
        fileLabel: 'File',
        waitingFile: 'Waiting for file',
        showSrtInput: 'Show input',
        hideSrtInput: 'Hide input',
        pasteSrt: 'Paste SRT here...',
        ready: 'Ready',
        loadingFile: 'Loading file...',
        processingSrt: 'Processing SRT...',
        applyingEdit: 'Applying edit...',
        translating: 'Translating...',
        processingGemini: 'Processing Gemini...',
        copied: 'Copied to clipboard.',
        editedApplied: 'Edited subtitle applied.',
        preparingBatches: 'Preparing batches...',
        processingBatch: 'Processing batch',
        assemblingSubtitle: 'Building subtitle',
        loadingCaptions: 'Subtitles loaded',
        cantReadSrt: 'Could not read SRT.',
        translationDone: 'Translation completed.',
        completed: 'Completed',
        canceled: 'Cancel',
        canceledState: 'Canceled',
        previous: 'Previous',
        next: 'Next',
        page: 'Page',
        translationCanceled: 'Translation canceled.',
        unknownTranslateError: 'Unknown translation error.',
        generatingTltw: 'Generating TLTW...',
        preparingPrompt: 'Preparing prompt...',
        sendingGemini: 'Sending to Gemini...',
        tltwSuccess: 'TLTW generated successfully.',
        tltwCanceled: 'TLTW canceled.',
        tltwError: 'Error generating TLTW.',
        tltwPhaseError: 'Generation error',
        result: 'Result',
        interfaceLanguage: 'Interface language',
        outputLanguage: 'Output language',
        language: 'Language',
        downloadSrt: 'Download SRT',
        startTranslation: 'Translate',
        engineLabel: 'Engine',
        engineGoogle: 'Google Translate (free)',
        engineGemini: 'Gemini (accurate)',
        isoHint: 'Type an ISO code (e.g., en, pt, es) or choose from the list.',
        geminiApiKey: 'Gemini API key',
        geminiModel: 'Gemini model',
        advancedSettings: 'Advanced settings',
        temperature: 'Temperature',
        topP: 'Top P',
        topK: 'Top K',
        maxOutputTokensAuto: 'Max output tokens (0 = auto)',
        maxOutputTokens: 'Max output tokens',
        thinkingMode: 'Thinking mode',
        thinkingBudget: 'Thinking budget (-1 = auto)',
        additionalPrompt: 'Additional prompt instructions',
        additionalPromptTranslatePlaceholder:
          'Optional: extra rules, tone, specific terminology...',
        additionalPromptTltwPlaceholder:
          'Optional: extra rules, focus, writing style...',
        additionalPromptTranslateHint:
          'These instructions are appended to the Gemini prompt without changing JSON format.',
        additionalPromptTltwHint:
          'These instructions are appended to the TLTW prompt and do not change the required structure.',
        debugPromptSent: 'Debug: prompt sent',
        debugGeminiRetry: 'Debug: Gemini retry',
        googlePublicHint: 'Uses a public endpoint without API key. Avoid abuse to prevent limits.',
        translationPlaceholder: 'Translation will appear here...',
        loadSrtToViewSegments: 'Load an SRT file to view segments.',
        converterLabel: 'Converter',
        converterTitle: 'Extract plain text (SRT → TXT)',
        converterDesc: 'No timestamps, ready for notes or scripts.',
        downloadTxt: 'Download TXT',
        copy: 'Copy',
        loadSrtToGenerateTxt: 'Load an SRT file to generate TXT.',
        tltwLabel: 'TLTW',
        tltwTitle: 'Quick summary to understand the video without watching all of it',
        tltwDesc: 'Generates a clear TLTW with key points, insights, and next steps.',
        generateTltw: 'Generate TLTW',
        downloadMd: 'Download MD',
        maxSrtChars: 'Max SRT chars',
        tltwStructureHint:
          'TLTW follows the structure: Title, Tags, Key Points, Actions and Summary.',
        editorLabel: 'Editor',
        editorTitle: 'Segments, timings, and text',
        editorDesc: 'Insert/remove operations reindex items and prevent overlap.',
        downloadSubtitle: 'Download subtitle',
        apply: 'Apply',
        insertSegment: 'Insert segment',
        above: 'Above',
        below: 'Below',
        remove: 'Remove',
        start: 'Start',
        end: 'End',
        text: 'Text',
        promptDebugTranslate: 'Gemini Prompt (Translation)',
        promptDebugTltw: 'Gemini Prompt (TLTW)',
        authTitle: 'Restricted Access',
        authDesc: 'This tool is exclusive to project supporters.',
        authPlaceholder: 'Enter access password',
        authButton: 'Access',
        authError: 'Incorrect password. Please try again.',
        authShowPassword: 'Show password',
        authHidePassword: 'Hide password',
        authTelegramHint:
          'The password is shared in the pinned message in the Telegram supporters group.',
        authGetPassword: 'How can I get the password?',
        authGetPasswordDesc:
          'The password is provided to project supporters in the Telegram supporters group.',
        authGetPasswordCta: 'Join the supporters group',
      },
      es: {
        appEyebrow: 'LeGen Web',
        appTitle: 'Traduce, resume y edita subtítulos en segundos.',
        appLead:
          'Importa tu SRT y usa traducción con Google o Gemini, genera un TLTW con un clic y ajusta cada línea en el editor.',
        importSrt: 'Importar SRT',
        segmentsCount: 'segmentos',
        segmentsDetected: 'segmentos detectados.',
        noFile: 'Ningún archivo',
        modes: 'Modos',
        modeTranslate: 'Traducción',
        modeTranslateDesc: 'Google / Gemini',
        modeTranslateInfo: 'Combina frases, traduce y recompone subtítulos sin perder el timing.',
        modeConverter: 'Convertir',
        modeConverterDesc: 'SRT → TXT',
        modeTltw: 'TLTW',
        modeTltwDesc: 'Gemini 2.5 Flash',
        modeEditor: 'Editor',
        modeEditorDesc: 'Segmentos y tiempos',
        srtInput: 'Entrada SRT',
        fileLabel: 'Archivo',
        waitingFile: 'Esperando archivo',
        showSrtInput: 'Mostrar entrada',
        hideSrtInput: 'Ocultar entrada',
        pasteSrt: 'Pega el SRT aquí...',
        ready: 'Listo',
        loadingFile: 'Cargando archivo...',
        processingSrt: 'Procesando SRT...',
        applyingEdit: 'Aplicando edición...',
        translating: 'Traduciendo...',
        processingGemini: 'Procesando Gemini...',
        copied: 'Copiado al portapapeles.',
        editedApplied: 'Subtítulo editado aplicado.',
        preparingBatches: 'Preparando lotes...',
        processingBatch: 'Procesando lote',
        assemblingSubtitle: 'Construyendo subtítulo',
        loadingCaptions: 'Subtítulos cargados',
        cantReadSrt: 'No se pudo leer el SRT.',
        translationDone: 'Traducción completada.',
        completed: 'Completado',
        canceled: 'Cancelar',
        canceledState: 'Cancelado',
        previous: 'Anterior',
        next: 'Siguiente',
        page: 'Página',
        translationCanceled: 'Traducción cancelada.',
        unknownTranslateError: 'Error desconocido al traducir.',
        generatingTltw: 'Generando TLTW...',
        preparingPrompt: 'Preparando prompt...',
        sendingGemini: 'Enviando a Gemini...',
        tltwSuccess: 'TLTW generado con éxito.',
        tltwCanceled: 'TLTW cancelado.',
        tltwError: 'Error al generar TLTW.',
        tltwPhaseError: 'Error al generar',
        result: 'Resultado',
        interfaceLanguage: 'Idioma de la interfaz',
        outputLanguage: 'Idioma de salida',
        language: 'Idioma',
        downloadSrt: 'Descargar SRT',
        startTranslation: 'Traducir',
        engineLabel: 'Motor',
        engineGoogle: 'Google Translate (gratis)',
        engineGemini: 'Gemini (preciso)',
        isoHint: 'Escribe un código ISO (ej.: en, pt, es) o elige de la lista.',
        geminiApiKey: 'Gemini API key',
        geminiModel: 'Modelo Gemini',
        advancedSettings: 'Configuración avanzada',
        temperature: 'Temperatura',
        topP: 'Top P',
        topK: 'Top K',
        maxOutputTokensAuto: 'Max output tokens (0 = auto)',
        maxOutputTokens: 'Max output tokens',
        thinkingMode: 'Thinking mode',
        thinkingBudget: 'Thinking budget (-1 = auto)',
        additionalPrompt: 'Instrucciones adicionales del prompt',
        additionalPromptTranslatePlaceholder:
          'Opcional: reglas extra, tono, terminología específica...',
        additionalPromptTltwPlaceholder:
          'Opcional: reglas extra, foco, estilo de escritura...',
        additionalPromptTranslateHint:
          'Estas instrucciones se agregan al prompt de Gemini sin cambiar el formato JSON.',
        additionalPromptTltwHint:
          'Estas instrucciones se agregan al prompt de TLTW y no cambian la estructura requerida.',
        debugPromptSent: 'Debug: prompt enviado',
        debugGeminiRetry: 'Debug: Gemini retry',
        googlePublicHint: 'Usa endpoint público sin API key. Evita abusos para no ser limitado.',
        translationPlaceholder: 'La traducción aparecerá aquí...',
        loadSrtToViewSegments: 'Carga un SRT para ver los segmentos.',
        converterLabel: 'Conversión',
        converterTitle: 'Extraer texto plano (SRT → TXT)',
        converterDesc: 'Sin tiempos, listo para notas o guiones.',
        downloadTxt: 'Descargar TXT',
        copy: 'Copiar',
        loadSrtToGenerateTxt: 'Carga un SRT para generar TXT.',
        tltwLabel: 'TLTW',
        tltwTitle: 'Resumen rápido para entender el video sin verlo completo',
        tltwDesc: 'Genera un TLTW claro con puntos clave, insights y próximos pasos.',
        generateTltw: 'Generar TLTW',
        downloadMd: 'Descargar MD',
        maxSrtChars: 'Max chars del SRT',
        tltwStructureHint:
          'TLTW sigue la estructura: Title, Tags, Key Points, Actions y Summary.',
        editorLabel: 'Editor',
        editorTitle: 'Segmentos, tiempos y texto',
        editorDesc: 'Insertar/quitar reindexa y evita superposiciones.',
        downloadSubtitle: 'Descargar subtítulo',
        apply: 'Aplicar',
        insertSegment: 'Insertar segmento',
        above: 'Arriba',
        below: 'Abajo',
        remove: 'Eliminar',
        start: 'Inicio',
        end: 'Fin',
        text: 'Texto',
        promptDebugTranslate: 'Prompt Gemini (Traducción)',
        promptDebugTltw: 'Prompt Gemini (TLTW)',
        authTitle: 'Acceso Restringido',
        authDesc: 'Esta herramienta es exclusiva para los seguidores del proyecto.',
        authPlaceholder: 'Introduce la contraseña de acceso',
        authButton: 'Acceder',
        authError: 'Contraseña incorrecta. Inténtalo de nuevo.',
        authShowPassword: 'Mostrar contraseña',
        authHidePassword: 'Ocultar contraseña',
        authTelegramHint:
          'La contraseña está publicada en el mensaje fijado del grupo de seguidores en Telegram.',
        authGetPassword: '¿Cómo obtener la contraseña?',
        authGetPasswordDesc:
          'La contraseña se comparte con los seguidores del proyecto en el grupo de Telegram de seguidores.',
        authGetPasswordCta: 'Unirme al grupo de seguidores',
      },
    }),
    [],
  )

  const t = (key: keyof (typeof i18n)['pt']) => i18n[uiLanguage][key] || i18n.en[key]

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
    const savedUiLanguage = readCache<UiLanguage>('legen_ui_language')
    const savedPassword = readCache<string>(AUTH_PASSWORD_CACHE_KEY)
    if (savedTranslate) setTranslateConfig((prev) => ({ ...prev, ...savedTranslate }))
    if (savedTltw) setTltwConfig((prev) => ({ ...prev, ...savedTltw }))
    if (savedTool) setActiveTool(savedTool)
    if (savedUiLanguage) setUiLanguage(savedUiLanguage)
    if (savedPassword) {
      const savedHash = CryptoJS.SHA256(savedPassword).toString()
      if (savedHash === EXPECTED_HASH) {
        setIsAuthenticated(true)
        setPasswordInput(savedPassword)
      }
    }
  }, [AUTH_PASSWORD_CACHE_KEY, EXPECTED_HASH])

  useEffect(() => {
    writeCache('legen_translate_config', translateConfig)
  }, [translateConfig])

  useEffect(() => {
    writeCache('legen_tltw_config', tltwConfig)
  }, [tltwConfig])

  useEffect(() => {
    writeCache('legen_active_tool', activeTool)
  }, [activeTool])

  useEffect(() => {
    writeCache('legen_ui_language', uiLanguage)
  }, [uiLanguage])

  useEffect(() => {
    if (isAuthenticated) {
      writeCache('legen_is_authenticated', true)
    }
  }, [isAuthenticated])

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return
    const media = window.matchMedia('(max-width: 1050px)')
    const apply = (matches: boolean) => {
      setIsMobileLayout(matches)
      setIsSrtInputOpen(!matches)
    }
    apply(media.matches)

    const onChange = (event: MediaQueryListEvent) => apply(event.matches)
    if (typeof media.addEventListener === 'function') {
      media.addEventListener('change', onChange)
      return () => media.removeEventListener('change', onChange)
    }

    media.addListener(onChange)
    return () => media.removeListener(onChange)
  }, [])

  const plainText = useMemo(() => (entries.length ? srtToPlainText(entries) : ''), [entries])

  const handleSourceChange = async (content: string, name?: string) => {
    setRawInput(content)
    setSourceName(name || 'SRT')
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
          setStatus(`${t('loadingCaptions')} (${parsed.length} ${t('segmentsCount')}).`)
        } catch (err) {
          setEntries([])
          setStatus('')
          setError(err instanceof Error ? err.message : t('cantReadSrt'))
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
    setStatus(t('copied'))
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
      .then(() => setStatus(t('editedApplied')))
      .finally(() => setIsApplyingEdits(false))
  }

  const runTranslation = async () => {
    if (translateInFlight.current) return
    translateInFlight.current = true
    translateAbort.current?.abort()
    translateAbort.current = new AbortController()
    setIsTranslating(true)
    setTranslateProgress(0)
    setTranslateStep(t('preparingBatches'))
    setStatus(t('translating'))
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
        onPrompt: (prompt: string) => {
          setPromptDebugTitle(t('promptDebugTranslate'))
          setPromptDebug(prompt)
        },
        onDebug: (info: string) => setGeminiDebug(info),
        signal: translateAbort.current.signal,
        onProgress: (done: number, total: number) => {
          setTranslateProgress(done / total)
          setTranslateStep(`${t('processingBatch')} ${done}/${total}`)
        },
        onPartialSrt: (text: string, pct: number) => {
          setTranslateProgress(pct)
          setTranslateStep(`${t('assemblingSubtitle')} ${Math.round(pct * 100)}%`)

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
      setStatus(t('translationDone'))
      setTranslateStep(t('completed'))
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        setStatus(t('translationCanceled'))
        setTranslateStep(t('canceledState'))
        setError(null)
        setTranslateProgress(0)
      } else {
        setError(err instanceof Error ? err.message : t('unknownTranslateError'))
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
    setStatus(t('generatingTltw'))
    setTltwProgress(0.15)
    setTltwPhase(t('preparingPrompt'))
    setError(null)
    setTltw('')
    setPromptDebug('')
    setPromptDebugTitle('')

    try {
      setTltwPhase(t('sendingGemini'))
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
        onPrompt: (prompt: string) => {
          setPromptDebugTitle(t('promptDebugTltw'))
          setPromptDebug(prompt)
        },
        onChunk: (chunkText: string) => {
          setTltw(chunkText.replace('<!-- END -->', '').trim())
        },
        signal: tltwAbort.current.signal,
      })
      setTltwProgress(1)
      setTltwPhase(t('completed'))
      setTltw(summary)
      setStatus(t('tltwSuccess'))
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        setStatus(t('tltwCanceled'))
        setError(null)
        setTltwProgress(0)
        setTltwPhase(t('canceledState'))
      } else {
        setError(err instanceof Error ? err.message : t('tltwError'))
        setStatus('')
        setTltwProgress(0)
        setTltwPhase(t('tltwPhaseError'))
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

  const handleAuthSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const hash = CryptoJS.SHA256(passwordInput).toString()
    if (hash === EXPECTED_HASH) {
      setIsAuthenticated(true)
      setAuthError('')
      writeCache(AUTH_PASSWORD_CACHE_KEY, passwordInput)
    } else {
      setAuthError(t('authError'))
    }
  }

  if (!isAuthenticated) {
    return (
      <div className="app-shell auth-shell">
        <div className="bg-gradients" aria-hidden />
        <div className="auth-container">
          <div className="auth-card panel">
            <p className="eyebrow">{t('appEyebrow')}</p>
            <h2>{t('authTitle')}</h2>
            <p className="muted">{t('authDesc')}</p>
            <div className="auth-support-box">
              <p className="label">{t('authGetPassword')}</p>
              <p className="muted small">{t('authGetPasswordDesc')}</p>
              <a
                className="auth-support-link"
                href="https://livepix.gg/legendonate/grupo-dos-apoiadores"
                target="_blank"
                rel="noreferrer"
              >
                {t('authGetPasswordCta')}
              </a>
            </div>
            
            <form onSubmit={handleAuthSubmit} className="auth-form">
              <div className="auth-password-field">
                <input
                  type={showPassword ? 'text' : 'password'}
                  className="text-input"
                  placeholder={t('authPlaceholder')}
                  value={passwordInput}
                  onChange={(e) => setPasswordInput(e.target.value)}
                  autoFocus
                />
                <button
                  type="button"
                  className="auth-eye-button"
                  onClick={() => setShowPassword((prev) => !prev)}
                  aria-label={showPassword ? t('authHidePassword') : t('authShowPassword')}
                  title={showPassword ? t('authHidePassword') : t('authShowPassword')}
                >
                  {showPassword ? '🙈' : '👁️'}
                </button>
              </div>
              <p className="muted small auth-telegram-hint">{t('authTelegramHint')}</p>
              {authError && <p className="error small">{authError}</p>}
              <button type="submit" className="primary full-width">
                {t('authButton')}
              </button>
            </form>

            <div className="auth-lang">
              <select
                className="text-input compact"
                value={uiLanguage}
                onChange={(e) => setUiLanguage(e.target.value as UiLanguage)}
              >
                <option value="pt">🇧🇷 Português</option>
                <option value="en">🇺🇸 English</option>
                <option value="es">🇪🇸 Español</option>
              </select>
            </div>
          </div>
        </div>
      </div>
    )
  }

  const hasInput = entries.length > 0

  return (
    <div className="app-shell">
      <div className="bg-gradients" aria-hidden />
      <header className="hero">
        <div className="hero-main">
          <div className="hero-copy">
            <p className="eyebrow">{t('appEyebrow')}</p>
            <h1>{t('appTitle')}</h1>
            <p className="lede">
              {t('appLead')}
            </p>
            <div className="hero-actions">
              <label className="file-button primary">
                {t('importSrt')}
                <input type="file" accept=".srt" onChange={(e) => handleFile(e.target.files?.[0])} />
              </label>
              <span className="pill">{hasInput ? `${entries.length} ${t('segmentsCount')}` : t('noFile')}</span>
            </div>
          </div>
          <label className="lang-select compact hero-lang-inline">
            <span className="muted small">{t('interfaceLanguage')}</span>
            <select
              className="text-input"
              value={uiLanguage}
              onChange={(e) => setUiLanguage(e.target.value as UiLanguage)}
            >
              <option value="pt">🇧🇷 Português</option>
              <option value="en">🇺🇸 English</option>
              <option value="es">🇪🇸 Español</option>
            </select>
          </label>
        </div>
      </header>

      <section className="workspace">
        <aside className="dock">
          <p className="label">{t('modes')}</p>
          <div className="dock-buttons">
            <button
              className={activeTool === 'translate' ? 'dock-btn active' : 'dock-btn'}
              onClick={() => setActiveTool('translate')}
            >
              {t('modeTranslate')}
              <span>{t('modeTranslateDesc')}</span>
            </button>
            <button
              className={activeTool === 'converter' ? 'dock-btn active' : 'dock-btn'}
              onClick={() => setActiveTool('converter')}
            >
              {t('modeConverter')}
              <span>{t('modeConverterDesc')}</span>
            </button>
            <button
              className={activeTool === 'tltw' ? 'dock-btn active' : 'dock-btn'}
              onClick={() => setActiveTool('tltw')}
            >
              {t('modeTltw')}
              <span>{t('modeTltwDesc')}</span>
            </button>
            <button
              className={activeTool === 'editor' ? 'dock-btn active' : 'dock-btn'}
              onClick={() => setActiveTool('editor')}
            >
              {t('modeEditor')}
              <span>{t('modeEditorDesc')}</span>
            </button>
          </div>

          <div className="panel upload compact">
            <div className="panel-top">
              <div>
                <p className="label">{t('srtInput')}</p>
                <h3>{sourceName}</h3>
                <p className="muted small">{hasInput ? `${t('fileLabel')}: ${sourceName}` : t('waitingFile')}</p>
              </div>
              {isMobileLayout ? (
                <button
                  className="ghost srt-toggle"
                  onClick={() => setIsSrtInputOpen((open) => !open)}
                >
                  {isSrtInputOpen ? t('hideSrtInput') : t('showSrtInput')}
                </button>
              ) : null}
            </div>
            {(!isMobileLayout || isSrtInputOpen) ? (
              <>
                <textarea
                  className="input-area"
                  placeholder={t('pasteSrt')}
                  value={rawInput}
                  onChange={(e) => handleSourceChange(e.target.value)}
                />
                {error ? <p className="error">{error}</p> : null}
                {!error && hasInput ? (
                  <p className="muted small">{entries.length} {t('segmentsDetected')}</p>
                ) : null}
                <div className="status-line">
                  <span className="pill subtle">{status || t('ready')}</span>
                  {(isLoadingInput || isParsing || isApplyingEdits) && (
                    <span className="pill subtle">{isLoadingInput ? t('loadingFile') : isParsing ? t('processingSrt') : t('applyingEdit')}</span>
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
                          ? translateStep || t('translating')
                          : tltwPhase || t('processingGemini')}
                      </p>
                    </div>
                  )}
                </div>
              </>
            ) : null}
          </div>
        </aside>

        <main className="surface">
          {activeTool === 'translate' && (
            <article className="tool">
              <div className="tool-head">
                <div className="head-status">
                  <div>
                    <p className="label">{t('modeTranslate')}</p>
                    <h3>{t('modeTranslateDesc')}</h3>
                    <p className="muted">
                      {t('modeTranslateInfo')}
                    </p>
                  </div>
                </div>
                <div className="tool-actions">
                  <button
                    className="primary"
                    onClick={runTranslation}
                    disabled={!hasInput || isTranslating}
                  >
                    {isTranslating ? t('translating') : t('startTranslation')}
                  </button>
                  {isTranslating ? (
                    <button className="ghost" onClick={() => translateAbort.current?.abort()}>
                      {t('canceled')}
                    </button>
                  ) : null}
                  <button
                    className="ghost"
                    onClick={() => translatedSrt && downloadText(translatedSrt, 'translated.srt')}
                    disabled={!translatedSrt}
                  >
                    {t('downloadSrt')}
                  </button>
                </div>
              </div>

              {(isTranslating || (translateProgress > 0 && translateProgress < 1)) && (
                <div className="progress-line">
                  <div className="progress">
                    <div style={{ width: `${Math.round(translateProgress * 100)}%` }} />
                  </div>
                  <p className="muted small">{translateStep || t('translating')}</p>
                </div>
              )}

              <div className="grid two">
                <div>
                  <p className="muted small">{t('engineLabel')}</p>
                  <div className="chips">
                    <button
                      className={translateConfig.engine === 'google' ? 'chip active' : 'chip'}
                      onClick={() => setTranslateConfig({ ...translateConfig, engine: 'google' })}
                    >
                      {t('engineGoogle')}
                    </button>
                    <button
                      className={translateConfig.engine === 'gemini' ? 'chip active' : 'chip'}
                      onClick={() => setTranslateConfig({ ...translateConfig, engine: 'gemini' })}
                    >
                      {t('engineGemini')}
                    </button>
                  </div>
                </div>
                <div>
                  <p className="muted small">{t('outputLanguage')}</p>
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
                  <p className="muted small">{t('isoHint')}</p>
                </div>
              </div>

              {translateConfig.engine === 'gemini' ? (
                <>
                  <div className="grid two">
                    <div>
                      <p className="muted small">{t('geminiApiKey')}</p>
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
                      <p className="muted small">{t('geminiModel')}</p>
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
                    <summary>{t('advancedSettings')}</summary>
                    <div className="grid two">
                      <label>
                        <span className="muted small">{t('temperature')}</span>
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
                        <span className="muted small">{t('topP')}</span>
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
                        <span className="muted small">{t('topK')}</span>
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
                        <span className="muted small">{t('maxOutputTokensAuto')}</span>
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
                        <span>{t('thinkingMode')}</span>
                      </label>
                      <label>
                        <span className="muted small">{t('thinkingBudget')}</span>
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
                      <summary>{t('additionalPrompt')}</summary>
                      <textarea
                        className="text-input"
                        rows={4}
                        placeholder={t('additionalPromptTranslatePlaceholder')}
                        value={translateConfig.geminiAdditionalPrompt}
                        onChange={(e) =>
                          setTranslateConfig({
                            ...translateConfig,
                            geminiAdditionalPrompt: e.target.value,
                          })
                        }
                      />
                      <p className="muted small">
                        {t('additionalPromptTranslateHint')}
                      </p>
                    </details>
                  </details>
                  {promptDebug && promptDebugTitle === t('promptDebugTranslate') ? (
                    <details className="prompt-details">
                      <summary>{t('debugPromptSent')}</summary>
                      <pre className="prompt-debug">{promptDebug}</pre>
                    </details>
                  ) : null}
                  {geminiDebug ? (
                    <details className="prompt-details">
                      <summary>{t('debugGeminiRetry')}</summary>
                      <pre className="prompt-debug">{geminiDebug}</pre>
                    </details>
                  ) : null}
                </>
              ) : (
                <p className="muted small">
                  {t('googlePublicHint')}
                </p>
              )}

              <p className="label">{t('result')}</p>


              <div className="pager">
                <span className="muted small">
                  {t('page')} {translatePage + 1} / {Math.max(1, Math.ceil(Math.max(1, entries.length) / PAGE_SIZE))}
                </span>
                <div className="pager-actions">
                  <button
                    className="ghost"
                    onClick={() => setTranslatePage((p) => Math.max(0, p - 1))}
                    disabled={translatePage === 0}
                  >
                    {t('previous')}
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
                    {t('next')}
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
                            placeholder={translatedEntries.length ? '' : t('translationPlaceholder')}
                          />
                        </div>
                      </div>
                    )
                  })
                ) : (
                  <p className="muted">{t('loadSrtToViewSegments')}</p>
                )}
              </div>
            </article>
          )}

          {activeTool === 'converter' && (
            <article className="tool">
              <div className="tool-head">
                <div>
                  <p className="label">{t('converterLabel')}</p>
                  <h3>{t('converterTitle')}</h3>
                  <p className="muted">{t('converterDesc')}</p>
                </div>
                <div className="tool-actions">
                  <button
                    className="ghost"
                    onClick={() => plainText && downloadText(plainText, 'captions.txt')}
                    disabled={!plainText}
                  >
                    {t('downloadTxt')}
                  </button>
                  <button className="ghost" onClick={() => copyToClipboard(plainText)} disabled={!plainText}>
                    {t('copy')}
                  </button>
                </div>
              </div>
              <div className="output">
                {plainText ? <pre>{plainText}</pre> : (
                  <p className="muted">{t('loadSrtToGenerateTxt')}</p>
                )}
              </div>
            </article>
          )}

          {activeTool === 'tltw' && (
            <article className="tool">
              <div className="tool-head">
                <div className="head-status">
                  <div>
                    <p className="label">{t('tltwLabel')}</p>
                    <h3>{t('tltwTitle')}</h3>
                    <p className="muted">{t('tltwDesc')}</p>
                  </div>
                </div>
                <div className="tool-actions">
                  <button className="primary" onClick={runTltw} disabled={!hasInput || isSummarizing}>
                    {isSummarizing ? t('generatingTltw') : t('generateTltw')}
                  </button>
                  {isSummarizing ? (
                    <button className="ghost" onClick={() => tltwAbort.current?.abort()}>
                      {t('canceled')}
                    </button>
                  ) : null}
                  <button
                    className="ghost"
                    onClick={() => tltw && downloadText(tltw, 'tltw.md')}
                    disabled={!tltw}
                  >
                    {t('downloadMd')}
                  </button>
                </div>
              </div>

              {(isSummarizing || (tltwProgress > 0 && tltwProgress < 1)) && (
                <div className="progress-line">
                  <div className="progress">
                    <div style={{ width: `${Math.round(tltwProgress * 100)}%` }} />
                  </div>
                  <p className="muted small">{tltwPhase || t('processingGemini')}</p>
                </div>
              )}

              <div className="grid three">
                <label>
                  <span className="muted small">{t('outputLanguage')}</span>
                  <input
                    className="text-input"
                    value={tltwConfig.language}
                    onChange={(e) => setTltwConfig({ ...tltwConfig, language: e.target.value.trim() })}
                  />
                </label>
                <label>
                  <span className="muted small">{t('geminiModel')}</span>
                  <input
                    className="text-input"
                    value={tltwConfig.geminiModel}
                    onChange={(e) => setTltwConfig({ ...tltwConfig, geminiModel: e.target.value })}
                  />
                </label>
                <label>
                  <span className="muted small">{t('maxSrtChars')}</span>
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
                  <span className="muted small">{t('geminiApiKey')}</span>
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
                <summary>{t('advancedSettings')}</summary>
                <div className="grid two">
                  <label>
                    <span className="muted small">{t('temperature')}</span>
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
                    <span className="muted small">{t('topP')}</span>
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
                    <span className="muted small">{t('topK')}</span>
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
                    <span className="muted small">{t('maxOutputTokens')}</span>
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
                    <span>{t('thinkingMode')}</span>
                  </label>
                  <label>
                    <span className="muted small">{t('thinkingBudget')}</span>
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
                  <summary>{t('additionalPrompt')}</summary>
                  <textarea
                    className="text-input"
                    rows={4}
                    placeholder={t('additionalPromptTltwPlaceholder')}
                    value={tltwConfig.geminiAdditionalPrompt}
                    onChange={(e) =>
                      setTltwConfig({
                        ...tltwConfig,
                        geminiAdditionalPrompt: e.target.value,
                      })
                    }
                  />
                  <p className="muted small">
                    {t('additionalPromptTltwHint')}
                  </p>
                </details>
              </details>
              {promptDebug && promptDebugTitle === t('promptDebugTltw') ? (
                <details className="prompt-details">
                  <summary>{t('debugPromptSent')}</summary>
                  <pre className="prompt-debug">{promptDebug}</pre>
                </details>
              ) : null}

              <p className="label">{t('result')}</p>
              <div className="output">
                {tltw ? (
                  <>
                    <pre>{tltw}</pre>
                    <div className="small-actions">
                      <button onClick={() => copyToClipboard(tltw)}>{t('copy')}</button>
                      <button onClick={() => downloadText(tltw, 'tltw.md')}>{t('downloadMd')}</button>
                    </div>
                  </>
                ) : (
                  <p className="muted">
                    {t('tltwStructureHint')}
                  </p>
                )}
              </div>
            </article>
          )}

          {activeTool === 'editor' && (
            <article className="tool">
              <div className="tool-head">
                <div>
                  <p className="label">{t('editorLabel')}</p>
                  <h3>{t('editorTitle')}</h3>
                  <p className="muted">{t('editorDesc')}</p>
                </div>
                <div className="tool-actions">
                  <button className="ghost" onClick={() => downloadText(formatSrt(entries), 'edited.srt')}>
                    {t('downloadSubtitle')}
                  </button>
                  <button className="primary" onClick={applyEditedCaptions}>
                    {t('apply')}
                  </button>
                </div>
              </div>
              <div className="pager">
                <span className="muted small">
                  {t('page')} {editorPage + 1} / {Math.max(1, Math.ceil(Math.max(1, entries.length) / PAGE_SIZE))}
                </span>
                <div className="pager-actions">
                  <button
                    className="ghost"
                    onClick={() => setEditorPage((p) => Math.max(0, p - 1))}
                    disabled={editorPage === 0}
                  >
                    {t('previous')}
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
                    {t('next')}
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
                          {t('insertSegment')}
                        </button>
                        {insertMenuIndex === idx ? (
                          <div className="insert-menu">
                            <button onClick={() => { addSegment(idx); setInsertMenuIndex(null) }}>{t('below')}</button>
                            <button onClick={() => { addSegmentBefore(idx); setInsertMenuIndex(null) }}>{t('above')}</button>
                          </div>
                        ) : null}
                        <button className="small danger" onClick={() => removeSegment(idx)}>
                          {t('remove')}
                        </button>
                      </div>
                    </div>
                    <div className="seg-grid">
                      <label>
                        <span>{t('start')}</span>
                        <input
                          type="text"
                          value={entry.start}
                          onChange={(e) => updateSegment(idx, { start: e.target.value })}
                        />
                      </label>
                      <label>
                        <span>{t('end')}</span>
                        <input
                          type="text"
                          value={entry.end}
                          onChange={(e) => updateSegment(idx, { end: e.target.value })}
                        />
                      </label>
                      <label className="full">
                        <span>{t('text')}</span>
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
