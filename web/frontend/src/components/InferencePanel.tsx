'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Cpu, Settings, Loader2, MessageSquare, Send, Trash2,
  RefreshCw, X, FileText, ChevronLeft, ChevronRight,
  HardDrive, Gauge, Image as ImageIcon, Mic, Video,
  ToggleLeft, ToggleRight, Layers, Download, Monitor,
  PanelLeftClose, PanelLeft, Wrench, Code, Copy, Check,
  AlertCircle, Zap, Bot, User, Sparkles, Upload, Plus
} from 'lucide-react'

// ============================================================
// TYPES
// ============================================================

interface ModelCapabilities {
  // Input capabilities
  supports_text_input: boolean
  supports_image_input: boolean
  supports_audio_input: boolean
  supports_video_input: boolean
  
  // Output capabilities
  supports_text_output: boolean
  supports_image_output: boolean
  supports_audio_output: boolean
  supports_video_output: boolean
  
  // Backend compatibility
  supported_backends: string[]
  
  // Other capabilities
  supports_streaming: boolean
  supports_system_prompt: boolean
  supports_tool_calls: boolean
  max_context_length: number
  
  // Model type: llm, vlm, mllm, asr, tts, t2i, i2i, t2v
  model_type: 'llm' | 'vlm' | 'mllm' | 'asr' | 'tts' | 't2i' | 'i2i' | 't2v'
}

interface ContentItem {
  type: 'text' | 'image_url' | 'audio' | 'video'
  text?: string
  image_url?: { url: string }
  audio?: { data: string; format: string }
  video?: { url: string }
}

interface ToolCall {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string
  }
}

interface ToolResult {
  tool_call_id: string
  content: string
}

interface ChatMessage {
  id: string
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string | ContentItem[]
  tool_calls?: ToolCall[]
  tool_call_id?: string
  timestamp: number
  isStreaming?: boolean
}

interface LoadedAdapter {
  id: string
  name: string
  path: string
  active: boolean
}

interface InferenceStatus {
  model_loaded: boolean
  model_path: string | null
  adapter_path: string | null
  backend: string | null
  capabilities: ModelCapabilities | null
  memory_used_gb: number
  loaded_adapters: string[]
}

interface SystemMetrics {
  gpu_utilization: number | null
  gpu_memory_used: number | null
  gpu_memory_total: number | null
  available: boolean
}

interface AvailableBackends {
  transformers: boolean
  vllm: boolean
  sglang: boolean
}

type AlertType = 'error' | 'success' | 'warning' | 'info'

interface InferencePanelProps {
  systemMetrics: SystemMetrics
  onRefreshMetrics: () => void
  lockedModels?: { name: string; path: string; modality: string }[]
  onShowAlert?: (message: string, type: AlertType, title?: string) => void
}

// ============================================================
// MARKDOWN RENDERER
// ============================================================

const MarkdownRenderer = ({ content }: { content: string }) => {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null)

  const copyToClipboard = (text: string, index: number) => {
    navigator.clipboard.writeText(text)
    setCopiedIndex(index)
    setTimeout(() => setCopiedIndex(null), 2000)
  }

  const renderMarkdown = (text: string) => {
    const lines = text.split('\n')
    const elements: JSX.Element[] = []
    let inCodeBlock = false
    let codeContent = ''
    let codeLanguage = ''
    let codeBlockIndex = 0

    lines.forEach((line, i) => {
      if (line.startsWith('```')) {
        if (inCodeBlock) {
          const currentIndex = codeBlockIndex++
          elements.push(
            <div key={`code-${currentIndex}`} className="my-2 rounded-lg overflow-hidden border border-slate-200">
              <div className="flex items-center justify-between px-3 py-1.5 bg-slate-100 border-b border-slate-200">
                <span className="text-xs font-medium text-slate-600">{codeLanguage || 'code'}</span>
                <button
                  onClick={() => copyToClipboard(codeContent, currentIndex)}
                  className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-700"
                >
                  {copiedIndex === currentIndex ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                  {copiedIndex === currentIndex ? 'Copied!' : 'Copy'}
                </button>
              </div>
              <pre className="p-3 bg-slate-900 text-slate-100 text-sm overflow-x-auto">
                <code>{codeContent}</code>
              </pre>
            </div>
          )
          codeContent = ''
          codeLanguage = ''
          inCodeBlock = false
        } else {
          codeLanguage = line.slice(3).trim()
          inCodeBlock = true
        }
        return
      }

      if (inCodeBlock) {
        codeContent += (codeContent ? '\n' : '') + line
        return
      }

      if (line.startsWith('# ')) {
        elements.push(<h1 key={i} className="text-xl font-bold mt-4 mb-2">{line.slice(2)}</h1>)
      } else if (line.startsWith('## ')) {
        elements.push(<h2 key={i} className="text-lg font-semibold mt-3 mb-2">{line.slice(3)}</h2>)
      } else if (line.startsWith('### ')) {
        elements.push(<h3 key={i} className="text-base font-semibold mt-2 mb-1">{line.slice(4)}</h3>)
      } else if (line.startsWith('- ') || line.startsWith('* ')) {
        elements.push(
          <li key={i} className="ml-4 list-disc">{renderInlineMarkdown(line.slice(2))}</li>
        )
      } else if (/^\d+\.\s/.test(line)) {
        const match = line.match(/^(\d+)\.\s(.*)/)
        if (match) {
          elements.push(
            <li key={i} className="ml-4 list-decimal">{renderInlineMarkdown(match[2])}</li>
          )
        }
      } else if (line.startsWith('> ')) {
        elements.push(
          <blockquote key={i} className="border-l-4 border-blue-300 pl-3 my-2 text-slate-600 italic">
            {renderInlineMarkdown(line.slice(2))}
          </blockquote>
        )
      } else if (line.trim() === '') {
        elements.push(<div key={i} className="h-2" />)
      } else {
        elements.push(<p key={i} className="my-1">{renderInlineMarkdown(line)}</p>)
      }
    })

    return elements
  }

  const renderInlineMarkdown = (text: string) => {
    const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|__[^_]+__|_[^_]+_)/g)
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={i}>{part.slice(2, -2)}</strong>
      } else if (part.startsWith('*') && part.endsWith('*')) {
        return <em key={i}>{part.slice(1, -1)}</em>
      } else if (part.startsWith('__') && part.endsWith('__')) {
        return <strong key={i}>{part.slice(2, -2)}</strong>
      } else if (part.startsWith('_') && part.endsWith('_')) {
        return <em key={i}>{part.slice(1, -1)}</em>
      } else if (part.startsWith('`') && part.endsWith('`')) {
        return <code key={i} className="px-1.5 py-0.5 bg-slate-100 rounded text-sm font-mono text-pink-600">{part.slice(1, -1)}</code>
      }
      return part
    })
  }

  return <div className="prose prose-sm max-w-none">{renderMarkdown(content)}</div>
}

// ============================================================
// TOOL CALL DISPLAY
// ============================================================

const ToolCallDisplay = ({ toolCall, result }: { toolCall: ToolCall; result?: string }) => {
  const [expanded, setExpanded] = useState(false)
  let args: Record<string, unknown> = {}
  try {
    args = JSON.parse(toolCall.function.arguments)
  } catch {
    args = { raw: toolCall.function.arguments }
  }

  return (
    <div className="my-2 border border-amber-200 rounded-lg overflow-hidden bg-amber-50">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-3 py-2 flex items-center gap-2 text-left hover:bg-amber-100 transition-colors"
      >
        <Wrench className="w-4 h-4 text-amber-600" />
        <span className="font-medium text-amber-800">{toolCall.function.name}</span>
        <ChevronRight className={`w-4 h-4 text-amber-600 ml-auto transition-transform ${expanded ? 'rotate-90' : ''}`} />
      </button>
      {expanded && (
        <div className="px-3 pb-3 border-t border-amber-200">
          <div className="mt-2">
            <span className="text-xs font-medium text-amber-700">Arguments:</span>
            <pre className="mt-1 p-2 bg-white rounded text-xs overflow-x-auto border border-amber-200">
              {JSON.stringify(args, null, 2)}
            </pre>
          </div>
          {result && (
            <div className="mt-2">
              <span className="text-xs font-medium text-green-700">Result:</span>
              <pre className="mt-1 p-2 bg-white rounded text-xs overflow-x-auto border border-green-200">
                {result}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ============================================================
// IMAGE GALLERY COMPONENT
// ============================================================

const ImageGallery = ({ images, isUser }: { images: string[]; isUser: boolean }) => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)

  if (images.length === 0) return null

  // Single image - larger display
  if (images.length === 1) {
    return (
      <>
        <div className="mb-2">
          <img
            src={images[0]}
            alt="Content"
            onClick={() => setSelectedImage(images[0])}
            className={`max-w-[300px] max-h-[300px] rounded-xl border cursor-pointer hover:opacity-90 transition-opacity ${
              isUser ? 'border-blue-400' : 'border-slate-200'
            }`}
          />
        </div>
        {/* Lightbox */}
        {selectedImage && (
          <div
            className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
            onClick={() => setSelectedImage(null)}
          >
            <img src={selectedImage} alt="Full size" className="max-w-full max-h-full rounded-lg" />
            <button
              onClick={() => setSelectedImage(null)}
              className="absolute top-4 right-4 p-2 bg-white/20 hover:bg-white/30 rounded-full text-white"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        )}
      </>
    )
  }

  // Multiple images - grid layout
  const gridCols = images.length === 2 ? 'grid-cols-2' : images.length === 3 ? 'grid-cols-3' : 'grid-cols-2'
  
  return (
    <>
      <div className={`grid ${gridCols} gap-2 mb-2 max-w-[400px]`}>
        {images.slice(0, 4).map((img, i) => (
          <div key={i} className="relative">
            <img
              src={img}
              alt={`Image ${i + 1}`}
              onClick={() => setSelectedImage(img)}
              className={`w-full h-[120px] object-cover rounded-lg border cursor-pointer hover:opacity-90 transition-opacity ${
                isUser ? 'border-blue-400' : 'border-slate-200'
              }`}
            />
            {i === 3 && images.length > 4 && (
              <div className="absolute inset-0 bg-black/50 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">+{images.length - 4}</span>
              </div>
            )}
          </div>
        ))}
      </div>
      {/* Lightbox */}
      {selectedImage && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
          onClick={() => setSelectedImage(null)}
        >
          <div className="relative max-w-4xl max-h-full">
            <img src={selectedImage} alt="Full size" className="max-w-full max-h-[80vh] rounded-lg" />
            <button
              onClick={() => setSelectedImage(null)}
              className="absolute top-2 right-2 p-2 bg-white/20 hover:bg-white/30 rounded-full text-white"
            >
              <X className="w-6 h-6" />
            </button>
            {/* Navigation through images */}
            <div className="flex gap-2 mt-4 justify-center overflow-x-auto pb-2">
              {images.map((img, i) => (
                <img
                  key={i}
                  src={img}
                  alt={`Thumbnail ${i + 1}`}
                  onClick={(e) => { e.stopPropagation(); setSelectedImage(img) }}
                  className={`w-16 h-16 object-cover rounded cursor-pointer border-2 ${
                    selectedImage === img ? 'border-blue-500' : 'border-transparent hover:border-white/50'
                  }`}
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </>
  )
}

// ============================================================
// AUDIO PLAYER COMPONENT
// ============================================================

const AudioPlayer = ({ audioData, format, isUser }: { audioData: string; format: string; isUser: boolean }) => {
  const [isPlaying, setIsPlaying] = useState(false)
  const [duration, setDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const audioRef = useRef<HTMLAudioElement>(null)

  const audioSrc = audioData.startsWith('data:') ? audioData : `data:audio/${format};base64,${audioData}`

  const togglePlay = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause()
      } else {
        audioRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const formatTime = (time: number) => {
    const mins = Math.floor(time / 60)
    const secs = Math.floor(time % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className={`flex items-center gap-3 p-3 rounded-xl mb-2 ${
      isUser ? 'bg-blue-400/30' : 'bg-slate-100'
    }`}>
      <audio
        ref={audioRef}
        src={audioSrc}
        onLoadedMetadata={() => setDuration(audioRef.current?.duration || 0)}
        onTimeUpdate={() => setCurrentTime(audioRef.current?.currentTime || 0)}
        onEnded={() => setIsPlaying(false)}
      />
      <button
        onClick={togglePlay}
        className={`w-10 h-10 rounded-full flex items-center justify-center ${
          isUser ? 'bg-white text-blue-600' : 'bg-blue-500 text-white'
        }`}
      >
        {isPlaying ? (
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
            <rect x="6" y="4" width="4" height="16" />
            <rect x="14" y="4" width="4" height="16" />
          </svg>
        ) : (
          <svg className="w-4 h-4 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
            <polygon points="5,3 19,12 5,21" />
          </svg>
        )}
      </button>
      <div className="flex-1">
        <div className={`h-1 rounded-full ${isUser ? 'bg-white/30' : 'bg-slate-300'}`}>
          <div
            className={`h-full rounded-full ${isUser ? 'bg-white' : 'bg-blue-500'}`}
            style={{ width: `${duration ? (currentTime / duration) * 100 : 0}%` }}
          />
        </div>
        <div className={`flex justify-between text-[10px] mt-1 ${isUser ? 'text-white/70' : 'text-slate-500'}`}>
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>
      <Mic className={`w-4 h-4 ${isUser ? 'text-white/70' : 'text-slate-400'}`} />
    </div>
  )
}

// ============================================================
// VIDEO PLAYER COMPONENT
// ============================================================

const VideoPlayer = ({ videoUrl, isUser }: { videoUrl: string; isUser: boolean }) => {
  const [isFullscreen, setIsFullscreen] = useState(false)

  return (
    <>
      <div className="mb-2">
        <video
          src={videoUrl}
          controls
          className={`max-w-[350px] max-h-[250px] rounded-xl border ${
            isUser ? 'border-blue-400' : 'border-slate-200'
          }`}
          onClick={(e) => e.stopPropagation()}
        />
        <button
          onClick={() => setIsFullscreen(true)}
          className={`mt-1 text-xs flex items-center gap-1 ${
            isUser ? 'text-blue-200 hover:text-white' : 'text-slate-500 hover:text-slate-700'
          }`}
        >
          <Video className="w-3 h-3" /> Expand
        </button>
      </div>
      {/* Fullscreen modal */}
      {isFullscreen && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4"
          onClick={() => setIsFullscreen(false)}
        >
          <video
            src={videoUrl}
            controls
            autoPlay
            className="max-w-full max-h-full rounded-lg"
            onClick={(e) => e.stopPropagation()}
          />
          <button
            onClick={() => setIsFullscreen(false)}
            className="absolute top-4 right-4 p-2 bg-white/20 hover:bg-white/30 rounded-full text-white"
          >
            <X className="w-6 h-6" />
          </button>
        </div>
      )}
    </>
  )
}

// ============================================================
// MESSAGE BUBBLE
// ============================================================

const MessageBubble = ({ message, toolResults }: { message: ChatMessage; toolResults: Map<string, string> }) => {
  const isUser = message.role === 'user'
  const isSystem = message.role === 'system'
  const isTool = message.role === 'tool'
  const isAssistant = message.role === 'assistant'

  // Extract text content
  const getContent = (): string => {
    if (typeof message.content === 'string') return message.content
    const textParts = (message.content as ContentItem[]).filter(c => c.type === 'text')
    return textParts.map(c => c.text || '').join('\n')
  }

  // Extract images
  const getImages = (): string[] => {
    if (typeof message.content === 'string') return []
    return (message.content as ContentItem[])
      .filter(c => c.type === 'image_url' && c.image_url?.url)
      .map(c => c.image_url!.url)
  }

  // Extract audio
  const getAudio = (): { data: string; format: string }[] => {
    if (typeof message.content === 'string') return []
    return (message.content as ContentItem[])
      .filter(c => c.type === 'audio' && c.audio?.data)
      .map(c => ({ data: c.audio!.data, format: c.audio!.format || 'wav' }))
  }

  // Extract videos
  const getVideos = (): string[] => {
    if (typeof message.content === 'string') return []
    return (message.content as ContentItem[])
      .filter(c => c.type === 'video' && c.video?.url)
      .map(c => c.video!.url)
  }

  if (isSystem) {
    return (
      <div className="flex justify-center my-2">
        <div className="px-4 py-2 bg-slate-100 rounded-full text-xs text-slate-600 flex items-center gap-2">
          <Settings className="w-3 h-3" />
          System: {getContent().slice(0, 50)}{getContent().length > 50 ? '...' : ''}
        </div>
      </div>
    )
  }

  if (isTool) {
    return (
      <div className="flex justify-start my-2">
        <div className="max-w-[85%] px-4 py-2 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-center gap-2 text-xs text-green-700 mb-1">
            <Code className="w-3 h-3" />
            Tool Result
          </div>
          <pre className="text-xs text-green-800 overflow-x-auto">{getContent()}</pre>
        </div>
      </div>
    )
  }

  const images = getImages()
  const audioItems = getAudio()
  const videos = getVideos()
  const textContent = getContent()
  const hasMedia = images.length > 0 || audioItems.length > 0 || videos.length > 0

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} my-3`}>
      <div className={`flex gap-3 max-w-[85%] ${isUser ? 'flex-row-reverse' : ''}`}>
        {/* Avatar */}
        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
          isUser ? 'bg-blue-500' : 'bg-gradient-to-br from-purple-500 to-blue-500'
        }`}>
          {isUser ? <User className="w-4 h-4 text-white" /> : <Bot className="w-4 h-4 text-white" />}
        </div>

        {/* Content */}
        <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
          {/* Images */}
          <ImageGallery images={images} isUser={isUser} />

          {/* Videos */}
          {videos.map((videoUrl, i) => (
            <VideoPlayer key={`video-${i}`} videoUrl={videoUrl} isUser={isUser} />
          ))}

          {/* Audio */}
          {audioItems.map((audio, i) => (
            <AudioPlayer key={`audio-${i}`} audioData={audio.data} format={audio.format} isUser={isUser} />
          ))}

          {/* Text content bubble - only show if there's text or no media */}
          {(textContent || !hasMedia) && (
            <div className={`px-4 py-3 rounded-2xl ${
              isUser 
                ? 'bg-blue-500 text-white rounded-br-md' 
                : 'bg-white border border-slate-200 text-slate-900 rounded-bl-md shadow-sm'
            }`}>
              {message.isStreaming ? (
                <div className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm">{textContent || 'Thinking...'}</span>
                </div>
              ) : isUser ? (
                <p className="text-sm whitespace-pre-wrap">{textContent || '📷 Image'}</p>
              ) : (
                <MarkdownRenderer content={textContent || 'No response'} />
              )}
            </div>
          )}

          {/* Tool calls */}
          {isAssistant && message.tool_calls && message.tool_calls.length > 0 && (
            <div className="mt-2 w-full">
              {message.tool_calls.map((tc, i) => (
                <ToolCallDisplay key={i} toolCall={tc} result={toolResults.get(tc.id)} />
              ))}
            </div>
          )}

          {/* Timestamp */}
          <span className="text-[10px] text-slate-400 mt-1">
            {new Date(message.timestamp).toLocaleTimeString()}
          </span>
        </div>
      </div>
    </div>
  )
}

// ============================================================
// MAIN COMPONENT
// ============================================================

export default function InferencePanel({ systemMetrics, onRefreshMetrics, lockedModels, onShowAlert }: InferencePanelProps) {
  // Sidebar state
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [isMobile, setIsMobile] = useState(false)

  // Model state
  const [inferenceModel, setInferenceModel] = useState('')
  const [adapterPath, setAdapterPath] = useState('')
  const [isModelLoading, setIsModelLoading] = useState(false)
  const [loadingMessage, setLoadingMessage] = useState('')
  const [loadedAdapters, setLoadedAdapters] = useState<LoadedAdapter[]>([])

  // Backend state
  const [inferenceBackend, setInferenceBackend] = useState<'transformers' | 'vllm' | 'sglang'>('transformers')
  const [availableBackends, setAvailableBackends] = useState<AvailableBackends>({
    transformers: true, vllm: false, sglang: false
  })

  // Inference status
  const [inferenceStatus, setInferenceStatus] = useState<InferenceStatus>({
    model_loaded: false,
    model_path: null,
    adapter_path: null,
    backend: null,
    capabilities: null,
    memory_used_gb: 0,
    loaded_adapters: []
  })

  // Chat state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [toolResults, setToolResults] = useState<Map<string, string>>(new Map())

  // Chat settings
  const [chatMode, setChatMode] = useState<'chat' | 'completion'>('chat')
  const [systemPrompt, setSystemPrompt] = useState('')
  const [keepHistory, setKeepHistory] = useState(true)
  const [streamingEnabled, setStreamingEnabled] = useState(true)

  // Generation settings
  const [inferenceSettings, setInferenceSettings] = useState({
    max_new_tokens: 512,
    temperature: 0.7,
    top_p: 0.9,
    repetition_penalty: 1.0,
  })

  // Multimodal state
  const [uploadedImages, setUploadedImages] = useState<string[]>([])
  const [uploadedAudio, setUploadedAudio] = useState<{ data: string; format: string; name: string }[]>([])
  const [uploadedVideos, setUploadedVideos] = useState<{ url: string; name: string }[]>([])
  const [isUploadingImage, setIsUploadingImage] = useState(false)
  const [isUploadingAudio, setIsUploadingAudio] = useState(false)
  const [isUploadingVideo, setIsUploadingVideo] = useState(false)

  // Refs
  const chatEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const audioInputRef = useRef<HTMLInputElement>(null)
  const videoInputRef = useRef<HTMLInputElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)

  // Detect mobile
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 1024)
      if (window.innerWidth < 1024) {
        setSidebarCollapsed(true)
      }
    }
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  // Scroll to bottom on new messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages])

  // Fetch available backends and capabilities
  useEffect(() => {
    fetchBackends()
    fetchInferenceStatus()
  }, [])

  // Fetch model capabilities when model path changes
  useEffect(() => {
    if (inferenceModel.trim()) {
      fetchModelCapabilities(inferenceModel)
    }
  }, [inferenceModel])

  const fetchModelCapabilities = async (modelPath: string) => {
    try {
      const source = modelPath.startsWith('/') ? 'local' : 'huggingface'
      const res = await fetch(`/api/models/capabilities?model_path=${encodeURIComponent(modelPath)}&source=${source}`)
      if (res.ok) {
        const data = await res.json()
        // Update inference status with capabilities if model not loaded yet
        if (!inferenceStatus.model_loaded) {
          setInferenceStatus(prev => ({
            ...prev,
            capabilities: {
              supports_text_input: data.supports_text_input ?? true,
              supports_image_input: data.supports_image_input ?? false,
              supports_audio_input: data.supports_audio_input ?? false,
              supports_video_input: data.supports_video_input ?? false,
              supports_text_output: data.supports_text_output ?? true,
              supports_image_output: data.supports_image_output ?? false,
              supports_audio_output: data.supports_audio_output ?? false,
              supports_video_output: data.supports_video_output ?? false,
              supported_backends: data.supported_backends ?? ['transformers'],
              supports_streaming: data.supports_streaming ?? true,
              supports_system_prompt: data.supports_system_prompt ?? true,
              supports_tool_calls: data.supports_tool_calls ?? false,
              max_context_length: data.max_context_length ?? 4096,
              model_type: data.model_type ?? 'llm'
            }
          }))
        }
      }
    } catch (e) {
      console.error('Failed to fetch model capabilities:', e)
    }
  }

  const fetchBackends = async () => {
    try {
      const res = await fetch('/api/inference/backends')
      if (res.ok) {
        const data = await res.json()
        setAvailableBackends(data.backends || { transformers: true, vllm: false, sglang: false })
      }
    } catch (e) {
      console.error('Failed to fetch backends:', e)
    }
  }

  const fetchInferenceStatus = async () => {
    try {
      const res = await fetch('/api/inference/status')
      if (res.ok) {
        const data = await res.json()
        setInferenceStatus({
          model_loaded: data.model_loaded || false,
          model_path: data.model_path || null,
          adapter_path: data.adapter_path || null,
          backend: data.backend || null,
          capabilities: data.capabilities || null,
          memory_used_gb: data.memory_used_gb || 0,
          loaded_adapters: data.loaded_adapters || []
        })
        if (data.model_path) {
          setInferenceModel(data.model_path)
        }
        if (data.backend) {
          setInferenceBackend(data.backend as 'transformers' | 'vllm' | 'sglang')
        }
        if (data.available_backends) {
          setAvailableBackends(data.available_backends)
        }
        // Sync loaded adapters from backend status
        if (data.adapter_path) {
          setAdapterPath('')
          const adapterName = data.adapter_path.split('/').pop() || 'adapter'
          setLoadedAdapters(prev => {
            const alreadyLoaded = prev.some(a => a.path === data.adapter_path)
            if (alreadyLoaded) return prev
            return [{ id: Date.now().toString(), name: adapterName, path: data.adapter_path, active: true }]
          })
        } else if (data.loaded_adapters && data.loaded_adapters.length > 0) {
          setLoadedAdapters(data.loaded_adapters.map((p: string, i: number) => ({
            id: `api-${i}`,
            name: p.split('/').pop() || 'adapter',
            path: p,
            active: true
          })))
        } else {
          setLoadedAdapters([])
        }
      }
    } catch (e) {
      console.error('Failed to fetch inference status:', e)
    }
  }

  // Check if current backend supports streaming
  const supportsStreaming = useCallback(() => {
    // All backends support streaming: transformers via infer_async, vLLM, SGLang
    return true
  }, [inferenceBackend, inferenceStatus.capabilities])

  // Check if model supports image input
  const supportsImages = useCallback(() => {
    return inferenceStatus.capabilities?.supports_image_input ?? false
  }, [inferenceStatus.capabilities])

  // Check if model supports audio input
  const supportsAudio = useCallback(() => {
    return inferenceStatus.capabilities?.supports_audio_input ?? false
  }, [inferenceStatus.capabilities])

  // Check if model supports video input
  const supportsVideo = useCallback(() => {
    return inferenceStatus.capabilities?.supports_video_input ?? false
  }, [inferenceStatus.capabilities])

  // Check if selected backend is supported by the model
  const isBackendSupported = useCallback((backend: string) => {
    if (!inferenceStatus.capabilities?.supported_backends) return true
    return inferenceStatus.capabilities.supported_backends.includes(backend)
  }, [inferenceStatus.capabilities])

  // Load model
  const loadModel = async () => {
    if (!inferenceModel.trim()) return

    setIsModelLoading(true)
    setLoadingMessage('Loading model...')

    try {
      const res = await fetch('/api/inference/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_path: inferenceModel,
          adapter_path: adapterPath || undefined,
          backend: inferenceBackend
        })
      })

      const data = await res.json()
      if (data.success) {
        await fetchInferenceStatus()
        setLoadingMessage('')
      } else {
        setLoadingMessage('')
        if (onShowAlert) {
          onShowAlert(data.error || 'Failed to load model', 'error', 'Model Load Failed')
        }
      }
    } catch (e) {
      console.error('Failed to load model:', e)
      setLoadingMessage('')
      if (onShowAlert) {
        onShowAlert('An unexpected error occurred while loading the model', 'error', 'Model Load Failed')
      }
    } finally {
      setIsModelLoading(false)
    }
  }

  // Load adapter
  const loadAdapter = async () => {
    if (!adapterPath.trim() || !inferenceStatus.model_loaded) return

    setIsModelLoading(true)
    setLoadingMessage('Loading adapter...')

    try {
      const res = await fetch('/api/inference/load-adapter', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ adapter_path: adapterPath })
      })

      const data = await res.json()
      if (data.success) {
        await fetchInferenceStatus()
        setLoadedAdapters(prev => [...prev, {
          id: Date.now().toString(),
          name: adapterPath.split('/').pop() || 'adapter',
          path: adapterPath,
          active: true
        }])
        setAdapterPath('')
      } else {
        if (onShowAlert) {
          onShowAlert(data.error || 'Failed to load adapter', 'error', 'Adapter Load Failed')
        }
      }
    } catch (e) {
      console.error('Failed to load adapter:', e)
      if (onShowAlert) {
        onShowAlert('An unexpected error occurred while loading the adapter', 'error', 'Adapter Load Failed')
      }
    } finally {
      setIsModelLoading(false)
      setLoadingMessage('')
    }
  }

  // Clear memory
  const clearMemory = async () => {
    try {
      const res = await fetch('/api/inference/deep-clear-memory', { method: 'POST' })
      const data = await res.json()
      if (data.success) {
        setInferenceStatus({
          model_loaded: false,
          model_path: null,
          adapter_path: null,
          backend: null,
          capabilities: null,
          memory_used_gb: 0,
          loaded_adapters: []
        })
        setLoadedAdapters([])
        setChatMessages([])
      }
    } catch (e) {
      console.error('Failed to clear memory:', e)
    }
  }

  // Handle image upload
  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    setIsUploadingImage(true)
    const newImages: string[] = []

    for (const file of Array.from(files)) {
      if (!file.type.startsWith('image/')) continue
      
      const reader = new FileReader()
      await new Promise<void>((resolve) => {
        reader.onload = () => {
          if (typeof reader.result === 'string') {
            newImages.push(reader.result)
          }
          resolve()
        }
        reader.readAsDataURL(file)
      })
    }

    setUploadedImages(prev => [...prev, ...newImages])
    setIsUploadingImage(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  // Remove uploaded image
  const removeImage = (index: number) => {
    setUploadedImages(prev => prev.filter((_, i) => i !== index))
  }

  // Handle audio upload
  const handleAudioUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    setIsUploadingAudio(true)
    const newAudio: { data: string; format: string; name: string }[] = []

    for (const file of Array.from(files)) {
      if (!file.type.startsWith('audio/')) continue
      
      const format = file.type.split('/')[1] || 'wav'
      const reader = new FileReader()
      await new Promise<void>((resolve) => {
        reader.onload = () => {
          if (typeof reader.result === 'string') {
            newAudio.push({ data: reader.result, format, name: file.name })
          }
          resolve()
        }
        reader.readAsDataURL(file)
      })
    }

    setUploadedAudio(prev => [...prev, ...newAudio])
    setIsUploadingAudio(false)
    if (audioInputRef.current) {
      audioInputRef.current.value = ''
    }
  }

  // Remove uploaded audio
  const removeAudio = (index: number) => {
    setUploadedAudio(prev => prev.filter((_, i) => i !== index))
  }

  // Handle video upload
  const handleVideoUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    setIsUploadingVideo(true)
    const newVideos: { url: string; name: string }[] = []

    for (const file of Array.from(files)) {
      if (!file.type.startsWith('video/')) continue
      
      const reader = new FileReader()
      await new Promise<void>((resolve) => {
        reader.onload = () => {
          if (typeof reader.result === 'string') {
            newVideos.push({ url: reader.result, name: file.name })
          }
          resolve()
        }
        reader.readAsDataURL(file)
      })
    }

    setUploadedVideos(prev => [...prev, ...newVideos])
    setIsUploadingVideo(false)
    if (videoInputRef.current) {
      videoInputRef.current.value = ''
    }
  }

  // Remove uploaded video
  const removeVideo = (index: number) => {
    setUploadedVideos(prev => prev.filter((_, i) => i !== index))
  }

  // Check if any media is uploaded
  const hasUploadedMedia = uploadedImages.length > 0 || uploadedAudio.length > 0 || uploadedVideos.length > 0

  // Send message
  const sendMessage = async () => {
    if ((!inputMessage.trim() && !hasUploadedMedia) || !inferenceStatus.model_loaded || isGenerating) return

    const messageId = Date.now().toString()
    let content: string | ContentItem[]

    // Build multimodal content if any media is uploaded
    if (hasUploadedMedia) {
      const contentItems: ContentItem[] = []
      
      // Add images
      uploadedImages.forEach(img => {
        contentItems.push({
          type: 'image_url' as const,
          image_url: { url: img }
        })
      })
      
      // Add audio
      uploadedAudio.forEach(audio => {
        contentItems.push({
          type: 'audio' as const,
          audio: { data: audio.data, format: audio.format }
        })
      })
      
      // Add videos
      uploadedVideos.forEach(video => {
        contentItems.push({
          type: 'video' as const,
          video: { url: video.url }
        })
      })
      
      // Add text if present
      if (inputMessage.trim()) {
        contentItems.push({ type: 'text' as const, text: inputMessage })
      }
      
      content = contentItems
    } else {
      content = inputMessage
    }

    const userMessage: ChatMessage = {
      id: messageId,
      role: 'user',
      content,
      timestamp: Date.now()
    }

    const newMessages = keepHistory ? [...chatMessages, userMessage] : [userMessage]
    setChatMessages(newMessages)
    setInputMessage('')
    setUploadedImages([])
    setUploadedAudio([])
    setUploadedVideos([])
    setIsGenerating(true)

    // Add placeholder for assistant response
    const assistantId = (Date.now() + 1).toString()
    const assistantPlaceholder: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      isStreaming: true
    }
    setChatMessages([...newMessages, assistantPlaceholder])

    try {
      // Build messages for API
      const apiMessages = []
      
      if (systemPrompt && chatMode === 'chat') {
        apiMessages.push({ role: 'system', content: systemPrompt })
      }

      for (const msg of newMessages) {
        if (msg.role === 'system') continue // System prompt handled above
        apiMessages.push({
          role: msg.role,
          content: msg.content
        })
      }

      const shouldStream = streamingEnabled && supportsStreaming()

      if (shouldStream) {
        // Streaming request
        const res = await fetch('/api/inference/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model_path: inferenceStatus.model_path,
            messages: apiMessages,
            backend: inferenceBackend,
            max_new_tokens: inferenceSettings.max_new_tokens,
            temperature: inferenceSettings.temperature,
            top_p: inferenceSettings.top_p,
            repetition_penalty: inferenceSettings.repetition_penalty,
            stream: true
          })
        })

        const reader = res.body?.getReader()
        const decoder = new TextDecoder()
        let fullResponse = ''

        while (reader) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value)
          const lines = chunk.split('\n')

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6)
              if (data === '[DONE]') break
              fullResponse += data
              setChatMessages(prev => prev.map(m => 
                m.id === assistantId 
                  ? { ...m, content: fullResponse, isStreaming: true }
                  : m
              ))
            }
          }
        }

        setChatMessages(prev => prev.map(m => 
          m.id === assistantId 
            ? { ...m, content: fullResponse, isStreaming: false }
            : m
        ))
      } else {
        // Non-streaming request
        const res = await fetch('/api/inference/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model_path: inferenceStatus.model_path,
            messages: apiMessages,
            backend: inferenceBackend,
            max_new_tokens: inferenceSettings.max_new_tokens,
            temperature: inferenceSettings.temperature,
            top_p: inferenceSettings.top_p,
            repetition_penalty: inferenceSettings.repetition_penalty,
            stream: false
          })
        })

        const data = await res.json()
        
        if (data.success) {
          // Build multimodal content from response
          let responseContent: string | ContentItem[] = data.response || ''
          
          // Check if response contains multimodal outputs
          const hasMultimodalOutput = data.images || data.audio || data.video
          
          if (hasMultimodalOutput) {
            const contentItems: ContentItem[] = []
            
            // Add generated images
            if (data.images && Array.isArray(data.images)) {
              data.images.forEach((img: { data: string; format?: string }) => {
                contentItems.push({
                  type: 'image_url',
                  image_url: { url: img.data.startsWith('data:') ? img.data : `data:image/${img.format || 'png'};base64,${img.data}` }
                })
              })
            }
            
            // Add generated audio
            if (data.audio && data.audio.data) {
              contentItems.push({
                type: 'audio',
                audio: { 
                  data: data.audio.data.startsWith('data:') ? data.audio.data : `data:audio/${data.audio.format || 'wav'};base64,${data.audio.data}`,
                  format: data.audio.format || 'wav'
                }
              })
            }
            
            // Add generated video
            if (data.video && (data.video.data || data.video.url)) {
              contentItems.push({
                type: 'video',
                video: { 
                  url: data.video.url || (data.video.data.startsWith('data:') ? data.video.data : `data:video/${data.video.format || 'mp4'};base64,${data.video.data}`)
                }
              })
            }
            
            // Add text response if present
            if (data.response) {
              contentItems.push({ type: 'text', text: data.response })
            }
            
            responseContent = contentItems
          }
          
          setChatMessages(prev => prev.map(m => 
            m.id === assistantId 
              ? { 
                  ...m, 
                  content: responseContent, 
                  isStreaming: false,
                  tool_calls: data.tool_calls
                }
              : m
          ))
        } else {
          setChatMessages(prev => prev.map(m => 
            m.id === assistantId 
              ? { ...m, content: `Error: ${data.error || 'Failed to generate response'}`, isStreaming: false }
              : m
          ))
        }
      }
    } catch (e) {
      console.error('Failed to send message:', e)
      setChatMessages(prev => prev.map(m => 
        m.id === assistantId 
          ? { ...m, content: 'Error: Failed to connect to server', isStreaming: false }
          : m
      ))
    } finally {
      setIsGenerating(false)
    }
  }

  // Get model display name
  const getModelDisplayName = (path: string | null): string => {
    if (!path) return 'No model loaded'
    const locked = lockedModels?.find(m => m.path === path)
    if (locked) return locked.name
    return path.split('/').pop() || path
  }

  // Get backend streaming info
  const getStreamingInfo = () => {
    if (!inferenceStatus.model_loaded) return { available: false, reason: 'Load a model first' }
    // All backends support streaming
    return { available: true, reason: `${inferenceBackend} supports streaming` }
  }

  const streamingInfo = getStreamingInfo()

  return (
    <div className="flex h-[calc(100vh-180px)] min-h-[500px] bg-slate-50 rounded-xl overflow-hidden border border-slate-200 shadow-lg">
      {/* Left Sidebar - Model Settings */}
      <div className={`${sidebarCollapsed ? 'w-0 lg:w-12' : 'w-full lg:w-80'} flex-shrink-0 bg-white border-r border-slate-200 flex flex-col transition-all duration-300 overflow-hidden ${isMobile && !sidebarCollapsed ? 'absolute inset-0 z-50' : ''}`}>
        {/* Sidebar Header */}
        <div className="flex-shrink-0 p-3 border-b border-slate-200 flex items-center justify-between bg-gradient-to-r from-blue-50 to-slate-50">
          {!sidebarCollapsed && (
            <h3 className="font-bold text-slate-900 flex items-center gap-2">
              <Cpu className="w-5 h-5 text-blue-500" /> Model Settings
            </h3>
          )}
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="p-1.5 hover:bg-slate-100 rounded-lg transition-colors"
          >
            {sidebarCollapsed ? <PanelLeft className="w-5 h-5 text-slate-600" /> : <PanelLeftClose className="w-5 h-5 text-slate-600" />}
          </button>
        </div>

        {/* Sidebar Content - Scrollable */}
        {!sidebarCollapsed && (
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {/* Model Loading Section */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-slate-700">Base Model</label>
              {lockedModels && lockedModels.length > 0 ? (
                <select
                  value={inferenceModel}
                  onChange={(e) => setInferenceModel(e.target.value)}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900 text-sm bg-white"
                >
                  <option value="">Select a model...</option>
                  {lockedModels.map((m, i) => (
                    <option key={i} value={m.path}>{m.name}</option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  value={inferenceModel}
                  onChange={(e) => setInferenceModel(e.target.value)}
                  placeholder="/path/to/model or org/model"
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900 text-sm placeholder-slate-400"
                />
              )}

              {/* Backend Selection */}
              <div className="space-y-1">
                <label className="text-xs text-slate-500">Backend:</label>
                <select
                  value={inferenceBackend}
                  onChange={(e) => setInferenceBackend(e.target.value as 'transformers' | 'vllm' | 'sglang')}
                  className="w-full px-2 py-1.5 border border-slate-300 rounded-lg text-slate-900 text-sm bg-white"
                >
                  <option 
                    value="transformers" 
                    disabled={!availableBackends.transformers || !isBackendSupported('transformers')}
                  >
                    Transformers {!availableBackends.transformers ? '(not installed)' : !isBackendSupported('transformers') ? '(unsupported)' : ''}
                  </option>
                  <option 
                    value="vllm" 
                    disabled={!availableBackends.vllm || !isBackendSupported('vllm')}
                  >
                    vLLM {!availableBackends.vllm ? '(not installed)' : !isBackendSupported('vllm') ? '(unsupported)' : ''}
                  </option>
                  <option 
                    value="sglang" 
                    disabled={!availableBackends.sglang || !isBackendSupported('sglang')}
                  >
                    SGLang {!availableBackends.sglang ? '(not installed)' : !isBackendSupported('sglang') ? '(unsupported)' : ''}
                  </option>
                </select>
                {inferenceStatus.capabilities?.supported_backends && (
                  <p className="text-[10px] text-slate-400">
                    Supported: {inferenceStatus.capabilities.supported_backends.join(', ')}
                  </p>
                )}
              </div>

              {inferenceBackend !== 'transformers' && (
                <p className="text-xs text-amber-600 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" />
                  LoRA adapters only work with Transformers
                </p>
              )}
              
              {!isBackendSupported(inferenceBackend) && inferenceStatus.capabilities && (
                <p className="text-xs text-red-600 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" />
                  This model type ({inferenceStatus.capabilities.model_type.toUpperCase()}) doesn't support {inferenceBackend}
                </p>
              )}

              <button
                onClick={loadModel}
                disabled={!inferenceModel.trim() || isModelLoading}
                className="w-full py-2 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600 disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {isModelLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    {loadingMessage || 'Loading...'}
                  </>
                ) : (
                  <>
                    <Download className="w-4 h-4" />
                    Load Model
                  </>
                )}
              </button>
            </div>

            {/* LoRA Adapters */}
            <div className="border-t border-slate-200 pt-4 space-y-2">
              <label className="text-sm font-medium text-slate-700 flex items-center gap-2">
                <Layers className="w-4 h-4 text-blue-500" /> LoRA Adapters
              </label>

              {/* Adapter Status Banner */}
              {(inferenceStatus.adapter_path || loadedAdapters.length > 0) ? (
                <div className="flex items-center gap-2 p-2.5 bg-purple-50 border border-purple-200 rounded-lg">
                  <div className="w-2.5 h-2.5 rounded-full bg-purple-500 animate-pulse" />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-purple-800">Adapter Loaded</p>
                    <p className="text-[10px] text-purple-600 truncate">
                      {loadedAdapters.length > 0 ? loadedAdapters[0].path : inferenceStatus.adapter_path}
                    </p>
                  </div>
                  <Check className="w-4 h-4 text-purple-600 flex-shrink-0" />
                </div>
              ) : inferenceStatus.model_loaded ? (
                <div className="flex items-center gap-2 p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                  <div className="w-2.5 h-2.5 rounded-full bg-slate-300" />
                  <p className="text-xs text-slate-500">No adapter loaded (using base model only)</p>
                </div>
              ) : null}

              <div className="flex gap-2">
                <input
                  type="text"
                  value={adapterPath}
                  onChange={(e) => setAdapterPath(e.target.value)}
                  placeholder="/path/to/adapter"
                  disabled={!inferenceStatus.model_loaded || inferenceBackend !== 'transformers'}
                  className="flex-1 px-3 py-2 border border-slate-300 rounded-lg text-slate-900 text-sm placeholder-slate-400 disabled:opacity-50"
                />
                <button
                  onClick={loadAdapter}
                  disabled={!adapterPath.trim() || !inferenceStatus.model_loaded || inferenceBackend !== 'transformers'}
                  className="px-3 py-2 bg-blue-50 text-blue-600 border border-blue-200 rounded-lg text-sm font-medium hover:bg-blue-100 disabled:opacity-50"
                >
                  Load
                </button>
              </div>
              {loadedAdapters.length > 0 && (
                <div className="space-y-1 mt-2">
                  {loadedAdapters.map(adapter => (
                    <div key={adapter.id} className={`flex items-center gap-2 p-2 rounded-lg text-xs ${adapter.active ? 'bg-purple-50 border border-purple-200' : 'bg-slate-50'}`}>
                      <div className={`w-2 h-2 rounded-full ${adapter.active ? 'bg-purple-500' : 'bg-slate-300'}`} />
                      <span className="flex-1 text-slate-700 truncate" title={adapter.path}>{adapter.name}</span>
                      {adapter.active && <span className="text-[10px] text-purple-600 font-medium">Active</span>}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Chat Mode */}
            <div className="border-t border-slate-200 pt-4">
              <label className="block text-sm font-medium text-slate-700 mb-2">Mode</label>
              <div className="flex gap-2">
                <button
                  onClick={() => setChatMode('chat')}
                  className={`flex-1 py-2 rounded-lg text-sm font-medium flex items-center justify-center gap-1 transition-colors ${
                    chatMode === 'chat' ? 'bg-blue-500 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  <MessageSquare className="w-4 h-4" /> Chat
                </button>
                <button
                  onClick={() => setChatMode('completion')}
                  className={`flex-1 py-2 rounded-lg text-sm font-medium flex items-center justify-center gap-1 transition-colors ${
                    chatMode === 'completion' ? 'bg-blue-500 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  <FileText className="w-4 h-4" /> Complete
                </button>
              </div>

              {/* Chat Options */}
              <div className="mt-3 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-600">Keep conversation history</span>
                  <button onClick={() => setKeepHistory(!keepHistory)} className="text-slate-500 hover:text-slate-900">
                    {keepHistory ? <ToggleRight className="w-5 h-5 text-blue-500" /> : <ToggleLeft className="w-5 h-5" />}
                  </button>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1">
                    <span className="text-xs text-slate-600">Streaming</span>
                    {!streamingInfo.available && (
                      <span className="text-[10px] text-slate-400">({streamingInfo.reason})</span>
                    )}
                  </div>
                  <button 
                    onClick={() => streamingInfo.available && setStreamingEnabled(!streamingEnabled)}
                    disabled={!streamingInfo.available}
                    className={`${!streamingInfo.available ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    {streamingEnabled && streamingInfo.available ? (
                      <ToggleRight className="w-5 h-5 text-blue-500" />
                    ) : (
                      <ToggleLeft className="w-5 h-5 text-slate-400" />
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Generation Settings */}
            <div className="border-t border-slate-200 pt-4">
              <h4 className="text-sm font-medium text-slate-700 mb-3">Generation</h4>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-xs text-slate-500 mb-1">
                    <span>Max Tokens</span>
                    <span>{inferenceSettings.max_new_tokens}</span>
                  </div>
                  <input
                    type="range"
                    min="64"
                    max={inferenceStatus.capabilities?.max_context_length || 8192}
                    value={inferenceSettings.max_new_tokens}
                    onChange={(e) => setInferenceSettings({ ...inferenceSettings, max_new_tokens: parseInt(e.target.value) })}
                    className="w-full accent-blue-500"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-xs text-slate-500 mb-1">
                    <span>Temperature</span>
                    <span>{inferenceSettings.temperature}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={inferenceSettings.temperature}
                    onChange={(e) => setInferenceSettings({ ...inferenceSettings, temperature: parseFloat(e.target.value) })}
                    className="w-full accent-blue-500"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-xs text-slate-500 mb-1">
                    <span>Top P</span>
                    <span>{inferenceSettings.top_p}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={inferenceSettings.top_p}
                    onChange={(e) => setInferenceSettings({ ...inferenceSettings, top_p: parseFloat(e.target.value) })}
                    className="w-full accent-blue-500"
                  />
                </div>
              </div>
            </div>

            {/* Model Capabilities */}
            {inferenceStatus.model_loaded && inferenceStatus.capabilities && (
              <div className="border-t border-slate-200 pt-4">
                <h4 className="text-sm font-medium text-slate-700 mb-2 flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-purple-500" /> Model Capabilities
                </h4>
                <div className="space-y-2">
                  {/* Input capabilities */}
                  <div>
                    <span className="text-[10px] text-slate-500 uppercase">Input</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {inferenceStatus.capabilities.supports_text_input && (
                        <span className="px-2 py-0.5 bg-green-100 text-green-700 rounded text-xs flex items-center gap-1">
                          <FileText className="w-3 h-3" /> Text
                        </span>
                      )}
                      {inferenceStatus.capabilities.supports_image_input && (
                        <span className="px-2 py-0.5 bg-purple-100 text-purple-700 rounded text-xs flex items-center gap-1">
                          <ImageIcon className="w-3 h-3" /> Image
                        </span>
                      )}
                      {inferenceStatus.capabilities.supports_audio_input && (
                        <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs flex items-center gap-1">
                          <Mic className="w-3 h-3" /> Audio
                        </span>
                      )}
                      {inferenceStatus.capabilities.supports_video_input && (
                        <span className="px-2 py-0.5 bg-orange-100 text-orange-700 rounded text-xs flex items-center gap-1">
                          <Video className="w-3 h-3" /> Video
                        </span>
                      )}
                    </div>
                  </div>
                  {/* Output capabilities */}
                  <div>
                    <span className="text-[10px] text-slate-500 uppercase">Output</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {inferenceStatus.capabilities.supports_text_output && (
                        <span className="px-2 py-0.5 bg-green-100 text-green-700 rounded text-xs flex items-center gap-1">
                          <FileText className="w-3 h-3" /> Text
                        </span>
                      )}
                      {inferenceStatus.capabilities.supports_image_output && (
                        <span className="px-2 py-0.5 bg-purple-100 text-purple-700 rounded text-xs flex items-center gap-1">
                          <ImageIcon className="w-3 h-3" /> Image
                        </span>
                      )}
                      {inferenceStatus.capabilities.supports_audio_output && (
                        <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs flex items-center gap-1">
                          <Mic className="w-3 h-3" /> Audio
                        </span>
                      )}
                      {inferenceStatus.capabilities.supports_video_output && (
                        <span className="px-2 py-0.5 bg-orange-100 text-orange-700 rounded text-xs flex items-center gap-1">
                          <Video className="w-3 h-3" /> Video
                        </span>
                      )}
                    </div>
                  </div>
                  {/* Model type badge */}
                  <div className="pt-1">
                    <span className="px-2 py-1 bg-slate-100 text-slate-600 rounded text-xs font-medium uppercase">
                      {inferenceStatus.capabilities.model_type}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* System Status */}
            <div className="border-t border-slate-200 pt-4">
              <h4 className="text-sm font-medium text-slate-700 mb-2 flex items-center gap-2">
                <Monitor className="w-4 h-4 text-green-500" /> System Status
              </h4>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-slate-50 rounded-lg p-2 text-center border border-slate-200">
                  <Gauge className={`w-4 h-4 mx-auto mb-1 ${systemMetrics.available ? 'text-blue-500' : 'text-slate-400'}`} />
                  <span className="text-[10px] text-slate-500 block">GPU</span>
                  <p className="font-medium text-sm text-slate-900">
                    {systemMetrics.available && systemMetrics.gpu_utilization !== null 
                      ? `${systemMetrics.gpu_utilization}%` 
                      : 'N/A'}
                  </p>
                </div>
                <div className="bg-slate-50 rounded-lg p-2 text-center border border-slate-200">
                  <HardDrive className={`w-4 h-4 mx-auto mb-1 ${systemMetrics.available ? 'text-blue-500' : 'text-slate-400'}`} />
                  <span className="text-[10px] text-slate-500 block">VRAM</span>
                  <p className="font-medium text-sm text-slate-900">
                    {systemMetrics.available && systemMetrics.gpu_memory_used !== null
                      ? `${systemMetrics.gpu_memory_used.toFixed(1)}GB`
                      : 'N/A'}
                  </p>
                </div>
              </div>
              <div className="flex gap-2 mt-2">
                <button
                  onClick={() => { fetchInferenceStatus(); onRefreshMetrics(); }}
                  className="flex-1 px-2 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-lg text-xs font-medium text-slate-600 flex items-center justify-center gap-1"
                >
                  <RefreshCw className="w-3 h-3" /> Refresh
                </button>
                <button
                  onClick={clearMemory}
                  className="flex-1 px-2 py-1.5 bg-red-50 hover:bg-red-100 text-red-600 rounded-lg text-xs font-medium flex items-center justify-center gap-1"
                >
                  <Trash2 className="w-3 h-3" /> Clear
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Right Panel - Chat Interface */}
      <div className="flex-1 flex flex-col min-w-0 bg-gradient-to-b from-slate-50 to-white">
        {/* Chat Header */}
        <div className="flex-shrink-0 p-4 border-b border-slate-200 bg-white flex items-center justify-between">
          <div className="flex items-center gap-3">
            {(sidebarCollapsed || isMobile) && (
              <button
                onClick={() => setSidebarCollapsed(false)}
                className="p-2 hover:bg-slate-100 rounded-lg lg:hidden"
              >
                <PanelLeft className="w-5 h-5 text-slate-600" />
              </button>
            )}
            <div>
              <h3 className="font-bold text-slate-900 flex items-center gap-2">
                {chatMode === 'chat' ? (
                  <><MessageSquare className="w-5 h-5 text-blue-500" /> Chat Interface</>
                ) : (
                  <><FileText className="w-5 h-5 text-blue-500" /> Text Completion</>
                )}
              </h3>
              <p className="text-xs text-slate-500">
                {inferenceStatus.model_loaded ? (
                  <>
                    <span className="inline-flex items-center gap-1">
                      <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                      {getModelDisplayName(inferenceStatus.model_path)}
                    </span>
                    <span className="text-slate-400 ml-2">• {inferenceBackend}</span>
                    {(inferenceStatus.adapter_path || loadedAdapters.length > 0) && (
                      <span className="ml-2 inline-flex items-center gap-1 px-1.5 py-0.5 bg-purple-100 text-purple-700 rounded text-[10px] font-medium">
                        <Layers className="w-3 h-3" />
                        LoRA: {loadedAdapters.length > 0 ? loadedAdapters[0].name : inferenceStatus.adapter_path?.split('/').pop()}
                      </span>
                    )}
                  </>
                ) : (
                  'Load a model to start'
                )}
              </p>
            </div>
          </div>
          {chatMode === 'chat' && chatMessages.length > 0 && (
            <button
              onClick={() => setChatMessages([])}
              className="p-2 text-slate-500 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-colors"
              title="Clear chat"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* System Prompt (Chat mode) */}
        {chatMode === 'chat' && (
          <div className="flex-shrink-0 px-4 py-2 border-b border-slate-100 bg-white">
            <input
              type="text"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="System prompt (optional)..."
              className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-slate-900 text-sm placeholder-slate-400"
            />
          </div>
        )}

        {/* Chat Messages - Scrollable */}
        <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4">
          {chatMessages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-400">
              <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mb-4">
                <MessageSquare className="w-8 h-8 text-slate-300" />
              </div>
              <p className="text-lg font-medium">{inferenceStatus.model_loaded ? 'Start a conversation' : 'Load a model to start'}</p>
              {inferenceStatus.model_loaded && (
                <p className="text-sm mt-2">Type a message below to begin chatting</p>
              )}
            </div>
          ) : (
            <>
              {chatMessages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} toolResults={toolResults} />
              ))}
            </>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Media Preview Area */}
        {hasUploadedMedia && (
          <div className="flex-shrink-0 px-4 py-2 border-t border-slate-100 bg-white">
            <div className="flex gap-2 overflow-x-auto pb-2 flex-wrap">
              {/* Image previews */}
              {uploadedImages.map((img, i) => (
                <div key={`img-${i}`} className="relative flex-shrink-0">
                  <img src={img} alt="Upload preview" className="w-16 h-16 object-cover rounded-lg border border-slate-200" />
                  <button
                    onClick={() => removeImage(i)}
                    className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white rounded-full flex items-center justify-center hover:bg-red-600"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
              
              {/* Audio previews */}
              {uploadedAudio.map((audio, i) => (
                <div key={`audio-${i}`} className="relative flex-shrink-0 bg-purple-50 border border-purple-200 rounded-lg p-2 flex items-center gap-2">
                  <Mic className="w-4 h-4 text-purple-600" />
                  <span className="text-xs text-purple-700 max-w-[80px] truncate">{audio.name}</span>
                  <button
                    onClick={() => removeAudio(i)}
                    className="w-4 h-4 bg-red-500 text-white rounded-full flex items-center justify-center hover:bg-red-600"
                  >
                    <X className="w-2 h-2" />
                  </button>
                </div>
              ))}
              
              {/* Video previews */}
              {uploadedVideos.map((video, i) => (
                <div key={`video-${i}`} className="relative flex-shrink-0">
                  <video src={video.url} className="w-20 h-16 object-cover rounded-lg border border-slate-200" />
                  <div className="absolute inset-0 flex items-center justify-center bg-black/30 rounded-lg">
                    <Video className="w-5 h-5 text-white" />
                  </div>
                  <button
                    onClick={() => removeVideo(i)}
                    className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white rounded-full flex items-center justify-center hover:bg-red-600"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="flex-shrink-0 p-4 border-t border-slate-200 bg-white">
          <div className="flex gap-2 items-end">
            {/* Image Upload Button - Only show if model supports images */}
            {supportsImages() && (
              <>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleImageUpload}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={!inferenceStatus.model_loaded || isGenerating || isUploadingImage}
                  className="p-3 bg-slate-100 hover:bg-slate-200 rounded-xl text-slate-600 disabled:opacity-50 transition-colors"
                  title="Upload image"
                >
                  {isUploadingImage ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <ImageIcon className="w-5 h-5" />
                  )}
                </button>
              </>
            )}

            {/* Audio Upload Button - Only show if model supports audio */}
            {supportsAudio() && (
              <>
                <input
                  ref={audioInputRef}
                  type="file"
                  accept="audio/*"
                  multiple
                  onChange={handleAudioUpload}
                  className="hidden"
                />
                <button
                  onClick={() => audioInputRef.current?.click()}
                  disabled={!inferenceStatus.model_loaded || isGenerating || isUploadingAudio}
                  className="p-3 bg-purple-50 hover:bg-purple-100 rounded-xl text-purple-600 disabled:opacity-50 transition-colors"
                  title="Upload audio"
                >
                  {isUploadingAudio ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Mic className="w-5 h-5" />
                  )}
                </button>
              </>
            )}

            {/* Video Upload Button - Only show if model supports video input */}
            {supportsVideo() && (
              <>
                <input
                  ref={videoInputRef}
                  type="file"
                  accept="video/*"
                  onChange={handleVideoUpload}
                  className="hidden"
                />
                <button
                  onClick={() => videoInputRef.current?.click()}
                  disabled={!inferenceStatus.model_loaded || isGenerating || isUploadingVideo}
                  className="p-3 bg-orange-50 hover:bg-orange-100 rounded-xl text-orange-600 disabled:opacity-50 transition-colors"
                  title="Upload video"
                >
                  {isUploadingVideo ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Video className="w-5 h-5" />
                  )}
                </button>
              </>
            )}

            {/* Clear Button */}
            <button
              onClick={() => setChatMessages([])}
              disabled={chatMessages.length === 0}
              className="p-3 bg-slate-100 hover:bg-slate-200 rounded-xl text-slate-500 disabled:opacity-50 transition-colors"
              title="Clear chat"
            >
              <Trash2 className="w-5 h-5" />
            </button>

            {/* Input Field */}
            <div className="flex-1 relative">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    sendMessage()
                  }
                }}
                placeholder={chatMode === 'chat' ? 'Type your message...' : 'Enter text to complete...'}
                disabled={!inferenceStatus.model_loaded || isGenerating}
                rows={1}
                className="w-full px-4 py-3 pr-12 border border-slate-300 rounded-xl text-slate-900 placeholder-slate-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50 resize-none"
                style={{ minHeight: '48px', maxHeight: '120px' }}
              />
            </div>

            {/* Send Button */}
            <button
              onClick={sendMessage}
              disabled={!inferenceStatus.model_loaded || (!inputMessage.trim() && !hasUploadedMedia) || isGenerating}
              className="p-3 bg-blue-500 text-white rounded-xl hover:bg-blue-600 disabled:opacity-50 transition-colors shadow-lg shadow-blue-500/20"
            >
              {isGenerating ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>

          {/* Streaming indicator */}
          {isGenerating && streamingEnabled && supportsStreaming() && (
            <div className="mt-2 flex items-center gap-2 text-xs text-blue-600">
              <Zap className="w-3 h-3" />
              Streaming response...
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
