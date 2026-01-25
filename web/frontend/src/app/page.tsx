'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { 
  Cpu, Database, Settings, Play, CheckCircle, 
  ChevronRight, ChevronLeft, Loader2,
  AlertCircle, Sparkles, Zap, BarChart3,
  MessageSquare, Send, Trash2, StopCircle,
  RefreshCw, Upload, X, FileText, Check,
  HardDrive, Thermometer, Clock, Activity,
  FolderOpen, Download, Layers, ToggleLeft, ToggleRight,
  History, Menu, Monitor, Gauge, XCircle, Lock
} from 'lucide-react'
import TrainingSettingsStep from '@/components/TrainingSettings'
import DatasetConfig from '@/components/DatasetConfig'
import AlertModal, { AlertType, ConfirmModal } from '@/components/AlertModal'

// ============================================================
// LOCKED MODEL CONFIGURATION
// Models are fetched from backend API - NOT hardcoded here.
// This ensures security - frontend cannot be modified to bypass.
// ============================================================
interface LockedModel {
  name: string      // Display name (shown to users)
  path: string      // Actual path (hidden from users)
  source: 'local' | 'huggingface' | 'modelscope'
  modality: 'text' | 'vision' | 'audio' | 'video'
  description?: string
}

// ============================================================
// SHIMMER/SKELETON LOADING COMPONENT
// Use this for loading states throughout the UI
// ============================================================
const Shimmer = ({ className = '', width = 'w-full', height = 'h-4' }: { 
  className?: string
  width?: string
  height?: string 
}) => (
  <div className={`animate-pulse bg-gradient-to-r from-slate-200 via-slate-300 to-slate-200 bg-[length:200%_100%] rounded ${width} ${height} ${className}`} 
    style={{ animation: 'shimmer 1.5s ease-in-out infinite' }} />
)

const ShimmerCard = ({ lines = 3 }: { lines?: number }) => (
  <div className="bg-white rounded-lg border border-slate-200 p-4 space-y-3">
    <Shimmer width="w-1/3" height="h-5" />
    {Array.from({ length: lines }).map((_, i) => (
      <Shimmer key={i} width={i === lines - 1 ? 'w-2/3' : 'w-full'} height="h-3" />
    ))}
  </div>
)

// Add shimmer animation CSS (injected via style tag in component)
const shimmerStyles = `
  @keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }
`

// Types
interface TrainingConfig {
  model_path: string
  model_source: 'huggingface' | 'modelscope' | 'local'
  modality: 'text' | 'vision' | 'audio' | 'video'
  training_method: 'sft' | 'pt' | 'rlhf'
  train_type: 'full' | 'lora' | 'qlora' | 'adalora'
  rlhf_type: 'dpo' | 'orpo' | 'simpo' | 'kto' | 'cpo' | 'rm' | 'ppo' | 'grpo' | 'gkd' | null
  // RLHF parameters
  beta: number | null
  max_completion_length: number
  label_smoothing: number
  rpo_alpha: number | null
  simpo_gamma: number
  desirable_weight: number
  undesirable_weight: number
  num_ppo_epochs: number
  kl_coef: number
  cliprange: number
  num_generations: number
  // Base parameters
  dataset_paths: string[]
  output_dir: string
  num_train_epochs: number
  learning_rate: number
  per_device_train_batch_size: number
  gradient_accumulation_steps: number
  max_length: number
  lora_rank: number
  lora_alpha: number
  lora_dropout: number
  target_modules: string
  quant_bits: number | null
  torch_dtype: string
  warmup_ratio: number
  attn_impl: string | null
  deepspeed: string | null
  fsdp: string | null
  gradient_checkpointing: boolean
  use_liger_kernel: boolean
  packing: boolean
  sequence_parallel_size: number
  lr_scheduler_type: string
  weight_decay: number
  adam_beta2: number
  gpu_ids: number[] | null
  num_gpus: number | null
  early_stop_interval: number | null
  // API Tokens for private models/datasets (optional)
  hf_token: string | null  // HuggingFace token for private models/datasets
  ms_token: string | null  // ModelScope token for private models/datasets
}

interface JobStatus {
  job_id: string
  job_name: string  // User-friendly name for the training
  status: 'pending' | 'initializing' | 'running' | 'completed' | 'failed' | 'stopped'
  current_step: number
  total_steps: number
  current_loss: number | null
  logs: string[]
  error: string | null
  learning_rate?: number
  epoch?: number
  total_epochs?: number
  samples_per_second?: number
  eta_seconds?: number
}

interface SystemMetrics {
  gpu_utilization: number | null
  gpu_memory_used: number | null
  gpu_memory_total: number | null
  gpu_temperature: number | null
  cpu_percent: number | null
  ram_used: number | null
  ram_total: number | null
  available: boolean
}

interface SystemStatus {
  status: 'live' | 'starting' | 'degraded' | 'offline' | 'error' | 'unknown'
  message: string
  gpu_available: boolean
  gpu_name: string | null
  cuda_available: boolean
  bios_installed: boolean
  backend_ready: boolean
  details: Record<string, string>
}

interface SystemCapabilities {
  supported_model: string | null
  supported_sources: string[]
  supported_model_sources: string[]
  supported_dataset_sources: string[]
  supported_architectures: string[] | null
  supported_modalities: string[]
  has_model_restriction: boolean
  has_architecture_restriction: boolean
  has_modality_restriction: boolean
  has_external_storage: boolean
  storage_path: string | null
  storage_writable: boolean
}

interface TrainingMetric {
  step: number
  epoch?: number
  loss: number
  learning_rate: number
  grad_norm?: number
  eval_loss?: number
  reward?: number
  kl_divergence?: number
  policy_loss?: number
  value_loss?: number
  entropy?: number
  chosen_rewards?: number
  rejected_rewards?: number
  reward_margin?: number
  extra_metrics?: Record<string, number>
  timestamp: number
}

interface LoadedAdapter {
  id: string
  name: string
  path: string
  active: boolean
}

interface TrainingHistoryItem {
  job_id: string
  job_name: string
  status: string
  created_at: string | null
  started_at: string | null
  completed_at: string | null
  current_step: number
  total_steps: number
  error: string | null
  config?: {
    train_type: string
    training_method: string
    num_epochs: number
    learning_rate: number
    batch_size: number
  }
  output_path: string
  output_exists: boolean
  has_adapter: boolean
  adapter_path?: string
  checkpoint_count: number
  final_metrics?: {
    loss: number | null
    learning_rate: number | null
    epoch: number | null
    step: number | null
  } | null
}

interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

interface UploadedDataset {
  id: string
  name: string
  filename: string
  path: string
  size: number
  size_human: string
  format: string
  total_samples: number
  created_at: number
  selected?: boolean
}

type MainTab = 'train' | 'inference'

// Training steps - dynamically configured based on locked models
// If single model is locked, skip the "Select Model" step entirely
const TRAIN_STEPS_FULL = [
  { id: 1, title: 'Select Model', icon: Cpu },
  { id: 2, title: 'Configure Dataset', icon: Database },
  { id: 3, title: 'Training Settings', icon: Settings },
  { id: 4, title: 'Review & Start', icon: Play },
]

const TRAIN_STEPS_LOCKED = [
  { id: 1, title: 'Configure Dataset', icon: Database },
  { id: 2, title: 'Training Settings', icon: Settings },
  { id: 3, title: 'Review & Start', icon: Play },
]

export default function Home() {
  const [mainTab, setMainTab] = useState<MainTab>('train')
  const [currentStep, setCurrentStep] = useState(1)
  const [isTraining, setIsTraining] = useState(false)
  const [isStartingTraining, setIsStartingTraining] = useState(false)
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [trainingLogs, setTrainingLogs] = useState<string[]>([])
  
  // Initialize config - will be updated when locked models are fetched
  const [config, setConfig] = useState<TrainingConfig>({
    model_path: '',
    model_source: 'local',
    modality: 'text',
    training_method: 'sft',
    train_type: 'lora',
    rlhf_type: null,
    // RLHF parameters with defaults
    beta: null,
    max_completion_length: 512,
    label_smoothing: 0,
    rpo_alpha: null,
    simpo_gamma: 1.0,
    desirable_weight: 1.0,
    undesirable_weight: 1.0,
    num_ppo_epochs: 4,
    kl_coef: 0.05,
    cliprange: 0.2,
    num_generations: 8,
    // Base parameters
    dataset_paths: [],
    output_dir: '',  // Empty by default - backend auto-generates when locked
    num_train_epochs: 3,
    learning_rate: 0.0001,
    per_device_train_batch_size: 1,
    gradient_accumulation_steps: 16,
    max_length: 2048,
    lora_rank: 8,
    lora_alpha: 32,
    lora_dropout: 0.05,
    target_modules: 'all-linear',
    quant_bits: null,
    torch_dtype: 'bfloat16',
    warmup_ratio: 0.03,
    attn_impl: null,
    deepspeed: null,
    fsdp: null,
    gradient_checkpointing: true,
    use_liger_kernel: false,
    packing: false,
    sequence_parallel_size: 1,
    lr_scheduler_type: 'cosine',
    weight_decay: 0.1,
    adam_beta2: 0.95,
    gpu_ids: null,
    num_gpus: null,
    early_stop_interval: null,
    // API Tokens for private models/datasets (optional)
    hf_token: null,
    ms_token: null,
  })

  // Dataset state
  const [uploadedDatasets, setUploadedDatasets] = useState<UploadedDataset[]>([])
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadName, setUploadName] = useState('')
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [showUploadForm, setShowUploadForm] = useState(false)
  const [uploadError, setUploadError] = useState('')
  
  // Delete confirmation state
  const [deleteTarget, setDeleteTarget] = useState<UploadedDataset | null>(null)
  const [deleteConfirmText, setDeleteConfirmText] = useState('')
  const [isDeleting, setIsDeleting] = useState(false)
  
  // Custom alert modal state (replaces browser alert)
  const [alertModal, setAlertModal] = useState<{
    isOpen: boolean
    title?: string
    message: string
    type: AlertType
  }>({ isOpen: false, message: '', type: 'info' })
  
  // Confirm modal state (replaces browser confirm)
  const [confirmModal, setConfirmModal] = useState<{
    isOpen: boolean
    title: string
    message: string
    type: 'danger' | 'warning' | 'info'
    onConfirm: () => void
    confirmText?: string
    isLoading?: boolean
  }>({ isOpen: false, title: '', message: '', type: 'warning', onConfirm: () => {} })
  
  // Helper function to show custom alert (replaces browser alert)
  const showAlert = (message: string, type: AlertType = 'error', title?: string) => {
    setAlertModal({ isOpen: true, message, type, title })
  }
  
  const closeAlert = () => {
    setAlertModal(prev => ({ ...prev, isOpen: false }))
  }
  
  // Helper function to show custom confirm dialog (replaces browser confirm)
  const showConfirm = (options: {
    title: string
    message: string
    type?: 'danger' | 'warning' | 'info'
    onConfirm: () => void
    confirmText?: string
  }) => {
    setConfirmModal({
      isOpen: true,
      title: options.title,
      message: options.message,
      type: options.type || 'warning',
      onConfirm: options.onConfirm,
      confirmText: options.confirmText,
      isLoading: false
    })
  }
  
  const closeConfirm = () => {
    setConfirmModal(prev => ({ ...prev, isOpen: false, isLoading: false }))
  }
  
  // Helper to extract clean error message
  const getErrorMessage = (error: unknown): string => {
    if (error instanceof Error) return error.message
    return String(error)
  }
  
  // Training name state (optional - auto-generated if empty)
  const [trainingName, setTrainingName] = useState('')

  // Inference state
  const [inferenceModel, setInferenceModel] = useState('')
  const [adapterPath, setAdapterPath] = useState('')
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [inferenceStatus, setInferenceStatus] = useState<{
    model_loaded: boolean
    model_path: string | null
    memory_used_gb: number
  }>({ model_loaded: false, model_path: null, memory_used_gb: 0 })
  const [inferenceSettings, setInferenceSettings] = useState({
    max_new_tokens: 512,
    temperature: 0.7,
    top_p: 0.9,
    repetition_penalty: 1.0,
  })
  
  // Enhanced inference state
  const [isModelLoading, setIsModelLoading] = useState(false)
  const [isCleaningMemory, setIsCleaningMemory] = useState(false)
  const [loadingMessage, setLoadingMessage] = useState('')
  const [loadedAdapters, setLoadedAdapters] = useState<LoadedAdapter[]>([])
  const [chatMode, setChatMode] = useState<'chat' | 'completion'>('chat')
  const [systemPrompt, setSystemPrompt] = useState('')
  const [keepHistory, setKeepHistory] = useState(true)
  
  // System metrics state - null means data not available
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    gpu_utilization: null, gpu_memory_used: null, gpu_memory_total: null,
    gpu_temperature: null, cpu_percent: null, ram_used: null, ram_total: null,
    available: false
  })
  
  // System status - tracks if system is ready for training
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    status: 'unknown',
    message: 'Checking system status...',
    gpu_available: false,
    gpu_name: null,
    cuda_available: false,
    bios_installed: false,
    backend_ready: false,
    details: {}
  })
  
  // System capabilities - what this system can fine-tune
  const [systemCapabilities, setSystemCapabilities] = useState<SystemCapabilities>({
    supported_model: null,
    supported_sources: ['local'],
    supported_model_sources: ['local'],
    supported_dataset_sources: ['local'],
    supported_architectures: null,
    supported_modalities: ['text2text', 'multimodal', 'speech2text', 'text2speech', 'vision', 'audio'],
    has_model_restriction: false,
    has_architecture_restriction: false,
    has_modality_restriction: false,
    has_external_storage: false,
    storage_path: null,
    storage_writable: false
  })
  
  // System expiration - COMPLETE LOCKDOWN when expired
  const [systemExpired, setSystemExpired] = useState(false)
  const [expirationChecked, setExpirationChecked] = useState(false)
  
  // Locked models - fetched from backend API (NOT hardcoded for security)
  const [lockedModels, setLockedModels] = useState<LockedModel[]>([])
  const [isModelLockLoaded, setIsModelLockLoaded] = useState(false)
  
  // Available GPUs - fetched from backend API for dynamic GPU selection
  const [availableGpus, setAvailableGpus] = useState<{
    id: number
    name: string
    memory_total_gb: number | null
    memory_free_gb: number | null
    utilization: number | null
  }[]>([])
  
  // Output path configuration - fetched from backend API
  const [outputPathConfig, setOutputPathConfig] = useState<{
    mode: 'locked' | 'base_locked' | 'free'
    base_path: string
    is_locked: boolean
    user_can_customize: boolean
    user_can_add_path: boolean
  }>({
    mode: 'locked',
    base_path: '/workspace/output',
    is_locked: true,
    user_can_customize: false,
    user_can_add_path: false
  })
  
  // Model context length - fetched dynamically based on selected model
  // This is used for dynamic max_length and max_completion_length ranges
  const [modelContextLength, setModelContextLength] = useState<number>(4096)
  const [modelInfo, setModelInfo] = useState<{
    model_type: string | null
    architecture: string | null
    isLoading: boolean
    error: string | null
  }>({ model_type: null, architecture: null, isLoading: false, error: null })
  
  // System API tokens status - check if HF_TOKEN/MS_TOKEN are configured on server
  const [systemTokens, setSystemTokens] = useState<{
    hf_token_available: boolean
    ms_token_available: boolean
    hf_token_masked: string | null
    ms_token_masked: string | null
  }>({ hf_token_available: false, ms_token_available: false, hf_token_masked: null, ms_token_masked: null })
  const [useSystemToken, setUseSystemToken] = useState(false)
  
  // Derived values from locked models (computed, not hardcoded)
  const IS_MODEL_LOCKED = lockedModels.length > 0
  const IS_SINGLE_MODEL = lockedModels.length === 1
  const DEFAULT_MODEL = lockedModels[0] || null
  
  // Dynamic training steps based on locked model configuration
  const TRAIN_STEPS = IS_SINGLE_MODEL ? TRAIN_STEPS_LOCKED : TRAIN_STEPS_FULL
  const TOTAL_STEPS = TRAIN_STEPS.length
  
  // Helper to get model name from path
  const getModelDisplayName = (path: string): string => {
    const lockedModel = lockedModels.find(m => m.path === path)
    return lockedModel ? lockedModel.name : path
  }
  
  // Training metrics for graphs
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetric[]>([])
  
  // Training history
  const [trainingHistory, setTrainingHistory] = useState<TrainingHistoryItem[]>([])
  const [showHistory, setShowHistory] = useState(false)
  const [isLoadingHistory, setIsLoadingHistory] = useState(false)
  
  // Mobile menu
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  
  const chatEndRef = useRef<HTMLDivElement>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Fetch datasets on mount and when on step 2
  useEffect(() => {
    if (mainTab === 'train' && currentStep === 2) {
      fetchDatasets()
    }
  }, [mainTab, currentStep])

  // Check system expiration - MUST run first before any other API calls
  const checkSystemExpiration = useCallback(async () => {
    try {
      const res = await fetch('/api/system/status')
      if (res.status === 503) {
        const data = await res.json()
        if (data.error === 'system_expired' || data.blocked) {
          setSystemExpired(true)
        }
      }
      setExpirationChecked(true)
    } catch (e) {
      // If we can't reach the server, still mark as checked
      setExpirationChecked(true)
    }
  }, [])

  // Fetch system status - check if system is ready for training
  const fetchSystemStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/system/status')
      if (res.status === 503) {
        const data = await res.json()
        if (data.error === 'system_expired' || data.blocked) {
          setSystemExpired(true)
          return
        }
      }
      if (res.ok) {
        const data = await res.json()
        setSystemStatus(data)
      } else {
        setSystemStatus(prev => ({
          ...prev,
          status: 'offline',
          message: 'Backend not responding',
          backend_ready: false
        }))
      }
    } catch (e) {
      setSystemStatus({
        status: 'offline',
        message: 'Cannot connect to backend server',
        gpu_available: false,
        gpu_name: null,
        cuda_available: false,
        bios_installed: false,
        backend_ready: false,
        details: { error: String(e) }
      })
    }
  }, [])

  // Fetch training history
  const fetchTrainingHistory = useCallback(async () => {
    setIsLoadingHistory(true)
    try {
      const res = await fetch('/api/jobs/history/all?limit=50&include_metrics=true')
      if (res.ok) {
        const data = await res.json()
        setTrainingHistory(data.history || [])
      }
    } catch (e) {
      console.error('Failed to fetch training history:', e)
    } finally {
      setIsLoadingHistory(false)
    }
  }, [])

  // Fetch locked models from backend API - CRITICAL for security
  // Models MUST come from backend, NOT hardcoded in frontend
  const fetchLockedModels = useCallback(async () => {
    try {
      const res = await fetch('/api/system/locked-models')
      if (res.ok) {
        const data = await res.json()
        if (data.is_locked && data.models && data.models.length > 0) {
          setLockedModels(data.models)
          // Auto-select first model if single model locked
          if (data.models.length === 1) {
            const model = data.models[0]
            setConfig(prev => ({
              ...prev,
              model_path: model.path,
              model_source: model.source,
              modality: model.modality || 'text'
            }))
          }
        }
        setIsModelLockLoaded(true)
      }
    } catch (e) {
      console.error('Failed to fetch locked models:', e)
      setIsModelLockLoaded(true)
    }
  }, [])

  // Fetch available GPUs from backend API for dynamic GPU selection
  const fetchAvailableGpus = useCallback(async () => {
    try {
      const res = await fetch('/api/system/gpus')
      if (res.ok) {
        const data = await res.json()
        if (data.available && data.gpus && data.gpus.length > 0) {
          setAvailableGpus(data.gpus)
        }
      }
    } catch (e) {
      console.error('Failed to fetch available GPUs:', e)
    }
  }, [])
  
  // Fetch model info to get context length for dynamic UI ranges
  const fetchModelInfo = useCallback(async (modelPath: string, source: string) => {
    if (!modelPath) return
    
    setModelInfo(prev => ({ ...prev, isLoading: true, error: null }))
    
    try {
      const res = await fetch(`/api/models/info?model_path=${encodeURIComponent(modelPath)}&source=${source}`)
      if (res.ok) {
        const data = await res.json()
        if (data.context_length && data.context_length > 0) {
          setModelContextLength(data.context_length)
        } else {
          setModelContextLength(4096) // Default fallback
        }
        setModelInfo({
          model_type: data.model_type || null,
          architecture: data.architecture || null,
          isLoading: false,
          error: null
        })
      } else {
        setModelInfo(prev => ({ ...prev, isLoading: false, error: 'Failed to fetch model info' }))
        setModelContextLength(4096)
      }
    } catch (e) {
      console.error('Failed to fetch model info:', e)
      setModelContextLength(4096) // Default fallback
      setModelInfo(prev => ({ ...prev, isLoading: false, error: 'Failed to fetch model info' }))
    }
  }, [])

  // Fetch output path configuration from backend API
  const fetchOutputPathConfig = useCallback(async () => {
    try {
      const res = await fetch('/api/system/output-path-config')
      if (res.ok) {
        const data = await res.json()
        setOutputPathConfig(data)
        // When output path is locked, clear any user-provided output_dir
        // The backend will auto-generate the path using job_id
        if (data.mode === 'locked') {
          setConfig(prev => ({ ...prev, output_dir: '' }))
        }
      }
    } catch (e) {
      console.error('Failed to fetch output path config:', e)
    }
  }, [])
  
  // Fetch system API token status (check if HF_TOKEN/MS_TOKEN are configured)
  const fetchSystemTokens = useCallback(async () => {
    try {
      const res = await fetch('/api/system/api-tokens')
      if (res.ok) {
        const data = await res.json()
        setSystemTokens(data)
      }
    } catch (e) {
      console.error('Failed to fetch system token status:', e)
    }
  }, [])

  // Fetch system capabilities - what this system can fine-tune
  const fetchSystemCapabilities = useCallback(async () => {
    try {
      const res = await fetch('/api/system/capabilities')
      if (res.ok) {
        const data = await res.json()
        setSystemCapabilities(data)
        
        // Get the supported model sources
        const supportedSources = data.supported_model_sources || data.supported_sources || ['local']
        
        // Always auto-select the first supported source by default
        // User can change it if they want, but first option is pre-selected
        if (data.has_model_restriction && data.supported_model) {
          setConfig(prev => ({
            ...prev,
            model_path: data.supported_model,
            model_source: supportedSources[0] || 'local'
          }))
        } else {
          // Auto-select first available source
          setConfig(prev => ({
            ...prev,
            model_source: supportedSources[0] || 'local'
          }))
        }
      }
    } catch (e) {
      console.error('Failed to fetch system capabilities:', e)
    }
  }, [])

  // Fetch system metrics periodically - only set if data is valid
  const fetchSystemMetrics = useCallback(async () => {
    try {
      const res = await fetch('/api/system/metrics')
      if (res.ok) {
        const data = await res.json()
        // Check if we have actual valid data (not null)
        // gpu_available and cpu_available indicate if hardware is detected
        const hasGpuData = data.gpu_available === true && (
          data.gpu_utilization !== null || 
          data.gpu_memory_used !== null
        )
        const hasCpuData = data.cpu_available === true && (
          data.cpu_percent !== null || 
          data.ram_used !== null
        )
        
        setSystemMetrics({
          gpu_utilization: data.gpu_utilization ?? null,
          gpu_memory_used: data.gpu_memory_used ?? null,
          gpu_memory_total: data.gpu_memory_total ?? null,
          gpu_temperature: data.gpu_temperature ?? null,
          cpu_percent: data.cpu_percent ?? null,
          ram_used: data.ram_used ?? null,
          ram_total: data.ram_total ?? null,
          available: hasGpuData || hasCpuData
        })
      }
    } catch (e) {
      console.error('Failed to fetch system metrics:', e)
      // Keep metrics as unavailable on error
    }
  }, [])

  // Check for active training job on page load - restores state after refresh
  const checkActiveTraining = useCallback(async () => {
    try {
      const res = await fetch('/api/jobs/current')
      if (res.ok) {
        const data = await res.json()
        
        // Case 1: We have job info - restore full state
        if (data.has_active_job && data.job) {
          const job = data.job
          setJobStatus({
            job_id: job.job_id,
            job_name: job.name || job.job_id,
            status: job.status,
            current_step: job.current_step || 0,
            total_steps: job.total_steps || 0,
            current_loss: job.current_loss,
            logs: data.logs || [],
            error: job.error,
            learning_rate: job.learning_rate,
            epoch: job.current_epoch,
            total_epochs: job.total_epochs,
          })
          setTrainingLogs(data.logs || [])
          setIsTraining(job.status === 'running' || job.status === 'initializing')
          setCurrentStep(5) // Go to training progress view
          console.log('Restored active training:', job.job_id)
        }
        // Case 2: Training process running but job state lost (fallback)
        else if (data.has_active_job && data.process_running && !data.job) {
          // Show a minimal training view - process is running in background
          setJobStatus({
            job_id: data.process_pid ? `pid-${data.process_pid}` : 'unknown',
            job_name: 'Training in Progress',
            status: 'running',
            current_step: 0,
            total_steps: 0,
            current_loss: null,
            logs: [],
            error: null,
          })
          setTrainingLogs([
            'Training process detected running in background.',
            data.message || 'Job state was lost but training continues.',
            `Process ID: ${data.process_pid || 'unknown'}`,
            'Please wait for training to complete or stop it manually.'
          ])
          setIsTraining(true)
          setCurrentStep(5)
          console.log('Detected running training process:', data.process_pid)
        }
      }
    } catch (e) {
      console.error('Failed to check active training:', e)
    }
  }, [])

  // Check system expiration FIRST on mount - blocks everything if expired
  useEffect(() => {
    checkSystemExpiration()
  }, [checkSystemExpiration])

  // Check for active training after expiration check passes
  useEffect(() => {
    if (expirationChecked && !systemExpired) {
      checkActiveTraining()
    }
  }, [expirationChecked, systemExpired, checkActiveTraining])

  // Fetch system status, capabilities, locked models, available GPUs, and output path config on mount (only if not expired)
  useEffect(() => {
    if (systemExpired) return
    fetchSystemStatus()
    fetchSystemCapabilities()
    fetchLockedModels()  // CRITICAL: Fetch locked models from backend API
    fetchAvailableGpus()  // Fetch available GPUs for dynamic selection
    fetchOutputPathConfig()  // Fetch output path configuration
    fetchSystemTokens()  // Check if system has HF_TOKEN/MS_TOKEN configured
    const interval = setInterval(fetchSystemStatus, 10000) // Check every 10 seconds
    return () => clearInterval(interval)
  }, [fetchSystemStatus, fetchSystemCapabilities, fetchLockedModels, fetchAvailableGpus, fetchOutputPathConfig, fetchSystemTokens, systemExpired])
  
  // Fetch model info when model_path changes to get dynamic context length
  useEffect(() => {
    if (config.model_path) {
      fetchModelInfo(config.model_path, config.model_source)
    }
  }, [config.model_path, config.model_source, fetchModelInfo])

  // Poll system metrics during training or inference
  useEffect(() => {
    if (isTraining || mainTab === 'inference') {
      fetchSystemMetrics()
      const interval = setInterval(fetchSystemMetrics, 3000)
      return () => clearInterval(interval)
    }
  }, [isTraining, mainTab, fetchSystemMetrics])

  // Poll for job status - CRITICAL for detecting failures
  // This is the primary way to detect when training completes/fails
  useEffect(() => {
    const jobId = jobStatus?.job_id
    if (!jobId || !isTraining) return
    
    console.log('[STATUS POLL] Starting status polling for job:', jobId)
    
    const checkJobStatus = async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`)
        if (res.ok) {
          const job = await res.json()
          console.log('[STATUS POLL] Job status:', job.status)
          
          // Update status from backend
          if (job.status === 'completed' || job.status === 'failed' || job.status === 'stopped') {
            console.log('[STATUS POLL] Training ended with status:', job.status)
            setIsTraining(false)
            setJobStatus(prev => prev ? { 
              ...prev, 
              status: job.status,
              error: job.error || null
            } : null)
          }
        }
      } catch (e) {
        console.error('[STATUS POLL] Error checking job status:', e)
      }
    }
    
    // Check immediately
    checkJobStatus()
    
    // Poll every 2 seconds
    const interval = setInterval(checkJobStatus, 2000)
    
    return () => {
      console.log('[STATUS POLL] Stopping status polling for job:', jobId)
      clearInterval(interval)
    }
  }, [jobStatus?.job_id, isTraining])

  // WebSocket connection state
  const [wsConnected, setWsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const wsReconnectAttempts = useRef(0)
  const maxWsReconnectAttempts = 5
  
  // WebSocket for training updates (real-time progress) with auto-reconnect
  useEffect(() => {
    if (jobStatus?.job_id && isTraining) {
      let reconnectTimeout: NodeJS.Timeout
      
      const connectWebSocket = () => {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        const ws = new WebSocket(`${wsProtocol}//${window.location.host}/api/jobs/ws/${jobStatus.job_id}`)
        wsRef.current = ws
        
        ws.onopen = () => {
          console.log('[WS] Connected')
          setWsConnected(true)
          wsReconnectAttempts.current = 0
        }
        
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            // WebSocket handles progress and status updates
            if (data.type === 'progress') {
              setJobStatus(prev => prev ? {
                ...prev,
                current_step: data.step || prev.current_step,
                total_steps: data.total_steps || prev.total_steps,
                current_loss: data.loss ?? prev.current_loss,
                learning_rate: data.learning_rate ?? prev.learning_rate,
                epoch: data.epoch ?? prev.epoch,
                total_epochs: data.total_epochs ?? prev.total_epochs,
                samples_per_second: data.samples_per_second ?? prev.samples_per_second,
                eta_seconds: data.eta_seconds ?? prev.eta_seconds,
              } : null)
            }
            if (data.type === 'status') {
              if (data.status === 'completed' || data.status === 'failed' || data.status === 'stopped') {
                setIsTraining(false)
              }
              setJobStatus(prev => prev ? { ...prev, status: data.status } : null)
            }
          } catch (e) {
            console.error('[WS] Parse error:', e)
          }
        }
        
        ws.onerror = (e) => {
          console.error('[WS] Error:', e)
          setWsConnected(false)
        }
        
        ws.onclose = () => {
          console.log('[WS] Closed')
          setWsConnected(false)
          wsRef.current = null
          
          // Auto-reconnect if still training and under max attempts
          if (isTraining && wsReconnectAttempts.current < maxWsReconnectAttempts) {
            wsReconnectAttempts.current++
            const delay = Math.min(1000 * Math.pow(2, wsReconnectAttempts.current), 10000)
            console.log(`[WS] Reconnecting in ${delay}ms (attempt ${wsReconnectAttempts.current})`)
            reconnectTimeout = setTimeout(connectWebSocket, delay)
          }
        }
      }
      
      connectWebSocket()
      
      return () => {
        clearTimeout(reconnectTimeout)
        if (wsRef.current) {
          wsRef.current.close()
          wsRef.current = null
        }
      }
    }
  }, [jobStatus?.job_id, isTraining])
  
  // Fallback polling for job status (when WebSocket fails or as backup)
  // Polls every 2 seconds - ensures we ALWAYS have updated status even if WS fails
  useEffect(() => {
    if (jobStatus?.job_id && isTraining) {
      const pollJobStatus = async () => {
        try {
          const res = await fetch(`/api/jobs/${jobStatus.job_id}`)
          if (res.ok) {
            const data = await res.json()
            if (data.job) {
              // Update status from polling
              setJobStatus(prev => prev ? {
                ...prev,
                status: data.job.status || prev.status,
                current_step: data.job.current_step ?? prev.current_step,
                total_steps: data.job.total_steps ?? prev.total_steps,
                current_loss: data.job.current_loss ?? prev.current_loss,
                epoch: data.job.epoch ?? prev.epoch,
              } : null)
              
              // Detect completion/failure
              if (data.job.status === 'completed' || data.job.status === 'failed' || data.job.status === 'stopped') {
                setIsTraining(false)
              }
            }
          }
        } catch (e) {
          console.error('[POLL] Status error:', e)
        }
      }
      
      // Poll every 2 seconds as backup
      const interval = setInterval(pollJobStatus, 2000)
      return () => clearInterval(interval)
    }
  }, [jobStatus?.job_id, isTraining])

  // Poll for terminal logs from file - SIMPLE AND ROBUST
  // Polls every second when there's an active job_id
  // Also detects training completion/failure from log patterns as fallback
  useEffect(() => {
    const jobId = jobStatus?.job_id
    if (!jobId) return
    
    console.log('[LOG POLL] Starting log polling for job:', jobId)
    
    const fetchTerminalLogs = async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}/terminal-logs?lines=500`)
        if (res.ok) {
          const data = await res.json()
          if (data.logs && Array.isArray(data.logs)) {
            // Check for failure/completion patterns in logs (fallback detection)
            const recentLogs = data.logs.slice(-20).join('\n')
            
            // Detect failure from log patterns
            if (isTraining && (
              recentLogs.includes('SESSION ENDED: FAILED') ||
              recentLogs.includes('Training failed with exit code') ||
              recentLogs.includes('[ERROR] Training failed') ||
              recentLogs.includes('Training was interrupted or failed')
            )) {
              console.log('[LOG POLL] Detected training failure from logs')
              setIsTraining(false)
              setJobStatus(prev => prev ? { ...prev, status: 'failed' } : null)
            }
            
            // Detect completion from log patterns
            if (isTraining && (
              recentLogs.includes('SESSION ENDED: COMPLETED') ||
              recentLogs.includes('Training completed successfully')
            )) {
              console.log('[LOG POLL] Detected training completion from logs')
              setIsTraining(false)
              setJobStatus(prev => prev ? { ...prev, status: 'completed' } : null)
            }
            
            setTrainingLogs(prevLogs => {
              // Only update if logs changed
              if (data.logs.length !== prevLogs.length || 
                  (data.logs.length > 0 && data.logs[data.logs.length - 1] !== prevLogs[prevLogs.length - 1])) {
                return data.logs
              }
              return prevLogs
            })
          }
        }
      } catch (e) {
        console.error('[LOG POLL] Error fetching logs:', e)
      }
    }
    
    // Fetch immediately
    fetchTerminalLogs()
    
    // Poll every 1 second
    const interval = setInterval(fetchTerminalLogs, 1000)
    
    return () => {
      console.log('[LOG POLL] Stopping log polling for job:', jobId)
      clearInterval(interval)
    }
  }, [jobStatus?.job_id, isTraining])

  // State for training type and available metrics
  const [trainTypeInfo, setTrainTypeInfo] = useState<{
    train_type: string
    display_name: string
    primary_metrics: string[]
    graph_configs: Array<{key: string, label: string, unit: string, color: string}>
  } | null>(null)
  
  // GPU metrics history for GPU utilization graph
  const [gpuMetricsHistory, setGpuMetricsHistory] = useState<Array<{
    timestamp: number
    gpu_utilization: number | null
    gpu_memory_used: number | null
    gpu_temperature: number | null
  }>>([])
  
  // Track GPU metrics over time for graphs
  useEffect(() => {
    if (isTraining && systemMetrics.available) {
      setGpuMetricsHistory(prev => {
        const newEntry = {
          timestamp: Date.now(),
          gpu_utilization: systemMetrics.gpu_utilization,
          gpu_memory_used: systemMetrics.gpu_memory_used,
          gpu_temperature: systemMetrics.gpu_temperature
        }
        // Keep last 100 data points
        const updated = [...prev, newEntry].slice(-100)
        return updated
      })
    }
  }, [systemMetrics, isTraining])

  // Data source tracking for metrics (tensorboard vs database)
  const [metricsDataSource, setMetricsDataSource] = useState<string>('none')
  const [metricsDataQuality, setMetricsDataQuality] = useState<number>(1.0)
  
  // Poll for training metrics using UNIFIED endpoint
  // Prefers TensorBoard data (most accurate), falls back to database
  // IMPORTANT: Only shows validated data - never shows wrong/corrupt data
  useEffect(() => {
    if (jobStatus?.job_id && (isTraining || jobStatus.status === 'completed')) {
      const fetchMetrics = async () => {
        try {
          // Use unified endpoint which combines TensorBoard + database with validation
          const res = await fetch(`/api/jobs/${jobStatus.job_id}/metrics/unified?limit=500&prefer_tensorboard=true`)
          if (res.ok) {
            const data = await res.json()
            
            // Track data source for transparency
            setMetricsDataSource(data.data_source || 'none')
            setMetricsDataQuality(data.data_quality || 1.0)
            
            // Store training type info for dynamic graph rendering
            if (data.training_type) {
              // Map training type to graph configs - ALL 20+ TRAINING TYPES
              const GRAPH_CONFIGS: Record<string, Array<{key: string, label: string, unit: string, color: string}>> = {
                // ============================================================
                // SUPERVISED FINE-TUNING (SFT) & VARIANTS
                // ============================================================
                sft: [
                  {key: 'loss', label: 'Training Loss', unit: '', color: '#3B82F6'},
                  {key: 'learning_rate', label: 'Learning Rate', unit: '', color: '#10B981'},
                  {key: 'grad_norm', label: 'Gradient Norm', unit: '', color: '#F59E0B'},
                ],
                lora: [
                  {key: 'loss', label: 'Training Loss', unit: '', color: '#3B82F6'},
                  {key: 'learning_rate', label: 'Learning Rate', unit: '', color: '#10B981'},
                  {key: 'grad_norm', label: 'Gradient Norm', unit: '', color: '#F59E0B'},
                ],
                qlora: [
                  {key: 'loss', label: 'Training Loss', unit: '', color: '#3B82F6'},
                  {key: 'learning_rate', label: 'Learning Rate', unit: '', color: '#10B981'},
                  {key: 'grad_norm', label: 'Gradient Norm', unit: '', color: '#F59E0B'},
                ],
                adalora: [
                  {key: 'loss', label: 'Training Loss', unit: '', color: '#3B82F6'},
                  {key: 'learning_rate', label: 'Learning Rate', unit: '', color: '#10B981'},
                  {key: 'grad_norm', label: 'Gradient Norm', unit: '', color: '#F59E0B'},
                ],
                full: [
                  {key: 'loss', label: 'Training Loss', unit: '', color: '#3B82F6'},
                  {key: 'learning_rate', label: 'Learning Rate', unit: '', color: '#10B981'},
                  {key: 'grad_norm', label: 'Gradient Norm', unit: '', color: '#F59E0B'},
                ],
                // ============================================================
                // PRE-TRAINING & CONTINUOUS PRE-TRAINING
                // ============================================================
                pt: [
                  {key: 'loss', label: 'Training Loss', unit: '', color: '#3B82F6'},
                  {key: 'perplexity', label: 'Perplexity', unit: '', color: '#8B5CF6'},
                  {key: 'learning_rate', label: 'Learning Rate', unit: '', color: '#10B981'},
                  {key: 'grad_norm', label: 'Gradient Norm', unit: '', color: '#F59E0B'},
                ],
                pretrain: [
                  {key: 'loss', label: 'Training Loss', unit: '', color: '#3B82F6'},
                  {key: 'perplexity', label: 'Perplexity', unit: '', color: '#8B5CF6'},
                  {key: 'learning_rate', label: 'Learning Rate', unit: '', color: '#10B981'},
                  {key: 'grad_norm', label: 'Gradient Norm', unit: '', color: '#F59E0B'},
                ],
                cpt: [
                  {key: 'loss', label: 'Training Loss', unit: '', color: '#3B82F6'},
                  {key: 'perplexity', label: 'Perplexity', unit: '', color: '#8B5CF6'},
                  {key: 'learning_rate', label: 'Learning Rate', unit: '', color: '#10B981'},
                  {key: 'tokens_per_second', label: 'Tokens/sec', unit: 'tok/s', color: '#06B6D4'},
                ],
                // ============================================================
                // DIRECT PREFERENCE OPTIMIZATION (DPO) & VARIANTS
                // ============================================================
                dpo: [
                  {key: 'loss', label: 'DPO Loss', unit: '', color: '#3B82F6'},
                  {key: 'chosen_rewards', label: 'Chosen Rewards', unit: '', color: '#10B981'},
                  {key: 'rejected_rewards', label: 'Rejected Rewards', unit: '', color: '#EF4444'},
                  {key: 'reward_margin', label: 'Reward Margin', unit: '', color: '#F59E0B'},
                ],
                kto: [
                  {key: 'loss', label: 'KTO Loss', unit: '', color: '#3B82F6'},
                  {key: 'chosen_rewards', label: 'Desirable Rewards', unit: '', color: '#10B981'},
                  {key: 'rejected_rewards', label: 'Undesirable Rewards', unit: '', color: '#EF4444'},
                  {key: 'kl_divergence', label: 'KL Divergence', unit: '', color: '#F59E0B'},
                ],
                simpo: [
                  {key: 'loss', label: 'SimPO Loss', unit: '', color: '#3B82F6'},
                  {key: 'chosen_rewards', label: 'Chosen Rewards', unit: '', color: '#10B981'},
                  {key: 'rejected_rewards', label: 'Rejected Rewards', unit: '', color: '#EF4444'},
                  {key: 'reward_margin', label: 'Margin', unit: '', color: '#F59E0B'},
                ],
                orpo: [
                  {key: 'loss', label: 'ORPO Loss', unit: '', color: '#3B82F6'},
                  {key: 'chosen_rewards', label: 'Chosen Log Probs', unit: '', color: '#10B981'},
                  {key: 'rejected_rewards', label: 'Rejected Log Probs', unit: '', color: '#EF4444'},
                  {key: 'log_odds', label: 'Log Odds Ratio', unit: '', color: '#F59E0B'},
                ],
                rpo: [
                  {key: 'loss', label: 'RPO Loss', unit: '', color: '#3B82F6'},
                  {key: 'chosen_rewards', label: 'Chosen Rewards', unit: '', color: '#10B981'},
                  {key: 'rejected_rewards', label: 'Rejected Rewards', unit: '', color: '#EF4444'},
                  {key: 'reward_margin', label: 'Margin', unit: '', color: '#F59E0B'},
                ],
                cpo: [
                  {key: 'loss', label: 'CPO Loss', unit: '', color: '#3B82F6'},
                  {key: 'chosen_rewards', label: 'Chosen Rewards', unit: '', color: '#10B981'},
                  {key: 'rejected_rewards', label: 'Rejected Rewards', unit: '', color: '#EF4444'},
                ],
                ipo: [
                  {key: 'loss', label: 'IPO Loss', unit: '', color: '#3B82F6'},
                  {key: 'chosen_rewards', label: 'Chosen Rewards', unit: '', color: '#10B981'},
                  {key: 'rejected_rewards', label: 'Rejected Rewards', unit: '', color: '#EF4444'},
                ],
                // ============================================================
                // PPO & REINFORCEMENT LEARNING
                // ============================================================
                ppo: [
                  {key: 'reward', label: 'Reward', unit: '', color: '#10B981'},
                  {key: 'policy_loss', label: 'Policy Loss', unit: '', color: '#3B82F6'},
                  {key: 'value_loss', label: 'Value Loss', unit: '', color: '#EF4444'},
                  {key: 'entropy', label: 'Entropy', unit: 'nats', color: '#8B5CF6'},
                  {key: 'kl_divergence', label: 'KL Divergence', unit: '', color: '#F59E0B'},
                ],
                grpo: [
                  {key: 'reward', label: 'Group Reward', unit: '', color: '#10B981'},
                  {key: 'loss', label: 'GRPO Loss', unit: '', color: '#3B82F6'},
                  {key: 'kl_divergence', label: 'KL Penalty', unit: '', color: '#F59E0B'},
                  {key: 'policy_loss', label: 'Policy Loss', unit: '', color: '#EF4444'},
                ],
                rlhf: [
                  {key: 'reward', label: 'Reward', unit: '', color: '#10B981'},
                  {key: 'kl_divergence', label: 'KL Divergence', unit: '', color: '#F59E0B'},
                  {key: 'policy_loss', label: 'Policy Loss', unit: '', color: '#3B82F6'},
                  {key: 'value_loss', label: 'Value Loss', unit: '', color: '#EF4444'},
                  {key: 'entropy', label: 'Entropy', unit: 'nats', color: '#8B5CF6'},
                ],
                reinforce: [
                  {key: 'reward', label: 'Reward', unit: '', color: '#10B981'},
                  {key: 'policy_loss', label: 'Policy Loss', unit: '', color: '#3B82F6'},
                  {key: 'entropy', label: 'Entropy', unit: 'nats', color: '#8B5CF6'},
                  {key: 'kl_divergence', label: 'KL Divergence', unit: '', color: '#F59E0B'},
                ],
                // ============================================================
                // KNOWLEDGE DISTILLATION
                // ============================================================
                gkd: [
                  {key: 'loss', label: 'Total Loss', unit: '', color: '#3B82F6'},
                  {key: 'distillation_loss', label: 'Distillation Loss', unit: '', color: '#8B5CF6'},
                  {key: 'student_loss', label: 'Student Loss', unit: '', color: '#10B981'},
                  {key: 'teacher_kl', label: 'Teacher KL', unit: '', color: '#F59E0B'},
                ],
                kd: [
                  {key: 'loss', label: 'Total Loss', unit: '', color: '#3B82F6'},
                  {key: 'distillation_loss', label: 'Distillation Loss', unit: '', color: '#8B5CF6'},
                  {key: 'student_loss', label: 'Student Loss', unit: '', color: '#10B981'},
                ],
                // ============================================================
                // REWARD MODEL TRAINING
                // ============================================================
                rm: [
                  {key: 'loss', label: 'RM Loss', unit: '', color: '#3B82F6'},
                  {key: 'accuracy', label: 'Accuracy', unit: '%', color: '#10B981'},
                  {key: 'chosen_rewards', label: 'Chosen Score', unit: '', color: '#8B5CF6'},
                  {key: 'rejected_rewards', label: 'Rejected Score', unit: '', color: '#EF4444'},
                ],
                // ============================================================
                // SPEECH & AUDIO (ASR/TTS)
                // ============================================================
                asr: [
                  {key: 'loss', label: 'CTC Loss', unit: '', color: '#3B82F6'},
                  {key: 'wer', label: 'WER', unit: '%', color: '#EF4444'},
                  {key: 'cer', label: 'CER', unit: '%', color: '#F59E0B'},
                  {key: 'learning_rate', label: 'Learning Rate', unit: '', color: '#10B981'},
                ],
                tts: [
                  {key: 'loss', label: 'Total Loss', unit: '', color: '#3B82F6'},
                  {key: 'mel_loss', label: 'Mel Loss', unit: '', color: '#8B5CF6'},
                  {key: 'duration_loss', label: 'Duration Loss', unit: '', color: '#F59E0B'},
                  {key: 'pitch_loss', label: 'Pitch Loss', unit: '', color: '#10B981'},
                ],
                // ============================================================
                // MULTIMODAL (Vision-Language)
                // ============================================================
                vlm: [
                  {key: 'loss', label: 'Total Loss', unit: '', color: '#3B82F6'},
                  {key: 'image_loss', label: 'Image Loss', unit: '', color: '#8B5CF6'},
                  {key: 'text_loss', label: 'Text Loss', unit: '', color: '#10B981'},
                  {key: 'contrastive_loss', label: 'Contrastive Loss', unit: '', color: '#F59E0B'},
                ],
                mllm: [
                  {key: 'loss', label: 'Training Loss', unit: '', color: '#3B82F6'},
                  {key: 'learning_rate', label: 'Learning Rate', unit: '', color: '#10B981'},
                  {key: 'grad_norm', label: 'Gradient Norm', unit: '', color: '#F59E0B'},
                ],
              }
              
              const displayNames: Record<string, string> = {
                // SFT variants
                sft: 'Supervised Fine-Tuning',
                lora: 'LoRA Fine-Tuning',
                qlora: 'QLoRA (4-bit) Fine-Tuning',
                adalora: 'AdaLoRA Fine-Tuning',
                full: 'Full Fine-Tuning',
                // Pre-training
                pt: 'Pre-Training',
                pretrain: 'Pre-Training',
                cpt: 'Continuous Pre-Training',
                // DPO variants
                dpo: 'DPO (Direct Preference Optimization)',
                kto: 'KTO (Kahneman-Tversky Optimization)',
                simpo: 'SimPO (Simple Preference Optimization)',
                orpo: 'ORPO (Odds Ratio Preference Optimization)',
                rpo: 'RPO (Relative Preference Optimization)',
                cpo: 'CPO (Contrastive Preference Optimization)',
                ipo: 'IPO (Identity Preference Optimization)',
                // PPO/RLHF
                ppo: 'PPO (Proximal Policy Optimization)',
                grpo: 'GRPO (Group Relative Policy Optimization)',
                rlhf: 'RLHF (Reinforcement Learning from Human Feedback)',
                reinforce: 'REINFORCE Policy Gradient',
                // Knowledge Distillation
                gkd: 'GKD (Generalized Knowledge Distillation)',
                kd: 'Knowledge Distillation',
                // Reward Model
                rm: 'Reward Model Training',
                // Speech
                asr: 'ASR (Automatic Speech Recognition)',
                tts: 'TTS (Text-to-Speech)',
                // Multimodal
                vlm: 'VLM (Vision-Language Model)',
                mllm: 'MLLM (Multimodal LLM)',
              }
              
              setTrainTypeInfo({
                train_type: data.training_type,
                display_name: displayNames[data.training_type] || data.training_type.toUpperCase(),
                primary_metrics: data.relevant_metrics || ['loss', 'learning_rate'],
                graph_configs: GRAPH_CONFIGS[data.training_type] || GRAPH_CONFIGS['sft']
              })
            }
            
            // Only update metrics if we have valid data
            // CRITICAL: Never show invalid/corrupt data to user
            if (data.metrics && data.metrics.length > 0 && data.data_source !== 'error') {
              // Filter out any remaining invalid values client-side as extra safety
              const isValidValue = (val: any): boolean => {
                if (val === null || val === undefined) return false
                if (typeof val === 'number' && (isNaN(val) || !isFinite(val))) return false
                return true
              }
              
              const validatedMetrics = data.metrics
                .filter((m: any) => m.step !== undefined && m.step !== null)
                .map((m: any) => {
                  const metric: any = {
                    step: m.step,
                    timestamp: m.timestamp ? new Date(m.timestamp).getTime() : Date.now()
                  }
                  // Only include valid values
                  if (isValidValue(m.epoch)) metric.epoch = m.epoch
                  if (isValidValue(m.loss)) metric.loss = m.loss
                  if (isValidValue(m.learning_rate)) metric.learning_rate = m.learning_rate
                  if (isValidValue(m.grad_norm)) metric.grad_norm = m.grad_norm
                  if (isValidValue(m.eval_loss)) metric.eval_loss = m.eval_loss
                  if (isValidValue(m.reward)) metric.reward = m.reward
                  if (isValidValue(m.kl_divergence)) metric.kl_divergence = m.kl_divergence
                  if (isValidValue(m.policy_loss)) metric.policy_loss = m.policy_loss
                  if (isValidValue(m.value_loss)) metric.value_loss = m.value_loss
                  if (isValidValue(m.entropy)) metric.entropy = m.entropy
                  if (isValidValue(m.chosen_rewards)) metric.chosen_rewards = m.chosen_rewards
                  if (isValidValue(m.rejected_rewards)) metric.rejected_rewards = m.rejected_rewards
                  if (isValidValue(m.reward_margin)) metric.reward_margin = m.reward_margin
                  if (isValidValue(m.perplexity)) metric.perplexity = m.perplexity
                  if (isValidValue(m.accuracy)) metric.accuracy = m.accuracy
                  if (m.extra_metrics) metric.extra_metrics = m.extra_metrics
                  return metric
                })
              
              setTrainingMetrics(validatedMetrics)
            }
          }
        } catch (e) {
          console.error('Failed to fetch metrics:', e)
          // Don't update metrics on error - keep showing last valid data
        }
      }
      
      // Fetch immediately
      fetchMetrics()
      
      // Poll every 3 seconds while training
      const interval = setInterval(fetchMetrics, 3000)
      
      // Stop polling after training completes
      if (!isTraining) {
        setTimeout(() => clearInterval(interval), 10000)
      }
      
      return () => clearInterval(interval)
    }
  }, [jobStatus?.job_id, isTraining, jobStatus?.status])

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [trainingLogs])
  
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages])

  // ============================================================
  // NAVIGATION LOCK: Prevent user from leaving during training
  // ============================================================
  useEffect(() => {
    if (isTraining) {
      // Warn user before page refresh/close
      const handleBeforeUnload = (e: BeforeUnloadEvent) => {
        e.preventDefault()
        e.returnValue = 'Training is in progress. Are you sure you want to leave?'
        return e.returnValue
      }
      
      window.addEventListener('beforeunload', handleBeforeUnload)
      return () => window.removeEventListener('beforeunload', handleBeforeUnload)
    }
  }, [isTraining])

  // Lock tab switching during training
  const handleTabChange = (tab: 'train' | 'inference') => {
    if (isTraining) {
      showAlert('Training is in progress. Please wait for training to complete or stop it first.', 'warning', 'Training In Progress')
      return
    }
    setMainTab(tab)
  }

  // Dataset functions
  const fetchDatasets = async () => {
    setIsLoadingDatasets(true)
    try {
      const res = await fetch('/api/datasets/list')
      if (res.ok) {
        const data = await res.json()
        // Preserve selection state
        const newDatasets = data.datasets.map((ds: UploadedDataset) => ({
          ...ds,
          selected: uploadedDatasets.find(d => d.id === ds.id)?.selected ?? true
        }))
        setUploadedDatasets(newDatasets)
        // Update config with selected paths
        const selectedPaths = newDatasets.filter((d: UploadedDataset) => d.selected).map((d: UploadedDataset) => d.path)
        setConfig(prev => ({ ...prev, dataset_paths: selectedPaths }))
      }
    } catch (e) {
      console.error('Failed to fetch datasets:', e)
    } finally {
      setIsLoadingDatasets(false)
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setUploadFile(file)
      // Auto-suggest name from filename
      if (!uploadName) {
        const name = file.name.replace(/\.(jsonl|json|csv)$/i, '')
        setUploadName(name)
      }
    }
  }

  const uploadDataset = async () => {
    if (!uploadFile || !uploadName.trim()) {
      setUploadError('Please provide both a file and a name')
      return
    }
    
    setIsUploading(true)
    setUploadError('')
    
    try {
      const formData = new FormData()
      formData.append('file', uploadFile)
      
      const res = await fetch(`/api/datasets/upload?dataset_name=${encodeURIComponent(uploadName.trim())}`, {
        method: 'POST',
        body: formData,
      })
      
      const data = await res.json()
      
      if (data.success) {
        // Add to list with selected=true
        const newDataset: UploadedDataset = {
          id: data.id,
          name: data.name,
          filename: data.filename,
          path: data.path,
          size: data.size,
          size_human: `${(data.size / 1024).toFixed(1)} KB`,
          format: data.format,
          total_samples: data.total_samples || 0,
          created_at: Date.now() / 1000,
          selected: true
        }
        setUploadedDatasets(prev => [newDataset, ...prev])
        setConfig(prev => ({ ...prev, dataset_paths: [...prev.dataset_paths, data.path] }))
        
        // Reset form
        setUploadName('')
        setUploadFile(null)
        setShowUploadForm(false)
        if (fileInputRef.current) fileInputRef.current.value = ''
        
        // Show validation errors if any
        if (data.errors && data.errors.length > 0) {
          showAlert(`Dataset uploaded but has warnings:\n${data.errors.join('\n')}`, 'warning', 'Upload Warnings')
        }
      } else {
        setUploadError(data.detail || 'Upload failed')
      }
    } catch (e) {
      setUploadError(`Upload failed: ${e}`)
    } finally {
      setIsUploading(false)
    }
  }

  const toggleDatasetSelection = (datasetId: string) => {
    setUploadedDatasets(prev => {
      const updated = prev.map(d => 
        d.id === datasetId ? { ...d, selected: !d.selected } : d
      )
      // Update config with selected paths
      const selectedPaths = updated.filter(d => d.selected).map(d => d.path)
      setConfig(c => ({ ...c, dataset_paths: selectedPaths }))
      return updated
    })
  }

  const confirmDelete = async () => {
    if (!deleteTarget || deleteConfirmText !== deleteTarget.name) return
    
    setIsDeleting(true)
    try {
      // Pass the dataset name as confirmation (not "delete")
      const res = await fetch(
        `/api/datasets/delete/${encodeURIComponent(deleteTarget.id)}?confirm=${encodeURIComponent(deleteTarget.name)}`,
        { method: 'DELETE' }
      )
      
      if (res.ok) {
        setUploadedDatasets(prev => prev.filter(d => d.id !== deleteTarget.id))
        setConfig(prev => ({
          ...prev,
          dataset_paths: prev.dataset_paths.filter(p => p !== deleteTarget.path)
        }))
        setDeleteTarget(null)
        setDeleteConfirmText('')
      } else {
        const data = await res.json()
        showAlert(data.detail || 'Delete failed', 'error', 'Delete Failed')
      }
    } catch (e) {
      showAlert(`Delete failed: ${getErrorMessage(e)}`, 'error', 'Delete Failed')
    } finally {
      setIsDeleting(false)
    }
  }

  const startTraining = async () => {
    if (config.dataset_paths.length === 0) {
      showAlert('Please select at least one dataset for training', 'warning', 'Dataset Required')
      return
    }
    
    // Prevent double submission
    if (isStartingTraining || isTraining) {
      console.log('[START] Already starting or training, ignoring duplicate click')
      return
    }
    
    try {
      setIsStartingTraining(true)
      setTrainingLogs([])
      setLoadingMessage('Preparing for training...')
      
      // Step 1: Clean GPU memory before training to ensure maximum available VRAM
      setLoadingMessage('Cleaning GPU memory before training...')
      await deepCleanMemory()
      
      // Create job with combined dataset path and optional custom name
      const jobConfig = {
        ...config,
        dataset_path: config.dataset_paths.join(','),
        // Include training name if provided (empty = auto-generate)
        name: trainingName.trim() || undefined,
      }
      
      const createRes = await fetch('/api/jobs/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(jobConfig),
      })
      if (!createRes.ok) {
        const errData = await createRes.json().catch(() => ({}))
        throw new Error(errData.detail || `Create failed: ${createRes.status}`)
      }
      const job = await createRes.json()
      
      const startRes = await fetch(`/api/jobs/${job.job_id}/start`, {
        method: 'POST',
      })
      if (!startRes.ok) throw new Error(`Start failed: ${startRes.status}`)
      
      setJobStatus({
        job_id: job.job_id,
        job_name: job.name, // Store the job name
        status: 'running',
        current_step: 0,
        total_steps: 0,
        current_loss: null,
        logs: [],
        error: null,
      })
      setIsTraining(true)
      setCurrentStep(5)
      setTrainingName('') // Clear the input after successful creation
      setLoadingMessage('') // Clear loading message
    } catch (e) {
      showAlert(`Failed to start training: ${getErrorMessage(e)}`, 'error', 'Training Failed')
    } finally {
      setIsStartingTraining(false)
      setLoadingMessage('')
    }
  }

  // Show confirmation before stopping training
  const confirmStopTraining = () => {
    showConfirm({
      title: 'Stop Training',
      message: 'Are you sure you want to stop the current training? This action cannot be undone and all progress will be lost.',
      type: 'danger',
      confirmText: 'Stop Training',
      onConfirm: executeStopTraining
    })
  }
  
  const executeStopTraining = async () => {
    if (!jobStatus?.job_id) return
    setConfirmModal(prev => ({ ...prev, isLoading: true }))
    try {
      await fetch(`/api/jobs/${jobStatus.job_id}/stop`, { method: 'POST' })
      setIsTraining(false)
      // Reset to step 1 so user can create a new training
      setCurrentStep(1)
      setJobStatus(null)
      setTrainingLogs([])
      setTrainingMetrics([])
      closeConfirm()
    } catch (e) {
      closeConfirm()
      showAlert(`Failed to stop training: ${getErrorMessage(e)}`, 'error', 'Stop Failed')
    }
  }

  // Reset training state when completed or failed (allow new training)
  const resetTrainingState = () => {
    setIsTraining(false)
    setCurrentStep(1)
    setJobStatus(null)
    setTrainingLogs([])
    setTrainingMetrics([])
  }

  // Inference functions
  const fetchInferenceStatus = async () => {
    try {
      const res = await fetch('/api/inference/status')
      if (res.ok) {
        const data = await res.json()
        setInferenceStatus(data)
      }
    } catch (e) {
      console.error('Failed to fetch inference status:', e)
    }
  }

  // Deep memory cleanup - use before loading new models or starting training
  const deepCleanMemory = async (): Promise<boolean> => {
    setIsCleaningMemory(true)
    setLoadingMessage('Cleaning GPU memory...')
    try {
      const res = await fetch('/api/inference/deep-clear-memory', { method: 'POST' })
      const data = await res.json()
      if (data.success) {
        await fetchInferenceStatus()
        return true
      }
      console.error('Memory cleanup failed:', data.error)
      return false
    } catch (e) {
      console.error('Memory cleanup error:', e)
      return false
    } finally {
      setIsCleaningMemory(false)
    }
  }

  const clearMemory = async () => {
    setIsCleaningMemory(true)
    setLoadingMessage('Clearing memory...')
    try {
      const res = await fetch('/api/inference/clear-memory', { method: 'POST' })
      if (res.ok) {
        await fetchInferenceStatus()
      }
    } catch (e) {
      console.error(`Failed to clear memory: ${e}`)
    } finally {
      setIsCleaningMemory(false)
      setLoadingMessage('')
    }
  }

  // Load model for inference (basic - used by inference page input)
  const loadModel = async () => {
    if (!inferenceModel.trim()) return
    setIsModelLoading(true)
    setLoadingMessage('Loading model...')
    try {
      // First clean memory
      await deepCleanMemory()
      
      setLoadingMessage('Loading base model...')
      const res = await fetch('/api/inference/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: inferenceModel })
      })
      const data = await res.json()
      if (data.success) {
        await fetchInferenceStatus()
      } else {
        showAlert(`Failed to load model: ${data.error || 'Unknown error'}`, 'error', 'Model Load Failed')
      }
    } catch (e) {
      showAlert(`Failed to load model: ${getErrorMessage(e)}`, 'error', 'Model Load Failed')
    } finally {
      setIsModelLoading(false)
      setLoadingMessage('')
    }
  }

  // Comprehensive load model with adapter - used after training or from history
  const loadModelForInference = async (modelPath: string, adapterPath?: string, switchToInference: boolean = true) => {
    setIsModelLoading(true)
    setLoadingMessage('Preparing to load model...')
    
    try {
      // Step 1: Deep clean GPU memory first
      setLoadingMessage('Cleaning GPU memory...')
      const cleanResult = await deepCleanMemory()
      if (!cleanResult) {
        console.warn('Memory cleanup may have failed, continuing anyway...')
      }
      
      // Step 2: Load base model
      setLoadingMessage(`Loading base model: ${getModelDisplayName(modelPath)}...`)
      const loadRes = await fetch('/api/inference/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: modelPath })
      })
      const loadData = await loadRes.json()
      
      if (!loadData.success) {
        showAlert(`Failed to load model: ${loadData.error || 'Unknown error'}`, 'error', 'Model Load Failed')
        return false
      }
      
      // Step 3: Load adapter if provided
      if (adapterPath) {
        setLoadingMessage('Loading fine-tuned adapter...')
        const adapterRes = await fetch('/api/inference/load', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            model_path: modelPath,
            adapter_path: adapterPath 
          })
        })
        const adapterData = await adapterRes.json()
        
        if (adapterData.success) {
          // Add to loaded adapters list
          const newAdapter: LoadedAdapter = {
            id: `adapter-${Date.now()}`,
            name: adapterPath.split('/').pop() || 'fine-tuned',
            path: adapterPath,
            active: true
          }
          setLoadedAdapters(prev => [...prev.map(a => ({ ...a, active: false })), newAdapter])
        } else {
          console.warn('Adapter load failed:', adapterData.error)
        }
      }
      
      // Step 4: Update UI state
      setInferenceModel(modelPath)
      if (adapterPath) {
        setAdapterPath(adapterPath)
      }
      await fetchInferenceStatus()
      
      // Step 5: Switch to inference tab if requested
      if (switchToInference) {
        setMainTab('inference')
        setChatMessages([]) // Clear chat for fresh start
      }
      
      setLoadingMessage('Model loaded successfully!')
      setTimeout(() => setLoadingMessage(''), 2000)
      return true
      
    } catch (e) {
      console.error('Error loading model:', e)
      showAlert(`Failed to load model: ${getErrorMessage(e)}`, 'error', 'Model Load Failed')
      return false
    } finally {
      setIsModelLoading(false)
    }
  }

  // Load adapter
  const loadAdapter = async () => {
    if (!adapterPath.trim() || !inferenceStatus.model_loaded) return
    try {
      const res = await fetch('/api/inference/load-adapter', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ adapter_path: adapterPath })
      })
      const data = await res.json()
      if (data.success) {
        const newAdapter: LoadedAdapter = {
          id: `adapter-${Date.now()}`,
          name: adapterPath.split('/').pop() || 'adapter',
          path: adapterPath,
          active: true
        }
        // Deactivate others, add new
        setLoadedAdapters(prev => [...prev.map(a => ({ ...a, active: false })), newAdapter])
        setAdapterPath('')
      } else {
        showAlert(`Failed to load adapter: ${data.error || 'Unknown error'}`, 'error', 'Adapter Load Failed')
      }
    } catch (e) {
      showAlert(`Failed to load adapter: ${getErrorMessage(e)}`, 'error', 'Adapter Load Failed')
    }
  }

  // Switch active adapter
  const switchAdapter = async (adapterId: string) => {
    const adapter = loadedAdapters.find(a => a.id === adapterId)
    if (!adapter) return
    try {
      const res = await fetch('/api/inference/switch-adapter', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ adapter_path: adapter.path })
      })
      if (res.ok) {
        setLoadedAdapters(prev => prev.map(a => ({ ...a, active: a.id === adapterId })))
      }
    } catch (e) {
      console.error('Failed to switch adapter:', e)
    }
  }

  // Remove adapter
  const removeAdapter = (adapterId: string) => {
    setLoadedAdapters(prev => prev.filter(a => a.id !== adapterId))
  }

  const sendMessage = async () => {
    if (!inputMessage.trim() || !inferenceStatus.model_loaded || isGenerating) return
    
    const userMessage: ChatMessage = { role: 'user', content: inputMessage }
    
    // Build messages array with optional system prompt and history
    let messagesToSend: ChatMessage[] = []
    
    // Add system prompt if set and in chat mode
    if (chatMode === 'chat' && systemPrompt.trim()) {
      messagesToSend.push({ role: 'system', content: systemPrompt })
    }
    
    // Add history if enabled, otherwise just current message
    if (keepHistory && chatMode === 'chat') {
      messagesToSend = [...messagesToSend, ...chatMessages, userMessage]
    } else {
      messagesToSend = [...messagesToSend, userMessage]
    }
    
    const newMessages = [...chatMessages, userMessage]
    setChatMessages(newMessages)
    setInputMessage('')
    setIsGenerating(true)
    
    try {
      const activeAdapter = loadedAdapters.find(a => a.active)
      const endpoint = chatMode === 'chat' ? '/api/inference/chat' : '/api/inference/generate'
      
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_path: inferenceModel,
          adapter_path: activeAdapter?.path || null,
          messages: messagesToSend.map(m => ({ role: m.role, content: m.content })),
          prompt: chatMode === 'completion' ? inputMessage : undefined,
          ...inferenceSettings,
          do_sample: inferenceSettings.temperature > 0,
        }),
      })
      
      const data = await res.json()
      
      if (data.success && data.response) {
        setChatMessages([...newMessages, { role: 'assistant', content: data.response }])
      } else {
        setChatMessages([...newMessages, { role: 'assistant', content: `Error: ${data.error || 'Unknown error'}` }])
      }
      
      await fetchInferenceStatus()
    } catch (e) {
      setChatMessages([...newMessages, { role: 'assistant', content: `Error: ${e}` }])
    } finally {
      setIsGenerating(false)
    }
  }

  const canProceed = () => {
    // Dynamic step validation based on locked model mode
    if (IS_SINGLE_MODEL) {
      // Steps: 1=Dataset, 2=Settings, 3=Review
      switch (currentStep) {
        case 1: return config.dataset_paths.length > 0  // Dataset step
        case 2: return true  // Settings step
        case 3: return true  // Review step
        default: return false
      }
    } else {
      // Steps: 1=Model, 2=Dataset, 3=Settings, 4=Review
      switch (currentStep) {
        case 1: return config.model_path.length > 0  // Model step
        case 2: return config.dataset_paths.length > 0  // Dataset step
        case 3: return true  // Settings step
        case 4: return true  // Review step
        default: return false
      }
    }
  }

  useEffect(() => {
    if (mainTab === 'inference') {
      fetchInferenceStatus()
    }
  }, [mainTab])

  const selectedCount = uploadedDatasets.filter(d => d.selected).length

  // Format time helper
  const formatTime = (seconds: number) => {
    if (!seconds || seconds < 0) return '--:--'
    const h = Math.floor(seconds / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    const s = Math.floor(seconds % 60)
    return h > 0 ? `${h}h ${m}m` : `${m}m ${s}s`
  }

  // SYSTEM LOCKDOWN - Complete block when expired
  if (systemExpired) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center p-8">
          <div className="w-20 h-20 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
            <XCircle className="w-10 h-10 text-red-500" />
          </div>
          <h1 className="text-2xl font-bold text-white mb-3">System Requires Upgrade</h1>
          <p className="text-slate-400 max-w-md">
            This system version has expired. Please contact support to upgrade your system.
          </p>
          <p className="text-slate-500 text-sm mt-6">
            USF BIOS - Powered by US Inc
          </p>
        </div>
      </div>
    )
  }

  // Wait for expiration check before showing anything
  if (!expirationChecked) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 text-blue-500 animate-spin mx-auto mb-4" />
          <p className="text-slate-400">Initializing system...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-white">
      {/* Shimmer Animation Styles */}
      <style dangerouslySetInnerHTML={{ __html: shimmerStyles }} />
      
      {/* Global Loading Overlay - Shows during model loading or memory cleanup */}
      {(isModelLoading || isCleaningMemory) && loadingMessage && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[60]">
          <div className="bg-white rounded-2xl p-8 max-w-md w-full mx-4 shadow-2xl">
            <div className="text-center space-y-4">
              {/* Animated Icon */}
              <div className="relative mx-auto w-20 h-20">
                <div className="absolute inset-0 rounded-full bg-blue-100 animate-ping opacity-75" />
                <div className="relative w-20 h-20 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg">
                  {isCleaningMemory ? (
                    <Trash2 className="w-8 h-8 text-white animate-pulse" />
                  ) : (
                    <Cpu className="w-8 h-8 text-white" />
                  )}
                </div>
              </div>
              
              {/* Loading Spinner */}
              <Loader2 className="w-6 h-6 text-blue-500 animate-spin mx-auto" />
              
              {/* Message */}
              <div>
                <p className="text-lg font-semibold text-slate-900">
                  {isCleaningMemory ? 'Cleaning Memory' : 'Loading Model'}
                </p>
                <p className="text-sm text-slate-500 mt-1">{loadingMessage}</p>
              </div>
              
              {/* Progress Indicator */}
              <div className="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full animate-pulse" style={{ width: '60%' }} />
              </div>
              
              <p className="text-xs text-slate-400">Please wait, this may take a moment...</p>
            </div>
          </div>
        </div>
      )}
      
      {/* Delete Confirmation Modal */}
      {deleteTarget && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 max-w-md w-full mx-4 shadow-2xl">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center">
                <Trash2 className="w-5 h-5 text-red-600" />
              </div>
              <div>
                <h3 className="font-bold text-slate-900">Delete Dataset</h3>
                <p className="text-sm text-slate-500">This action cannot be undone</p>
              </div>
            </div>
            
            <div className="bg-slate-50 rounded-lg p-3 mb-4">
              <p className="text-sm text-slate-700">
                <strong>Dataset:</strong> {deleteTarget.name}
              </p>
              <p className="text-sm text-slate-500">
                {deleteTarget.total_samples} samples • {deleteTarget.size_human}
              </p>
            </div>
            
            <p className="text-sm text-slate-600 mb-2">
              To confirm deletion, type the dataset name:
            </p>
            <p className="text-sm font-mono bg-red-50 text-red-700 px-2 py-1 rounded mb-3 select-all">
              {deleteTarget.name}
            </p>
            <input
              type="text"
              value={deleteConfirmText}
              onChange={(e) => setDeleteConfirmText(e.target.value)}
              placeholder={`Type "${deleteTarget.name}" to confirm`}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg mb-4 focus:ring-2 focus:ring-red-500 focus:border-red-500"
              autoFocus
            />
            
            <div className="flex gap-2">
              <button
                onClick={() => { setDeleteTarget(null); setDeleteConfirmText('') }}
                className="flex-1 px-4 py-2 border border-slate-300 rounded-lg font-medium text-slate-700 hover:bg-slate-50"
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                disabled={deleteConfirmText !== deleteTarget.name || isDeleting}
                className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg font-medium hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isDeleting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Header - Light Theme with Blue Accents */}
      <header className="bg-white border-b border-slate-200 shadow-sm sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/30">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div className="hidden sm:block">
                <h1 className="text-lg font-bold text-slate-900">USF BIOS</h1>
                <p className="text-xs text-slate-500">AI Fine-tuning Platform</p>
              </div>
            </div>
            
            {/* Mobile menu button */}
            <button onClick={() => setMobileMenuOpen(!mobileMenuOpen)} className="sm:hidden p-2 text-slate-600 hover:text-slate-900">
              <Menu className="w-6 h-6" />
            </button>
            
            {/* Desktop tabs - Disabled during training */}
            <div className="hidden sm:flex bg-slate-100 rounded-lg p-1">
              <button
                onClick={() => handleTabChange('train')}
                disabled={isTraining}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${
                  mainTab === 'train' ? 'bg-blue-500 text-white shadow-md' : 'text-slate-600 hover:text-slate-900'
                } ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <Zap className="w-4 h-4" />Fine-tuning
                {isTraining && <Loader2 className="w-3 h-3 animate-spin" />}
              </button>
              <button
                onClick={() => handleTabChange('inference')}
                disabled={isTraining}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${
                  mainTab === 'inference' ? 'bg-blue-500 text-white shadow-md' : 'text-slate-600 hover:text-slate-900'
                } ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <MessageSquare className="w-4 h-4" />Inference
              </button>
            </div>
            
            {/* System metrics mini display - only show if data available */}
            <div className="hidden lg:flex items-center gap-4 text-xs">
              {systemMetrics.available && systemMetrics.gpu_utilization !== null && (
                <div className="flex items-center gap-1 text-slate-600">
                  <Gauge className="w-4 h-4 text-blue-500" />
                  <span>GPU: {systemMetrics.gpu_utilization}%</span>
                </div>
              )}
              {systemMetrics.available && systemMetrics.gpu_memory_used !== null && systemMetrics.gpu_memory_total !== null && (
                <div className="flex items-center gap-1 text-slate-600">
                  <HardDrive className="w-4 h-4 text-blue-500" />
                  <span>VRAM: {systemMetrics.gpu_memory_used.toFixed(1)}/{systemMetrics.gpu_memory_total.toFixed(0)}GB</span>
                </div>
              )}
              <div className="text-slate-500">Powered by US Inc</div>
            </div>
          </div>
          
          {/* Mobile menu - Disabled during training */}
          {mobileMenuOpen && (
            <div className="sm:hidden mt-3 pt-3 border-t border-slate-200 flex gap-2">
              <button 
                onClick={() => { handleTabChange('train'); setMobileMenuOpen(false) }}
                disabled={isTraining}
                className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium ${mainTab === 'train' ? 'bg-blue-500 text-white' : 'bg-slate-100 text-slate-700'} ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}>
                <Zap className="w-4 h-4 inline mr-1" />Fine-tuning
                {isTraining && <Loader2 className="w-3 h-3 inline ml-1 animate-spin" />}
              </button>
              <button 
                onClick={() => { handleTabChange('inference'); setMobileMenuOpen(false) }}
                disabled={isTraining}
                className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium ${mainTab === 'inference' ? 'bg-blue-500 text-white' : 'bg-slate-100 text-slate-700'} ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}>
                <MessageSquare className="w-4 h-4 inline mr-1" />Inference
              </button>
            </div>
          )}
        </div>
      </header>

      {/* System Status Banner - Shows when system is not live */}
      {systemStatus.status !== 'live' && !isTraining && (
        <div className={`px-4 py-3 text-center text-sm font-medium ${
          systemStatus.status === 'offline' ? 'bg-red-100 text-red-800 border-b border-red-200' :
          systemStatus.status === 'degraded' ? 'bg-yellow-100 text-yellow-800 border-b border-yellow-200' :
          systemStatus.status === 'error' ? 'bg-red-100 text-red-800 border-b border-red-200' :
          systemStatus.status === 'starting' ? 'bg-blue-100 text-blue-800 border-b border-blue-200' :
          'bg-slate-100 text-slate-800 border-b border-slate-200'
        }`}>
          <div className="max-w-6xl mx-auto flex items-center justify-center gap-2">
            {systemStatus.status === 'offline' && <XCircle className="w-4 h-4" />}
            {systemStatus.status === 'degraded' && <AlertCircle className="w-4 h-4" />}
            {systemStatus.status === 'error' && <XCircle className="w-4 h-4" />}
            {systemStatus.status === 'starting' && <Loader2 className="w-4 h-4 animate-spin" />}
            {systemStatus.status === 'unknown' && <AlertCircle className="w-4 h-4" />}
            <span>
              <strong>System Status: {systemStatus.status.toUpperCase()}</strong> — {systemStatus.message}
            </span>
            <button onClick={fetchSystemStatus} className="ml-2 p-1 hover:bg-black/10 rounded">
              <RefreshCw className="w-3 h-3" />
            </button>
          </div>
        </div>
      )}

      {/* Training In Progress Banner - Shows when training is active */}
      {isTraining && (
        <div className="bg-blue-600 text-white px-4 py-3 text-center text-sm font-medium border-b border-blue-700">
          <div className="max-w-6xl mx-auto flex items-center justify-center gap-2">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>
              <strong>Training in Progress</strong> — Navigation is locked until training completes or is stopped
            </span>
          </div>
        </div>
      )}

      <main className="max-w-6xl mx-auto px-4 py-6">
        
        {/* ===================== TRAINING VIEW - PROGRESS OR RESULT ===================== */}
        {/* Show when training is active OR when we have a job result (completed/failed) */}
        {(isTraining || (jobStatus && (jobStatus.status === 'completed' || jobStatus.status === 'failed' || jobStatus.status === 'stopped'))) && jobStatus && (
          <div className="bg-white rounded-xl shadow-lg border border-slate-200 p-4 sm:p-6">
            <div className="space-y-4">
              {/* Model Info Banner - Shows which model is being trained */}
              <div className="bg-gradient-to-r from-slate-50 to-blue-50 border border-slate-200 rounded-lg p-3 flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center flex-shrink-0">
                  <Cpu className="w-5 h-5 text-blue-600" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-slate-500 uppercase tracking-wide font-medium">Model</p>
                  <p className="font-semibold text-slate-900 truncate">{getModelDisplayName(config.model_path)}</p>
                </div>
                {IS_MODEL_LOCKED && <Lock className="w-4 h-4 text-slate-400 flex-shrink-0" />}
              </div>
              
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <h2 className="text-xl font-bold text-slate-900">Training Progress</h2>
                  <p className="text-slate-500 text-sm">
                    <span className="font-medium text-slate-700">{jobStatus.job_name}</span>
                    <span className="mx-2">•</span>
                    <span className="font-mono text-xs">{jobStatus.job_id}</span>
                  </p>
                </div>
                <div className={`px-3 py-1 rounded-full text-sm font-medium self-start ${
                  jobStatus.status === 'running' ? 'bg-blue-100 text-blue-700' :
                  jobStatus.status === 'completed' ? 'bg-green-100 text-green-700' :
                  jobStatus.status === 'failed' ? 'bg-red-100 text-red-700' : 
                  jobStatus.status === 'stopped' ? 'bg-yellow-100 text-yellow-700' : 'bg-slate-100 text-slate-600'
                }`}>
                  {jobStatus.status === 'running' && <Loader2 className="w-4 h-4 inline mr-1 animate-spin" />}
                  {jobStatus.status === 'completed' && <CheckCircle className="w-4 h-4 inline mr-1" />}
                  {jobStatus.status === 'failed' && <XCircle className="w-4 h-4 inline mr-1" />}
                  {jobStatus.status === 'stopped' && <StopCircle className="w-4 h-4 inline mr-1" />}
                  {jobStatus.status.toUpperCase()}
                </div>
              </div>
              
              {/* Progress bar */}
              {jobStatus.total_steps > 0 && (
                <div className="bg-slate-50 rounded-lg p-3 border border-slate-200">
                  <div className="flex justify-between text-sm text-slate-600 mb-2">
                    <span>Step {jobStatus.current_step} / {jobStatus.total_steps}</span>
                    <span>{Math.round((jobStatus.current_step / jobStatus.total_steps) * 100)}%</span>
                  </div>
                  <div className="w-full h-3 bg-slate-200 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-500 transition-all"
                      style={{ width: `${(jobStatus.current_step / jobStatus.total_steps) * 100}%` }} />
                  </div>
                  {jobStatus.eta_seconds && jobStatus.eta_seconds > 0 && (
                    <p className="text-xs text-slate-500 mt-2 flex items-center gap-1">
                      <Clock className="w-3 h-3" /> ETA: {formatTime(jobStatus.eta_seconds)}
                    </p>
                  )}
                </div>
              )}
              
              {/* Real-time Metrics - Use trainingMetrics for final values when training completes */}
              {(() => {
                // Get final values from trainingMetrics if available (more reliable after training)
                const lastMetric = trainingMetrics.length > 0 ? trainingMetrics[trainingMetrics.length - 1] : null
                const finalLoss = lastMetric?.loss ?? jobStatus.current_loss
                const finalEpoch = lastMetric?.epoch ?? jobStatus.epoch
                const finalLR = lastMetric?.learning_rate ?? jobStatus.learning_rate
                
                return (
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-3 text-center border border-blue-200">
                  <BarChart3 className="w-5 h-5 mx-auto text-blue-600 mb-1" />
                  <span className="text-[10px] text-blue-600 font-medium uppercase">Loss</span>
                  <p className="text-xl font-bold text-blue-900">{finalLoss !== null && finalLoss !== undefined ? finalLoss.toFixed(4) : '--'}</p>
                </div>
                <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-3 text-center border border-purple-200">
                  <Layers className="w-5 h-5 mx-auto text-purple-600 mb-1" />
                  <span className="text-[10px] text-purple-600 font-medium uppercase">Epoch</span>
                  <p className="text-xl font-bold text-purple-900">{finalEpoch !== null && finalEpoch !== undefined ? finalEpoch : '--'}</p>
                </div>
                <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-3 text-center border border-green-200">
                  <Activity className="w-5 h-5 mx-auto text-green-600 mb-1" />
                  <span className="text-[10px] text-green-600 font-medium uppercase">LR</span>
                  <p className="text-xl font-bold text-green-900">{finalLR ? finalLR.toExponential(1) : '--'}</p>
                </div>
                <div className={`bg-gradient-to-br ${systemMetrics.available && systemMetrics.gpu_utilization !== null ? 'from-cyan-50 to-cyan-100 border-cyan-200' : 'from-slate-50 to-slate-100 border-slate-200'} rounded-lg p-3 text-center border`}>
                  <Cpu className="w-5 h-5 mx-auto text-cyan-600 mb-1" />
                  <span className="text-[10px] text-cyan-600 font-medium uppercase">GPU %</span>
                  <p className={`text-xl font-bold ${systemMetrics.available && systemMetrics.gpu_utilization !== null ? 'text-cyan-900' : 'text-slate-400'}`}>
                    {systemMetrics.available && systemMetrics.gpu_utilization !== null ? `${systemMetrics.gpu_utilization}%` : '--'}
                  </p>
                </div>
                <div className={`bg-gradient-to-br ${systemMetrics.available && systemMetrics.gpu_memory_used !== null ? 'from-amber-50 to-amber-100 border-amber-200' : 'from-slate-50 to-slate-100 border-slate-200'} rounded-lg p-3 text-center border`}>
                  <HardDrive className="w-5 h-5 mx-auto text-amber-600 mb-1" />
                  <span className="text-[10px] text-amber-600 font-medium uppercase">VRAM</span>
                  <p className={`text-xl font-bold ${systemMetrics.available && systemMetrics.gpu_memory_used !== null ? 'text-amber-900' : 'text-slate-400'}`}>
                    {systemMetrics.available && systemMetrics.gpu_memory_used !== null && systemMetrics.gpu_memory_total !== null 
                      ? <>{systemMetrics.gpu_memory_used.toFixed(1)}<span className="text-xs font-normal">/{systemMetrics.gpu_memory_total.toFixed(0)}G</span></>
                      : '--'}
                  </p>
                </div>
                <div className={`bg-gradient-to-br ${systemMetrics.available && systemMetrics.gpu_temperature !== null ? 'from-orange-50 to-orange-100 border-orange-200' : 'from-slate-50 to-slate-100 border-slate-200'} rounded-lg p-3 text-center border`}>
                  <Thermometer className="w-5 h-5 mx-auto text-orange-600 mb-1" />
                  <span className="text-[10px] text-orange-600 font-medium uppercase">Temp</span>
                  <p className={`text-xl font-bold ${systemMetrics.available && systemMetrics.gpu_temperature !== null ? 'text-orange-900' : 'text-slate-400'}`}>
                    {systemMetrics.available && systemMetrics.gpu_temperature !== null ? `${systemMetrics.gpu_temperature}°` : '--'}
                  </p>
                </div>
              </div>
                )
              })()}

              {/* ============================================================ */}
              {/* PROFESSIONAL TRAINING GRAPHS - Dynamic based on training type */}
              {/* Data validated: Only shows accurate data, hides if unavailable */}
              {/* ============================================================ */}
              {trainingMetrics.length > 1 && (
                <div className="space-y-4">
                  {/* Training Type Header with Data Source Indicator */}
                  {trainTypeInfo && (
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                      <h3 className="text-sm font-semibold text-slate-700">
                        {trainTypeInfo.display_name} Metrics
                      </h3>
                      <div className="flex items-center gap-2">
                        {/* Data Source Badge */}
                        <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                          metricsDataSource === 'tensorboard' 
                            ? 'bg-green-100 text-green-700' 
                            : metricsDataSource === 'database'
                            ? 'bg-blue-100 text-blue-700'
                            : 'bg-slate-100 text-slate-500'
                        }`}>
                          {metricsDataSource === 'tensorboard' ? '● TensorBoard' : 
                           metricsDataSource === 'database' ? '● Parsed Logs' : '○ Loading...'}
                        </span>
                        <span className="text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded">
                          {trainingMetrics.length} points
                        </span>
                      </div>
                    </div>
                  )}
                  
                  {/* Dynamic Metric Graphs based on training type */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {(trainTypeInfo?.graph_configs || [
                      {key: 'loss', label: 'Training Loss', unit: '', color: '#3B82F6'},
                      {key: 'learning_rate', label: 'Learning Rate', unit: '', color: '#10B981'}
                    ]).map((graphConfig) => {
                      const metricKey = graphConfig.key as keyof TrainingMetric
                      const data = trainingMetrics.filter(m => {
                        const val = m[metricKey]
                        return val !== undefined && val !== null && typeof val === 'number' && !isNaN(val)
                      })
                      if (data.length < 2) return null
                      
                      const values = data.map(m => m[metricKey] as number)
                      const maxVal = Math.max(...values)
                      const minVal = Math.min(...values)
                      const range = maxVal - minVal || 1
                      const currentVal = values[values.length - 1]
                      const isExponential = Math.abs(maxVal) < 0.01 || Math.abs(maxVal) > 1000
                      
                      const formatValue = (v: number) => {
                        if (isExponential) return v.toExponential(2)
                        if (v < 1) return v.toFixed(4)
                        return v.toFixed(2)
                      }
                      
                      // Calculate Y-axis ticks (5 ticks)
                      const yTicks = Array.from({length: 5}, (_, i) => minVal + (range * i / 4))
                      
                      return (
                        <div key={graphConfig.key} className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
                          {/* Graph Header */}
                          <div className="px-4 py-2 border-b border-slate-100 bg-slate-50 flex items-center justify-between">
                            <h4 className="text-sm font-medium text-slate-700 flex items-center gap-2">
                              <div className="w-3 h-3 rounded-full" style={{backgroundColor: graphConfig.color}} />
                              {graphConfig.label}
                              {graphConfig.unit && <span className="text-slate-400 text-xs">({graphConfig.unit})</span>}
                            </h4>
                            <span className="text-xs font-mono bg-white px-2 py-0.5 rounded border border-slate-200" style={{color: graphConfig.color}}>
                              {formatValue(currentVal)}
                            </span>
                          </div>
                          
                          {/* Graph Body */}
                          <div className="p-3">
                            <div className="flex">
                              {/* Y-Axis Labels */}
                              <div className="flex flex-col justify-between text-[9px] text-slate-400 pr-2 h-32 w-12 flex-shrink-0">
                                {yTicks.reverse().map((tick, i) => (
                                  <span key={i} className="text-right font-mono">{formatValue(tick)}</span>
                                ))}
                              </div>
                              
                              {/* Graph Area */}
                              <div className="flex-1 h-32 relative">
                                {/* Grid Lines */}
                                <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
                                  {[0, 25, 50, 75, 100].map(y => (
                                    <line key={y} x1="0" y1={`${y}%`} x2="100%" y2={`${y}%`} stroke="#e2e8f0" strokeWidth="1" />
                                  ))}
                                  {[0, 25, 50, 75, 100].map(x => (
                                    <line key={x} x1={`${x}%`} y1="0" x2={`${x}%`} y2="100%" stroke="#e2e8f0" strokeWidth="1" strokeDasharray="2,2" />
                                  ))}
                                </svg>
                                
                                {/* Data Line */}
                                <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                                  {/* Area fill */}
                                  <path
                                    fill={graphConfig.color}
                                    fillOpacity="0.1"
                                    d={`M 0,100 ${data.map((m, i) => {
                                      const x = (i / (data.length - 1)) * 100
                                      const y = 100 - ((m[metricKey] as number - minVal) / range) * 95 - 2.5
                                      return `L ${x},${y}`
                                    }).join(' ')} L 100,100 Z`}
                                  />
                                  {/* Line */}
                                  <polyline
                                    fill="none"
                                    stroke={graphConfig.color}
                                    strokeWidth="2"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    points={data.map((m, i) => {
                                      const x = (i / (data.length - 1)) * 100
                                      const y = 100 - ((m[metricKey] as number - minVal) / range) * 95 - 2.5
                                      return `${x},${y}`
                                    }).join(' ')}
                                  />
                                </svg>
                              </div>
                            </div>
                            
                            {/* X-Axis Labels */}
                            <div className="flex justify-between text-[9px] text-slate-400 mt-1 ml-12 font-mono">
                              <span>Step {data[0]?.step || 0}</span>
                              <span>Step {data[Math.floor(data.length / 2)]?.step || 0}</span>
                              <span>Step {data[data.length - 1]?.step || 0}</span>
                            </div>
                          </div>
                          
                          {/* Graph Footer - Stats */}
                          <div className="px-4 py-2 bg-slate-50 border-t border-slate-100 grid grid-cols-3 gap-2 text-xs">
                            <div className="text-center">
                              <span className="text-slate-400 block">Min</span>
                              <span className="font-mono text-slate-600">{formatValue(minVal)}</span>
                            </div>
                            <div className="text-center">
                              <span className="text-slate-400 block">Max</span>
                              <span className="font-mono text-slate-600">{formatValue(maxVal)}</span>
                            </div>
                            <div className="text-center">
                              <span className="text-slate-400 block">Current</span>
                              <span className="font-mono font-semibold" style={{color: graphConfig.color}}>{formatValue(currentVal)}</span>
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                  
                  {/* ============================================================ */}
                  {/* SYSTEM METRICS GRAPHS (Weights & Biases Style) */}
                  {/* GPU Utilization, VRAM, Temperature over time */}
                  {/* ============================================================ */}
                  {gpuMetricsHistory.length > 3 && (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                          <Cpu className="w-4 h-4 text-cyan-500" />
                          System Metrics
                        </h3>
                        <span className="text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded">
                          {gpuMetricsHistory.length} samples
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {/* GPU Utilization Graph */}
                        {(() => {
                          const data = gpuMetricsHistory.filter(m => m.gpu_utilization !== null)
                          if (data.length < 2) return null
                          const values = data.map(m => m.gpu_utilization || 0)
                          const minVal = Math.min(...values)
                          const maxVal = Math.max(...values)
                          const avgVal = values.reduce((a, b) => a + b, 0) / values.length
                          const currentVal = values[values.length - 1]
                          
                          return (
                            <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
                              <div className="px-4 py-2 border-b border-slate-100 bg-gradient-to-r from-cyan-50 to-slate-50 flex items-center justify-between">
                                <h4 className="text-sm font-medium text-slate-700 flex items-center gap-2">
                                  <div className="w-3 h-3 rounded-full bg-cyan-500" />
                                  GPU Utilization
                                  <span className="text-slate-400 text-xs">(%)</span>
                                </h4>
                                <span className="text-xs font-mono bg-white px-2 py-0.5 rounded border border-cyan-200 text-cyan-600">
                                  {currentVal.toFixed(0)}%
                                </span>
                              </div>
                              <div className="p-3">
                                <div className="flex">
                                  <div className="flex flex-col justify-between text-[9px] text-slate-400 pr-2 h-24 w-10 flex-shrink-0 font-mono">
                                    <span>100%</span>
                                    <span>50%</span>
                                    <span>0%</span>
                                  </div>
                                  <div className="flex-1 h-24 relative bg-slate-50 rounded">
                                    <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
                                      {[0, 25, 50, 75, 100].map(y => (
                                        <line key={y} x1="0" y1={`${y}%`} x2="100%" y2={`${y}%`} stroke="#e2e8f0" strokeWidth="1" />
                                      ))}
                                    </svg>
                                    <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                                      <path fill="#06b6d4" fillOpacity="0.15"
                                        d={`M 0,100 ${data.map((m, i) => `L ${(i / (data.length - 1)) * 100},${100 - (m.gpu_utilization || 0)}`).join(' ')} L 100,100 Z`}
                                      />
                                      <polyline fill="none" stroke="#06b6d4" strokeWidth="2" strokeLinecap="round"
                                        points={data.map((m, i) => `${(i / (data.length - 1)) * 100},${100 - (m.gpu_utilization || 0)}`).join(' ')}
                                      />
                                    </svg>
                                  </div>
                                </div>
                              </div>
                              <div className="px-4 py-2 bg-slate-50 border-t border-slate-100 grid grid-cols-3 gap-2 text-xs">
                                <div className="text-center">
                                  <span className="text-slate-400 block">Min</span>
                                  <span className="font-mono text-slate-600">{minVal.toFixed(0)}%</span>
                                </div>
                                <div className="text-center">
                                  <span className="text-slate-400 block">Avg</span>
                                  <span className="font-mono text-slate-600">{avgVal.toFixed(0)}%</span>
                                </div>
                                <div className="text-center">
                                  <span className="text-slate-400 block">Max</span>
                                  <span className="font-mono text-cyan-600 font-semibold">{maxVal.toFixed(0)}%</span>
                                </div>
                              </div>
                            </div>
                          )
                        })()}
                        
                        {/* VRAM Usage Graph */}
                        {(() => {
                          const data = gpuMetricsHistory.filter(m => m.gpu_memory_used !== null)
                          if (data.length < 2) return null
                          const maxMem = systemMetrics.gpu_memory_total || 24
                          const values = data.map(m => m.gpu_memory_used || 0)
                          const minVal = Math.min(...values)
                          const maxVal = Math.max(...values)
                          const avgVal = values.reduce((a, b) => a + b, 0) / values.length
                          const currentVal = values[values.length - 1]
                          
                          return (
                            <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
                              <div className="px-4 py-2 border-b border-slate-100 bg-gradient-to-r from-amber-50 to-slate-50 flex items-center justify-between">
                                <h4 className="text-sm font-medium text-slate-700 flex items-center gap-2">
                                  <div className="w-3 h-3 rounded-full bg-amber-500" />
                                  VRAM Usage
                                  <span className="text-slate-400 text-xs">(GB)</span>
                                </h4>
                                <span className="text-xs font-mono bg-white px-2 py-0.5 rounded border border-amber-200 text-amber-600">
                                  {currentVal.toFixed(1)}/{maxMem.toFixed(0)}G
                                </span>
                              </div>
                              <div className="p-3">
                                <div className="flex">
                                  <div className="flex flex-col justify-between text-[9px] text-slate-400 pr-2 h-24 w-10 flex-shrink-0 font-mono">
                                    <span>{maxMem.toFixed(0)}G</span>
                                    <span>{(maxMem / 2).toFixed(0)}G</span>
                                    <span>0G</span>
                                  </div>
                                  <div className="flex-1 h-24 relative bg-slate-50 rounded">
                                    <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
                                      {[0, 25, 50, 75, 100].map(y => (
                                        <line key={y} x1="0" y1={`${y}%`} x2="100%" y2={`${y}%`} stroke="#e2e8f0" strokeWidth="1" />
                                      ))}
                                    </svg>
                                    <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                                      <path fill="#f59e0b" fillOpacity="0.15"
                                        d={`M 0,100 ${data.map((m, i) => `L ${(i / (data.length - 1)) * 100},${100 - ((m.gpu_memory_used || 0) / maxMem) * 100}`).join(' ')} L 100,100 Z`}
                                      />
                                      <polyline fill="none" stroke="#f59e0b" strokeWidth="2" strokeLinecap="round"
                                        points={data.map((m, i) => `${(i / (data.length - 1)) * 100},${100 - ((m.gpu_memory_used || 0) / maxMem) * 100}`).join(' ')}
                                      />
                                    </svg>
                                  </div>
                                </div>
                              </div>
                              <div className="px-4 py-2 bg-slate-50 border-t border-slate-100 grid grid-cols-3 gap-2 text-xs">
                                <div className="text-center">
                                  <span className="text-slate-400 block">Min</span>
                                  <span className="font-mono text-slate-600">{minVal.toFixed(1)}G</span>
                                </div>
                                <div className="text-center">
                                  <span className="text-slate-400 block">Avg</span>
                                  <span className="font-mono text-slate-600">{avgVal.toFixed(1)}G</span>
                                </div>
                                <div className="text-center">
                                  <span className="text-slate-400 block">Peak</span>
                                  <span className="font-mono text-amber-600 font-semibold">{maxVal.toFixed(1)}G</span>
                                </div>
                              </div>
                            </div>
                          )
                        })()}
                        
                        {/* GPU Temperature Graph */}
                        {(() => {
                          const data = gpuMetricsHistory.filter(m => m.gpu_temperature !== null)
                          if (data.length < 2) return null
                          const values = data.map(m => m.gpu_temperature || 0)
                          const minVal = Math.min(...values)
                          const maxVal = Math.max(...values)
                          const avgVal = values.reduce((a, b) => a + b, 0) / values.length
                          const currentVal = values[values.length - 1]
                          // Temperature scale: 30-90°C
                          const tempMin = 30, tempMax = 90, tempRange = tempMax - tempMin
                          
                          return (
                            <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
                              <div className="px-4 py-2 border-b border-slate-100 bg-gradient-to-r from-orange-50 to-slate-50 flex items-center justify-between">
                                <h4 className="text-sm font-medium text-slate-700 flex items-center gap-2">
                                  <div className="w-3 h-3 rounded-full bg-orange-500" />
                                  GPU Temperature
                                  <span className="text-slate-400 text-xs">(°C)</span>
                                </h4>
                                <span className={`text-xs font-mono bg-white px-2 py-0.5 rounded border ${
                                  currentVal > 80 ? 'border-red-200 text-red-600' : 
                                  currentVal > 70 ? 'border-orange-200 text-orange-600' : 
                                  'border-green-200 text-green-600'
                                }`}>
                                  {currentVal.toFixed(0)}°C
                                </span>
                              </div>
                              <div className="p-3">
                                <div className="flex">
                                  <div className="flex flex-col justify-between text-[9px] text-slate-400 pr-2 h-24 w-10 flex-shrink-0 font-mono">
                                    <span>{tempMax}°</span>
                                    <span>{(tempMin + tempMax) / 2}°</span>
                                    <span>{tempMin}°</span>
                                  </div>
                                  <div className="flex-1 h-24 relative bg-slate-50 rounded">
                                    <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
                                      {[0, 25, 50, 75, 100].map(y => (
                                        <line key={y} x1="0" y1={`${y}%`} x2="100%" y2={`${y}%`} stroke="#e2e8f0" strokeWidth="1" />
                                      ))}
                                      {/* Warning zone (>70°C) */}
                                      <rect x="0" y="0" width="100%" height={`${((tempMax - 70) / tempRange) * 100}%`} fill="#fef3c7" fillOpacity="0.5" />
                                      {/* Danger zone (>80°C) */}
                                      <rect x="0" y="0" width="100%" height={`${((tempMax - 80) / tempRange) * 100}%`} fill="#fee2e2" fillOpacity="0.5" />
                                    </svg>
                                    <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                                      <path fill="#f97316" fillOpacity="0.15"
                                        d={`M 0,100 ${data.map((m, i) => {
                                          const y = 100 - (((m.gpu_temperature || tempMin) - tempMin) / tempRange) * 100
                                          return `L ${(i / (data.length - 1)) * 100},${Math.max(0, Math.min(100, y))}`
                                        }).join(' ')} L 100,100 Z`}
                                      />
                                      <polyline fill="none" stroke="#f97316" strokeWidth="2" strokeLinecap="round"
                                        points={data.map((m, i) => {
                                          const y = 100 - (((m.gpu_temperature || tempMin) - tempMin) / tempRange) * 100
                                          return `${(i / (data.length - 1)) * 100},${Math.max(0, Math.min(100, y))}`
                                        }).join(' ')}
                                      />
                                    </svg>
                                  </div>
                                </div>
                              </div>
                              <div className="px-4 py-2 bg-slate-50 border-t border-slate-100 grid grid-cols-3 gap-2 text-xs">
                                <div className="text-center">
                                  <span className="text-slate-400 block">Min</span>
                                  <span className="font-mono text-slate-600">{minVal.toFixed(0)}°C</span>
                                </div>
                                <div className="text-center">
                                  <span className="text-slate-400 block">Avg</span>
                                  <span className="font-mono text-slate-600">{avgVal.toFixed(0)}°C</span>
                                </div>
                                <div className="text-center">
                                  <span className="text-slate-400 block">Peak</span>
                                  <span className={`font-mono font-semibold ${
                                    maxVal > 80 ? 'text-red-600' : maxVal > 70 ? 'text-orange-600' : 'text-green-600'
                                  }`}>{maxVal.toFixed(0)}°C</span>
                                </div>
                              </div>
                            </div>
                          )
                        })()}
                      </div>
                    </div>
                  )}
                </div>
              )}
              
              {/* Placeholder when no metrics yet */}
              {trainingMetrics.length <= 1 && isTraining && (
                <div className="bg-slate-50 rounded-lg border border-slate-200 p-6 text-center">
                  <Loader2 className="w-8 h-8 animate-spin text-blue-500 mx-auto mb-2" />
                  <p className="text-sm text-slate-600">Waiting for training metrics...</p>
                  <p className="text-xs text-slate-400 mt-1">Metrics will appear after the first training step</p>
                </div>
              )}
              
              {/* Terminal Logs */}
              <div className="bg-slate-900 rounded-lg border border-slate-700 flex flex-col h-64">
                <div className="flex-shrink-0 px-3 py-2 border-b border-slate-700 text-slate-500 text-[10px] bg-slate-900 rounded-t-lg">
                  TERMINAL OUTPUT ({trainingLogs.length} lines)
                </div>
                <div className="flex-1 overflow-y-auto p-3 font-mono text-xs text-green-400">
                  {trainingLogs.length === 0 ? (
                    <div className="text-slate-500 text-center py-4">Waiting for training output...</div>
                  ) : (
                    trainingLogs.map((log, i) => (
                      <div key={i} className="hover:bg-slate-800/50 py-0.5 whitespace-pre-wrap break-all">{log}</div>
                    ))
                  )}
                  <div ref={logsEndRef} />
                </div>
              </div>
              
              {/* Training Completion Summary - show when completed */}
              {jobStatus.status === 'completed' && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <h4 className="font-semibold text-green-800 flex items-center gap-2 mb-3">
                    <CheckCircle className="w-5 h-5" /> Training Completed Successfully
                  </h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-start gap-2">
                      <FolderOpen className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                      <div>
                        <span className="text-green-700 font-medium">Output Path:</span>
                        <code className="block text-green-800 bg-green-100 px-2 py-1 rounded mt-1 text-xs break-all">
                          /app/data/outputs/{jobStatus.job_id}
                        </code>
                      </div>
                    </div>
                    {trainingMetrics.length > 0 && (
                      <div className="grid grid-cols-2 gap-2 mt-3 pt-3 border-t border-green-200">
                        <div className="text-green-700">
                          <span className="font-medium">Final Loss:</span> {trainingMetrics[trainingMetrics.length - 1]?.loss?.toFixed(4) || '--'}
                        </div>
                        <div className="text-green-700">
                          <span className="font-medium">Total Steps:</span> {jobStatus.current_step || trainingMetrics[trainingMetrics.length - 1]?.step || '--'}
                        </div>
                        <div className="text-green-700">
                          <span className="font-medium">Epochs:</span> {trainingMetrics[trainingMetrics.length - 1]?.epoch || jobStatus.total_epochs || '--'}
                        </div>
                        <div className="text-green-700">
                          <span className="font-medium">Training Type:</span> {trainTypeInfo?.display_name || 'SFT'}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              {/* Error message when failed */}
              {jobStatus.status === 'failed' && jobStatus.error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
                  <strong>Error:</strong> {jobStatus.error}
                </div>
              )}
              
              {/* Stop Training Button - only show when running */}
              {isTraining && (
                <button onClick={confirmStopTraining}
                  className="w-full py-3 bg-red-500 text-white rounded-lg font-medium hover:bg-red-600 flex items-center justify-center gap-2">
                  <StopCircle className="w-5 h-5" /> Stop Training
                </button>
              )}
              
              {/* Start New Training Button - show when completed/failed/stopped */}
              {!isTraining && (jobStatus.status === 'completed' || jobStatus.status === 'failed' || jobStatus.status === 'stopped') && (
                <div className="space-y-3">
                  {/* Load for Inference - Only show for completed training with adapter */}
                  {jobStatus.status === 'completed' && (
                    <button 
                      onClick={() => loadModelForInference(config.model_path, config.output_dir)}
                      disabled={isModelLoading || isCleaningMemory}
                      className="w-full py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg font-medium hover:from-green-600 hover:to-emerald-700 disabled:opacity-50 flex items-center justify-center gap-2 shadow-lg shadow-green-500/20 transition-all">
                      {isModelLoading ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin" />
                          {loadingMessage || 'Loading...'}
                        </>
                      ) : (
                        <>
                          <MessageSquare className="w-5 h-5" />
                          Load for Inference
                        </>
                      )}
                    </button>
                  )}
                  
                  <div className="flex gap-2">
                    <button onClick={resetTrainingState}
                      disabled={isModelLoading}
                      className="flex-1 py-3 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 disabled:opacity-50 flex items-center justify-center gap-2">
                      <Play className="w-5 h-5" /> Start New Training
                    </button>
                    <button onClick={() => { setShowHistory(true); fetchTrainingHistory(); }}
                      disabled={isModelLoading}
                      className="px-4 py-3 bg-slate-100 text-slate-700 rounded-lg font-medium hover:bg-slate-200 disabled:opacity-50 flex items-center justify-center gap-2 border border-slate-200">
                      <History className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ===================== TRAINING HISTORY MODAL ===================== */}
        {showHistory && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-2 sm:p-4">
            <div className="bg-white rounded-xl shadow-2xl w-full max-w-5xl max-h-[90vh] sm:max-h-[85vh] overflow-hidden flex flex-col">
              {/* Header */}
              <div className="flex items-center justify-between p-3 sm:p-4 border-b border-slate-200 bg-gradient-to-r from-blue-50 to-slate-50">
                <div>
                  <h2 className="text-lg sm:text-xl font-bold text-slate-900 flex items-center gap-2">
                    <History className="w-5 h-5 text-blue-500" /> Training History
                  </h2>
                  <p className="text-xs text-slate-500 mt-0.5">{trainingHistory.length} training runs</p>
                </div>
                <button onClick={() => setShowHistory(false)} className="p-2 hover:bg-white rounded-lg transition-colors">
                  <X className="w-5 h-5 text-slate-500" />
                </button>
              </div>
              
              {/* Content */}
              <div className="flex-1 overflow-y-auto p-3 sm:p-4">
                {isLoadingHistory ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
                  </div>
                ) : trainingHistory.length === 0 ? (
                  <div className="text-center py-12 text-slate-500">
                    <History className="w-12 h-12 mx-auto mb-3 opacity-30" />
                    <p className="font-medium">No training history found</p>
                    <p className="text-sm mt-1">Complete a training run to see it here</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {trainingHistory.map((item) => (
                      <div key={item.job_id} className={`border rounded-xl overflow-hidden transition-all hover:shadow-md ${
                        item.status === 'completed' ? 'border-green-200 bg-gradient-to-r from-green-50/80 to-white' :
                        item.status === 'failed' ? 'border-red-200 bg-gradient-to-r from-red-50/80 to-white' :
                        item.status === 'running' ? 'border-blue-200 bg-gradient-to-r from-blue-50/80 to-white' :
                        'border-slate-200 bg-gradient-to-r from-slate-50/80 to-white'
                      }`}>
                        {/* Training Info Header */}
                        <div className="p-3 sm:p-4">
                          <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 sm:gap-4">
                            <div className="flex-1 min-w-0">
                              {/* Name and Status */}
                              <div className="flex flex-wrap items-center gap-2 mb-2">
                                <span className="font-semibold text-slate-900 truncate">{item.job_name}</span>
                                <span className={`text-[10px] sm:text-xs px-2 py-0.5 rounded-full font-medium whitespace-nowrap ${
                                  item.status === 'completed' ? 'bg-green-100 text-green-700' :
                                  item.status === 'failed' ? 'bg-red-100 text-red-700' :
                                  item.status === 'running' ? 'bg-blue-100 text-blue-700 animate-pulse' :
                                  'bg-slate-100 text-slate-600'
                                }`}>
                                  {item.status === 'running' && <span className="inline-block w-1.5 h-1.5 bg-blue-500 rounded-full mr-1 animate-pulse" />}
                                  {item.status.toUpperCase()}
                                </span>
                                {item.config && (
                                  <span className="text-[10px] sm:text-xs px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 font-medium whitespace-nowrap">
                                    {item.config.training_method.toUpperCase()} • {item.config.train_type.toUpperCase()}
                                  </span>
                                )}
                              </div>
                              
                              {/* Date and Config Details */}
                              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500">
                                {item.created_at && (
                                  <span className="flex items-center gap-1">
                                    <Clock className="w-3 h-3" />
                                    {new Date(item.created_at).toLocaleDateString()} {new Date(item.created_at).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                                  </span>
                                )}
                                {item.config && (
                                  <>
                                    <span>LR: {item.config.learning_rate}</span>
                                    <span>Batch: {item.config.batch_size}</span>
                                    <span>Epochs: {item.config.num_epochs}</span>
                                  </>
                                )}
                              </div>
                            </div>
                            
                            {/* Status Icons - Right Side */}
                            <div className="flex sm:flex-col items-center sm:items-end gap-2 sm:gap-1 text-xs">
                              {item.has_adapter && (
                                <div className="flex items-center gap-1 text-green-600 bg-green-50 px-2 py-1 rounded-full">
                                  <CheckCircle className="w-3 h-3" /> 
                                  <span className="hidden sm:inline">Adapter</span>
                                </div>
                              )}
                              {item.checkpoint_count > 0 && (
                                <div className="flex items-center gap-1 text-blue-600 bg-blue-50 px-2 py-1 rounded-full">
                                  <Layers className="w-3 h-3" />
                                  <span>{item.checkpoint_count} ckpt</span>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                        
                        {/* Metrics and Output - Footer */}
                        {(item.final_metrics || item.output_exists) && (
                          <div className="px-3 sm:px-4 py-2 bg-slate-50/50 border-t border-slate-100 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                            {/* Final Metrics */}
                            {item.final_metrics && item.final_metrics.loss !== null && (
                              <div className="flex flex-wrap items-center gap-3 text-xs">
                                <div className="flex items-center gap-1">
                                  <span className="text-slate-400">Final Loss:</span>
                                  <span className="font-mono font-semibold text-slate-700">{item.final_metrics.loss.toFixed(4)}</span>
                                </div>
                                {item.final_metrics.epoch !== null && (
                                  <div className="flex items-center gap-1">
                                    <span className="text-slate-400">Epochs:</span>
                                    <span className="font-mono font-semibold text-slate-700">{item.final_metrics.epoch}</span>
                                  </div>
                                )}
                                {item.final_metrics.step !== null && (
                                  <div className="flex items-center gap-1">
                                    <span className="text-slate-400">Steps:</span>
                                    <span className="font-mono font-semibold text-slate-700">{item.final_metrics.step}</span>
                                  </div>
                                )}
                              </div>
                            )}
                            
                            {/* Output Path */}
                            {item.output_exists && (
                              <div className="flex items-center gap-1 text-xs">
                                <FolderOpen className="w-3 h-3 text-slate-400 flex-shrink-0" />
                                <code className="text-[10px] text-slate-500 truncate max-w-[250px] sm:max-w-[300px] bg-white px-2 py-0.5 rounded border border-slate-200" title={item.output_path}>
                                  {item.output_path}
                                </code>
                              </div>
                            )}
                          </div>
                        )}
                        
                        {/* Adapter Path - if available */}
                        {item.adapter_path && (
                          <div className="px-3 sm:px-4 py-2 bg-green-50/50 border-t border-green-100 flex items-center gap-2 text-xs">
                            <Sparkles className="w-3 h-3 text-green-500 flex-shrink-0" />
                            <span className="text-green-600 font-medium">Adapter:</span>
                            <code className="text-[10px] text-green-700 truncate flex-1 bg-white px-2 py-0.5 rounded border border-green-200" title={item.adapter_path}>
                              {item.adapter_path}
                            </code>
                          </div>
                        )}
                        
                        {/* Action Buttons for completed jobs */}
                        {item.status === 'completed' && item.has_adapter && (
                          <div className="px-3 sm:px-4 py-3 bg-slate-50 border-t border-slate-200">
                            <button
                              onClick={() => {
                                setShowHistory(false)
                                loadModelForInference(config.model_path, item.adapter_path)
                              }}
                              disabled={isModelLoading || isCleaningMemory}
                              className="w-full py-2.5 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg text-sm font-medium hover:from-blue-600 hover:to-indigo-700 disabled:opacity-50 flex items-center justify-center gap-2 shadow-md transition-all"
                            >
                              {isModelLoading ? (
                                <>
                                  <Loader2 className="w-4 h-4 animate-spin" />
                                  {loadingMessage || 'Loading...'}
                                </>
                              ) : (
                                <>
                                  <MessageSquare className="w-4 h-4" />
                                  Load for Inference
                                </>
                              )}
                            </button>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
              
              {/* Footer */}
              <div className="p-3 sm:p-4 border-t border-slate-200 bg-slate-50 flex gap-2">
                <button onClick={() => fetchTrainingHistory()}
                  className="px-4 py-2 bg-white text-slate-600 rounded-lg font-medium hover:bg-slate-100 border border-slate-200 flex items-center gap-2">
                  <RefreshCw className="w-4 h-4" /> Refresh
                </button>
                <button onClick={() => setShowHistory(false)}
                  className="flex-1 py-2 bg-slate-700 text-white rounded-lg font-medium hover:bg-slate-800">
                  Close
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ===================== TRAINING TAB ===================== */}
        {/* Only show when NOT training and no job result to display */}
        {mainTab === 'train' && !isTraining && !jobStatus && (
          <>
            {currentStep <= 4 && (
              <div className="mb-6">
                {/* Training History Quick Access */}
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-slate-800">New Training</h2>
                  <button 
                    onClick={() => { setShowHistory(true); fetchTrainingHistory(); }}
                    className="flex items-center gap-2 px-3 py-1.5 text-sm text-slate-600 hover:text-slate-800 hover:bg-slate-100 rounded-lg transition-colors border border-slate-200"
                  >
                    <History className="w-4 h-4" />
                    <span className="hidden sm:inline">Training History</span>
                  </button>
                </div>
                <div className="flex items-center justify-between overflow-x-auto pb-2">
                  {TRAIN_STEPS.map((step, idx) => (
                    <div key={step.id} className="flex items-center flex-shrink-0">
                      <div className={`flex items-center gap-2 ${currentStep >= step.id ? 'text-blue-600' : 'text-slate-400'}`}>
                        <div className={`w-9 h-9 sm:w-10 sm:h-10 rounded-full flex items-center justify-center border-2 transition-all ${
                          currentStep > step.id ? 'bg-blue-500 border-blue-500 text-white' :
                          currentStep === step.id ? 'border-blue-500 text-blue-600 bg-blue-50' :
                          'border-slate-300 text-slate-400'
                        }`}>
                          {currentStep > step.id ? <CheckCircle className="w-4 h-4 sm:w-5 sm:h-5" /> : <step.icon className="w-4 h-4 sm:w-5 sm:h-5" />}
                        </div>
                        <span className={`hidden sm:block text-sm font-medium ${currentStep >= step.id ? 'text-slate-900' : 'text-slate-400'}`}>
                          {step.title}
                        </span>
                      </div>
                      {idx < TRAIN_STEPS.length - 1 && (
                        <div className={`w-8 sm:w-16 h-0.5 mx-2 ${currentStep > step.id ? 'bg-blue-500' : 'bg-slate-200'}`} />
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="bg-white rounded-xl shadow-lg border border-slate-200 p-4 sm:p-6">
              
              {/* Step 1: Model Selection (Only shown if multiple models available) */}
              {currentStep === 1 && !IS_SINGLE_MODEL && (
                <div className="space-y-5">
                  <div>
                    <h2 className="text-xl font-bold text-slate-900 mb-1">Select Model</h2>
                    <p className="text-slate-600 text-sm">Choose the base model for fine-tuning</p>
                  </div>
                  
                  {/* Locked Models Selection */}
                  {IS_MODEL_LOCKED && lockedModels.length > 1 && (
                    <div className="space-y-3">
                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                        <div className="flex items-start gap-3">
                          <Lock className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                          <div>
                            <p className="font-medium text-blue-800">Available Models</p>
                            <p className="text-sm text-blue-700 mt-1">
                              Select one of the supported models below for fine-tuning.
                            </p>
                          </div>
                        </div>
                      </div>
                      
                      <div className="grid gap-3">
                        {lockedModels.map((model: LockedModel, idx: number) => (
                          <button
                            key={idx}
                            onClick={() => setConfig(prev => ({
                              ...prev,
                              model_path: model.path,
                              model_source: model.source,
                              modality: model.modality
                            }))}
                            className={`p-4 rounded-lg border-2 text-left transition-all ${
                              config.model_path === model.path 
                                ? 'border-blue-500 bg-blue-50' 
                                : 'border-slate-200 hover:border-slate-300 bg-white'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <div>
                                <p className="font-semibold text-slate-900">{model.name}</p>
                                {model.description && (
                                  <p className="text-sm text-slate-500 mt-1">{model.description}</p>
                                )}
                                <div className="flex gap-2 mt-2">
                                  <span className="text-xs bg-slate-100 text-slate-600 px-2 py-0.5 rounded capitalize">
                                    {model.modality}
                                  </span>
                                </div>
                              </div>
                              {config.model_path === model.path && (
                                <CheckCircle className="w-6 h-6 text-blue-500" />
                              )}
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Unlocked Model Input (fallback if no locked models) */}
                  {!IS_MODEL_LOCKED && (
                    <div className="grid gap-4">
                      {(() => {
                        const supportedSources = (systemCapabilities.supported_model_sources || systemCapabilities.supported_sources || ['local']);
                        const hasMultipleSources = supportedSources.length > 1;
                        
                        const getLabel = () => {
                          if (config.model_source === 'local') return 'Local Model Path';
                          if (config.model_source === 'huggingface') return 'HuggingFace Model ID';
                          if (config.model_source === 'modelscope') return 'ModelScope Model ID';
                          return 'Model Path';
                        };
                        
                        const getPlaceholder = () => {
                          if (config.model_source === 'local') return '/path/to/model';
                          if (config.model_source === 'huggingface') return 'organization/model-name';
                          if (config.model_source === 'modelscope') return 'organization/model-name';
                          return 'Enter model path or ID';
                        };
                        
                        return (
                          <>
                            {hasMultipleSources && (
                              <div>
                                <label className="block text-sm font-medium text-slate-700 mb-2">Model Source</label>
                                <div className={`grid gap-2`} style={{ gridTemplateColumns: `repeat(${supportedSources.length}, 1fr)` }}>
                                  {supportedSources.map((source) => (
                                    <button key={source} 
                                      onClick={() => setConfig({ ...config, model_source: source as any })}
                                      className={`p-3 rounded-lg border-2 text-center transition-all ${
                                        config.model_source === source ? 'border-blue-500 bg-blue-50 text-blue-600' : 'border-slate-200 text-slate-600 hover:border-slate-300'
                                      }`}>
                                      <span className="capitalize font-medium text-sm">{source}</span>
                                    </button>
                                  ))}
                                </div>
                              </div>
                            )}
                            
                            <div>
                              <label className="block text-sm font-medium text-slate-700 mb-2">{getLabel()}</label>
                              <input type="text" value={config.model_path}
                                onChange={(e) => setConfig({ ...config, model_path: e.target.value })}
                                placeholder={getPlaceholder()}
                                className="w-full px-4 py-3 border border-slate-300 rounded-lg text-slate-900 placeholder-slate-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                              />
                              
                              {/* Model Info Display - shows after model path is entered */}
                              {config.model_path && (
                                <div className="mt-2">
                                  {modelInfo.isLoading ? (
                                    <div className="flex items-center gap-2 text-sm text-slate-500">
                                      <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                                      <span>Loading model info...</span>
                                    </div>
                                  ) : modelInfo.error ? (
                                    <p className="text-sm text-amber-600">⚠️ Could not fetch model info. Using defaults.</p>
                                  ) : (modelInfo.model_type || modelContextLength !== 4096) ? (
                                    <div className="flex flex-wrap gap-2">
                                      {modelInfo.model_type && (
                                        <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
                                          {modelInfo.model_type}
                                        </span>
                                      )}
                                      <span className="text-xs bg-emerald-100 text-emerald-700 px-2 py-1 rounded-full">
                                        Context: {modelContextLength.toLocaleString()} tokens
                                      </span>
                                      {modelInfo.architecture && (
                                        <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-full">
                                          {modelInfo.architecture.replace('ForCausalLM', '')}
                                        </span>
                                      )}
                                    </div>
                                  ) : null}
                                </div>
                              )}
                            </div>
                            
                            {/* API Token for private models - shown only for HuggingFace/ModelScope */}
                            {(config.model_source === 'huggingface' || config.model_source === 'modelscope') && (
                              <div className="mt-4 p-4 bg-slate-50 rounded-lg border border-slate-200">
                                <div className="flex items-center gap-2 mb-2">
                                  <span className="text-sm font-medium text-slate-700">
                                    {config.model_source === 'huggingface' ? 'HuggingFace' : 'ModelScope'} Token
                                  </span>
                                  <span className="text-xs bg-slate-200 text-slate-600 px-2 py-0.5 rounded">Optional</span>
                                </div>
                                
                                {/* Show "Use system token" option if available */}
                                {((config.model_source === 'huggingface' && systemTokens.hf_token_available) ||
                                  (config.model_source === 'modelscope' && systemTokens.ms_token_available)) && (
                                  <label className="flex items-center gap-2 mb-3 p-2 bg-green-50 rounded-lg border border-green-200 cursor-pointer">
                                    <input 
                                      type="checkbox"
                                      checked={useSystemToken}
                                      onChange={(e) => {
                                        setUseSystemToken(e.target.checked);
                                        if (e.target.checked) {
                                          // Clear manual token when using system token
                                          if (config.model_source === 'huggingface') {
                                            setConfig({ ...config, hf_token: null });
                                          } else {
                                            setConfig({ ...config, ms_token: null });
                                          }
                                        }
                                      }}
                                      className="rounded text-green-600 focus:ring-green-500"
                                    />
                                    <span className="text-sm text-green-700">
                                      Use system token ({config.model_source === 'huggingface' ? systemTokens.hf_token_masked : systemTokens.ms_token_masked})
                                    </span>
                                  </label>
                                )}
                                
                                {!useSystemToken && (
                                  <>
                                    <input 
                                      type="password"
                                      value={config.model_source === 'huggingface' ? (config.hf_token || '') : (config.ms_token || '')}
                                      onChange={(e) => {
                                        if (config.model_source === 'huggingface') {
                                          setConfig({ ...config, hf_token: e.target.value || null });
                                        } else {
                                          setConfig({ ...config, ms_token: e.target.value || null });
                                        }
                                      }}
                                      placeholder={config.model_source === 'huggingface' ? 'hf_xxxxxxxxxxxxxxxxxxxx' : 'Your ModelScope token'}
                                      className={`w-full px-3 py-2 border rounded-lg text-sm font-mono focus:ring-2 focus:ring-blue-500 ${
                                        config.model_source === 'huggingface' && config.hf_token && !config.hf_token.startsWith('hf_')
                                          ? 'border-amber-400 bg-amber-50'
                                          : 'border-slate-300'
                                      }`}
                                    />
                                    {/* Token format validation */}
                                    {config.model_source === 'huggingface' && config.hf_token && !config.hf_token.startsWith('hf_') && (
                                      <p className="text-xs text-amber-600 mt-1">⚠️ HuggingFace tokens typically start with "hf_"</p>
                                    )}
                                    <p className="text-xs text-slate-500 mt-2">
                                      🔒 Only needed for private models. Get your token from{' '}
                                      {config.model_source === 'huggingface' ? (
                                        <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">huggingface.co/settings/tokens</a>
                                      ) : (
                                        <a href="https://modelscope.cn/my/myaccesstoken" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">modelscope.cn/my/myaccesstoken</a>
                                      )}
                                    </p>
                                  </>
                                )}
                              </div>
                            )}
                          </>
                        );
                      })()}
                    </div>
                  )}
                </div>
              )}

              {/* Dataset Configuration Step */}
              {/* Step 1 if single model locked, Step 2 otherwise */}
              {((IS_SINGLE_MODEL && currentStep === 1) || (!IS_SINGLE_MODEL && currentStep === 2)) && (
                <div className="space-y-5">
                  {/* Show selected model banner when in locked single model mode */}
                  {IS_SINGLE_MODEL && DEFAULT_MODEL && (
                    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-4 mb-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center">
                          <Cpu className="w-5 h-5 text-blue-600" />
                        </div>
                        <div>
                          <p className="text-xs text-blue-600 font-medium uppercase tracking-wide">Selected Model</p>
                          <p className="font-semibold text-slate-900">{DEFAULT_MODEL.name}</p>
                        </div>
                        <Lock className="w-4 h-4 text-blue-400 ml-auto" />
                      </div>
                    </div>
                  )}
                  <DatasetConfig 
                    selectedPaths={config.dataset_paths}
                    onSelectionChange={(paths) => setConfig(prev => ({ ...prev, dataset_paths: paths }))}
                    onShowAlert={showAlert}
                  />
                </div>
              )}

              {/* Training Settings Step */}
              {/* Step 2 if single model locked, Step 3 otherwise */}
              {((IS_SINGLE_MODEL && currentStep === 2) || (!IS_SINGLE_MODEL && currentStep === 3)) && (
                <div className="space-y-5">
                  {/* Show selected model banner */}
                  {IS_MODEL_LOCKED && (
                    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-4 mb-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center">
                          <Cpu className="w-5 h-5 text-blue-600" />
                        </div>
                        <div>
                          <p className="text-xs text-blue-600 font-medium uppercase tracking-wide">Training Model</p>
                          <p className="font-semibold text-slate-900">{getModelDisplayName(config.model_path)}</p>
                        </div>
                        <Lock className="w-4 h-4 text-blue-400 ml-auto" />
                      </div>
                    </div>
                  )}
                  <TrainingSettingsStep 
                    config={config} 
                    setConfig={(fn) => setConfig(prev => ({ ...prev, ...fn(prev) }))}
                    availableGpus={availableGpus}
                    modelContextLength={modelContextLength}
                  />
                </div>
              )}

              {/* Review & Start Step */}
              {/* Step 3 if single model locked, Step 4 otherwise */}
              {((IS_SINGLE_MODEL && currentStep === 3) || (!IS_SINGLE_MODEL && currentStep === 4)) && (
                <div className="space-y-5">
                  <div>
                    <h2 className="text-xl font-bold text-slate-900 mb-1">Review & Start</h2>
                    <p className="text-slate-600 text-sm">Review your configuration before training</p>
                  </div>
                  
                  {/* Output Path Configuration - Hidden when locked */}
                  {outputPathConfig.is_locked ? (
                    /* Locked mode - show info only, no input */
                    <div className="bg-emerald-50 rounded-lg p-4 border border-emerald-200">
                      <div className="flex items-center gap-2">
                        <FolderOpen className="w-5 h-5 text-emerald-600" />
                        <span className="text-sm font-medium text-emerald-900">Output Directory</span>
                        <Lock className="w-4 h-4 text-emerald-500" />
                      </div>
                      <p className="text-sm text-emerald-700 mt-2">
                        Training outputs will be saved to <code className="bg-emerald-100 px-2 py-0.5 rounded font-mono text-xs">{outputPathConfig.base_path}/{'<training-id>'}</code>
                      </p>
                      <p className="text-xs text-emerald-600 mt-1">Output path is managed automatically for each training job</p>
                    </div>
                  ) : outputPathConfig.user_can_add_path ? (
                    /* Base locked mode - user can add intermediate path */
                    <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                      <div className="flex items-center gap-2 mb-3">
                        <FolderOpen className="w-5 h-5 text-blue-600" />
                        <label className="text-sm font-medium text-slate-900">Output Subdirectory</label>
                        <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">Optional</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-slate-500 font-mono">{outputPathConfig.base_path}/</span>
                        <input type="text" value={config.output_dir}
                          onChange={(e) => setConfig({ ...config, output_dir: e.target.value })}
                          placeholder="my-project"
                          className="flex-1 px-3 py-2 bg-white border border-slate-300 rounded-lg text-slate-900 placeholder-slate-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 text-sm" />
                        <span className="text-sm text-slate-500 font-mono">/{'<training-id>'}</span>
                      </div>
                      <p className="text-xs text-slate-500 mt-2">Add an optional subdirectory to organize your outputs</p>
                    </div>
                  ) : systemCapabilities.has_external_storage ? (
                    /* Free mode with external storage */
                    <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                      <div className="flex items-center gap-2 mb-3">
                        <FolderOpen className="w-5 h-5 text-blue-600" />
                        <label className="text-sm font-medium text-slate-900">Output Directory</label>
                        <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">External Storage</span>
                      </div>
                      <input type="text" value={config.output_dir}
                        onChange={(e) => setConfig({ ...config, output_dir: e.target.value })}
                        placeholder={`${systemCapabilities.storage_path}/finetuned`}
                        className="w-full px-4 py-2 bg-white border border-slate-300 rounded-lg text-slate-900 placeholder-slate-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500" />
                      <p className="text-xs text-slate-500 mt-2">
                        External storage detected at <code className="bg-slate-200 px-1 rounded">{systemCapabilities.storage_path}</code>. 
                        Use absolute path to save to persistent storage.
                      </p>
                    </div>
                  ) : (
                    /* Free mode without external storage */
                    <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                      <div className="flex items-center gap-2">
                        <FolderOpen className="w-5 h-5 text-slate-500" />
                        <span className="text-sm text-slate-600">Output will be saved to internal storage (not persistent)</span>
                      </div>
                      <p className="text-xs text-slate-400 mt-2">Attach a network volume for persistent storage</p>
                    </div>
                  )}
                  
                  <div className="bg-slate-50 rounded-lg p-4 space-y-3 text-sm border border-slate-200">
                    <h4 className="font-medium text-slate-900 mb-2">Configuration Summary</h4>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      <div className="flex items-center gap-2">
                        <span className="text-slate-500">Model:</span> 
                        <span className="font-medium text-slate-900">{getModelDisplayName(config.model_path)}</span>
                        {IS_MODEL_LOCKED && <Lock className="w-3 h-3 text-slate-400" />}
                      </div>
                      <div><span className="text-slate-500">Training Type:</span> <span className="font-medium text-blue-600 uppercase">{config.train_type}</span></div>
                      <div><span className="text-slate-500">Epochs:</span> <span className="font-medium text-slate-900">{config.num_train_epochs}</span></div>
                      <div><span className="text-slate-500">Learning Rate:</span> <span className="font-medium text-slate-900">{config.learning_rate.toExponential(0)}</span></div>
                      <div><span className="text-slate-500">Effective Batch:</span> <span className="font-medium text-slate-900">{config.per_device_train_batch_size} × {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps}</span></div>
                      <div><span className="text-slate-500">Max Length:</span> <span className="font-medium text-slate-900">{config.max_length}</span></div>
                      {['lora', 'qlora', 'adalora'].includes(config.train_type) && (
                        <>
                          <div><span className="text-slate-500">LoRA Rank:</span> <span className="font-medium text-slate-900">{config.lora_rank}</span></div>
                          <div><span className="text-slate-500">LoRA Alpha:</span> <span className="font-medium text-slate-900">{config.lora_alpha}</span></div>
                        </>
                      )}
                      {config.train_type === 'qlora' && (
                        <div><span className="text-slate-500">Quantization:</span> <span className="font-medium text-blue-600">{config.quant_bits}-bit</span></div>
                      )}
                    </div>
                    <div className="pt-3 border-t border-slate-200">
                      <span className="text-slate-500">Datasets ({config.dataset_paths.length}):</span>
                      <ul className="mt-1 space-y-1">
                        {config.dataset_paths.map((path, idx) => (
                          <li key={idx} className="font-medium text-slate-900 flex items-center gap-2">
                            <Check className="w-4 h-4 text-green-600" /> {path.split('/').pop()}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                  
                  {/* Training Name (optional) */}
                  <div className="bg-slate-50 rounded-lg p-4">
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                      Training Name <span className="text-slate-400 font-normal">(optional)</span>
                    </label>
                    <input
                      type="text"
                      value={trainingName}
                      onChange={(e) => setTrainingName(e.target.value)}
                      placeholder="e.g., my-training-run (auto-generated if empty)"
                      className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      maxLength={255}
                    />
                    <p className="text-xs text-slate-500 mt-1">
                      A unique name to identify this training. Leave empty to auto-generate.
                    </p>
                  </div>
                  
                  {/* System status warning */}
                  {systemStatus.status !== 'live' && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700 flex items-center gap-2">
                      <XCircle className="w-5 h-5 flex-shrink-0" />
                      <div>
                        <strong>Cannot start training:</strong> {systemStatus.message}
                        {systemStatus.gpu_name && <span className="block text-xs mt-1">GPU: {systemStatus.gpu_name}</span>}
                      </div>
                    </div>
                  )}
                  
                  <button onClick={startTraining}
                    disabled={config.dataset_paths.length === 0 || systemStatus.status !== 'live' || isStartingTraining}
                    className="w-full py-4 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-semibold text-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-lg">
                    <Zap className="w-5 h-5" /> {isStartingTraining ? 'Starting...' : (systemStatus.status === 'live' ? 'Start Training' : 'System Not Ready')}
                  </button>
                </div>
              )}

              {/* Navigation - Dynamic based on TOTAL_STEPS */}
              {currentStep <= TOTAL_STEPS && (
                <div className="flex justify-between mt-6 pt-4 border-t border-slate-200">
                  <button onClick={() => setCurrentStep(Math.max(1, currentStep - 1))} disabled={currentStep === 1}
                    className="flex items-center gap-2 px-4 py-2 text-slate-600 font-medium disabled:opacity-50 hover:text-slate-900 transition-colors">
                    <ChevronLeft className="w-5 h-5" /> Back
                  </button>
                  {currentStep < TOTAL_STEPS && (
                    <button onClick={() => setCurrentStep(currentStep + 1)} disabled={!canProceed()}
                      className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg font-medium disabled:opacity-50 hover:bg-blue-600 transition-colors">
                      Next <ChevronRight className="w-5 h-5" />
                    </button>
                  )}
                </div>
              )}
            </div>
          </>
        )}

        {/* ===================== INFERENCE TAB ===================== */}
        {/* Only show when NOT training */}
        {mainTab === 'inference' && !isTraining && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Left Panel - Model & Settings */}
            <div className="bg-white rounded-xl border border-slate-200 shadow-lg p-4 space-y-4">
              <h3 className="font-bold text-slate-900 flex items-center gap-2">
                <Cpu className="w-5 h-5 text-blue-500" /> Model Settings
              </h3>
              
              {/* Model Loading */}
              <div className="space-y-2">
                <label className="block text-sm font-medium text-slate-700">Base Model</label>
                <input type="text" value={inferenceModel}
                  onChange={(e) => setInferenceModel(e.target.value)}
                  placeholder="/path/to/model"
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900 text-sm placeholder-slate-400" />
                <button onClick={loadModel} disabled={!inferenceModel.trim() || isModelLoading}
                  className="w-full py-2 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600 disabled:opacity-50 flex items-center justify-center gap-2">
                  {isModelLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                  {isModelLoading ? 'Loading...' : 'Load Model'}
                </button>
              </div>
              
              {/* Adapter Management */}
              <div className="border-t border-slate-200 pt-4 space-y-2">
                <label className="text-sm font-medium text-slate-700 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-blue-500" /> LoRA Adapters
                </label>
                <div className="flex gap-2">
                  <input type="text" value={adapterPath}
                    onChange={(e) => setAdapterPath(e.target.value)}
                    placeholder="/path/to/adapter"
                    disabled={!inferenceStatus.model_loaded}
                    className="flex-1 px-3 py-2 border border-slate-300 rounded-lg text-slate-900 text-sm placeholder-slate-400 disabled:opacity-50" />
                  <button onClick={loadAdapter} disabled={!adapterPath.trim() || !inferenceStatus.model_loaded}
                    className="px-3 py-2 bg-blue-50 text-blue-600 border border-blue-200 rounded-lg text-sm font-medium hover:bg-blue-100 disabled:opacity-50">
                    Load
                  </button>
                </div>
                {loadedAdapters.length > 0 && (
                  <div className="space-y-1 mt-2">
                    {loadedAdapters.map(adapter => (
                      <div key={adapter.id} className={`flex items-center gap-2 p-2 rounded-lg text-xs ${adapter.active ? 'bg-blue-50 border border-blue-200' : 'bg-slate-50'}`}>
                        <button onClick={() => switchAdapter(adapter.id)} className={`w-4 h-4 rounded-full border-2 ${adapter.active ? 'bg-blue-500 border-blue-500' : 'border-slate-300'}`} />
                        <span className="flex-1 text-slate-700 truncate">{adapter.name}</span>
                        <button onClick={() => removeAdapter(adapter.id)} className="text-slate-400 hover:text-red-500"><X className="w-3 h-3" /></button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              
              {/* Chat Mode Toggle */}
              <div className="border-t border-slate-200 pt-4">
                <label className="block text-sm font-medium text-slate-700 mb-2">Mode</label>
                <div className="flex gap-2">
                  <button onClick={() => setChatMode('chat')}
                    className={`flex-1 py-2 rounded-lg text-sm font-medium flex items-center justify-center gap-1 ${chatMode === 'chat' ? 'bg-blue-500 text-white' : 'bg-slate-100 text-slate-600'}`}>
                    <MessageSquare className="w-4 h-4" /> Chat
                  </button>
                  <button onClick={() => setChatMode('completion')}
                    className={`flex-1 py-2 rounded-lg text-sm font-medium flex items-center justify-center gap-1 ${chatMode === 'completion' ? 'bg-blue-500 text-white' : 'bg-slate-100 text-slate-600'}`}>
                    <FileText className="w-4 h-4" /> Complete
                  </button>
                </div>
                {chatMode === 'chat' && (
                  <div className="mt-2 flex items-center gap-2">
                    <button onClick={() => setKeepHistory(!keepHistory)} className="text-slate-500 hover:text-slate-900">
                      {keepHistory ? <ToggleRight className="w-5 h-5 text-blue-500" /> : <ToggleLeft className="w-5 h-5" />}
                    </button>
                    <span className="text-xs text-slate-500">Keep conversation history</span>
                  </div>
                )}
              </div>
              
              {/* Generation Settings */}
              <div className="border-t border-slate-200 pt-4">
                <h4 className="text-sm font-medium text-slate-700 mb-2">Generation</h4>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-xs text-slate-500 mb-1">
                      <span>Max Tokens</span><span>{inferenceSettings.max_new_tokens} / {modelContextLength.toLocaleString()}</span>
                    </div>
                    <input type="range" min="64" max={modelContextLength} value={inferenceSettings.max_new_tokens}
                      onChange={(e) => setInferenceSettings({ ...inferenceSettings, max_new_tokens: parseInt(e.target.value) })}
                      className="w-full accent-blue-500" />
                  </div>
                  <div>
                    <div className="flex justify-between text-xs text-slate-500 mb-1">
                      <span>Temperature</span><span>{inferenceSettings.temperature}</span>
                    </div>
                    <input type="range" min="0" max="2" step="0.1" value={inferenceSettings.temperature}
                      onChange={(e) => setInferenceSettings({ ...inferenceSettings, temperature: parseFloat(e.target.value) })}
                      className="w-full accent-blue-500" />
                  </div>
                </div>
              </div>
              
              {/* System Status - only show actual data */}
              <div className="border-t border-slate-200 pt-4">
                <h4 className="text-sm font-medium text-slate-700 mb-2 flex items-center gap-2">
                  <Monitor className="w-4 h-4 text-green-500" /> System Status
                </h4>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="bg-slate-50 rounded-lg p-2 text-center border border-slate-200">
                    <Gauge className={`w-4 h-4 mx-auto mb-1 ${systemMetrics.available && systemMetrics.gpu_utilization !== null ? 'text-blue-500' : 'text-slate-400'}`} />
                    <span className="text-slate-500">GPU</span>
                    <p className={`font-medium ${systemMetrics.available && systemMetrics.gpu_utilization !== null ? 'text-slate-900' : 'text-slate-400'}`}>
                      {systemMetrics.available && systemMetrics.gpu_utilization !== null ? `${systemMetrics.gpu_utilization}%` : 'N/A'}
                    </p>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-2 text-center border border-slate-200">
                    <HardDrive className={`w-4 h-4 mx-auto mb-1 ${systemMetrics.available && systemMetrics.gpu_memory_used !== null ? 'text-blue-500' : 'text-slate-400'}`} />
                    <span className="text-slate-500">VRAM</span>
                    <p className={`font-medium ${systemMetrics.available && systemMetrics.gpu_memory_used !== null ? 'text-slate-900' : 'text-slate-400'}`}>
                      {systemMetrics.available && systemMetrics.gpu_memory_used !== null ? `${systemMetrics.gpu_memory_used.toFixed(1)}GB` : 'N/A'}
                    </p>
                  </div>
                </div>
                <div className="flex gap-2 mt-2">
                  <button onClick={fetchInferenceStatus}
                    className="flex-1 px-2 py-1 bg-slate-100 hover:bg-slate-200 rounded text-xs font-medium text-slate-600 flex items-center justify-center gap-1">
                    <RefreshCw className="w-3 h-3" /> Refresh
                  </button>
                  <button onClick={clearMemory}
                    className="flex-1 px-2 py-1 bg-red-50 hover:bg-red-100 text-red-600 rounded text-xs font-medium flex items-center justify-center gap-1">
                    <Trash2 className="w-3 h-3" /> Clear
                  </button>
                </div>
              </div>
            </div>
            
            {/* Right Panel - Chat Interface */}
            <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 shadow-lg flex flex-col h-[500px] sm:h-[600px]">
              <div className="p-4 border-b border-slate-200 flex items-center justify-between">
                <div>
                  <h3 className="font-bold text-slate-900">{chatMode === 'chat' ? 'Chat Interface' : 'Text Completion'}</h3>
                  <p className="text-xs text-slate-500">
                    {inferenceStatus.model_loaded ? `Model: ${inferenceStatus.model_path?.split('/').pop()}` : 'Load a model to start'}
                  </p>
                </div>
                {chatMode === 'chat' && (
                  <button onClick={() => setChatMessages([])} className="p-2 text-slate-500 hover:text-slate-900 hover:bg-slate-100 rounded-lg">
                    <History className="w-4 h-4" />
                  </button>
                )}
              </div>
              
              {/* System Prompt (Chat mode only) */}
              {chatMode === 'chat' && (
                <div className="px-4 pt-2">
                  <input type="text" value={systemPrompt}
                    onChange={(e) => setSystemPrompt(e.target.value)}
                    placeholder="System prompt (optional)..."
                    className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-slate-900 text-xs placeholder-slate-400" />
                </div>
              )}
              
              <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {chatMessages.length === 0 && (
                  <div className="text-center text-slate-400 py-10">
                    <MessageSquare className="w-10 h-10 mx-auto mb-2 opacity-50" />
                    <p>{inferenceStatus.model_loaded ? 'Start a conversation' : 'Load a model first'}</p>
                  </div>
                )}
                {chatMessages.map((msg, i) => (
                  <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[85%] px-4 py-2 rounded-lg ${
                      msg.role === 'user' ? 'bg-blue-500 text-white' : 'bg-slate-100 text-slate-900'
                    }`}>
                      <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                    </div>
                  </div>
                ))}
                {isGenerating && (
                  <div className="flex justify-start">
                    <div className="bg-slate-100 px-4 py-2 rounded-lg">
                      <Loader2 className="w-5 h-5 animate-spin text-blue-500" />
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
              
              <div className="p-4 border-t border-slate-200">
                <div className="flex gap-2">
                  <button onClick={() => setChatMessages([])}
                    className="px-3 py-2 bg-slate-100 hover:bg-slate-200 rounded-lg">
                    <Trash2 className="w-5 h-5 text-slate-500" />
                  </button>
                  <input type="text" value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
                    placeholder={chatMode === 'chat' ? 'Type your message...' : 'Enter text to complete...'}
                    disabled={!inferenceStatus.model_loaded || isGenerating}
                    className="flex-1 px-4 py-2 border border-slate-300 rounded-lg text-slate-900 placeholder-slate-400 focus:ring-2 focus:ring-blue-500 disabled:opacity-50" />
                  <button onClick={sendMessage} disabled={!inferenceStatus.model_loaded || !inputMessage.trim() || isGenerating}
                    className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50">
                    <Send className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="mt-6 py-4 text-center text-xs text-slate-500 border-t border-slate-200">
        USF BIOS v1.0.0 - Copyright 2024-2026 US Inc. All rights reserved.
      </footer>
      
      {/* Custom Alert Modal - replaces browser alert() */}
      <AlertModal
        isOpen={alertModal.isOpen}
        onClose={closeAlert}
        title={alertModal.title}
        message={alertModal.message}
        type={alertModal.type}
      />
      
      {/* Custom Confirm Modal - replaces browser confirm() */}
      <ConfirmModal
        isOpen={confirmModal.isOpen}
        onClose={closeConfirm}
        onConfirm={confirmModal.onConfirm}
        title={confirmModal.title}
        message={confirmModal.message}
        type={confirmModal.type}
        confirmText={confirmModal.confirmText}
        isLoading={confirmModal.isLoading}
      />
    </div>
  )
}
