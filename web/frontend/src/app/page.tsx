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

// Types
interface TrainingConfig {
  model_path: string
  model_source: 'huggingface' | 'modelscope' | 'local'
  modality: 'text' | 'vision' | 'audio' | 'video'
  train_type: 'full' | 'lora' | 'qlora' | 'adalora'
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
  deepspeed: string | null
  fsdp: string | null
  early_stop_interval: number | null
}

interface JobStatus {
  job_id: string
  job_name: string  // User-friendly name for the training
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped'
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
  loss: number
  learning_rate: number
  timestamp: number
}

interface LoadedAdapter {
  id: string
  name: string
  path: string
  active: boolean
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

const TRAIN_STEPS = [
  { id: 1, title: 'Select Model', icon: Cpu },
  { id: 2, title: 'Configure Dataset', icon: Database },
  { id: 3, title: 'Training Settings', icon: Settings },
  { id: 4, title: 'Review & Start', icon: Play },
]

export default function Home() {
  const [mainTab, setMainTab] = useState<MainTab>('train')
  const [currentStep, setCurrentStep] = useState(1)
  const [isTraining, setIsTraining] = useState(false)
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [trainingLogs, setTrainingLogs] = useState<string[]>([])
  
  const [config, setConfig] = useState<TrainingConfig>({
    model_path: '',
    model_source: 'local',
    modality: 'text',
    train_type: 'lora',
    dataset_paths: [],
    output_dir: 'output/finetuned',
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
    deepspeed: null,
    fsdp: null,
    early_stop_interval: null,
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
  
  // Training metrics for graphs
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetric[]>([])
  
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

  // Fetch system capabilities - what this system can fine-tune
  const fetchSystemCapabilities = useCallback(async () => {
    try {
      const res = await fetch('/api/system/capabilities')
      if (res.ok) {
        const data = await res.json()
        setSystemCapabilities(data)
        
        // Get the supported model sources
        const supportedSources = data.supported_model_sources || data.supported_sources || ['local']
        
        // If system is designed for a specific model, pre-fill the config
        if (data.has_model_restriction && data.supported_model) {
          setConfig(prev => ({
            ...prev,
            model_path: data.supported_model,
            model_source: supportedSources[0] || 'local'
          }))
        } else if (!supportedSources.includes('huggingface')) {
          // If huggingface is not supported, switch to first available source
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
        // Only update if we have actual data, mark as available
        setSystemMetrics({
          gpu_utilization: data.gpu_utilization ?? null,
          gpu_memory_used: data.gpu_memory_used ?? null,
          gpu_memory_total: data.gpu_memory_total ?? null,
          gpu_temperature: data.gpu_temperature ?? null,
          cpu_percent: data.cpu_percent ?? null,
          ram_used: data.ram_used ?? null,
          ram_total: data.ram_total ?? null,
          available: data.gpu_memory_total > 0 || data.cpu_percent > 0
        })
      }
    } catch (e) {
      console.error('Failed to fetch system metrics:', e)
      // Keep metrics as unavailable on error
    }
  }, [])

  // Check system expiration FIRST on mount - blocks everything if expired
  useEffect(() => {
    checkSystemExpiration()
  }, [checkSystemExpiration])

  // Fetch system status and capabilities on mount (only if not expired)
  useEffect(() => {
    if (systemExpired) return
    fetchSystemStatus()
    fetchSystemCapabilities()
    const interval = setInterval(fetchSystemStatus, 10000) // Check every 10 seconds
    return () => clearInterval(interval)
  }, [fetchSystemStatus, fetchSystemCapabilities, systemExpired])

  // Poll system metrics during training or inference
  useEffect(() => {
    if (isTraining || mainTab === 'inference') {
      fetchSystemMetrics()
      const interval = setInterval(fetchSystemMetrics, 3000)
      return () => clearInterval(interval)
    }
  }, [isTraining, mainTab, fetchSystemMetrics])

  // WebSocket for training updates
  useEffect(() => {
    if (jobStatus?.job_id && isTraining) {
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(`${wsProtocol}//${window.location.host}/api/jobs/ws/${jobStatus.job_id}`)
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'log' && data.message) {
            setTrainingLogs(prev => [...prev.slice(-500), data.message])
          }
          if (data.type === 'progress') {
            // Update job status
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
            
            // Collect metrics for graphs
            if (data.step && data.loss !== undefined) {
              setTrainingMetrics(prev => [...prev.slice(-200), {
                step: data.step,
                loss: data.loss,
                learning_rate: data.learning_rate || 0,
                timestamp: Date.now()
              }])
            }
          }
          if (data.type === 'status') {
            if (data.status === 'completed' || data.status === 'failed' || data.status === 'stopped') {
              setIsTraining(false)
            }
            setJobStatus(prev => prev ? { ...prev, status: data.status } : null)
          }
        } catch (e) {
          console.error('WebSocket parse error:', e)
        }
      }
      
      ws.onerror = (e) => console.error('WebSocket error:', e)
      ws.onclose = () => console.log('WebSocket closed')
      
      return () => ws.close()
    }
  }, [jobStatus?.job_id, isTraining])

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [trainingLogs])
  
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages])

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
          alert(`Dataset uploaded but has warnings:\n${data.errors.join('\n')}`)
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
        alert(data.detail || 'Delete failed')
      }
    } catch (e) {
      alert(`Delete failed: ${e}`)
    } finally {
      setIsDeleting(false)
    }
  }

  const startTraining = async () => {
    if (config.dataset_paths.length === 0) {
      alert('Please select at least one dataset for training')
      return
    }
    
    try {
      setTrainingLogs([])
      
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
    } catch (e) {
      alert(`Failed to start training: ${e}`)
    }
  }

  const stopTraining = async () => {
    if (!jobStatus?.job_id) return
    try {
      await fetch(`/api/jobs/${jobStatus.job_id}/stop`, { method: 'POST' })
      setIsTraining(false)
    } catch (e) {
      alert(`Failed to stop: ${e}`)
    }
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

  const clearMemory = async () => {
    try {
      const res = await fetch('/api/inference/clear-memory', { method: 'POST' })
      if (res.ok) {
        await fetchInferenceStatus()
        alert('Memory cleared successfully')
      }
    } catch (e) {
      alert(`Failed to clear memory: ${e}`)
    }
  }

  // Load model for inference
  const loadModel = async () => {
    if (!inferenceModel.trim()) return
    setIsModelLoading(true)
    try {
      const res = await fetch('/api/inference/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: inferenceModel })
      })
      const data = await res.json()
      if (data.success) {
        await fetchInferenceStatus()
      } else {
        alert(`Failed to load model: ${data.error || 'Unknown error'}`)
      }
    } catch (e) {
      alert(`Failed to load model: ${e}`)
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
        alert(`Failed to load adapter: ${data.error || 'Unknown error'}`)
      }
    } catch (e) {
      alert(`Failed to load adapter: ${e}`)
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
    switch (currentStep) {
      case 1: return config.model_path.length > 0
      case 2: return config.dataset_paths.length > 0
      case 3: return true
      case 4: return true
      default: return false
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
            
            {/* Desktop tabs */}
            <div className="hidden sm:flex bg-slate-100 rounded-lg p-1">
              <button
                onClick={() => setMainTab('train')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${
                  mainTab === 'train' ? 'bg-blue-500 text-white shadow-md' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Zap className="w-4 h-4" />Fine-tuning
              </button>
              <button
                onClick={() => setMainTab('inference')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${
                  mainTab === 'inference' ? 'bg-blue-500 text-white shadow-md' : 'text-slate-600 hover:text-slate-900'
                }`}
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
          
          {/* Mobile menu */}
          {mobileMenuOpen && (
            <div className="sm:hidden mt-3 pt-3 border-t border-slate-200 flex gap-2">
              <button onClick={() => { setMainTab('train'); setMobileMenuOpen(false) }}
                className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium ${mainTab === 'train' ? 'bg-blue-500 text-white' : 'bg-slate-100 text-slate-700'}`}>
                <Zap className="w-4 h-4 inline mr-1" />Fine-tuning
              </button>
              <button onClick={() => { setMainTab('inference'); setMobileMenuOpen(false) }}
                className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium ${mainTab === 'inference' ? 'bg-blue-500 text-white' : 'bg-slate-100 text-slate-700'}`}>
                <MessageSquare className="w-4 h-4 inline mr-1" />Inference
              </button>
            </div>
          )}
        </div>
      </header>

      {/* System Status Banner - Shows when system is not live */}
      {systemStatus.status !== 'live' && (
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

      <main className="max-w-6xl mx-auto px-4 py-6">
        
        {/* ===================== TRAINING TAB ===================== */}
        {mainTab === 'train' && (
          <>
            {currentStep <= 4 && (
              <div className="mb-6">
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
              
              {/* Step 1: Model */}
              {currentStep === 1 && (
                <div className="space-y-5">
                  <div>
                    <h2 className="text-xl font-bold text-slate-900 mb-1">Select Model</h2>
                    <p className="text-slate-600 text-sm">Choose the base model for fine-tuning</p>
                  </div>
                  
                  {/* System Capability Notice - Only show when specific model path is locked */}
                  {systemCapabilities.has_model_restriction && systemCapabilities.supported_model && (
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                      <div className="flex items-start gap-3">
                        <Lock className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                        <div>
                          <p className="font-medium text-blue-800">Supported Model</p>
                          <p className="text-sm text-blue-700 mt-1">
                            This system only supports the model shown below. Please use the supported model for fine-tuning.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  <div className="grid gap-4">
                    {/* Model Source - Only show if multiple sources are supported */}
                    {(() => {
                      const supportedSources = (systemCapabilities.supported_model_sources || systemCapabilities.supported_sources || ['local']);
                      const hasMultipleSources = supportedSources.length > 1;
                      
                      return hasMultipleSources ? (
                        <div>
                          <label className="block text-sm font-medium text-slate-700 mb-2">
                            Model Source
                            {systemCapabilities.has_model_restriction && systemCapabilities.supported_model && <Lock className="w-3 h-3 inline ml-1 text-slate-400" />}
                          </label>
                          <div className="grid grid-cols-3 gap-2">
                            {supportedSources.map((source) => {
                              const isLocked = !!(systemCapabilities.has_model_restriction && systemCapabilities.supported_model)
                              return (
                              <button key={source} 
                                onClick={() => !isLocked && setConfig({ ...config, model_source: source as any })}
                                disabled={isLocked}
                                className={`p-3 rounded-lg border-2 text-center transition-all ${
                                  config.model_source === source ? 'border-blue-500 bg-blue-50 text-blue-600' : 'border-slate-200 text-slate-600 hover:border-slate-300'
                                } ${isLocked ? 'opacity-60 cursor-not-allowed' : ''}`}>
                                <span className="capitalize font-medium text-sm">{source}</span>
                              </button>
                            )})}
                          </div>
                        </div>
                      ) : null;
                    })()}
                    
                    <div>
                      <label className="block text-sm font-medium text-slate-700 mb-2">
                        Local Model Path
                        {systemCapabilities.has_model_restriction && systemCapabilities.supported_model && <Lock className="w-3 h-3 inline ml-1 text-slate-400" />}
                      </label>
                      <input type="text" value={config.model_path}
                        onChange={(e) => !(systemCapabilities.has_model_restriction && systemCapabilities.supported_model) && setConfig({ ...config, model_path: e.target.value })}
                        disabled={!!(systemCapabilities.has_model_restriction && systemCapabilities.supported_model)}
                        placeholder="/path/to/model (e.g., /mnt/storage/models/usf-omega)"
                        className={`w-full px-4 py-3 border border-slate-300 rounded-lg text-slate-900 placeholder-slate-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                          (systemCapabilities.has_model_restriction && systemCapabilities.supported_model) ? 'bg-slate-100 cursor-not-allowed' : ''
                        }`}
                      />
                      {!(systemCapabilities.has_model_restriction && systemCapabilities.supported_model) && (
                        <p className="text-xs text-slate-500 mt-1">
                          Enter the full path to your model directory on the server
                        </p>
                      )}
                      {systemCapabilities.has_model_restriction && systemCapabilities.supported_model && (
                        <p className="text-xs text-slate-500 mt-1">
                          Supported model: {systemCapabilities.supported_model}
                        </p>
                      )}
                    </div>
                    
                    {/* Local Model Info */}
                    {!systemCapabilities.has_model_restriction && (
                      <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                        <p className="text-sm text-slate-700">
                          <strong>Local Model:</strong> Enter the path to your model directory on the server. 
                          This should be a directory containing the model files (config.json, model weights, etc.).
                        </p>
                        <p className="text-xs text-slate-500 mt-2">
                          Example paths:
                        </p>
                        <ul className="text-xs text-slate-500 mt-1 space-y-1">
                          <li>• <code className="bg-slate-200 px-1 rounded">/root/models/usf-omega-40b</code></li>
                          <li>• <code className="bg-slate-200 px-1 rounded">/mnt/storage/models/llama-7b</code></li>
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Step 2: Dataset */}
              {currentStep === 2 && (
                <DatasetConfig 
                  selectedPaths={config.dataset_paths}
                  onSelectionChange={(paths) => setConfig(prev => ({ ...prev, dataset_paths: paths }))}
                />
              )}

              {/* Step 3: Training Settings */}
              {currentStep === 3 && (
                <TrainingSettingsStep 
                  config={config} 
                  setConfig={(fn) => setConfig(prev => ({ ...prev, ...fn(prev) }))} 
                />
              )}

              {/* Step 4: Review */}
              {currentStep === 4 && (
                <div className="space-y-5">
                  <div>
                    <h2 className="text-xl font-bold text-slate-900 mb-1">Review & Start</h2>
                    <p className="text-slate-600 text-sm">Review your configuration before training</p>
                  </div>
                  
                  {/* Output Path Configuration - Only show if external storage is attached */}
                  {systemCapabilities.has_external_storage ? (
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
                      <div><span className="text-slate-500">Model:</span> <span className="font-medium text-slate-900 break-all">{config.model_path}</span></div>
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
                      placeholder="e.g., my-qwen-finetune (auto-generated if empty)"
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
                    disabled={config.dataset_paths.length === 0 || systemStatus.status !== 'live'}
                    className="w-full py-4 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-semibold text-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-lg">
                    <Zap className="w-5 h-5" /> {systemStatus.status === 'live' ? 'Start Training' : 'System Not Ready'}
                  </button>
                </div>
              )}

              {/* Step 5: Training Progress */}
              {currentStep === 5 && jobStatus && (
                <div className="space-y-4">
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
                      jobStatus.status === 'failed' ? 'bg-red-100 text-red-700' : 'bg-slate-100 text-slate-600'
                    }`}>
                      {isTraining && <Loader2 className="w-4 h-4 inline mr-1 animate-spin" />}
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
                  
                  {/* Real-time Metrics - Prominent Display */}
                  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
                    {/* Loss - Most Important */}
                    <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-3 text-center border border-blue-200">
                      <BarChart3 className="w-5 h-5 mx-auto text-blue-600 mb-1" />
                      <span className="text-[10px] text-blue-600 font-medium uppercase">Loss</span>
                      <p className="text-xl font-bold text-blue-900">{jobStatus.current_loss !== null && jobStatus.current_loss !== undefined ? jobStatus.current_loss.toFixed(4) : '--'}</p>
                    </div>
                    {/* Epoch */}
                    <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-3 text-center border border-purple-200">
                      <Layers className="w-5 h-5 mx-auto text-purple-600 mb-1" />
                      <span className="text-[10px] text-purple-600 font-medium uppercase">Epoch</span>
                      <p className="text-xl font-bold text-purple-900">{jobStatus.epoch !== null && jobStatus.epoch !== undefined ? jobStatus.epoch : '--'}</p>
                    </div>
                    {/* Speed */}
                    <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-3 text-center border border-green-200">
                      <Activity className="w-5 h-5 mx-auto text-green-600 mb-1" />
                      <span className="text-[10px] text-green-600 font-medium uppercase">Speed</span>
                      <p className="text-xl font-bold text-green-900">{jobStatus.samples_per_second ? `${jobStatus.samples_per_second.toFixed(1)}` : '--'}<span className="text-xs font-normal"> s/s</span></p>
                    </div>
                    {/* GPU Utilization */}
                    <div className={`bg-gradient-to-br ${systemMetrics.available && systemMetrics.gpu_utilization !== null ? 'from-cyan-50 to-cyan-100 border-cyan-200' : 'from-slate-50 to-slate-100 border-slate-200'} rounded-lg p-3 text-center border`}>
                      <Cpu className="w-5 h-5 mx-auto text-cyan-600 mb-1" />
                      <span className="text-[10px] text-cyan-600 font-medium uppercase">GPU %</span>
                      <p className={`text-xl font-bold ${systemMetrics.available && systemMetrics.gpu_utilization !== null ? 'text-cyan-900' : 'text-slate-400'}`}>
                        {systemMetrics.available && systemMetrics.gpu_utilization !== null ? `${systemMetrics.gpu_utilization}%` : '--'}
                      </p>
                    </div>
                    {/* GPU Memory */}
                    <div className={`bg-gradient-to-br ${systemMetrics.available && systemMetrics.gpu_memory_used !== null ? 'from-amber-50 to-amber-100 border-amber-200' : 'from-slate-50 to-slate-100 border-slate-200'} rounded-lg p-3 text-center border`}>
                      <HardDrive className="w-5 h-5 mx-auto text-amber-600 mb-1" />
                      <span className="text-[10px] text-amber-600 font-medium uppercase">VRAM</span>
                      <p className={`text-xl font-bold ${systemMetrics.available && systemMetrics.gpu_memory_used !== null ? 'text-amber-900' : 'text-slate-400'}`}>
                        {systemMetrics.available && systemMetrics.gpu_memory_used !== null && systemMetrics.gpu_memory_total !== null 
                          ? <>{systemMetrics.gpu_memory_used.toFixed(1)}<span className="text-xs font-normal">/{systemMetrics.gpu_memory_total.toFixed(0)}G</span></>
                          : '--'}
                      </p>
                    </div>
                    {/* GPU Temperature */}
                    <div className={`bg-gradient-to-br ${systemMetrics.available && systemMetrics.gpu_temperature !== null ? 'from-orange-50 to-orange-100 border-orange-200' : 'from-slate-50 to-slate-100 border-slate-200'} rounded-lg p-3 text-center border`}>
                      <Thermometer className="w-5 h-5 mx-auto text-orange-600 mb-1" />
                      <span className="text-[10px] text-orange-600 font-medium uppercase">Temp</span>
                      <p className={`text-xl font-bold ${systemMetrics.available && systemMetrics.gpu_temperature !== null ? 'text-orange-900' : 'text-slate-400'}`}>
                        {systemMetrics.available && systemMetrics.gpu_temperature !== null ? `${systemMetrics.gpu_temperature}°` : '--'}
                      </p>
                    </div>
                  </div>

                  {/* Loss Graph */}
                  {trainingMetrics.length > 1 && (
                    <div className="bg-slate-50 rounded-lg p-3 border border-slate-200">
                      <h4 className="text-sm font-medium text-slate-700 mb-2 flex items-center gap-2">
                        <BarChart3 className="w-4 h-4 text-blue-500" /> Training Loss
                      </h4>
                      <div className="h-32 flex items-end gap-px bg-white rounded p-2">
                        {trainingMetrics.slice(-50).map((m, i) => {
                          const maxLoss = Math.max(...trainingMetrics.slice(-50).map(x => x.loss))
                          const height = (m.loss / maxLoss) * 100
                          return (
                            <div key={i} className="flex-1 bg-blue-500 rounded-t opacity-80 hover:opacity-100 transition-opacity"
                              style={{ height: `${height}%` }} title={`Step ${m.step}: ${m.loss.toFixed(4)}`} />
                          )
                        })}
                      </div>
                      <div className="flex justify-between text-xs text-slate-500 mt-1">
                        <span>Step {trainingMetrics[Math.max(0, trainingMetrics.length - 50)]?.step || 0}</span>
                        <span>Step {trainingMetrics[trainingMetrics.length - 1]?.step || 0}</span>
                      </div>
                    </div>
                  )}
                  
                  {/* Terminal Logs - Real-time output */}
                  <div className="bg-slate-900 rounded-lg p-3 h-64 overflow-y-auto font-mono text-xs text-green-400 border border-slate-700">
                    <div className="sticky top-0 bg-slate-900 pb-2 mb-2 border-b border-slate-700 text-slate-500 text-[10px]">
                      TERMINAL OUTPUT ({trainingLogs.length} lines)
                    </div>
                    {trainingLogs.length === 0 ? (
                      <div className="text-slate-500 text-center py-4">Waiting for training output...</div>
                    ) : (
                      trainingLogs.map((log, i) => (
                        <div key={i} className="hover:bg-slate-800/50 py-0.5 whitespace-pre-wrap break-all">{log}</div>
                      ))
                    )}
                    <div ref={logsEndRef} />
                  </div>
                  
                  {isTraining && (
                    <button onClick={stopTraining}
                      className="w-full py-3 bg-red-500 text-white rounded-lg font-medium hover:bg-red-600 flex items-center justify-center gap-2">
                      <StopCircle className="w-5 h-5" /> Stop Training
                    </button>
                  )}
                  
                  {jobStatus.status === 'completed' && (
                    <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-center">
                      <CheckCircle className="w-8 h-8 mx-auto text-green-600 mb-2" />
                      <h3 className="font-semibold text-green-800">Training Complete!</h3>
                      <p className="text-green-600 text-sm">Output: {config.output_dir}</p>
                      <button onClick={() => { setMainTab('inference'); setAdapterPath(config.output_dir); }}
                        className="mt-3 px-4 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700">
                        Test Model in Inference
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* Navigation */}
              {currentStep <= 4 && (
                <div className="flex justify-between mt-6 pt-4 border-t border-slate-200">
                  <button onClick={() => setCurrentStep(Math.max(1, currentStep - 1))} disabled={currentStep === 1}
                    className="flex items-center gap-2 px-4 py-2 text-slate-600 font-medium disabled:opacity-50 hover:text-slate-900">
                    <ChevronLeft className="w-5 h-5" /> Back
                  </button>
                  {currentStep < 4 && (
                    <button onClick={() => setCurrentStep(currentStep + 1)} disabled={!canProceed()}
                      className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg font-medium disabled:opacity-50 hover:bg-blue-600">
                      Next <ChevronRight className="w-5 h-5" />
                    </button>
                  )}
                </div>
              )}
            </div>
          </>
        )}

        {/* ===================== INFERENCE TAB ===================== */}
        {mainTab === 'inference' && (
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
                  placeholder="Qwen/Qwen2.5-7B-Instruct"
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-slate-900 text-sm placeholder-slate-400" />
                <button onClick={loadModel} disabled={!inferenceModel.trim() || isModelLoading}
                  className="w-full py-2 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600 disabled:opacity-50 flex items-center justify-center gap-2">
                  {isModelLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                  {isModelLoading ? 'Loading...' : 'Load Model'}
                </button>
              </div>
              
              {/* Adapter Management */}
              <div className="border-t border-slate-200 pt-4 space-y-2">
                <label className="block text-sm font-medium text-slate-700 flex items-center gap-2">
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
                      <span>Max Tokens</span><span>{inferenceSettings.max_new_tokens}</span>
                    </div>
                    <input type="range" min="64" max="2048" value={inferenceSettings.max_new_tokens}
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
    </div>
  )
}
