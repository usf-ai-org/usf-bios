'use client'

import { useState, useEffect, useRef } from 'react'
import { 
  Cpu, Database, Settings, Play, CheckCircle, 
  ChevronRight, ChevronLeft, Loader2,
  AlertCircle, Sparkles, Zap, BarChart3,
  MessageSquare, Send, Trash2, StopCircle,
  MemoryStick, RefreshCw
} from 'lucide-react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Types
interface TrainingConfig {
  model_path: string
  model_source: 'huggingface' | 'modelscope' | 'local'
  modality: 'text' | 'vision' | 'audio' | 'video'
  train_type: 'full' | 'lora' | 'qlora' | 'adalora'
  dataset_path: string
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
  deepspeed: string | null
  fsdp: string | null
  early_stop_interval: number | null
}

interface JobStatus {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped'
  current_step: number
  total_steps: number
  current_loss: number | null
  logs: string[]
  error: string | null
}

interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

// Main tabs
type MainTab = 'train' | 'inference'

// Training wizard steps
const TRAIN_STEPS = [
  { id: 1, title: 'Select Model', icon: Cpu },
  { id: 2, title: 'Configure Dataset', icon: Database },
  { id: 3, title: 'Training Settings', icon: Settings },
  { id: 4, title: 'Review & Start', icon: Play },
]

export default function Home() {
  // Main tab
  const [mainTab, setMainTab] = useState<MainTab>('train')
  
  // Training state
  const [currentStep, setCurrentStep] = useState(1)
  const [isTraining, setIsTraining] = useState(false)
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [trainingLogs, setTrainingLogs] = useState<string[]>([])
  
  const [config, setConfig] = useState<TrainingConfig>({
    model_path: '',
    model_source: 'huggingface',
    modality: 'text',
    train_type: 'lora',
    dataset_path: '',
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
    deepspeed: null,
    fsdp: null,
    early_stop_interval: null,
  })

  const [datasetValidation, setDatasetValidation] = useState<{
    valid: boolean
    total_samples: number
    errors: string[]
  } | null>(null)

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
  
  const chatEndRef = useRef<HTMLDivElement>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)

  // WebSocket for training updates
  useEffect(() => {
    if (jobStatus?.job_id && isTraining) {
      const wsUrl = API_URL.replace('http', 'ws')
      const ws = new WebSocket(`${wsUrl}/api/jobs/ws/${jobStatus.job_id}`)
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'log' && data.message) {
            setTrainingLogs(prev => [...prev.slice(-500), data.message])
          }
          if (data.type === 'progress') {
            setJobStatus(prev => prev ? {
              ...prev,
              current_step: data.step || prev.current_step,
              total_steps: data.total_steps || prev.total_steps,
              current_loss: data.loss ?? prev.current_loss,
            } : null)
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

  // Auto-scroll logs and chat
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [trainingLogs])
  
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages])

  // API functions with error handling
  const validateDataset = async () => {
    if (!config.dataset_path) return
    try {
      const res = await fetch(`${API_URL}/api/datasets/validate?dataset_path=${encodeURIComponent(config.dataset_path)}`, {
        method: 'POST',
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setDatasetValidation(data)
    } catch (e) {
      setDatasetValidation({ valid: false, total_samples: 0, errors: [`Validation failed: ${e}`] })
    }
  }

  const startTraining = async () => {
    try {
      setTrainingLogs([])
      
      // Create job
      const createRes = await fetch(`${API_URL}/api/jobs/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      if (!createRes.ok) throw new Error(`Create failed: ${createRes.status}`)
      const job = await createRes.json()
      
      // Start job
      const startRes = await fetch(`${API_URL}/api/jobs/${job.job_id}/start`, {
        method: 'POST',
      })
      if (!startRes.ok) throw new Error(`Start failed: ${startRes.status}`)
      
      setJobStatus({
        job_id: job.job_id,
        status: 'running',
        current_step: 0,
        total_steps: 0,
        current_loss: null,
        logs: [],
        error: null,
      })
      setIsTraining(true)
      setCurrentStep(5)
    } catch (e) {
      alert(`Failed to start training: ${e}`)
    }
  }

  const stopTraining = async () => {
    if (!jobStatus?.job_id) return
    try {
      await fetch(`${API_URL}/api/jobs/${jobStatus.job_id}/stop`, { method: 'POST' })
      setIsTraining(false)
    } catch (e) {
      alert(`Failed to stop: ${e}`)
    }
  }

  // Inference functions
  const fetchInferenceStatus = async () => {
    try {
      const res = await fetch(`${API_URL}/api/inference/status`)
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
      const res = await fetch(`${API_URL}/api/inference/clear-memory`, { method: 'POST' })
      if (res.ok) {
        await fetchInferenceStatus()
        alert('Memory cleared successfully')
      }
    } catch (e) {
      alert(`Failed to clear memory: ${e}`)
    }
  }

  const sendMessage = async () => {
    if (!inputMessage.trim() || !inferenceModel || isGenerating) return
    
    const userMessage: ChatMessage = { role: 'user', content: inputMessage }
    const newMessages = [...chatMessages, userMessage]
    setChatMessages(newMessages)
    setInputMessage('')
    setIsGenerating(true)
    
    try {
      const res = await fetch(`${API_URL}/api/inference/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_path: inferenceModel,
          adapter_path: adapterPath || null,
          messages: newMessages.map(m => ({ role: m.role, content: m.content })),
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
      case 2: return config.dataset_path.length > 0
      case 3: return true
      case 4: return true
      default: return false
    }
  }

  // Fetch inference status on mount and tab change
  useEffect(() => {
    if (mainTab === 'inference') {
      fetchInferenceStatus()
    }
  }, [mainTab])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Sparkles className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">USF BIOS</h1>
                <p className="text-xs text-slate-500">AI Fine-tuning Platform</p>
              </div>
            </div>
            
            {/* Main Tab Switcher */}
            <div className="flex bg-slate-100 rounded-lg p-1">
              <button
                onClick={() => setMainTab('train')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  mainTab === 'train' 
                    ? 'bg-white text-slate-900 shadow-sm' 
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Zap className="w-4 h-4 inline mr-2" />
                Fine-tuning
              </button>
              <button
                onClick={() => setMainTab('inference')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  mainTab === 'inference' 
                    ? 'bg-white text-slate-900 shadow-sm' 
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <MessageSquare className="w-4 h-4 inline mr-2" />
                Inference
              </button>
            </div>
            
            <div className="text-sm text-slate-500">Powered by US Inc</div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6">
        
        {/* ===================== TRAINING TAB ===================== */}
        {mainTab === 'train' && (
          <>
            {/* Progress Steps */}
            {currentStep <= 4 && (
              <div className="mb-6">
                <div className="flex items-center justify-between">
                  {TRAIN_STEPS.map((step, idx) => (
                    <div key={step.id} className="flex items-center">
                      <div className={`flex items-center gap-2 ${currentStep >= step.id ? 'text-primary-600' : 'text-slate-400'}`}>
                        <div className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all ${
                          currentStep > step.id ? 'bg-primary-600 border-primary-600 text-white' :
                          currentStep === step.id ? 'border-primary-600 text-primary-600 bg-primary-50' :
                          'border-slate-300 text-slate-400'
                        }`}>
                          {currentStep > step.id ? <CheckCircle className="w-5 h-5" /> : <step.icon className="w-5 h-5" />}
                        </div>
                        <span className={`hidden sm:block font-medium ${currentStep >= step.id ? 'text-slate-900' : 'text-slate-400'}`}>
                          {step.title}
                        </span>
                      </div>
                      {idx < TRAIN_STEPS.length - 1 && (
                        <div className={`w-12 sm:w-20 h-0.5 mx-2 ${currentStep > step.id ? 'bg-primary-600' : 'bg-slate-200'}`} />
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="bg-white rounded-xl shadow-lg border border-slate-200 p-6">
              
              {/* Step 1: Model */}
              {currentStep === 1 && (
                <div className="space-y-5">
                  <div>
                    <h2 className="text-xl font-bold text-slate-900 mb-1">Select Model</h2>
                    <p className="text-slate-600 text-sm">Choose the base model for fine-tuning</p>
                  </div>
                  
                  <div className="grid gap-4">
                    <div>
                      <label className="block text-sm font-medium text-slate-700 mb-2">Model Source</label>
                      <div className="grid grid-cols-3 gap-2">
                        {['huggingface', 'modelscope', 'local'].map((source) => (
                          <button key={source} onClick={() => setConfig({ ...config, model_source: source as any })}
                            className={`p-3 rounded-lg border-2 text-center transition-all ${
                              config.model_source === source ? 'border-primary-500 bg-primary-50' : 'border-slate-200 hover:border-slate-300'
                            }`}>
                            <span className="capitalize font-medium text-sm">{source}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-slate-700 mb-2">Model Path / ID</label>
                      <input type="text" value={config.model_path}
                        onChange={(e) => setConfig({ ...config, model_path: e.target.value })}
                        placeholder={config.model_source === 'local' ? '/path/to/model' : 'organization/model-name'}
                        className="w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-slate-700 mb-2">Quick Select</label>
                      <div className="grid grid-cols-2 gap-2">
                        {[
                          { id: 'Qwen/Qwen2.5-7B-Instruct', name: 'Qwen 2.5 7B' },
                          { id: 'meta-llama/Llama-3.1-8B-Instruct', name: 'Llama 3.1 8B' },
                          { id: 'mistralai/Mistral-7B-Instruct-v0.3', name: 'Mistral 7B' },
                          { id: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', name: 'DeepSeek R1 7B' },
                        ].map((model) => (
                          <button key={model.id} onClick={() => setConfig({ ...config, model_path: model.id, model_source: 'huggingface' })}
                            className={`p-2 rounded-lg border text-left transition-all ${
                              config.model_path === model.id ? 'border-primary-500 bg-primary-50' : 'border-slate-200 hover:border-slate-300'
                            }`}>
                            <span className="font-medium text-sm">{model.name}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Step 2: Dataset */}
              {currentStep === 2 && (
                <div className="space-y-5">
                  <div>
                    <h2 className="text-xl font-bold text-slate-900 mb-1">Configure Dataset</h2>
                    <p className="text-slate-600 text-sm">Specify your training dataset</p>
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-slate-700 mb-2">Dataset Path</label>
                      <div className="flex gap-2">
                        <input type="text" value={config.dataset_path}
                          onChange={(e) => setConfig({ ...config, dataset_path: e.target.value })}
                          placeholder="/path/to/dataset.jsonl"
                          className="flex-1 px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                        />
                        <button onClick={validateDataset}
                          className="px-4 py-3 bg-slate-100 hover:bg-slate-200 rounded-lg font-medium text-slate-700">
                          Validate
                        </button>
                      </div>
                    </div>
                    
                    {datasetValidation && (
                      <div className={`p-4 rounded-lg ${datasetValidation.valid ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}`}>
                        {datasetValidation.valid ? (
                          <div className="flex items-center gap-2 text-green-700">
                            <CheckCircle className="w-5 h-5" />
                            <span>Valid: {datasetValidation.total_samples} samples</span>
                          </div>
                        ) : (
                          <div className="flex items-center gap-2 text-red-700">
                            <AlertCircle className="w-5 h-5" />
                            <span>{datasetValidation.errors.join(', ')}</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Step 3: Settings */}
              {currentStep === 3 && (
                <div className="space-y-5">
                  <div>
                    <h2 className="text-xl font-bold text-slate-900 mb-1">Training Settings</h2>
                    <p className="text-slate-600 text-sm">Configure hyperparameters</p>
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-slate-700 mb-2">Training Type</label>
                      <div className="grid grid-cols-4 gap-2">
                        {[
                          { id: 'lora', name: 'LoRA' },
                          { id: 'qlora', name: 'QLoRA' },
                          { id: 'adalora', name: 'AdaLoRA' },
                          { id: 'full', name: 'Full' },
                        ].map((type) => (
                          <button key={type.id} onClick={() => setConfig({ ...config, train_type: type.id as any, quant_bits: type.id === 'qlora' ? 4 : null })}
                            className={`p-2 rounded-lg border-2 text-center transition-all ${
                              config.train_type === type.id ? 'border-primary-500 bg-primary-50' : 'border-slate-200 hover:border-slate-300'
                            }`}>
                            <span className="font-medium text-sm">{type.name}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-3">
                      <div>
                        <label className="block text-sm font-medium text-slate-700 mb-1">Epochs</label>
                        <input type="number" value={config.num_train_epochs}
                          onChange={(e) => setConfig({ ...config, num_train_epochs: parseInt(e.target.value) || 1 })}
                          className="w-full px-3 py-2 border border-slate-300 rounded-lg" />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-slate-700 mb-1">Learning Rate</label>
                        <input type="number" step="0.00001" value={config.learning_rate}
                          onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) || 0.0001 })}
                          className="w-full px-3 py-2 border border-slate-300 rounded-lg" />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-slate-700 mb-1">Batch Size</label>
                        <input type="number" value={config.per_device_train_batch_size}
                          onChange={(e) => setConfig({ ...config, per_device_train_batch_size: parseInt(e.target.value) || 1 })}
                          className="w-full px-3 py-2 border border-slate-300 rounded-lg" />
                      </div>
                    </div>
                    
                    {['lora', 'qlora', 'adalora'].includes(config.train_type) && (
                      <div className="bg-slate-50 rounded-lg p-4">
                        <h4 className="font-medium text-slate-900 mb-3">LoRA Settings</h4>
                        <div className="grid grid-cols-3 gap-3">
                          <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Rank</label>
                            <input type="number" value={config.lora_rank}
                              onChange={(e) => setConfig({ ...config, lora_rank: parseInt(e.target.value) || 8 })}
                              className="w-full px-3 py-2 border border-slate-300 rounded-lg" />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Alpha</label>
                            <input type="number" value={config.lora_alpha}
                              onChange={(e) => setConfig({ ...config, lora_alpha: parseInt(e.target.value) || 32 })}
                              className="w-full px-3 py-2 border border-slate-300 rounded-lg" />
                          </div>
                          <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Dropout</label>
                            <input type="number" step="0.01" value={config.lora_dropout}
                              onChange={(e) => setConfig({ ...config, lora_dropout: parseFloat(e.target.value) || 0.05 })}
                              className="w-full px-3 py-2 border border-slate-300 rounded-lg" />
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Step 4: Review */}
              {currentStep === 4 && (
                <div className="space-y-5">
                  <div>
                    <h2 className="text-xl font-bold text-slate-900 mb-1">Review & Start</h2>
                    <p className="text-slate-600 text-sm">Review your configuration</p>
                  </div>
                  
                  <div className="bg-slate-50 rounded-lg p-4 grid grid-cols-2 gap-3 text-sm">
                    <div><span className="text-slate-500">Model:</span> <span className="font-medium">{config.model_path}</span></div>
                    <div><span className="text-slate-500">Type:</span> <span className="font-medium uppercase">{config.train_type}</span></div>
                    <div><span className="text-slate-500">Dataset:</span> <span className="font-medium">{config.dataset_path}</span></div>
                    <div><span className="text-slate-500">Epochs:</span> <span className="font-medium">{config.num_train_epochs}</span></div>
                  </div>
                  
                  <button onClick={startTraining}
                    className="w-full py-4 bg-gradient-to-r from-primary-600 to-purple-600 text-white rounded-lg font-semibold text-lg hover:opacity-90 flex items-center justify-center gap-2">
                    <Zap className="w-5 h-5" /> Start Training
                  </button>
                </div>
              )}

              {/* Step 5: Training Progress */}
              {currentStep === 5 && jobStatus && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h2 className="text-xl font-bold text-slate-900">Training Progress</h2>
                      <p className="text-slate-600 text-sm">Job: {jobStatus.job_id}</p>
                    </div>
                    <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                      jobStatus.status === 'running' ? 'bg-blue-100 text-blue-700' :
                      jobStatus.status === 'completed' ? 'bg-green-100 text-green-700' :
                      jobStatus.status === 'failed' ? 'bg-red-100 text-red-700' : 'bg-slate-100 text-slate-700'
                    }`}>
                      {isTraining && <Loader2 className="w-4 h-4 inline mr-1 animate-spin" />}
                      {jobStatus.status.toUpperCase()}
                    </div>
                  </div>
                  
                  {jobStatus.total_steps > 0 && (
                    <div>
                      <div className="flex justify-between text-sm text-slate-600 mb-1">
                        <span>Step {jobStatus.current_step} / {jobStatus.total_steps}</span>
                        <span>{Math.round((jobStatus.current_step / jobStatus.total_steps) * 100)}%</span>
                      </div>
                      <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-primary-500 to-purple-500 transition-all"
                          style={{ width: `${(jobStatus.current_step / jobStatus.total_steps) * 100}%` }} />
                      </div>
                    </div>
                  )}
                  
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-slate-50 rounded-lg p-3 text-center">
                      <BarChart3 className="w-5 h-5 mx-auto text-slate-400 mb-1" />
                      <span className="text-xs text-slate-500">Loss</span>
                      <p className="text-lg font-bold">{jobStatus.current_loss?.toFixed(4) || '-'}</p>
                    </div>
                    <div className="bg-slate-50 rounded-lg p-3 text-center">
                      <Zap className="w-5 h-5 mx-auto text-slate-400 mb-1" />
                      <span className="text-xs text-slate-500">Step</span>
                      <p className="text-lg font-bold">{jobStatus.current_step}</p>
                    </div>
                  </div>
                  
                  <div className="bg-slate-900 rounded-lg p-3 h-48 overflow-y-auto font-mono text-xs text-green-400">
                    {trainingLogs.map((log, i) => <div key={i}>{log}</div>)}
                    <div ref={logsEndRef} />
                  </div>
                  
                  {isTraining && (
                    <button onClick={stopTraining}
                      className="w-full py-3 bg-red-600 text-white rounded-lg font-medium hover:bg-red-700 flex items-center justify-center gap-2">
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
                    className="flex items-center gap-2 px-4 py-2 text-slate-600 font-medium disabled:opacity-50">
                    <ChevronLeft className="w-5 h-5" /> Back
                  </button>
                  {currentStep < 4 && (
                    <button onClick={() => setCurrentStep(currentStep + 1)} disabled={!canProceed()}
                      className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg font-medium disabled:opacity-50">
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
          <div className="grid grid-cols-3 gap-4">
            {/* Settings Panel */}
            <div className="bg-white rounded-xl shadow-lg border border-slate-200 p-4 space-y-4">
              <h3 className="font-bold text-slate-900">Model Settings</h3>
              
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Model Path</label>
                <input type="text" value={inferenceModel}
                  onChange={(e) => setInferenceModel(e.target.value)}
                  placeholder="Qwen/Qwen2.5-7B-Instruct"
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm" />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">LoRA Adapter (optional)</label>
                <input type="text" value={adapterPath}
                  onChange={(e) => setAdapterPath(e.target.value)}
                  placeholder="/path/to/adapter"
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm" />
              </div>
              
              <div className="border-t pt-4">
                <h4 className="text-sm font-medium text-slate-700 mb-2">Generation Settings</h4>
                <div className="space-y-2">
                  <div>
                    <label className="text-xs text-slate-600">Max Tokens: {inferenceSettings.max_new_tokens}</label>
                    <input type="range" min="64" max="2048" value={inferenceSettings.max_new_tokens}
                      onChange={(e) => setInferenceSettings({ ...inferenceSettings, max_new_tokens: parseInt(e.target.value) })}
                      className="w-full" />
                  </div>
                  <div>
                    <label className="text-xs text-slate-600">Temperature: {inferenceSettings.temperature}</label>
                    <input type="range" min="0" max="2" step="0.1" value={inferenceSettings.temperature}
                      onChange={(e) => setInferenceSettings({ ...inferenceSettings, temperature: parseFloat(e.target.value) })}
                      className="w-full" />
                  </div>
                </div>
              </div>
              
              <div className="border-t pt-4">
                <h4 className="text-sm font-medium text-slate-700 mb-2">Memory Status</h4>
                <div className="text-xs text-slate-600 space-y-1">
                  <p>Model Loaded: {inferenceStatus.model_loaded ? 'Yes' : 'No'}</p>
                  {inferenceStatus.model_path && <p className="truncate">Path: {inferenceStatus.model_path}</p>}
                  <p>GPU Memory: {inferenceStatus.memory_used_gb.toFixed(2)} GB</p>
                </div>
                <div className="flex gap-2 mt-2">
                  <button onClick={fetchInferenceStatus}
                    className="flex-1 px-2 py-1 bg-slate-100 hover:bg-slate-200 rounded text-xs font-medium flex items-center justify-center gap-1">
                    <RefreshCw className="w-3 h-3" /> Refresh
                  </button>
                  <button onClick={clearMemory}
                    className="flex-1 px-2 py-1 bg-red-100 hover:bg-red-200 text-red-700 rounded text-xs font-medium flex items-center justify-center gap-1">
                    <Trash2 className="w-3 h-3" /> Clear
                  </button>
                </div>
              </div>
            </div>
            
            {/* Chat Panel */}
            <div className="col-span-2 bg-white rounded-xl shadow-lg border border-slate-200 flex flex-col h-[600px]">
              <div className="p-4 border-b border-slate-200">
                <h3 className="font-bold text-slate-900">Chat Interface</h3>
                <p className="text-xs text-slate-500">Test your model with chat messages</p>
              </div>
              
              <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {chatMessages.length === 0 && (
                  <div className="text-center text-slate-400 py-10">
                    <MessageSquare className="w-10 h-10 mx-auto mb-2 opacity-50" />
                    <p>Enter a model path and start chatting</p>
                  </div>
                )}
                {chatMessages.map((msg, i) => (
                  <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[80%] px-4 py-2 rounded-lg ${
                      msg.role === 'user' 
                        ? 'bg-primary-600 text-white' 
                        : 'bg-slate-100 text-slate-900'
                    }`}>
                      <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                    </div>
                  </div>
                ))}
                {isGenerating && (
                  <div className="flex justify-start">
                    <div className="bg-slate-100 px-4 py-2 rounded-lg">
                      <Loader2 className="w-5 h-5 animate-spin text-slate-400" />
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
              
              <div className="p-4 border-t border-slate-200">
                <div className="flex gap-2">
                  <button onClick={() => setChatMessages([])}
                    className="px-3 py-2 bg-slate-100 hover:bg-slate-200 rounded-lg">
                    <Trash2 className="w-5 h-5 text-slate-600" />
                  </button>
                  <input type="text" value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
                    placeholder="Type your message..."
                    disabled={!inferenceModel || isGenerating}
                    className="flex-1 px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-500 disabled:opacity-50" />
                  <button onClick={sendMessage} disabled={!inferenceModel || !inputMessage.trim() || isGenerating}
                    className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50">
                    <Send className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="mt-6 py-4 text-center text-xs text-slate-500">
        USF BIOS v1.0.0 - Copyright 2024-2026 US Inc. All rights reserved.
      </footer>
    </div>
  )
}
