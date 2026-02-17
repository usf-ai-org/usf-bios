'use client'

import React, { useState, useEffect, useCallback } from 'react'
import {
  Repeat,
  Play,
  Pause,
  Square,
  Trash2,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  Loader2,
  FolderOpen,
  Settings,
  BarChart3,
  History,
  Sparkles,
  Brain,
  Target,
  Zap,
  Info,
  X,
  Globe,
  Code,
  Calculator,
  BookOpen,
  Database,
  GraduationCap,
  Layers
} from 'lucide-react'

// ==================== Types ====================

interface RoundMetrics {
  round_number: number
  started_at: string
  completed_at: string | null
  num_prompts: number
  num_generated: number
  generation_time_seconds: number
  num_judged: number
  judging_time_seconds: number
  mean_reward_score: number
  min_reward_score: number
  max_reward_score: number
  num_filtered: number
  filter_threshold_used: number
  training_loss_start: number
  training_loss_end: number
  training_time_seconds: number
  checkpoint_path: string | null
  peak_vram_gb: number
}

interface IterativeJob {
  id: string
  name: string
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled'
  current_phase: 'idle' | 'generating' | 'judging' | 'filtering' | 'training' | 'cleanup' | 'completed' | 'failed' | 'cancelled'
  current_round: number
  total_rounds: number
  progress_percent: number
  created_at: string
  started_at: string | null
  completed_at: string | null
  error_message: string | null
  final_model_path: string | null
  round_history: RoundMetrics[]
  config?: any
}

interface RewardConfig {
  type: 'local' | 'api' | 'script' | 'rule_based'
  model_path: string
  api_endpoint: string
  api_key: string
  api_timeout: number
  api_batch_size: number
  script_path: string
  script_function: string
  rule_type: string
}

interface DatasetConfigItem {
  path: string
  name: string
  difficulty: 'easy' | 'medium' | 'hard' | 'expert'
  weight: number
}

interface IterativeTrainingProps {
  isOpen: boolean
  onClose: () => void
  availableModels: Array<{ path: string; name: string }>
  availableDatasets: Array<{ path: string; name: string }>
}

// ==================== API Functions ====================

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

async function fetchJobs(): Promise<IterativeJob[]> {
  const response = await fetch(`${API_BASE}/iterative/jobs`)
  if (!response.ok) throw new Error('Failed to fetch jobs')
  const data = await response.json()
  return data.jobs
}

async function createJob(config: any): Promise<IterativeJob> {
  const response = await fetch(`${API_BASE}/iterative/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config)
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to create job')
  }
  return response.json()
}

async function startJob(jobId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/iterative/jobs/${jobId}/start`, {
    method: 'POST'
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to start job')
  }
}

async function pauseJob(jobId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/iterative/jobs/${jobId}/pause`, {
    method: 'POST'
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to pause job')
  }
}

async function cancelJob(jobId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/iterative/jobs/${jobId}/cancel`, {
    method: 'POST'
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to cancel job')
  }
}

async function deleteJob(jobId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/iterative/jobs/${jobId}`, {
    method: 'DELETE'
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to delete job')
  }
}

// ==================== Helper Functions ====================

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleString()
}

function getStatusColor(status: string): string {
  switch (status) {
    case 'completed': return 'text-green-400'
    case 'running': return 'text-blue-400'
    case 'paused': return 'text-yellow-400'
    case 'failed': return 'text-red-400'
    case 'cancelled': return 'text-gray-400'
    default: return 'text-slate-400'
  }
}

function getStatusBgColor(status: string): string {
  switch (status) {
    case 'completed': return 'bg-green-500/20 border-green-500/30'
    case 'running': return 'bg-blue-500/20 border-blue-500/30'
    case 'paused': return 'bg-yellow-500/20 border-yellow-500/30'
    case 'failed': return 'bg-red-500/20 border-red-500/30'
    case 'cancelled': return 'bg-gray-500/20 border-gray-500/30'
    default: return 'bg-slate-500/20 border-slate-500/30'
  }
}

function getPhaseIcon(phase: string) {
  switch (phase) {
    case 'generating': return <Sparkles className="w-4 h-4" />
    case 'judging': return <Brain className="w-4 h-4" />
    case 'filtering': return <Target className="w-4 h-4" />
    case 'training': return <Zap className="w-4 h-4" />
    case 'completed': return <CheckCircle className="w-4 h-4" />
    case 'failed': return <XCircle className="w-4 h-4" />
    default: return <Clock className="w-4 h-4" />
  }
}

// ==================== Sub-Components ====================

function PhaseProgress({ phase, progress }: { phase: string; progress: number }) {
  const phases = ['generating', 'judging', 'filtering', 'training']
  const currentIndex = phases.indexOf(phase)
  
  return (
    <div className="flex items-center gap-1">
      {phases.map((p, i) => (
        <div
          key={p}
          className={`h-1.5 flex-1 rounded-full transition-colors ${
            i < currentIndex
              ? 'bg-green-500'
              : i === currentIndex
              ? 'bg-blue-500'
              : 'bg-slate-600'
          }`}
        />
      ))}
    </div>
  )
}

function RoundCard({ round, isExpanded, onToggle }: { 
  round: RoundMetrics
  isExpanded: boolean
  onToggle: () => void 
}) {
  return (
    <div className="border border-slate-700 rounded-lg overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full px-4 py-3 flex items-center justify-between bg-slate-800/50 hover:bg-slate-800 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-slate-300">
            Round {round.round_number + 1}
          </span>
          <span className="text-xs text-slate-500">
            {round.completed_at ? '✓ Completed' : 'In Progress'}
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-xs text-slate-400">
            Score: {round.mean_reward_score.toFixed(3)}
          </span>
          <span className="text-xs text-slate-400">
            {round.num_filtered} samples
          </span>
          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </button>
      
      {isExpanded && (
        <div className="px-4 py-3 bg-slate-900/50 space-y-3">
          <div className="grid grid-cols-4 gap-4 text-xs">
            <div>
              <span className="text-slate-500">Generated</span>
              <p className="text-slate-300">{round.num_generated} samples</p>
              <p className="text-slate-500">{formatDuration(round.generation_time_seconds)}</p>
            </div>
            <div>
              <span className="text-slate-500">Judged</span>
              <p className="text-slate-300">{round.num_judged} samples</p>
              <p className="text-slate-500">{formatDuration(round.judging_time_seconds)}</p>
            </div>
            <div>
              <span className="text-slate-500">Filtered</span>
              <p className="text-slate-300">{round.num_filtered} kept</p>
              <p className="text-slate-500">Threshold: {round.filter_threshold_used}</p>
            </div>
            <div>
              <span className="text-slate-500">Trained</span>
              <p className="text-slate-300">Loss: {round.training_loss_end.toFixed(4)}</p>
              <p className="text-slate-500">{formatDuration(round.training_time_seconds)}</p>
            </div>
          </div>
          
          <div className="pt-2 border-t border-slate-700">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-500">Reward Score Range</span>
              <span className="text-slate-300">
                {round.min_reward_score.toFixed(3)} - {round.max_reward_score.toFixed(3)}
                <span className="text-slate-500 ml-2">(avg: {round.mean_reward_score.toFixed(3)})</span>
              </span>
            </div>
            {round.checkpoint_path && (
              <div className="flex items-center justify-between text-xs mt-1">
                <span className="text-slate-500">Checkpoint</span>
                <span className="text-slate-400 font-mono text-[10px]">{round.checkpoint_path}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function JobCard({ job, onStart, onPause, onCancel, onDelete, onViewDetails }: {
  job: IterativeJob
  onStart: () => void
  onPause: () => void
  onCancel: () => void
  onDelete: () => void
  onViewDetails: () => void
}) {
  const isRunning = job.status === 'running'
  const isPaused = job.status === 'paused'
  const isPending = job.status === 'pending'
  const isCompleted = job.status === 'completed'
  const isFailed = job.status === 'failed'
  
  return (
    <div className={`border rounded-lg p-4 ${getStatusBgColor(job.status)}`}>
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="font-medium text-white">{job.name}</h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Created {formatDate(job.created_at)}
          </p>
        </div>
        <span className={`text-xs font-medium px-2 py-1 rounded ${getStatusColor(job.status)} bg-black/20`}>
          {job.status?.toUpperCase() || 'UNKNOWN'}
        </span>
      </div>
      
      {/* Progress */}
      <div className="mb-3">
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="text-slate-400">
            Round {job.current_round + 1} / {job.total_rounds}
          </span>
          <span className="text-slate-400">
            {job.progress_percent.toFixed(0)}%
          </span>
        </div>
        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
          <div 
            className="h-full bg-blue-500 transition-all duration-300"
            style={{ width: `${job.progress_percent}%` }}
          />
        </div>
        {isRunning && (
          <div className="mt-2">
            <div className="flex items-center gap-2 text-xs text-slate-400">
              {getPhaseIcon(job.current_phase)}
              <span className="capitalize">{job.current_phase}</span>
            </div>
            <PhaseProgress phase={job.current_phase} progress={0} />
          </div>
        )}
      </div>
      
      {/* Error message */}
      {job.error_message && (
        <div className="mb-3 p-2 bg-red-500/10 border border-red-500/30 rounded text-xs text-red-400">
          {job.error_message}
        </div>
      )}
      
      {/* Final model path */}
      {job.final_model_path && (
        <div className="mb-3 p-2 bg-green-500/10 border border-green-500/30 rounded">
          <p className="text-xs text-green-400">Final Model:</p>
          <p className="text-xs text-slate-300 font-mono truncate">{job.final_model_path}</p>
        </div>
      )}
      
      {/* Actions */}
      <div className="flex items-center gap-2">
        {(isPending || isPaused) && (
          <button
            onClick={onStart}
            className="flex items-center gap-1 px-3 py-1.5 bg-green-600 hover:bg-green-500 text-white text-xs rounded transition-colors"
          >
            <Play className="w-3 h-3" />
            {isPaused ? 'Resume' : 'Start'}
          </button>
        )}
        
        {isRunning && (
          <>
            <button
              onClick={onPause}
              className="flex items-center gap-1 px-3 py-1.5 bg-yellow-600 hover:bg-yellow-500 text-white text-xs rounded transition-colors"
            >
              <Pause className="w-3 h-3" />
              Pause
            </button>
            <button
              onClick={onCancel}
              className="flex items-center gap-1 px-3 py-1.5 bg-red-600 hover:bg-red-500 text-white text-xs rounded transition-colors"
            >
              <Square className="w-3 h-3" />
              Cancel
            </button>
          </>
        )}
        
        <button
          onClick={onViewDetails}
          className="flex items-center gap-1 px-3 py-1.5 bg-slate-600 hover:bg-slate-500 text-white text-xs rounded transition-colors"
        >
          <History className="w-3 h-3" />
          History
        </button>
        
        {!isRunning && (
          <button
            onClick={onDelete}
            className="flex items-center gap-1 px-3 py-1.5 bg-slate-700 hover:bg-red-600 text-slate-300 hover:text-white text-xs rounded transition-colors ml-auto"
          >
            <Trash2 className="w-3 h-3" />
          </button>
        )}
      </div>
    </div>
  )
}

// ==================== Main Component ====================

export default function IterativeTraining({
  isOpen,
  onClose,
  availableModels,
  availableDatasets
}: IterativeTrainingProps) {
  // State
  const [jobs, setJobs] = useState<IterativeJob[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [selectedJob, setSelectedJob] = useState<IterativeJob | null>(null)
  const [expandedRounds, setExpandedRounds] = useState<Set<number>>(new Set())
  
  // Form state
  const [formData, setFormData] = useState({
    name: '',
    base_model_path: '',
    reward_model_path: '',
    prompts_dataset_path: '',
    num_rounds: 5,
    num_generations_per_prompt: 8,
    max_new_tokens: 512,
    temperature: 0.7,
    top_p: 0.9,
    filter_strategy: 'top_k_percent',
    filter_top_k_percent: 20,
    filter_threshold: 0.5,
    filter_top_n: 1000,
    training_method: 'lora',
    learning_rate: 0.00001,
    num_train_epochs: 1,
    batch_size: 4,
    gradient_accumulation_steps: 4,
    use_previous_round_model: true,
    // Advanced reward config
    reward_type: 'local' as 'local' | 'api' | 'script' | 'rule_based',
    api_endpoint: '',
    api_key: '',
    api_timeout: 30,
    api_batch_size: 10,
    script_path: '',
    script_function: 'score',
    rule_type: 'math',
    // Dataset selection
    dataset_selection_strategy: 'sequential',
    enable_difficulty_curriculum: false,
    samples_per_round: 1000,
    // Multi-dataset support
    datasets: [] as DatasetConfigItem[]
  })
  
  // UI state for advanced options
  const [showAdvancedReward, setShowAdvancedReward] = useState(false)
  const [showDatasetFormat, setShowDatasetFormat] = useState(false)
  
  // Fetch jobs
  const loadJobs = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const data = await fetchJobs()
      setJobs(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load jobs')
    } finally {
      setIsLoading(false)
    }
  }, [])
  
  // Auto-refresh for running jobs
  useEffect(() => {
    if (!isOpen) return
    
    loadJobs()
    
    const hasRunningJobs = jobs.some(j => j.status === 'running')
    if (hasRunningJobs) {
      const interval = setInterval(loadJobs, 3000)
      return () => clearInterval(interval)
    }
  }, [isOpen, loadJobs])
  
  // Handlers
  const handleCreateJob = async () => {
    // Validate required fields based on reward type
    if (!formData.name || !formData.base_model_path || !formData.prompts_dataset_path) {
      setError('Please fill in all required fields (name, base model, dataset)')
      return
    }
    
    // Validate reward-specific requirements
    if (formData.reward_type === 'local' && !formData.reward_model_path) {
      setError('Please provide a reward model path for local model scoring')
      return
    }
    if (formData.reward_type === 'api' && !formData.api_endpoint) {
      setError('Please provide an API endpoint for API scoring')
      return
    }
    if (formData.reward_type === 'script' && !formData.script_path) {
      setError('Please provide a script path for script-based scoring')
      return
    }
    
    setIsLoading(true)
    setError(null)
    try {
      // Build reward config based on type
      const rewardConfig = formData.reward_type !== 'local' ? {
        type: formData.reward_type,
        model_path: formData.reward_model_path,
        api_endpoint: formData.api_endpoint,
        api_key: formData.api_key,
        api_timeout: formData.api_timeout,
        api_batch_size: formData.api_batch_size,
        script_path: formData.script_path,
        script_function: formData.script_function,
        rule_type: formData.rule_type,
      } : null
      
      // Build the complete job config
      const jobConfig = {
        name: formData.name,
        base_model_path: formData.base_model_path,
        reward_model_path: formData.reward_model_path,
        prompts_dataset_path: formData.prompts_dataset_path,
        num_rounds: formData.num_rounds,
        num_generations_per_prompt: formData.num_generations_per_prompt,
        max_new_tokens: formData.max_new_tokens,
        temperature: formData.temperature,
        top_p: formData.top_p,
        filter_strategy: formData.filter_strategy,
        filter_top_k_percent: formData.filter_top_k_percent,
        filter_threshold: formData.filter_threshold,
        filter_top_n: formData.filter_top_n,
        training_method: formData.training_method,
        learning_rate: formData.learning_rate,
        num_train_epochs: formData.num_train_epochs,
        batch_size: formData.batch_size,
        gradient_accumulation_steps: formData.gradient_accumulation_steps,
        use_previous_round_model: formData.use_previous_round_model,
        // Advanced reward config
        reward_config: rewardConfig,
        // Dataset selection
        dataset_selection_strategy: formData.dataset_selection_strategy,
        enable_difficulty_curriculum: formData.enable_difficulty_curriculum,
        samples_per_round: formData.samples_per_round,
        datasets: formData.datasets.length > 0 ? formData.datasets : undefined,
      }
      
      await createJob(jobConfig)
      await loadJobs()
      setShowCreateForm(false)
      // Reset form
      setFormData({
        ...formData,
        name: '',
        base_model_path: '',
        reward_model_path: '',
        prompts_dataset_path: '',
        api_endpoint: '',
        api_key: '',
        script_path: '',
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create job')
    } finally {
      setIsLoading(false)
    }
  }
  
  const handleStartJob = async (jobId: string) => {
    try {
      await startJob(jobId)
      await loadJobs()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start job')
    }
  }
  
  const handlePauseJob = async (jobId: string) => {
    try {
      await pauseJob(jobId)
      await loadJobs()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to pause job')
    }
  }
  
  const handleCancelJob = async (jobId: string) => {
    try {
      await cancelJob(jobId)
      await loadJobs()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel job')
    }
  }
  
  const handleDeleteJob = async (jobId: string) => {
    if (!confirm('Are you sure you want to delete this job and all its data?')) return
    try {
      await deleteJob(jobId)
      await loadJobs()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete job')
    }
  }
  
  const toggleRoundExpanded = (roundNum: number) => {
    setExpandedRounds(prev => {
      const next = new Set(prev)
      if (next.has(roundNum)) {
        next.delete(roundNum)
      } else {
        next.add(roundNum)
      }
      return next
    })
  }
  
  if (!isOpen) return null
  
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-slate-900 border border-slate-700 rounded-xl w-full max-w-5xl max-h-[90vh] overflow-hidden shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700">
          <div className="flex items-center gap-3">
            <Repeat className="w-6 h-6 text-blue-400" />
            <div>
              <h2 className="text-lg font-semibold text-white">Iterative Self-Training</h2>
              <p className="text-xs text-slate-400">ReST / STaR / Expert Iteration</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={loadJobs}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
              title="Refresh"
            >
              <RefreshCw className={`w-4 h-4 text-slate-400 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-slate-400" />
            </button>
          </div>
        </div>
        
        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
          {/* Error */}
          {error && (
            <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2 text-red-400">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span className="text-sm">{error}</span>
              <button onClick={() => setError(null)} className="ml-auto">
                <X className="w-4 h-4" />
              </button>
            </div>
          )}
          
          {/* Info Banner */}
          <div className="mb-6 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-slate-300">
                <p className="font-medium text-blue-400 mb-1">Generate → Judge → Train → Repeat</p>
                <p>Iterative self-training improves your model by generating responses, scoring them with a reward model, filtering the best ones, and training on them. Each round produces a better model.</p>
              </div>
            </div>
          </div>
          
          {/* Create New Job */}
          {!showCreateForm ? (
            <button
              onClick={() => setShowCreateForm(true)}
              className="w-full mb-6 p-4 border-2 border-dashed border-slate-600 rounded-lg hover:border-blue-500 hover:bg-blue-500/5 transition-colors text-slate-400 hover:text-blue-400"
            >
              <div className="flex items-center justify-center gap-2">
                <Sparkles className="w-5 h-5" />
                <span>Create New Iterative Training Job</span>
              </div>
            </button>
          ) : (
            <div className="mb-6 p-4 bg-slate-800/50 border border-slate-700 rounded-lg">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-medium text-white">New Iterative Training Job</h3>
                <button onClick={() => setShowCreateForm(false)} className="text-slate-400 hover:text-white">
                  <X className="w-4 h-4" />
                </button>
              </div>
              
              <div className="space-y-4">
                {/* Basic Settings */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Job Name *</label>
                    <input
                      type="text"
                      value={formData.name}
                      onChange={e => setFormData({ ...formData, name: e.target.value })}
                      placeholder="my-iterative-training"
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white placeholder-slate-500 focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Number of Rounds</label>
                    <input
                      type="number"
                      value={formData.num_rounds}
                      onChange={e => setFormData({ ...formData, num_rounds: parseInt(e.target.value) || 5 })}
                      min={1}
                      max={100}
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                </div>
                
                {/* Base Model */}
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Base Model Path *</label>
                  <input
                    type="text"
                    value={formData.base_model_path}
                    onChange={e => setFormData({ ...formData, base_model_path: e.target.value })}
                    placeholder="/path/to/model or huggingface/model-name"
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white placeholder-slate-500 focus:border-blue-500 focus:outline-none"
                  />
                </div>
                
                {/* Reward Model Configuration */}
                <div className="border border-slate-700 rounded-lg p-4 bg-slate-800/30">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Brain className="w-4 h-4 text-purple-400" />
                      <h4 className="text-sm font-medium text-white">Reward Model / Judge</h4>
                    </div>
                    <button
                      type="button"
                      onClick={() => setShowAdvancedReward(!showAdvancedReward)}
                      className="text-xs text-blue-400 hover:text-blue-300"
                    >
                      {showAdvancedReward ? 'Simple Mode' : 'Advanced Options'}
                    </button>
                  </div>
                  
                  {/* Reward Type Selection */}
                  <div className="grid grid-cols-4 gap-2 mb-3">
                    {[
                      { id: 'local', label: 'Local Model', icon: Database, desc: 'HuggingFace reward model' },
                      { id: 'api', label: 'External API', icon: Globe, desc: 'Send to scoring API' },
                      { id: 'script', label: 'Custom Script', icon: Code, desc: 'Run Python script' },
                      { id: 'rule_based', label: 'Rule-Based', icon: Calculator, desc: 'Verify math/code' },
                    ].map(opt => (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => setFormData({ ...formData, reward_type: opt.id as any })}
                        className={`p-2 rounded-lg border text-left transition-all ${
                          formData.reward_type === opt.id
                            ? 'border-purple-500 bg-purple-500/20'
                            : 'border-slate-600 hover:border-slate-500'
                        }`}
                      >
                        <opt.icon className={`w-4 h-4 mb-1 ${formData.reward_type === opt.id ? 'text-purple-400' : 'text-slate-400'}`} />
                        <p className={`text-xs font-medium ${formData.reward_type === opt.id ? 'text-purple-300' : 'text-slate-300'}`}>{opt.label}</p>
                        <p className="text-[10px] text-slate-500">{opt.desc}</p>
                      </button>
                    ))}
                  </div>
                  
                  {/* Type-specific config */}
                  {formData.reward_type === 'local' && (
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Reward Model Path *</label>
                      <input
                        type="text"
                        value={formData.reward_model_path}
                        onChange={e => setFormData({ ...formData, reward_model_path: e.target.value })}
                        placeholder="OpenAssistant/reward-model-deberta-v3-large-v2"
                        className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white placeholder-slate-500 focus:border-blue-500 focus:outline-none"
                      />
                    </div>
                  )}
                  
                  {formData.reward_type === 'api' && (
                    <div className="space-y-3">
                      <div>
                        <label className="block text-xs text-slate-400 mb-1">API Endpoint *</label>
                        <input
                          type="text"
                          value={formData.api_endpoint}
                          onChange={e => setFormData({ ...formData, api_endpoint: e.target.value })}
                          placeholder="https://your-server.com/score"
                          className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white placeholder-slate-500 focus:border-blue-500 focus:outline-none"
                        />
                      </div>
                      {showAdvancedReward && (
                        <div className="grid grid-cols-3 gap-3">
                          <div>
                            <label className="block text-xs text-slate-400 mb-1">API Key (optional)</label>
                            <input
                              type="password"
                              value={formData.api_key}
                              onChange={e => setFormData({ ...formData, api_key: e.target.value })}
                              placeholder="Bearer token"
                              className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white placeholder-slate-500 focus:border-blue-500 focus:outline-none"
                            />
                          </div>
                          <div>
                            <label className="block text-xs text-slate-400 mb-1">Timeout (sec)</label>
                            <input
                              type="number"
                              value={formData.api_timeout}
                              onChange={e => setFormData({ ...formData, api_timeout: parseInt(e.target.value) || 30 })}
                              className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white focus:border-blue-500 focus:outline-none"
                            />
                          </div>
                          <div>
                            <label className="block text-xs text-slate-400 mb-1">Batch Size</label>
                            <input
                              type="number"
                              value={formData.api_batch_size}
                              onChange={e => setFormData({ ...formData, api_batch_size: parseInt(e.target.value) || 10 })}
                              className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white focus:border-blue-500 focus:outline-none"
                            />
                          </div>
                        </div>
                      )}
                      <div className="p-2 bg-blue-500/10 border border-blue-500/30 rounded text-xs text-blue-300">
                        <strong>API Format:</strong> POST with {`{"items": [{"prompt": "...", "response": "..."}]}`}. 
                        Returns {`{"scores": [0.85, 0.72]}`}. Retries automatically on failure.
                      </div>
                    </div>
                  )}
                  
                  {formData.reward_type === 'script' && (
                    <div className="space-y-3">
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="block text-xs text-slate-400 mb-1">Script Path *</label>
                          <input
                            type="text"
                            value={formData.script_path}
                            onChange={e => setFormData({ ...formData, script_path: e.target.value })}
                            placeholder="/path/to/my_scorer.py"
                            className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white placeholder-slate-500 focus:border-blue-500 focus:outline-none"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-slate-400 mb-1">Function Name</label>
                          <input
                            type="text"
                            value={formData.script_function}
                            onChange={e => setFormData({ ...formData, script_function: e.target.value })}
                            placeholder="score"
                            className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white placeholder-slate-500 focus:border-blue-500 focus:outline-none"
                          />
                        </div>
                      </div>
                      <div className="p-2 bg-slate-700/50 border border-slate-600 rounded text-xs text-slate-400">
                        Script receives --input (JSONL) and --output paths. Each line needs "reward_score" field added.
                      </div>
                    </div>
                  )}
                  
                  {formData.reward_type === 'rule_based' && (
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Rule Type</label>
                      <select
                        value={formData.rule_type}
                        onChange={e => setFormData({ ...formData, rule_type: e.target.value })}
                        className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white focus:border-blue-500 focus:outline-none"
                      >
                        <option value="math">Math - Extract & verify numerical answers</option>
                        <option value="code">Code - Execute & check test cases</option>
                        <option value="json">JSON - Validate format correctness</option>
                        <option value="regex">Regex - Match against pattern</option>
                      </select>
                      <p className="mt-2 text-xs text-slate-500">
                        Dataset must include "metadata.expected_answer" or "metadata.test_cases" for verification.
                      </p>
                    </div>
                  )}
                </div>
                
                {/* Dataset Configuration */}
                <div className="border border-slate-700 rounded-lg p-4 bg-slate-800/30">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Database className="w-4 h-4 text-green-400" />
                      <h4 className="text-sm font-medium text-white">Dataset Configuration</h4>
                    </div>
                    <button
                      type="button"
                      onClick={() => setShowDatasetFormat(!showDatasetFormat)}
                      className="flex items-center gap-1 text-xs text-green-400 hover:text-green-300"
                    >
                      <BookOpen className="w-3 h-3" />
                      {showDatasetFormat ? 'Hide Format Info' : 'View Format Info'}
                    </button>
                  </div>
                  
                  {showDatasetFormat && (
                    <div className="mb-4 p-3 bg-slate-900/50 border border-slate-600 rounded-lg text-xs space-y-2">
                      <p className="text-slate-300 font-medium">Supported Dataset Formats:</p>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="p-2 bg-slate-800 rounded">
                          <p className="text-green-400 font-medium">Basic Prompts</p>
                          <code className="text-slate-400">{`{"prompt": "Question..."}`}</code>
                        </div>
                        <div className="p-2 bg-slate-800 rounded">
                          <p className="text-green-400 font-medium">With Difficulty</p>
                          <code className="text-slate-400">{`{"prompt": "...", "difficulty": "easy"}`}</code>
                        </div>
                        <div className="p-2 bg-slate-800 rounded">
                          <p className="text-green-400 font-medium">Verifiable (Math)</p>
                          <code className="text-slate-400">{`{"prompt": "...", "metadata": {"expected_answer": "42"}}`}</code>
                        </div>
                        <div className="p-2 bg-slate-800 rounded">
                          <p className="text-green-400 font-medium">Code with Tests</p>
                          <code className="text-slate-400">{`{"prompt": "...", "metadata": {"test_cases": [...]}}`}</code>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Prompts Dataset Path *</label>
                    <input
                      type="text"
                      value={formData.prompts_dataset_path}
                      onChange={e => setFormData({ ...formData, prompts_dataset_path: e.target.value })}
                      placeholder="/path/to/prompts.jsonl"
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white placeholder-slate-500 focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  
                  {/* Curriculum Learning Toggle */}
                  <div className="mt-3 flex items-center gap-3">
                    <input
                      type="checkbox"
                      id="curriculum"
                      checked={formData.enable_difficulty_curriculum}
                      onChange={e => setFormData({ ...formData, enable_difficulty_curriculum: e.target.checked })}
                      className="rounded border-slate-600 bg-slate-900 text-green-500 focus:ring-green-500"
                    />
                    <label htmlFor="curriculum" className="text-sm text-slate-300 flex items-center gap-2">
                      <GraduationCap className="w-4 h-4 text-green-400" />
                      Enable difficulty curriculum (easy → hard as rounds progress)
                    </label>
                  </div>
                  
                  {formData.enable_difficulty_curriculum && (
                    <div className="mt-3 grid grid-cols-2 gap-3">
                      <div>
                        <label className="block text-xs text-slate-400 mb-1">Selection Strategy</label>
                        <select
                          value={formData.dataset_selection_strategy}
                          onChange={e => setFormData({ ...formData, dataset_selection_strategy: e.target.value })}
                          className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white focus:border-blue-500 focus:outline-none"
                        >
                          <option value="sequential">Sequential</option>
                          <option value="difficulty_curriculum">Difficulty Curriculum</option>
                          <option value="round_robin">Round Robin</option>
                          <option value="random_weighted">Random Weighted</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-xs text-slate-400 mb-1">Samples per Round</label>
                        <input
                          type="number"
                          value={formData.samples_per_round}
                          onChange={e => setFormData({ ...formData, samples_per_round: parseInt(e.target.value) || 1000 })}
                          min={10}
                          className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white focus:border-blue-500 focus:outline-none"
                        />
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Generation Settings */}
                <div className="grid grid-cols-4 gap-4">
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Generations/Prompt</label>
                    <input
                      type="number"
                      value={formData.num_generations_per_prompt}
                      onChange={e => setFormData({ ...formData, num_generations_per_prompt: parseInt(e.target.value) || 8 })}
                      min={1}
                      max={64}
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Max Tokens</label>
                    <input
                      type="number"
                      value={formData.max_new_tokens}
                      onChange={e => setFormData({ ...formData, max_new_tokens: parseInt(e.target.value) || 512 })}
                      min={32}
                      max={4096}
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Temperature</label>
                    <input
                      type="number"
                      value={formData.temperature}
                      onChange={e => setFormData({ ...formData, temperature: parseFloat(e.target.value) || 0.7 })}
                      min={0}
                      max={2}
                      step={0.1}
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Top P</label>
                    <input
                      type="number"
                      value={formData.top_p}
                      onChange={e => setFormData({ ...formData, top_p: parseFloat(e.target.value) || 0.9 })}
                      min={0}
                      max={1}
                      step={0.05}
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                </div>
                
                {/* Filter Settings */}
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Filter Strategy</label>
                    <select
                      value={formData.filter_strategy}
                      onChange={e => setFormData({ ...formData, filter_strategy: e.target.value })}
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white focus:border-blue-500 focus:outline-none"
                    >
                      <option value="top_k_percent">Top K%</option>
                      <option value="threshold">Threshold</option>
                      <option value="top_n">Top N</option>
                      <option value="best_of_n">Best of N</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Top K %</label>
                    <input
                      type="number"
                      value={formData.filter_top_k_percent}
                      onChange={e => setFormData({ ...formData, filter_top_k_percent: parseFloat(e.target.value) || 20 })}
                      min={1}
                      max={100}
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Training Method</label>
                    <select
                      value={formData.training_method}
                      onChange={e => setFormData({ ...formData, training_method: e.target.value })}
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white focus:border-blue-500 focus:outline-none"
                    >
                      <option value="lora">LoRA</option>
                      <option value="qlora">QLoRA</option>
                      <option value="full">Full Fine-tune</option>
                    </select>
                  </div>
                </div>
                
                {/* Training Settings */}
                <div className="grid grid-cols-4 gap-4">
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Learning Rate</label>
                    <input
                      type="number"
                      value={formData.learning_rate}
                      onChange={e => setFormData({ ...formData, learning_rate: parseFloat(e.target.value) || 0.00001 })}
                      step={0.000001}
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Epochs/Round</label>
                    <input
                      type="number"
                      value={formData.num_train_epochs}
                      onChange={e => setFormData({ ...formData, num_train_epochs: parseInt(e.target.value) || 1 })}
                      min={1}
                      max={10}
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Batch Size</label>
                    <input
                      type="number"
                      value={formData.batch_size}
                      onChange={e => setFormData({ ...formData, batch_size: parseInt(e.target.value) || 4 })}
                      min={1}
                      max={64}
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Grad Accum</label>
                    <input
                      type="number"
                      value={formData.gradient_accumulation_steps}
                      onChange={e => setFormData({ ...formData, gradient_accumulation_steps: parseInt(e.target.value) || 4 })}
                      min={1}
                      max={64}
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-sm text-white focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                </div>
                
                {/* Checkbox */}
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="use_previous"
                    checked={formData.use_previous_round_model}
                    onChange={e => setFormData({ ...formData, use_previous_round_model: e.target.checked })}
                    className="rounded border-slate-600 bg-slate-900 text-blue-500 focus:ring-blue-500"
                  />
                  <label htmlFor="use_previous" className="text-sm text-slate-300">
                    Train each round from previous round's model (recommended)
                  </label>
                </div>
                
                {/* Submit */}
                <div className="flex justify-end gap-2 pt-2">
                  <button
                    onClick={() => setShowCreateForm(false)}
                    className="px-4 py-2 text-sm text-slate-300 hover:text-white transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleCreateJob}
                    disabled={isLoading}
                    className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-600 text-white text-sm rounded transition-colors"
                  >
                    {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
                    Create Job
                  </button>
                </div>
              </div>
            </div>
          )}
          
          {/* Jobs List */}
          {selectedJob ? (
            // Job Details View
            <div>
              <button
                onClick={() => setSelectedJob(null)}
                className="mb-4 text-sm text-slate-400 hover:text-white flex items-center gap-1"
              >
                ← Back to Jobs
              </button>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-medium text-white">{selectedJob.name}</h3>
                    <p className="text-xs text-slate-400">
                      {selectedJob.round_history.length} / {selectedJob.total_rounds} rounds completed
                    </p>
                  </div>
                  <span className={`text-sm font-medium px-3 py-1 rounded ${getStatusColor(selectedJob.status)} ${getStatusBgColor(selectedJob.status)}`}>
                    {selectedJob.status?.toUpperCase() || 'UNKNOWN'}
                  </span>
                </div>
                
                {/* Round History */}
                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-slate-300">Round History</h4>
                  {selectedJob.round_history.length === 0 ? (
                    <p className="text-sm text-slate-500 italic">No rounds completed yet</p>
                  ) : (
                    selectedJob.round_history.map((round, idx) => (
                      <RoundCard
                        key={idx}
                        round={round}
                        isExpanded={expandedRounds.has(idx)}
                        onToggle={() => toggleRoundExpanded(idx)}
                      />
                    ))
                  )}
                </div>
              </div>
            </div>
          ) : (
            // Jobs List View
            <div className="space-y-4">
              {isLoading && jobs.length === 0 ? (
                <div className="text-center py-8">
                  <Loader2 className="w-8 h-8 animate-spin text-slate-400 mx-auto mb-2" />
                  <p className="text-slate-400">Loading jobs...</p>
                </div>
              ) : jobs.length === 0 ? (
                <div className="text-center py-8">
                  <Repeat className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                  <p className="text-slate-400">No iterative training jobs yet</p>
                  <p className="text-sm text-slate-500">Create one to get started</p>
                </div>
              ) : (
                jobs.map(job => (
                  <JobCard
                    key={job.id}
                    job={job}
                    onStart={() => handleStartJob(job.id)}
                    onPause={() => handlePauseJob(job.id)}
                    onCancel={() => handleCancelJob(job.id)}
                    onDelete={() => handleDeleteJob(job.id)}
                    onViewDetails={() => setSelectedJob(job)}
                  />
                ))
              )}
            </div>
          )}
        </div>
        
        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-700 bg-slate-800/50">
          <p className="text-xs text-slate-500 text-center">
            VRAM-safe sequential execution • Full logging and history • Automatic checkpointing
          </p>
        </div>
      </div>
    </div>
  )
}
