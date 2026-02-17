'use client'

import React, { useState, useEffect, useCallback, useMemo } from 'react'
import {
  X,
  Loader2,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  Layers,
  FolderOpen,
  Sparkles,
  MessageSquare,
  RefreshCw,
  ChevronLeft,
  Terminal,
  BarChart3,
  Settings,
  FileText,
  Pause,
  Play,
  TrendingDown,
  Activity,
  Zap,
  Info,
  Download,
  Copy,
  ExternalLink
} from 'lucide-react'

// ==================== Types ====================
interface TrainingMetric {
  step: number
  epoch?: number
  loss?: number
  learning_rate?: number
  grad_norm?: number
  reward?: number
  chosen_rewards?: number
  rejected_rewards?: number
  kl_divergence?: number
  policy_loss?: number
  value_loss?: number
  timestamp?: string
}

interface TrainingDetailsProps {
  jobId: string
  jobName: string
  isOpen: boolean
  onClose: () => void
  onLoadForInference?: (modelPath: string, adapterPath?: string) => void
  modelPath?: string
}

interface JobDetails {
  job_id: string
  name: string
  status: string
  config: any
  created_at: string | null
  started_at: string | null
  completed_at: string | null
  error: string | null
  current_step: number
  total_steps: number
  current_loss: number | null
  output_path?: string
  adapter_path?: string
  has_adapter?: boolean
}

interface GraphConfig {
  key: string
  label: string
  color: string
}

// ==================== Component ====================
export default function TrainingDetails({
  jobId,
  jobName,
  isOpen,
  onClose,
  onLoadForInference,
  modelPath
}: TrainingDetailsProps) {
  // State
  const [jobDetails, setJobDetails] = useState<JobDetails | null>(null)
  const [metrics, setMetrics] = useState<TrainingMetric[]>([])
  const [terminalLogs, setTerminalLogs] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<'overview' | 'graphs' | 'logs'>('overview')
  const [error, setError] = useState<string | null>(null)
  const [graphHover, setGraphHover] = useState<{ x: number; y: number; step: number; value: number } | null>(null)

  // ==================== Data Fetching ====================
  const fetchJobDetails = useCallback(async () => {
    try {
      const res = await fetch(`/api/jobs/debug/${jobId}`)
      if (res.ok) {
        const data = await res.json()
        setJobDetails(data)
      }
    } catch (e) {
      console.error('Failed to fetch job details:', e)
    }
  }, [jobId])

  const fetchMetrics = useCallback(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}/metrics?limit=1000`)
      if (res.ok) {
        const data = await res.json()
        setMetrics(data.metrics || [])
      }
    } catch (e) {
      console.error('Failed to fetch metrics:', e)
    }
  }, [jobId])

  const fetchLogs = useCallback(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}/terminal-logs?lines=500`)
      if (res.ok) {
        const data = await res.json()
        setTerminalLogs(data.logs || [])
      }
    } catch (e) {
      console.error('Failed to fetch logs:', e)
    }
  }, [jobId])

  const fetchAllData = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      await Promise.all([fetchJobDetails(), fetchMetrics(), fetchLogs()])
    } catch (e) {
      setError('Failed to load training details')
    } finally {
      setIsLoading(false)
    }
  }, [fetchJobDetails, fetchMetrics, fetchLogs])

  useEffect(() => {
    if (isOpen && jobId) {
      fetchAllData()
    }
  }, [isOpen, jobId, fetchAllData])

  // ==================== Computed Values ====================
  const graphConfigs = useMemo((): GraphConfig[] => {
    if (!metrics.length) return []
    
    const configs: GraphConfig[] = []
    const sample = metrics[0]
    
    // Check which metrics are available
    if (metrics.some(m => m.loss !== undefined && m.loss !== null)) {
      configs.push({ key: 'loss', label: 'Training Loss', color: '#3B82F6' })
    }
    if (metrics.some(m => m.learning_rate !== undefined && m.learning_rate !== null)) {
      configs.push({ key: 'learning_rate', label: 'Learning Rate', color: '#10B981' })
    }
    if (metrics.some(m => m.grad_norm !== undefined && m.grad_norm !== null)) {
      configs.push({ key: 'grad_norm', label: 'Gradient Norm', color: '#F59E0B' })
    }
    if (metrics.some(m => m.reward !== undefined && m.reward !== null)) {
      configs.push({ key: 'reward', label: 'Reward', color: '#8B5CF6' })
    }
    if (metrics.some(m => m.chosen_rewards !== undefined && m.chosen_rewards !== null)) {
      configs.push({ key: 'chosen_rewards', label: 'Chosen Rewards', color: '#10B981' })
    }
    if (metrics.some(m => m.rejected_rewards !== undefined && m.rejected_rewards !== null)) {
      configs.push({ key: 'rejected_rewards', label: 'Rejected Rewards', color: '#EF4444' })
    }
    if (metrics.some(m => m.kl_divergence !== undefined && m.kl_divergence !== null)) {
      configs.push({ key: 'kl_divergence', label: 'KL Divergence', color: '#F59E0B' })
    }
    if (metrics.some(m => m.policy_loss !== undefined && m.policy_loss !== null)) {
      configs.push({ key: 'policy_loss', label: 'Policy Loss', color: '#3B82F6' })
    }
    if (metrics.some(m => m.value_loss !== undefined && m.value_loss !== null)) {
      configs.push({ key: 'value_loss', label: 'Value Loss', color: '#EC4899' })
    }
    
    return configs
  }, [metrics])

  const finalMetrics = useMemo(() => {
    if (!metrics.length) return null
    const last = metrics[metrics.length - 1]
    return {
      finalLoss: last.loss,
      finalLR: last.learning_rate,
      totalSteps: last.step,
      finalEpoch: last.epoch
    }
  }, [metrics])

  const duration = useMemo(() => {
    if (!jobDetails?.started_at) return null
    const start = new Date(jobDetails.started_at)
    const end = jobDetails.completed_at ? new Date(jobDetails.completed_at) : new Date()
    const diff = end.getTime() - start.getTime()
    const hours = Math.floor(diff / 3600000)
    const minutes = Math.floor((diff % 3600000) / 60000)
    const seconds = Math.floor((diff % 60000) / 1000)
    if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`
    if (minutes > 0) return `${minutes}m ${seconds}s`
    return `${seconds}s`
  }, [jobDetails])

  // ==================== Helpers ====================
  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return '--'
    const date = new Date(dateStr)
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString()
  }

  const formatNumber = (num: number | null | undefined, decimals: number = 4) => {
    if (num === null || num === undefined) return '--'
    if (Math.abs(num) < 0.0001 && num !== 0) return num.toExponential(2)
    return num.toFixed(decimals)
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'failed': return <XCircle className="w-5 h-5 text-red-500" />
      case 'running': return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
      case 'stopped': return <Pause className="w-5 h-5 text-amber-500" />
      default: return <Clock className="w-5 h-5 text-slate-400" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-700 border-green-200'
      case 'failed': return 'bg-red-100 text-red-700 border-red-200'
      case 'running': return 'bg-blue-100 text-blue-700 border-blue-200'
      case 'stopped': return 'bg-amber-100 text-amber-700 border-amber-200'
      default: return 'bg-slate-100 text-slate-600 border-slate-200'
    }
  }

  // ==================== Graph Rendering ====================
  const renderGraph = (config: GraphConfig, isLarge: boolean = false) => {
    const data = metrics.filter(m => (m as any)[config.key] !== undefined && (m as any)[config.key] !== null)
    if (data.length < 2) {
      return (
        <div className={`${isLarge ? 'h-64' : 'h-48'} flex items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100 rounded-xl border border-slate-200`}>
          <div className="text-center text-slate-400">
            <BarChart3 className="w-10 h-10 mx-auto mb-3 opacity-40" />
            <p className="text-sm font-medium">No data for {config.label}</p>
            <p className="text-xs mt-1 opacity-70">Data will appear during training</p>
          </div>
        </div>
      )
    }

    const values = data.map(m => (m as any)[config.key] as number)
    const steps = data.map(m => m.step)
    const minVal = Math.min(...values)
    const maxVal = Math.max(...values)
    const currentVal = values[values.length - 1]
    const range = maxVal - minVal || 1

    const width = 100
    const height = 50
    const padding = 3

    const points = data.map((m, i) => {
      const x = padding + ((steps[i] - steps[0]) / (steps[steps.length - 1] - steps[0] || 1)) * (width - padding * 2)
      const y = height - padding - ((values[i] - minVal) / range) * (height - padding * 2)
      return `${x},${y}`
    }).join(' ')

    return (
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        {/* Header */}
        <div className="px-4 py-3 bg-gradient-to-r from-slate-50 to-white border-b border-slate-100">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: config.color }} />
              <h4 className="text-sm font-semibold text-slate-800">{config.label}</h4>
            </div>
            <div className="text-right">
              <p className="text-lg font-bold" style={{ color: config.color }}>
                {formatNumber(currentVal)}
              </p>
              <p className="text-[10px] text-slate-400 uppercase tracking-wide">Current</p>
            </div>
          </div>
        </div>
        
        {/* Graph */}
        <div className="p-4 relative">
          <svg 
            viewBox={`0 0 ${width} ${height}`} 
            className={`w-full ${isLarge ? 'h-52' : 'h-40'}`}
            preserveAspectRatio="none"
            onMouseMove={(e) => {
              const rect = e.currentTarget.getBoundingClientRect()
              const x = ((e.clientX - rect.left) / rect.width) * width
              const idx = Math.round(((x - padding) / (width - padding * 2)) * (data.length - 1))
              if (idx >= 0 && idx < data.length) {
                setGraphHover({
                  x: e.clientX - rect.left,
                  y: e.clientY - rect.top,
                  step: data[idx].step,
                  value: values[idx]
                })
              }
            }}
            onMouseLeave={() => setGraphHover(null)}
          >
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map(pct => (
              <line
                key={pct}
                x1={padding}
                y1={padding + pct * (height - padding * 2)}
                x2={width - padding}
                y2={padding + pct * (height - padding * 2)}
                stroke="#f1f5f9"
                strokeWidth="0.3"
              />
            ))}
            
            {/* Area fill gradient */}
            <defs>
              <linearGradient id={`gradient-${config.key}`} x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor={config.color} stopOpacity="0.3" />
                <stop offset="100%" stopColor={config.color} stopOpacity="0.02" />
              </linearGradient>
            </defs>
            
            {/* Area fill */}
            <polygon
              points={`${padding},${height - padding} ${points} ${width - padding},${height - padding}`}
              fill={`url(#gradient-${config.key})`}
            />
            
            {/* Line */}
            <polyline
              points={points}
              fill="none"
              stroke={config.color}
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            
            {/* End dot */}
            {data.length > 0 && (
              <circle
                cx={width - padding}
                cy={height - padding - ((currentVal - minVal) / range) * (height - padding * 2)}
                r="3"
                fill={config.color}
                stroke="white"
                strokeWidth="1.5"
              />
            )}
          </svg>
          
          {/* Hover tooltip */}
          {graphHover && (
            <div 
              className="absolute bg-slate-900 text-white text-xs px-3 py-2 rounded-lg shadow-xl pointer-events-none z-10 whitespace-nowrap"
              style={{ left: Math.min(graphHover.x + 10, 200), top: Math.max(graphHover.y - 40, 10) }}
            >
              <div className="font-semibold">Step {graphHover.step}</div>
              <div className="text-slate-300">{formatNumber(graphHover.value)}</div>
            </div>
          )}
        </div>
        
        {/* Footer Stats */}
        <div className="px-4 py-3 bg-slate-50 border-t border-slate-100 grid grid-cols-3 gap-4">
          <div className="text-center">
            <p className="text-xs text-slate-400 mb-0.5">Min</p>
            <p className="text-sm font-semibold text-slate-700">{formatNumber(minVal)}</p>
          </div>
          <div className="text-center border-x border-slate-200">
            <p className="text-xs text-slate-400 mb-0.5">Max</p>
            <p className="text-sm font-semibold text-slate-700">{formatNumber(maxVal)}</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-slate-400 mb-0.5">Points</p>
            <p className="text-sm font-semibold text-slate-700">{data.length}</p>
          </div>
        </div>
      </div>
    )
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-2 sm:p-4 lg:p-6">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-6xl max-h-[95vh] overflow-hidden flex flex-col">
        {/* ==================== Header ==================== */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between p-4 lg:p-6 border-b border-slate-200 bg-gradient-to-r from-slate-50 to-blue-50 gap-3">
          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              className="p-2 hover:bg-white rounded-lg transition-colors"
            >
              <ChevronLeft className="w-5 h-5 text-slate-500" />
            </button>
            <div>
              <h2 className="text-xl lg:text-2xl font-bold text-slate-900 flex items-center gap-2">
                {jobName}
              </h2>
              <p className="text-sm text-slate-500 flex items-center gap-2 mt-1">
                <code className="bg-slate-200 px-2 py-0.5 rounded text-xs">{jobId}</code>
                {jobDetails && (
                  <span className={`text-xs px-2 py-0.5 rounded-full border ${getStatusColor(jobDetails.status)}`}>
                    {jobDetails.status?.toUpperCase() || 'UNKNOWN'}
                  </span>
                )}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={fetchAllData}
              disabled={isLoading}
              className="p-2 lg:px-4 lg:py-2 bg-white text-slate-600 rounded-lg hover:bg-slate-100 border border-slate-200 transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              <span className="hidden lg:inline">Refresh</span>
            </button>
            <button
              onClick={onClose}
              className="p-2 hover:bg-white rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-slate-500" />
            </button>
          </div>
        </div>

        {/* ==================== Tabs ==================== */}
        <div className="flex border-b border-slate-200 bg-slate-50/50 px-4 lg:px-6">
          {[
            { id: 'overview', label: 'Overview', icon: Settings },
            { id: 'graphs', label: 'Graphs', icon: BarChart3 },
            { id: 'logs', label: 'Logs', icon: Terminal }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600 bg-white'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* ==================== Content ==================== */}
        <div className="flex-1 overflow-y-auto p-4 lg:p-6">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center py-16">
              <Loader2 className="w-10 h-10 animate-spin text-blue-500 mb-4" />
              <p className="text-slate-500">Loading training details...</p>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center py-16 text-red-500">
              <AlertCircle className="w-12 h-12 mb-4 opacity-50" />
              <p className="font-medium">{error}</p>
              <button
                onClick={fetchAllData}
                className="mt-4 px-4 py-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors"
              >
                Try Again
              </button>
            </div>
          ) : (
            <>
              {/* ==================== Overview Tab ==================== */}
              {activeTab === 'overview' && (
                <div className="space-y-6">
                  {/* Status Banner */}
                  {jobDetails?.status === 'failed' && jobDetails.error && (
                    <div className="bg-red-50 border border-red-200 rounded-xl p-4">
                      <div className="flex items-start gap-3">
                        <XCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                        <div>
                          <h4 className="font-medium text-red-800">Training Failed</h4>
                          <p className="text-sm text-red-600 mt-1">{jobDetails.error}</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Metrics Summary */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <div className="bg-white rounded-xl border border-slate-200 p-4">
                      <div className="flex items-center gap-2 text-slate-500 text-sm mb-1">
                        <TrendingDown className="w-4 h-4" />
                        Final Loss
                      </div>
                      <p className="text-2xl font-bold text-slate-900">
                        {finalMetrics?.finalLoss !== undefined ? formatNumber(finalMetrics.finalLoss) : '--'}
                      </p>
                    </div>
                    <div className="bg-white rounded-xl border border-slate-200 p-4">
                      <div className="flex items-center gap-2 text-slate-500 text-sm mb-1">
                        <Activity className="w-4 h-4" />
                        Total Steps
                      </div>
                      <p className="text-2xl font-bold text-slate-900">
                        {jobDetails?.current_step || finalMetrics?.totalSteps || '--'}
                      </p>
                    </div>
                    <div className="bg-white rounded-xl border border-slate-200 p-4">
                      <div className="flex items-center gap-2 text-slate-500 text-sm mb-1">
                        <Clock className="w-4 h-4" />
                        Duration
                      </div>
                      <p className="text-2xl font-bold text-slate-900">{duration || '--'}</p>
                    </div>
                    <div className="bg-white rounded-xl border border-slate-200 p-4">
                      <div className="flex items-center gap-2 text-slate-500 text-sm mb-1">
                        <Zap className="w-4 h-4" />
                        Final LR
                      </div>
                      <p className="text-2xl font-bold text-slate-900">
                        {finalMetrics?.finalLR !== undefined ? finalMetrics.finalLR.toExponential(1) : '--'}
                      </p>
                    </div>
                  </div>

                  {/* Configuration Details */}
                  {jobDetails?.config && (
                    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
                      <div className="px-4 py-3 bg-slate-50 border-b border-slate-200">
                        <h3 className="font-semibold text-slate-900 flex items-center gap-2">
                          <Settings className="w-4 h-4" />
                          Training Configuration
                        </h3>
                      </div>
                      <div className="p-4 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
                        {[
                          { label: 'Training Method', value: jobDetails.config.training_method?.toUpperCase() },
                          { label: 'Training Type', value: jobDetails.config.train_type?.toUpperCase() },
                          { label: 'Epochs', value: jobDetails.config.num_train_epochs },
                          { label: 'Learning Rate', value: jobDetails.config.learning_rate },
                          { label: 'Batch Size', value: jobDetails.config.per_device_train_batch_size },
                          { label: 'Max Length', value: jobDetails.config.max_length },
                          { label: 'Grad Accum', value: jobDetails.config.gradient_accumulation_steps },
                          { label: 'Warmup Ratio', value: jobDetails.config.warmup_ratio },
                        ].filter(item => item.value !== undefined && item.value !== null).map(item => (
                          <div key={item.label}>
                            <span className="text-xs text-slate-400 block">{item.label}</span>
                            <span className="text-sm font-medium text-slate-900">{item.value}</span>
                          </div>
                        ))}
                        
                        {/* LoRA params if applicable */}
                        {['lora', 'qlora', 'adalora'].includes(jobDetails.config.train_type?.toLowerCase()) && (
                          <>
                            {jobDetails.config.lora_rank && (
                              <div>
                                <span className="text-xs text-slate-400 block">LoRA Rank</span>
                                <span className="text-sm font-medium text-slate-900">{jobDetails.config.lora_rank}</span>
                              </div>
                            )}
                            {jobDetails.config.lora_alpha && (
                              <div>
                                <span className="text-xs text-slate-400 block">LoRA Alpha</span>
                                <span className="text-sm font-medium text-slate-900">{jobDetails.config.lora_alpha}</span>
                              </div>
                            )}
                          </>
                        )}
                      </div>
                      
                      {/* RLHF Configuration - show if training_method is rlhf */}
                      {jobDetails.config.training_method === 'rlhf' && (
                        <div className="px-4 pb-4 pt-2 border-t border-slate-100">
                          <h4 className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-3">RLHF Configuration</h4>
                          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
                            {jobDetails.config.rlhf_type && (
                              <div>
                                <span className="text-xs text-slate-400 block">Algorithm</span>
                                <span className="text-sm font-medium text-slate-900">{jobDetails.config.rlhf_type.toUpperCase()}</span>
                              </div>
                            )}
                            {jobDetails.config.beta !== undefined && jobDetails.config.beta !== null && (
                              <div>
                                <span className="text-xs text-slate-400 block">Beta</span>
                                <span className="text-sm font-medium text-slate-900">{jobDetails.config.beta}</span>
                              </div>
                            )}
                            {jobDetails.config.num_generations && (
                              <div>
                                <span className="text-xs text-slate-400 block">Num Generations</span>
                                <span className="text-sm font-medium text-slate-900">{jobDetails.config.num_generations}</span>
                              </div>
                            )}
                            {jobDetails.config.max_completion_length && (
                              <div>
                                <span className="text-xs text-slate-400 block">Max Completion</span>
                                <span className="text-sm font-medium text-slate-900">{jobDetails.config.max_completion_length}</span>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                      
                      {/* vLLM Configuration - show if online RL with vLLM */}
                      {jobDetails.config.training_method === 'rlhf' && 
                       ['grpo', 'ppo', 'gkd'].includes(jobDetails.config.rlhf_type?.toLowerCase()) && 
                       jobDetails.config.use_vllm && (
                        <div className="px-4 pb-4 pt-2 border-t border-slate-100">
                          <h4 className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-3">vLLM Configuration</h4>
                          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
                            <div>
                              <span className="text-xs text-slate-400 block">Mode</span>
                              <span className="text-sm font-medium text-slate-900 capitalize">{jobDetails.config.vllm_mode || 'Not Set'}</span>
                            </div>
                            {jobDetails.config.vllm_mode === 'server' && (
                              <>
                                <div>
                                  <span className="text-xs text-slate-400 block">Server Host</span>
                                  <span className="text-sm font-medium text-slate-900">{jobDetails.config.vllm_server_host || '--'}</span>
                                </div>
                                <div>
                                  <span className="text-xs text-slate-400 block">Server Port</span>
                                  <span className="text-sm font-medium text-slate-900">{jobDetails.config.vllm_server_port || 8000}</span>
                                </div>
                                <div>
                                  <span className="text-xs text-slate-400 block">Verified</span>
                                  <span className={`text-sm font-medium ${jobDetails.config.vllm_server_verified ? 'text-green-600' : 'text-red-600'}`}>
                                    {jobDetails.config.vllm_server_verified ? 'Yes' : 'No'}
                                  </span>
                                </div>
                              </>
                            )}
                            {jobDetails.config.vllm_mode === 'colocate' && (
                              <>
                                <div>
                                  <span className="text-xs text-slate-400 block">Tensor Parallel</span>
                                  <span className="text-sm font-medium text-slate-900">{jobDetails.config.vllm_tensor_parallel_size || 1}</span>
                                </div>
                                <div>
                                  <span className="text-xs text-slate-400 block">GPU Memory</span>
                                  <span className="text-sm font-medium text-slate-900">{((jobDetails.config.vllm_gpu_memory_utilization || 0.9) * 100).toFixed(0)}%</span>
                                </div>
                                <div>
                                  <span className="text-xs text-slate-400 block">Sleep Level</span>
                                  <span className="text-sm font-medium text-slate-900">{jobDetails.config.sleep_level || 0}</span>
                                </div>
                                {jobDetails.config.offload_model && (
                                  <div>
                                    <span className="text-xs text-slate-400 block">Offload Model</span>
                                    <span className="text-sm font-medium text-green-600">Enabled</span>
                                  </div>
                                )}
                                {jobDetails.config.offload_optimizer && (
                                  <div>
                                    <span className="text-xs text-slate-400 block">Offload Optimizer</span>
                                    <span className="text-sm font-medium text-green-600">Enabled</span>
                                  </div>
                                )}
                              </>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Timestamps */}
                  <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
                    <div className="px-4 py-3 bg-slate-50 border-b border-slate-200">
                      <h3 className="font-semibold text-slate-900 flex items-center gap-2">
                        <Clock className="w-4 h-4" />
                        Timeline
                      </h3>
                    </div>
                    <div className="p-4 grid grid-cols-1 sm:grid-cols-3 gap-4">
                      <div>
                        <span className="text-xs text-slate-400 block">Created</span>
                        <span className="text-sm font-medium text-slate-900">{formatDate(jobDetails?.created_at || null)}</span>
                      </div>
                      <div>
                        <span className="text-xs text-slate-400 block">Started</span>
                        <span className="text-sm font-medium text-slate-900">{formatDate(jobDetails?.started_at || null)}</span>
                      </div>
                      <div>
                        <span className="text-xs text-slate-400 block">Completed</span>
                        <span className="text-sm font-medium text-slate-900">{formatDate(jobDetails?.completed_at || null)}</span>
                      </div>
                    </div>
                  </div>

                  {/* Output Path */}
                  {jobDetails?.config?.output_dir && (
                    <div className="bg-white rounded-xl border border-slate-200 p-4">
                      <div className="flex items-center gap-2 text-slate-500 text-sm mb-2">
                        <FolderOpen className="w-4 h-4" />
                        Output Directory
                      </div>
                      <code className="text-sm bg-slate-100 px-3 py-2 rounded-lg block overflow-x-auto">
                        {jobDetails.config.output_dir}
                      </code>
                    </div>
                  )}

                  {/* Load for Inference Button */}
                  {jobDetails?.status === 'completed' && onLoadForInference && modelPath && (
                    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-200 p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="font-medium text-blue-900">Ready for Inference</h4>
                          <p className="text-sm text-blue-600 mt-1">
                            {jobDetails.config?.train_type?.toLowerCase().includes('lora')
                              ? 'Load base model with trained adapter'
                              : 'Load fine-tuned model'}
                          </p>
                        </div>
                        <button
                          onClick={async () => {
                            // Resolve correct adapter/model path via checkpoints API
                            try {
                              const ckptRes = await fetch(`/api/inference/checkpoints/${jobId}`)
                              const ckptData = await ckptRes.json()
                              
                              if (ckptData.success && ckptData.best_adapter_path) {
                                const resolvedPath = ckptData.best_adapter_path
                                const finalCkpt = ckptData.checkpoints?.find((c: any) => c.is_final)
                                const ckptType = finalCkpt?.type || (jobDetails.config?.train_type?.toLowerCase().includes('lora') ? 'lora' : 'full')
                                
                                onClose()
                                if (ckptType === 'lora') {
                                  onLoadForInference(modelPath, resolvedPath)
                                } else {
                                  onLoadForInference(resolvedPath, undefined)
                                }
                              } else {
                                // Fallback to config output_dir
                                onClose()
                                const isLoraType = jobDetails.config?.train_type?.toLowerCase().includes('lora')
                                if (isLoraType) {
                                  onLoadForInference(modelPath, jobDetails.config?.output_dir)
                                } else {
                                  onLoadForInference(jobDetails.config?.output_dir, undefined)
                                }
                              }
                            } catch (err) {
                              console.error('Failed to resolve checkpoint path:', err)
                              onClose()
                              const isLoraType = jobDetails.config?.train_type?.toLowerCase().includes('lora')
                              if (isLoraType) {
                                onLoadForInference(modelPath, jobDetails.config?.output_dir)
                              } else {
                                onLoadForInference(jobDetails.config?.output_dir, undefined)
                              }
                            }
                          }}
                          className="px-6 py-3 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl text-sm font-semibold hover:from-blue-600 hover:to-indigo-700 flex items-center gap-2 shadow-lg"
                        >
                          <MessageSquare className="w-4 h-4" />
                          Load for Inference
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* ==================== Graphs Tab ==================== */}
              {activeTab === 'graphs' && (
                <div className="space-y-6">
                  {/* Data Source Info */}
                  <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 flex items-start gap-3">
                    <Info className="w-5 h-5 text-blue-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <h4 className="font-medium text-blue-900 text-sm">Metrics Source</h4>
                      <p className="text-xs text-blue-700 mt-1">
                        Data is parsed in real-time from training output logs. Each point represents a logged training step with metrics like loss, learning rate, and gradient norm.
                      </p>
                    </div>
                  </div>

                  {graphConfigs.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-16 text-slate-500 bg-slate-50 rounded-xl border border-slate-200">
                      <BarChart3 className="w-16 h-16 mb-4 opacity-20" />
                      <p className="font-medium text-lg">No metrics data available</p>
                      <p className="text-sm mt-1 text-slate-400">Metrics will appear here once training starts logging</p>
                    </div>
                  ) : (
                    <>
                      {/* Metrics Summary Row */}
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                        <div className="bg-white rounded-lg border border-slate-200 p-3 text-center">
                          <p className="text-xs text-slate-400 uppercase tracking-wide">Data Points</p>
                          <p className="text-xl font-bold text-slate-800">{metrics.length}</p>
                        </div>
                        <div className="bg-white rounded-lg border border-slate-200 p-3 text-center">
                          <p className="text-xs text-slate-400 uppercase tracking-wide">Metrics</p>
                          <p className="text-xl font-bold text-slate-800">{graphConfigs.length}</p>
                        </div>
                        <div className="bg-white rounded-lg border border-slate-200 p-3 text-center">
                          <p className="text-xs text-slate-400 uppercase tracking-wide">First Step</p>
                          <p className="text-xl font-bold text-slate-800">{metrics[0]?.step || '--'}</p>
                        </div>
                        <div className="bg-white rounded-lg border border-slate-200 p-3 text-center">
                          <p className="text-xs text-slate-400 uppercase tracking-wide">Last Step</p>
                          <p className="text-xl font-bold text-slate-800">{metrics[metrics.length - 1]?.step || '--'}</p>
                        </div>
                      </div>

                      {/* Graphs Grid - Responsive */}
                      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                        {graphConfigs.map((config, idx) => (
                          <div key={config.key} className={graphConfigs.length === 1 ? 'xl:col-span-2' : ''}>
                            {renderGraph(config, graphConfigs.length <= 2)}
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}

              {/* ==================== Logs Tab ==================== */}
              {activeTab === 'logs' && (
                <div className="space-y-4">
                  {terminalLogs.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-16 text-slate-500 bg-slate-50 rounded-xl border border-slate-200">
                      <Terminal className="w-16 h-16 mb-4 opacity-20" />
                      <p className="font-medium text-lg">No logs available</p>
                      <p className="text-sm mt-1 text-slate-400">Training logs will appear here</p>
                    </div>
                  ) : (
                    <div className="bg-slate-900 rounded-xl overflow-hidden shadow-xl">
                      {/* Logs Header */}
                      <div className="px-4 py-3 bg-slate-800 border-b border-slate-700 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                        <div className="flex items-center gap-3">
                          <div className="flex items-center gap-1.5">
                            <div className="w-3 h-3 rounded-full bg-red-500" />
                            <div className="w-3 h-3 rounded-full bg-yellow-500" />
                            <div className="w-3 h-3 rounded-full bg-green-500" />
                          </div>
                          <span className="text-sm text-slate-300 font-medium flex items-center gap-2">
                            <Terminal className="w-4 h-4" />
                            Terminal Output
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-slate-500 bg-slate-700 px-2 py-1 rounded">
                            {terminalLogs.length} lines
                          </span>
                          <button
                            onClick={() => {
                              navigator.clipboard.writeText(terminalLogs.join('\n'))
                            }}
                            className="text-xs text-slate-400 hover:text-white bg-slate-700 hover:bg-slate-600 px-2 py-1 rounded flex items-center gap-1 transition-colors"
                          >
                            <Copy className="w-3 h-3" />
                            Copy
                          </button>
                        </div>
                      </div>
                      
                      {/* Logs Content */}
                      <div className="p-4 max-h-[60vh] overflow-y-auto font-mono text-xs leading-relaxed">
                        {terminalLogs.map((line, i) => (
                          <div 
                            key={i} 
                            className={`whitespace-pre-wrap break-all px-2 py-1 rounded transition-colors ${
                              line.includes('ERROR') || line.includes('error') 
                                ? 'text-red-400 bg-red-900/20' 
                                : line.includes('WARNING') || line.includes('warning')
                                ? 'text-yellow-400 bg-yellow-900/10'
                                : line.includes('INFO') || line.includes('[INFO]')
                                ? 'text-blue-400'
                                : line.includes('loss') || line.includes('Loss')
                                ? 'text-green-400'
                                : 'text-slate-300 hover:bg-slate-800/50'
                            }`}
                          >
                            <span className="text-slate-600 select-none mr-4 inline-block w-8 text-right">
                              {i + 1}
                            </span>
                            {line}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
