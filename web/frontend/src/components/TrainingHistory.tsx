'use client'

import React, { useState, useEffect, useCallback, useMemo } from 'react'
import {
  History,
  X,
  Loader2,
  Clock,
  CheckCircle,
  XCircle,
  Layers,
  FolderOpen,
  Sparkles,
  MessageSquare,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  Filter,
  Search,
  Calendar,
  ArrowUpDown,
  AlertCircle,
  Play,
  Pause,
  RotateCcw,
  Eye
} from 'lucide-react'
import TrainingDetails from './TrainingDetails'

// ==================== Types ====================
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
  has_full_model?: boolean
  adapter_path?: string
  model_path?: string
  checkpoint_count: number
  final_metrics?: {
    loss: number | null
    learning_rate: number | null
    epoch: number | null
    step: number | null
  } | null
}

interface TrainingHistoryProps {
  isOpen: boolean
  onClose: () => void
  onLoadForInference: (modelPath: string, adapterPath?: string) => void
  modelPath: string
  isModelLoading: boolean
  isCleaningMemory: boolean
  loadingMessage: string
}

// ==================== Filter Types ====================
type StatusFilter = 'all' | 'completed' | 'failed' | 'running' | 'stopped'
type TrainingTypeFilter = 'all' | 'lora' | 'qlora' | 'full' | 'adalora'
type TrainingMethodFilter = 'all' | 'sft' | 'pt' | 'rlhf'
type SortField = 'date' | 'name' | 'status'
type SortOrder = 'asc' | 'desc'

// ==================== Constants ====================
const ITEMS_PER_PAGE_OPTIONS = [5, 10, 20, 50]

const STATUS_OPTIONS: { value: StatusFilter; label: string; color: string }[] = [
  { value: 'all', label: 'All Status', color: 'slate' },
  { value: 'completed', label: 'Completed', color: 'green' },
  { value: 'failed', label: 'Failed', color: 'red' },
  { value: 'running', label: 'Running', color: 'blue' },
  { value: 'stopped', label: 'Stopped', color: 'amber' }
]

const TRAINING_TYPE_OPTIONS: { value: TrainingTypeFilter; label: string; description: string }[] = [
  { value: 'all', label: 'All Types', description: 'Show all training types' },
  { value: 'lora', label: 'LoRA', description: 'Low-Rank Adaptation' },
  { value: 'qlora', label: 'QLoRA', description: 'Quantized LoRA' },
  { value: 'adalora', label: 'AdaLoRA', description: 'Adaptive LoRA' },
  { value: 'full', label: 'Full Fine-tune', description: 'Full parameter training' }
]

const TRAINING_METHOD_OPTIONS: { value: TrainingMethodFilter; label: string; description: string }[] = [
  { value: 'all', label: 'All Methods', description: 'Show all training methods' },
  { value: 'sft', label: 'SFT', description: 'Supervised Fine-Tuning' },
  { value: 'pt', label: 'Pre-Training', description: 'Continuous Pre-Training' },
  { value: 'rlhf', label: 'RLHF', description: 'Reinforcement Learning from Human Feedback' }
]

// ==================== Component ====================
export default function TrainingHistory({
  isOpen,
  onClose,
  onLoadForInference,
  modelPath,
  isModelLoading,
  isCleaningMemory,
  loadingMessage
}: TrainingHistoryProps) {
  // Data state
  const [history, setHistory] = useState<TrainingHistoryItem[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Filter state
  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')
  const [typeFilter, setTypeFilter] = useState<TrainingTypeFilter>('all')
  const [methodFilter, setMethodFilter] = useState<TrainingMethodFilter>('all')
  const [showFilters, setShowFilters] = useState(false)

  // Sort state
  const [sortField, setSortField] = useState<SortField>('date')
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc')

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1)
  const [itemsPerPage, setItemsPerPage] = useState(10)

  // Details view state
  const [selectedJob, setSelectedJob] = useState<TrainingHistoryItem | null>(null)

  // ==================== Data Fetching ====================
  const fetchHistory = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/jobs/history/all?limit=100&include_metrics=true')
      if (res.ok) {
        const data = await res.json()
        setHistory(data.history || [])
      } else {
        setError('Failed to fetch training history')
      }
    } catch (e) {
      console.error('Failed to fetch training history:', e)
      setError('Network error. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Fetch on open
  useEffect(() => {
    if (isOpen) {
      fetchHistory()
    }
  }, [isOpen, fetchHistory])

  // ==================== Filtering & Sorting ====================
  const filteredAndSortedHistory = useMemo(() => {
    let result = [...history]

    // Search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase()
      result = result.filter(item =>
        item.job_name.toLowerCase().includes(query) ||
        item.job_id.toLowerCase().includes(query) ||
        item.output_path?.toLowerCase().includes(query)
      )
    }

    // Status filter
    if (statusFilter !== 'all') {
      result = result.filter(item => item.status === statusFilter)
    }

    // Training type filter
    if (typeFilter !== 'all') {
      result = result.filter(item => {
        const trainType = item.config?.train_type?.toLowerCase() || ''
        return trainType === typeFilter
      })
    }

    // Training method filter
    if (methodFilter !== 'all') {
      result = result.filter(item => {
        const method = item.config?.training_method?.toLowerCase() || ''
        return method === methodFilter
      })
    }

    // Sorting
    result.sort((a, b) => {
      let comparison = 0
      switch (sortField) {
        case 'date':
          const dateA = a.created_at ? new Date(a.created_at).getTime() : 0
          const dateB = b.created_at ? new Date(b.created_at).getTime() : 0
          comparison = dateA - dateB
          break
        case 'name':
          comparison = a.job_name.localeCompare(b.job_name)
          break
        case 'status':
          comparison = a.status.localeCompare(b.status)
          break
      }
      return sortOrder === 'asc' ? comparison : -comparison
    })

    return result
  }, [history, searchQuery, statusFilter, typeFilter, methodFilter, sortField, sortOrder])

  // ==================== Pagination ====================
  const totalPages = Math.ceil(filteredAndSortedHistory.length / itemsPerPage)
  const paginatedHistory = useMemo(() => {
    const start = (currentPage - 1) * itemsPerPage
    return filteredAndSortedHistory.slice(start, start + itemsPerPage)
  }, [filteredAndSortedHistory, currentPage, itemsPerPage])

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1)
  }, [searchQuery, statusFilter, typeFilter, methodFilter, itemsPerPage])

  // ==================== Stats ====================
  const stats = useMemo(() => ({
    total: history.length,
    completed: history.filter(h => h.status === 'completed').length,
    failed: history.filter(h => h.status === 'failed').length,
    running: history.filter(h => h.status === 'running').length,
    filtered: filteredAndSortedHistory.length
  }), [history, filteredAndSortedHistory])

  // ==================== Helpers ====================
  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return '--'
    const date = new Date(dateStr)
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'failed': return <XCircle className="w-4 h-4 text-red-500" />
      case 'running': return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
      case 'stopped': return <Pause className="w-4 h-4 text-amber-500" />
      default: return <Clock className="w-4 h-4 text-slate-400" />
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

  const clearFilters = () => {
    setSearchQuery('')
    setStatusFilter('all')
    setTypeFilter('all')
    setMethodFilter('all')
    setSortField('date')
    setSortOrder('desc')
  }

  const hasActiveFilters = searchQuery || statusFilter !== 'all' || typeFilter !== 'all' || methodFilter !== 'all'

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-2 sm:p-4 lg:p-6">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-6xl max-h-[95vh] overflow-hidden flex flex-col">
        {/* ==================== Header ==================== */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between p-4 lg:p-6 border-b border-slate-200 bg-gradient-to-r from-blue-50 via-indigo-50 to-slate-50 gap-3">
          <div>
            <h2 className="text-xl lg:text-2xl font-bold text-slate-900 flex items-center gap-2">
              <History className="w-6 h-6 text-blue-600" />
              Training History
            </h2>
            <p className="text-sm text-slate-500 mt-1">
              {stats.total} total runs • {stats.completed} completed • {stats.failed} failed
              {stats.running > 0 && ` • ${stats.running} running`}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={fetchHistory}
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

        {/* ==================== Filters Bar ==================== */}
        <div className="px-4 lg:px-6 py-3 border-b border-slate-100 bg-slate-50/50">
          {/* Search and Filter Toggle */}
          <div className="flex flex-col sm:flex-row gap-3">
            {/* Search */}
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search by name, ID, or path..."
                className="w-full pl-10 pr-4 py-2.5 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {/* Filter Toggle & Actions */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={`px-4 py-2.5 rounded-lg border text-sm font-medium flex items-center gap-2 transition-colors ${
                  showFilters || hasActiveFilters
                    ? 'bg-blue-50 border-blue-200 text-blue-700'
                    : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                }`}
              >
                <Filter className="w-4 h-4" />
                <span>Filters</span>
                {hasActiveFilters && (
                  <span className="w-5 h-5 bg-blue-500 text-white text-xs rounded-full flex items-center justify-center">
                    {[statusFilter !== 'all', typeFilter !== 'all', methodFilter !== 'all', searchQuery].filter(Boolean).length}
                  </span>
                )}
              </button>

              {hasActiveFilters && (
                <button
                  onClick={clearFilters}
                  className="px-3 py-2.5 text-sm text-slate-500 hover:text-slate-700 flex items-center gap-1"
                >
                  <RotateCcw className="w-4 h-4" />
                  <span className="hidden sm:inline">Clear</span>
                </button>
              )}
            </div>
          </div>

          {/* Expanded Filters */}
          {showFilters && (
            <div className="mt-4 pt-4 border-t border-slate-200 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Status Filter */}
              <div>
                <label className="block text-xs font-medium text-slate-500 mb-1.5">Status</label>
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value as StatusFilter)}
                  className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
                >
                  {STATUS_OPTIONS.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>

              {/* Training Type Filter */}
              <div>
                <label className="block text-xs font-medium text-slate-500 mb-1.5">Training Type</label>
                <select
                  value={typeFilter}
                  onChange={(e) => setTypeFilter(e.target.value as TrainingTypeFilter)}
                  className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
                >
                  {TRAINING_TYPE_OPTIONS.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>

              {/* Training Method Filter */}
              <div>
                <label className="block text-xs font-medium text-slate-500 mb-1.5">Method</label>
                <select
                  value={methodFilter}
                  onChange={(e) => setMethodFilter(e.target.value as TrainingMethodFilter)}
                  className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
                >
                  {TRAINING_METHOD_OPTIONS.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>

              {/* Sort */}
              <div>
                <label className="block text-xs font-medium text-slate-500 mb-1.5">Sort By</label>
                <div className="flex gap-2">
                  <select
                    value={sortField}
                    onChange={(e) => setSortField(e.target.value as SortField)}
                    className="flex-1 px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
                  >
                    <option value="date">Date</option>
                    <option value="name">Name</option>
                    <option value="status">Status</option>
                  </select>
                  <button
                    onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                    className="px-3 py-2 border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors"
                    title={sortOrder === 'asc' ? 'Ascending' : 'Descending'}
                  >
                    <ArrowUpDown className={`w-4 h-4 text-slate-500 ${sortOrder === 'asc' ? 'rotate-180' : ''}`} />
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ==================== Content ==================== */}
        <div className="flex-1 overflow-y-auto p-4 lg:p-6">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center py-16">
              <Loader2 className="w-10 h-10 animate-spin text-blue-500 mb-4" />
              <p className="text-slate-500">Loading training history...</p>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center py-16 text-red-500">
              <AlertCircle className="w-12 h-12 mb-4 opacity-50" />
              <p className="font-medium">{error}</p>
              <button
                onClick={fetchHistory}
                className="mt-4 px-4 py-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors"
              >
                Try Again
              </button>
            </div>
          ) : filteredAndSortedHistory.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-slate-500">
              <History className="w-16 h-16 mb-4 opacity-20" />
              {hasActiveFilters ? (
                <>
                  <p className="font-medium text-lg">No matching training runs</p>
                  <p className="text-sm mt-1">Try adjusting your filters</p>
                  <button
                    onClick={clearFilters}
                    className="mt-4 px-4 py-2 bg-slate-100 text-slate-700 rounded-lg hover:bg-slate-200 transition-colors flex items-center gap-2"
                  >
                    <RotateCcw className="w-4 h-4" />
                    Clear Filters
                  </button>
                </>
              ) : (
                <>
                  <p className="font-medium text-lg">No training history found</p>
                  <p className="text-sm mt-1">Complete a training run to see it here</p>
                </>
              )}
            </div>
          ) : (
            <div className="space-y-4">
              {/* Results count */}
              {hasActiveFilters && (
                <p className="text-sm text-slate-500">
                  Showing {paginatedHistory.length} of {filteredAndSortedHistory.length} filtered results
                </p>
              )}

              {/* Training Items */}
              {paginatedHistory.map((item) => (
                <div
                  key={item.job_id}
                  className={`border rounded-xl overflow-hidden transition-all hover:shadow-lg ${
                    item.status === 'completed' ? 'border-green-200 bg-gradient-to-r from-green-50/50 to-white' :
                    item.status === 'failed' ? 'border-red-200 bg-gradient-to-r from-red-50/50 to-white' :
                    item.status === 'running' ? 'border-blue-200 bg-gradient-to-r from-blue-50/50 to-white' :
                    item.status === 'stopped' ? 'border-amber-200 bg-gradient-to-r from-amber-50/50 to-white' :
                    'border-slate-200 bg-gradient-to-r from-slate-50/50 to-white'
                  }`}
                >
                  {/* Main Info */}
                  <div className="p-4 lg:p-5">
                    <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-4">
                      {/* Left - Name & Details */}
                      <div className="flex-1 min-w-0">
                        {/* Name Row */}
                        <div className="flex flex-wrap items-center gap-2 mb-3">
                          {getStatusIcon(item.status)}
                          <span className="font-semibold text-slate-900 text-lg truncate">{item.job_name}</span>
                          <span className={`text-xs px-2.5 py-1 rounded-full font-medium border ${getStatusColor(item.status)}`}>
                            {item.status?.toUpperCase() || 'UNKNOWN'}
                          </span>
                          {item.config && (
                            <>
                              <span className="text-xs px-2.5 py-1 rounded-full bg-indigo-100 text-indigo-700 border border-indigo-200 font-medium">
                                {item.config.training_method?.toUpperCase() || 'N/A'}
                              </span>
                              <span className="text-xs px-2.5 py-1 rounded-full bg-purple-100 text-purple-700 border border-purple-200 font-medium">
                                {item.config.train_type?.toUpperCase() || 'N/A'}
                              </span>
                            </>
                          )}
                        </div>

                        {/* Date & Config Grid */}
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-6 gap-y-2 text-sm">
                          <div>
                            <span className="text-slate-400 text-xs">Created</span>
                            <p className="text-slate-700 font-medium">{formatDate(item.created_at)}</p>
                          </div>
                          {item.config && (
                            <>
                              <div>
                                <span className="text-slate-400 text-xs">Learning Rate</span>
                                <p className="text-slate-700 font-mono">{item.config.learning_rate}</p>
                              </div>
                              <div>
                                <span className="text-slate-400 text-xs">Batch Size</span>
                                <p className="text-slate-700 font-mono">{item.config.batch_size}</p>
                              </div>
                              <div>
                                <span className="text-slate-400 text-xs">Epochs</span>
                                <p className="text-slate-700 font-mono">{item.config.num_epochs}</p>
                              </div>
                            </>
                          )}
                        </div>
                      </div>

                      {/* Right - Badges */}
                      <div className="flex lg:flex-col items-center lg:items-end gap-2">
                        {item.has_adapter && (
                          <div className="flex items-center gap-1.5 text-green-600 bg-green-50 px-3 py-1.5 rounded-full border border-green-200 text-sm">
                            <CheckCircle className="w-4 h-4" />
                            <span className="font-medium">Adapter Ready</span>
                          </div>
                        )}
                        {item.checkpoint_count > 0 && (
                          <div className="flex items-center gap-1.5 text-blue-600 bg-blue-50 px-3 py-1.5 rounded-full border border-blue-200 text-sm">
                            <Layers className="w-4 h-4" />
                            <span>{item.checkpoint_count} checkpoints</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Metrics Row */}
                  {(item.final_metrics || item.output_exists) && (
                    <div className="px-4 lg:px-5 py-3 bg-slate-50/80 border-t border-slate-100 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                      {/* Final Metrics */}
                      {item.final_metrics && item.final_metrics.loss !== null && (
                        <div className="flex flex-wrap items-center gap-4 text-sm">
                          <div className="flex items-center gap-2">
                            <span className="text-slate-400">Final Loss:</span>
                            <span className="font-mono font-bold text-slate-800 bg-white px-2 py-0.5 rounded border border-slate-200">
                              {item.final_metrics.loss.toFixed(4)}
                            </span>
                          </div>
                          {item.final_metrics.epoch !== null && (
                            <div className="flex items-center gap-2">
                              <span className="text-slate-400">Epochs:</span>
                              <span className="font-mono font-semibold text-slate-700">{item.final_metrics.epoch}</span>
                            </div>
                          )}
                          {item.final_metrics.step !== null && (
                            <div className="flex items-center gap-2">
                              <span className="text-slate-400">Steps:</span>
                              <span className="font-mono font-semibold text-slate-700">{item.final_metrics.step}</span>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Output Path */}
                      {item.output_exists && (
                        <div className="flex items-center gap-2 text-sm">
                          <FolderOpen className="w-4 h-4 text-slate-400 flex-shrink-0" />
                          <code className="text-xs text-slate-500 truncate max-w-[300px] lg:max-w-[400px] bg-white px-2 py-1 rounded border border-slate-200" title={item.output_path}>
                            {item.output_path}
                          </code>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Adapter Path */}
                  {item.adapter_path && (
                    <div className="px-4 lg:px-5 py-3 bg-green-50/50 border-t border-green-100 flex items-center gap-2 text-sm">
                      <Sparkles className="w-4 h-4 text-green-500 flex-shrink-0" />
                      <span className="text-green-600 font-medium">Adapter:</span>
                      <code className="text-xs text-green-700 truncate flex-1 bg-white px-2 py-1 rounded border border-green-200" title={item.adapter_path}>
                        {item.adapter_path}
                      </code>
                    </div>
                  )}

                  {/* Error Message */}
                  {item.status === 'failed' && item.error && (
                    <div className="px-4 lg:px-5 py-3 bg-red-50/50 border-t border-red-100 flex items-start gap-2 text-sm">
                      <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
                      <span className="text-red-600">{item.error}</span>
                    </div>
                  )}

                  {/* Action Buttons */}
                  <div className="px-4 lg:px-5 py-4 bg-slate-50 border-t border-slate-200">
                    <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
                      {/* View Details Button - Always show */}
                      <button
                        onClick={() => setSelectedJob(item)}
                        className="px-4 py-2.5 bg-white text-slate-700 border border-slate-200 rounded-xl text-sm font-medium hover:bg-slate-100 flex items-center gap-2 transition-all"
                      >
                        <Eye className="w-4 h-4" />
                        View Details
                      </button>
                      
                      {/* Load for Inference - Only for completed with output */}
                      {item.status === 'completed' && (item.has_adapter || item.has_full_model || item.output_exists) && (
                        <>
                          <button
                            onClick={async () => {
                              // Resolve the correct adapter/model path via checkpoints API
                              try {
                                const ckptRes = await fetch(`/api/inference/checkpoints/${item.job_id}`)
                                const ckptData = await ckptRes.json()
                                
                                if (!ckptData.success || !ckptData.best_adapter_path) {
                                  // Fallback to raw paths from history data
                                  onClose()
                                  const isLoraType = item.config?.train_type?.toLowerCase().includes('lora')
                                  if (isLoraType && item.adapter_path) {
                                    onLoadForInference(modelPath, item.adapter_path)
                                  } else if (item.model_path) {
                                    onLoadForInference(item.model_path, undefined)
                                  } else if (item.output_exists) {
                                    onLoadForInference(item.output_path, undefined)
                                  }
                                  return
                                }
                                
                                const resolvedPath = ckptData.best_adapter_path
                                const finalCkpt = ckptData.checkpoints?.find((c: any) => c.is_final)
                                const ckptType = finalCkpt?.type || (item.config?.train_type?.toLowerCase().includes('lora') ? 'lora' : 'full')
                                
                                onClose()
                                if (ckptType === 'lora') {
                                  onLoadForInference(modelPath, resolvedPath)
                                } else {
                                  onLoadForInference(resolvedPath, undefined)
                                }
                              } catch (err) {
                                // Fallback on error
                                console.error('Failed to resolve checkpoint path:', err)
                                onClose()
                                const isLoraType = item.config?.train_type?.toLowerCase().includes('lora')
                                if (isLoraType && item.adapter_path) {
                                  onLoadForInference(modelPath, item.adapter_path)
                                } else if (item.output_exists) {
                                  onLoadForInference(item.output_path, undefined)
                                }
                              }
                            }}
                            disabled={isModelLoading || isCleaningMemory}
                            className="px-5 py-2.5 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl text-sm font-semibold hover:from-blue-600 hover:to-indigo-700 disabled:opacity-50 flex items-center gap-2 shadow-md transition-all hover:shadow-lg"
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
                          <div className="text-xs text-slate-500 hidden sm:block">
                            {item.config?.train_type?.toLowerCase().includes('lora') ? (
                              <span className="flex items-center gap-1">
                                <Sparkles className="w-3 h-3 text-purple-500" />
                                Base + {item.config?.train_type?.toUpperCase()}
                              </span>
                            ) : (
                              <span className="flex items-center gap-1">
                                <CheckCircle className="w-3 h-3 text-green-500" />
                                Fine-tuned Model
                              </span>
                            )}
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ==================== Pagination Footer ==================== */}
        {!isLoading && filteredAndSortedHistory.length > 0 && (
          <div className="px-4 lg:px-6 py-4 border-t border-slate-200 bg-slate-50 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            {/* Items per page */}
            <div className="flex items-center gap-3 text-sm">
              <span className="text-slate-500">Show:</span>
              <select
                value={itemsPerPage}
                onChange={(e) => setItemsPerPage(Number(e.target.value))}
                className="px-3 py-1.5 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
              >
                {ITEMS_PER_PAGE_OPTIONS.map(opt => (
                  <option key={opt} value={opt}>{opt} per page</option>
                ))}
              </select>
              <span className="text-slate-400 hidden sm:inline">
                {((currentPage - 1) * itemsPerPage) + 1}-{Math.min(currentPage * itemsPerPage, filteredAndSortedHistory.length)} of {filteredAndSortedHistory.length}
              </span>
            </div>

            {/* Pagination Controls */}
            {totalPages > 1 && (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setCurrentPage(1)}
                  disabled={currentPage === 1}
                  className="p-2 border border-slate-200 rounded-lg hover:bg-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  title="First page"
                >
                  <ChevronLeft className="w-4 h-4" />
                  <ChevronLeft className="w-4 h-4 -ml-2" />
                </button>
                <button
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="p-2 border border-slate-200 rounded-lg hover:bg-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  title="Previous page"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>

                {/* Page Numbers */}
                <div className="flex items-center gap-1">
                  {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                    let pageNum: number
                    if (totalPages <= 5) {
                      pageNum = i + 1
                    } else if (currentPage <= 3) {
                      pageNum = i + 1
                    } else if (currentPage >= totalPages - 2) {
                      pageNum = totalPages - 4 + i
                    } else {
                      pageNum = currentPage - 2 + i
                    }
                    return (
                      <button
                        key={pageNum}
                        onClick={() => setCurrentPage(pageNum)}
                        className={`w-9 h-9 rounded-lg text-sm font-medium transition-colors ${
                          currentPage === pageNum
                            ? 'bg-blue-500 text-white'
                            : 'hover:bg-white border border-slate-200 text-slate-600'
                        }`}
                      >
                        {pageNum}
                      </button>
                    )
                  })}
                </div>

                <button
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                  className="p-2 border border-slate-200 rounded-lg hover:bg-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  title="Next page"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setCurrentPage(totalPages)}
                  disabled={currentPage === totalPages}
                  className="p-2 border border-slate-200 rounded-lg hover:bg-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  title="Last page"
                >
                  <ChevronRight className="w-4 h-4" />
                  <ChevronRight className="w-4 h-4 -ml-2" />
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ==================== Training Details Modal ==================== */}
      {selectedJob && (
        <TrainingDetails
          jobId={selectedJob.job_id}
          jobName={selectedJob.job_name}
          isOpen={!!selectedJob}
          onClose={() => setSelectedJob(null)}
          onLoadForInference={onLoadForInference}
          modelPath={modelPath}
        />
      )}
    </div>
  )
}
