'use client'

import { useState, useRef, useEffect } from 'react'
import { 
  Upload, X, FileText, Check, Trash2, RefreshCw, 
  Database, Cloud, FolderOpen, Loader2, AlertCircle,
  Plus, ExternalLink, Info, Download, ChevronDown, ChevronUp, BookOpen
} from 'lucide-react'
import SampleDatasetsViewer from './SampleDatasetsViewer'

type DatasetSource = 'upload' | 'huggingface' | 'modelscope' | 'local_path'

interface Dataset {
  id: string
  name: string
  source: DatasetSource
  path: string
  subset?: string | null
  split?: string | null
  total_samples: number
  size_human: string
  format: string
  created_at: number
  selected: boolean
  max_samples?: number | null // null or 0 = all
  dataset_type?: string | null // sft, rlhf_offline, rlhf_online, pt, kto, unknown
  dataset_type_display?: string | null // Human-readable type name
  compatible_training_methods?: string[] | null
}

// Dataset type detection result
interface DatasetTypeInfo {
  dataset_type: string
  confidence: number
  detected_fields: string[]
  sample_count: number
  compatible_training_methods: string[]
  incompatible_training_methods: string[]
  compatible_rlhf_types: string[]
  display_name: string
  message: string
  file_size_bytes?: number
  is_large_file?: boolean
  format_warning?: string | null
}

// Dataset type display configuration
const DATASET_TYPE_CONFIG: Record<string, { label: string; color: string; bgColor: string; borderColor: string }> = {
  sft: { label: 'SFT', color: 'text-blue-700', bgColor: 'bg-blue-100', borderColor: 'border-blue-200' },
  rlhf_offline: { label: 'RLHF Offline', color: 'text-purple-700', bgColor: 'bg-purple-100', borderColor: 'border-purple-200' },
  rlhf_online: { label: 'RLHF Online', color: 'text-orange-700', bgColor: 'bg-orange-100', borderColor: 'border-orange-200' },
  pt: { label: 'Pre-Training', color: 'text-green-700', bgColor: 'bg-green-100', borderColor: 'border-green-200' },
  kto: { label: 'KTO', color: 'text-amber-700', bgColor: 'bg-amber-100', borderColor: 'border-amber-200' },
  unknown: { label: 'Unknown', color: 'text-slate-600', bgColor: 'bg-slate-100', borderColor: 'border-slate-200' },
}

interface DatasetInfo {
  subsets: string[]
  splits: { [subset: string]: { [split: string]: number } }
  isPrivate: boolean
  error?: string
}

interface SystemCapabilities {
  supported_dataset_sources: string[]
  supported_model_sources: string[]
}

interface Props {
  selectedPaths: string[]
  onSelectionChange: (paths: string[]) => void
  onShowAlert?: (message: string, type: 'error' | 'warning' | 'info' | 'success', title?: string) => void
  onDatasetTypeChange?: (typeInfo: DatasetTypeInfo | null) => void
}

const ALL_SOURCE_TABS: { id: DatasetSource; label: string; icon: any; desc: string; sourceKey: string }[] = [
  { id: 'upload', label: 'Upload', icon: Upload, desc: 'Upload from your computer', sourceKey: 'local' },
  { id: 'huggingface', label: 'HuggingFace', icon: Cloud, desc: 'Use HuggingFace datasets', sourceKey: 'huggingface' },
  { id: 'modelscope', label: 'ModelScope', icon: Cloud, desc: 'Use ModelScope datasets', sourceKey: 'modelscope' },
  { id: 'local_path', label: 'Local Path', icon: FolderOpen, desc: 'Use server local path', sourceKey: 'local' },
]


export default function DatasetConfig({ selectedPaths, onSelectionChange, onShowAlert, onDatasetTypeChange }: Props) {
  const [activeTab, setActiveTab] = useState<DatasetSource>('upload')
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [isLoading, setIsLoading] = useState(false)
  
  // System capabilities - what dataset sources are available
  const [capabilities, setCapabilities] = useState<SystemCapabilities>({
    supported_dataset_sources: ['local'],
    supported_model_sources: ['local']
  })
  
  // Upload state
  const [uploadName, setUploadName] = useState('')
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'processing' | 'success' | 'error'>('idle')
  const [uploadResult, setUploadResult] = useState<{ samples: number; size: string } | null>(null)
  const [uploadError, setUploadError] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  // Register state (for HF/MS/Local)
  const [registerName, setRegisterName] = useState('')
  const [registerPath, setRegisterPath] = useState('')
  const [registerSubset, setRegisterSubset] = useState('')
  const [registerSplit, setRegisterSplit] = useState('train')
  const [registerMaxSamples, setRegisterMaxSamples] = useState<string>('')
  const [isRegistering, setIsRegistering] = useState(false)
  const [registerError, setRegisterError] = useState('')
  
  // Dataset info from API
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null)
  const [isFetchingInfo, setIsFetchingInfo] = useState(false)
  const [infoFetched, setInfoFetched] = useState(false)
  
  // Delete confirmation
  const [deleteTarget, setDeleteTarget] = useState<Dataset | null>(null)
  const [deleteConfirm, setDeleteConfirm] = useState('')
  const [isDeleting, setIsDeleting] = useState(false)
  
  // Dataset type detection
  const [detectedTypeInfo, setDetectedTypeInfo] = useState<DatasetTypeInfo | null>(null)
  const [isDetectingType, setIsDetectingType] = useState(false)
  
  // Dataset type filter
  const [typeFilter, setTypeFilter] = useState<string>('all')
  
  // Sample datasets viewer modal
  const [showSampleViewer, setShowSampleViewer] = useState(false)

  // Fetch system capabilities and datasets on mount
  useEffect(() => {
    fetchCapabilities()
    fetchDatasets()
  }, [])
  
  const fetchCapabilities = async () => {
    try {
      const res = await fetch('/api/system/capabilities')
      if (res.ok) {
        const data = await res.json()
        setCapabilities({
          supported_dataset_sources: data.supported_dataset_sources || ['local'],
          supported_model_sources: data.supported_model_sources || ['local']
        })
      }
    } catch (e) {
      console.error('Failed to fetch capabilities:', e)
    }
  }
  
  // Filter tabs based on supported sources
  const SOURCE_TABS = ALL_SOURCE_TABS.filter(tab => 
    capabilities.supported_dataset_sources.includes(tab.sourceKey)
  )
  
  // Always auto-select first available tab when capabilities load
  // User can change it if they want, but first option is pre-selected
  useEffect(() => {
    if (SOURCE_TABS.length > 0) {
      setActiveTab(SOURCE_TABS[0].id)
    }
  }, [capabilities.supported_dataset_sources])

  // Sync selection from parent when selectedPaths prop changes (fixes Next button bug)
  useEffect(() => {
    if (datasets.length > 0) {
      const updated = datasets.map(d => ({
        ...d,
        selected: selectedPaths.includes(d.path)
      }))
      // Only update if there's an actual difference
      const hasChange = datasets.some((d, i) => d.selected !== updated[i].selected)
      if (hasChange) {
        setDatasets(updated)
      }
    }
  }, [selectedPaths])

  // Notify parent when datasets selection changes
  useEffect(() => {
    if (datasets.length > 0) {
      const paths = datasets.filter(d => d.selected).map(d => d.path)
      // Only notify if paths actually changed
      if (JSON.stringify(paths) !== JSON.stringify(selectedPaths)) {
        onSelectionChange(paths)
      }
    }
  }, [datasets])

  // Detect dataset type when selection changes
  useEffect(() => {
    const selectedDatasets = datasets.filter(d => d.selected)
    if (selectedDatasets.length > 0) {
      detectDatasetTypes(selectedDatasets.map(d => d.path))
    } else {
      setDetectedTypeInfo(null)
      if (onDatasetTypeChange) {
        onDatasetTypeChange(null)
      }
    }
  }, [datasets.filter(d => d.selected).map(d => d.path).join(',')])

  // Detect dataset types from selected paths
  const detectDatasetTypes = async (paths: string[]) => {
    if (paths.length === 0) return
    
    setIsDetectingType(true)
    try {
      const res = await fetch(`/api/datasets/detect-type-bulk?paths=${encodeURIComponent(paths.join(','))}`)
      if (res.ok) {
        const data = await res.json()
        const typeInfo: DatasetTypeInfo = {
          dataset_type: data.detected_type || 'unknown',
          confidence: data.datasets?.[0]?.confidence || 0,
          detected_fields: data.datasets?.[0]?.detected_fields || [],
          sample_count: data.datasets?.[0]?.sample_count || 0,
          compatible_training_methods: data.combined_compatible_methods || ['sft', 'pt', 'rlhf'],
          incompatible_training_methods: data.combined_incompatible_methods || [],
          compatible_rlhf_types: data.datasets?.[0]?.compatible_rlhf_types || [],
          display_name: data.datasets?.[0]?.display_name || DATASET_TYPE_CONFIG[data.detected_type || 'unknown']?.label || 'Unknown',
          message: data.message || ''
        }
        setDetectedTypeInfo(typeInfo)
        if (onDatasetTypeChange) {
          onDatasetTypeChange(typeInfo)
        }
      }
    } catch (e) {
      console.error('Failed to detect dataset type:', e)
    } finally {
      setIsDetectingType(false)
    }
  }

  // Fetch HuggingFace dataset info when path changes
  useEffect(() => {
    if (activeTab === 'huggingface' && registerPath.trim()) {
      const timer = setTimeout(() => fetchDatasetInfo(registerPath.trim()), 500)
      return () => clearTimeout(timer)
    } else {
      setDatasetInfo(null)
      setInfoFetched(false)
    }
  }, [registerPath, activeTab])

  const fetchDatasetInfo = async (datasetId: string) => {
    setIsFetchingInfo(true)
    setDatasetInfo(null)
    try {
      // Use HuggingFace datasets API
      const res = await fetch(`https://datasets-server.huggingface.co/info?dataset=${encodeURIComponent(datasetId)}`)
      if (res.ok) {
        const data = await res.json()
        const subsets = Object.keys(data.dataset_info || {})
        const splits: { [subset: string]: { [split: string]: number } } = {}
        
        for (const subset of subsets) {
          const subsetInfo = data.dataset_info[subset]
          if (subsetInfo?.splits) {
            splits[subset] = {}
            for (const [splitName, splitInfo] of Object.entries(subsetInfo.splits)) {
              splits[subset][splitName] = (splitInfo as any).num_examples || 0
            }
          }
        }
        
        setDatasetInfo({ subsets, splits, isPrivate: false })
        // Auto-select first subset if available
        if (subsets.length > 0 && !registerSubset) {
          setRegisterSubset(subsets[0] === 'default' ? '' : subsets[0])
        }
      } else if (res.status === 401 || res.status === 403 || res.status === 404) {
        setDatasetInfo({ subsets: [], splits: {}, isPrivate: true, error: 'Private or not found - enter details manually' })
      } else {
        setDatasetInfo({ subsets: [], splits: {}, isPrivate: true, error: 'Could not fetch info - enter details manually' })
      }
    } catch (e) {
      setDatasetInfo({ subsets: [], splits: {}, isPrivate: true, error: 'API unavailable - enter details manually' })
    } finally {
      setIsFetchingInfo(false)
      setInfoFetched(true)
    }
  }

  const getSampleCount = (subset: string, split: string): number | null => {
    if (!datasetInfo || datasetInfo.isPrivate) return null
    const subsetKey = subset || datasetInfo.subsets[0] || 'default'
    return datasetInfo.splits[subsetKey]?.[split] ?? null
  }

  const fetchDatasets = async () => {
    setIsLoading(true)
    try {
      const res = await fetch('/api/datasets/list-all')
      if (res.ok) {
        const data = await res.json()
        // Preserve selection state, but default to NOT selected
        // Also enforce same-type selection
        const newDatasets = data.datasets.map((ds: Dataset) => ({
          ...ds,
          selected: selectedPaths.includes(ds.path) || 
                    (datasets.find(d => d.id === ds.id)?.selected ?? false) // Default to false, not true
        }))
        
        // Enforce same-type selection: if multiple types are selected, keep only first type
        const selectedDatasets = newDatasets.filter((d: Dataset) => d.selected)
        if (selectedDatasets.length > 1) {
          const firstSelectedType = selectedDatasets[0].dataset_type || 'unknown'
          // Deselect any datasets that don't match the first selected type
          newDatasets.forEach((ds: Dataset) => {
            if (ds.selected) {
              const dsType = ds.dataset_type || 'unknown'
              if (dsType !== firstSelectedType || dsType === 'unknown') {
                ds.selected = false
              }
            }
          })
        }
        
        setDatasets(newDatasets)
      }
    } catch (e) {
      console.error('Failed to fetch datasets:', e)
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setUploadFile(file)
      if (!uploadName) setUploadName(file.name.replace(/\.(jsonl|json|csv|txt)$/i, ''))
    }
  }

  const checkNameAvailable = async (name: string): Promise<boolean> => {
    try {
      const res = await fetch(`/api/datasets/check-name?name=${encodeURIComponent(name)}`)
      if (res.ok) {
        const data = await res.json()
        return data.available
      }
    } catch (e) {
      console.error('Name check failed:', e)
    }
    return true // Allow if check fails
  }

  // Helper to format user-friendly error messages
  const formatUploadError = (error: any, context: string): string => {
    const errorStr = error?.message || String(error)
    
    // Network/connection errors
    if (errorStr.includes('Failed to fetch') || errorStr.includes('NetworkError') || errorStr.includes('ERR_')) {
      return `Connection error: Unable to reach the server. Please check:\n• Backend service is running\n• Network connection is stable\n• Try refreshing the page`
    }
    
    // File size errors
    if (errorStr.includes('too large') || errorStr.includes('size') || errorStr.includes('413')) {
      return `File too large: The file exceeds the maximum upload size. For large datasets, use JSONL format or the Local Path option.`
    }
    
    // Format errors
    if (errorStr.includes('format') || errorStr.includes('parse') || errorStr.includes('JSON') || errorStr.includes('invalid')) {
      return `Invalid format: The file format is not recognized. Please ensure:\n• JSONL files have one JSON object per line\n• JSON files are valid JSON arrays\n• CSV files have proper headers`
    }
    
    // Permission errors
    if (errorStr.includes('permission') || errorStr.includes('access') || errorStr.includes('403')) {
      return `Permission denied: Cannot write to the data directory. Please check server file permissions.`
    }
    
    // Disk space errors
    if (errorStr.includes('space') || errorStr.includes('disk') || errorStr.includes('storage')) {
      return `Storage error: Insufficient disk space. Please free up space on the server.`
    }
    
    // Timeout errors
    if (errorStr.includes('timeout') || errorStr.includes('timed out')) {
      return `Upload timeout: The upload took too long. For large files, try:\n• Using JSONL format (faster streaming)\n• Splitting into smaller files\n• Using Local Path option`
    }
    
    // Server errors
    if (errorStr.includes('500') || errorStr.includes('Internal Server Error')) {
      return `Server error: An unexpected error occurred. Please check the backend logs for details.`
    }
    
    // Return original error if no specific match
    return `${context} failed: ${errorStr}`
  }

  const uploadDataset = async () => {
    if (!uploadFile || !uploadName.trim()) {
      setUploadError('Please provide both a file and a name')
      return
    }
    
    // Validate file extension
    const fileName = uploadFile.name.toLowerCase()
    const validExtensions = ['.jsonl', '.json', '.csv', '.txt']
    const hasValidExtension = validExtensions.some(ext => fileName.endsWith(ext))
    if (!hasValidExtension) {
      setUploadError(`Unsupported file type. Please upload one of: ${validExtensions.join(', ')}`)
      return
    }
    
    // Warn about large files
    const fileSizeMB = uploadFile.size / (1024 * 1024)
    if (fileSizeMB > 500 && !fileName.endsWith('.jsonl')) {
      setUploadError(`Large files (${fileSizeMB.toFixed(1)}MB) should use JSONL format for better performance. Please convert to JSONL or use Local Path option.`)
      return
    }
    
    setIsUploading(true)
    setUploadError('')
    setUploadStatus('uploading')
    setUploadProgress(0)
    setUploadResult(null)
    
    try {
      // Check name availability first
      const nameAvailable = await checkNameAvailable(uploadName.trim())
      if (!nameAvailable) {
        setUploadError(`Dataset name "${uploadName}" is already in use. Please choose a different name.`)
        setIsUploading(false)
        setUploadStatus('error')
        return
      }

      // Simulate progress for better UX (actual upload progress would need XMLHttpRequest)
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + Math.random() * 15
        })
      }, 200)

      const formData = new FormData()
      formData.append('file', uploadFile)
      const res = await fetch(`/api/datasets/upload?dataset_name=${encodeURIComponent(uploadName.trim())}`, {
        method: 'POST', body: formData
      })
      
      clearInterval(progressInterval)
      setUploadProgress(95)
      setUploadStatus('processing')
      
      if (!res.ok) {
        let errorDetail = `Upload failed (HTTP ${res.status})`
        try {
          const data = await res.json()
          errorDetail = data.detail || data.error || data.message || errorDetail
        } catch {
          // Response not JSON
        }
        throw new Error(errorDetail)
      }
      
      const data = await res.json()
      if (data.success) {
        setUploadProgress(100)
        setUploadStatus('success')
        setUploadResult({
          samples: data.total_samples || 0,
          size: data.size_human || `${(uploadFile.size / 1024).toFixed(1)} KB`
        })
        
        // Refresh datasets immediately
        await fetchDatasets()
        
        // Clear form after delay to show success state (no alert modal needed - inline success shown)
        setTimeout(() => {
          setUploadName('')
          setUploadFile(null)
          if (fileInputRef.current) fileInputRef.current.value = ''
          setUploadStatus('idle')
          setUploadProgress(0)
          setUploadResult(null)
        }, 3000)
      } else {
        setUploadStatus('error')
        setUploadError(data.detail || data.error || 'Upload failed - please check file format')
      }
    } catch (e: any) {
      setUploadStatus('error')
      setUploadError(formatUploadError(e, 'Upload'))
    } finally {
      setIsUploading(false)
    }
  }

  const registerDataset = async () => {
    if (!registerName.trim() || !registerPath.trim()) {
      setRegisterError('Please provide a name and dataset ID/path')
      return
    }
    
    // Validate HuggingFace dataset ID format
    if (activeTab === 'huggingface') {
      const hfPattern = /^[\w.-]+\/[\w.-]+$/
      if (!hfPattern.test(registerPath.trim()) && !registerPath.includes('/')) {
        setRegisterError('Invalid HuggingFace dataset ID. Format should be: owner/dataset-name (e.g., "tatsu-lab/alpaca")')
        return
      }
    }
    
    // Validate local path format
    if (activeTab === 'local_path') {
      if (!registerPath.startsWith('/')) {
        setRegisterError('Local path must be an absolute path starting with / (e.g., "/data/my_dataset.jsonl")')
        return
      }
    }
    
    setIsRegistering(true)
    setRegisterError('')
    try {
      // Check name availability first
      const nameAvailable = await checkNameAvailable(registerName.trim())
      if (!nameAvailable) {
        setRegisterError(`Dataset name "${registerName}" is already in use. Please choose a different name.`)
        setIsRegistering(false)
        return
      }

      const maxSamples = registerMaxSamples.trim() ? parseInt(registerMaxSamples) : null
      
      const res = await fetch('/api/datasets/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: registerName.trim(),
          source: activeTab,
          dataset_id: registerPath.trim(),
          subset: registerSubset.trim() || null,
          split: registerSplit || 'train',
          max_samples: maxSamples && maxSamples > 0 ? maxSamples : null
        })
      })
      
      if (!res.ok) {
        let errorDetail = `Registration failed (HTTP ${res.status})`
        try {
          const data = await res.json()
          errorDetail = data.detail || data.error || data.message || errorDetail
          
          // Provide specific guidance for common errors
          if (res.status === 404) {
            if (activeTab === 'huggingface') {
              errorDetail = `Dataset not found on HuggingFace: "${registerPath}". Please verify:\n• Dataset ID is correct (owner/dataset-name)\n• Dataset is public or you have access\n• HuggingFace token is configured if private`
            } else if (activeTab === 'local_path') {
              errorDetail = `File not found: "${registerPath}". Please verify:\n• Path exists on the server\n• Path is accessible by the backend service\n• File has correct permissions`
            }
          } else if (res.status === 401 || res.status === 403) {
            errorDetail = `Access denied for "${registerPath}". For private datasets, configure HF_TOKEN in environment.`
          }
        } catch {
          // Response not JSON
        }
        throw new Error(errorDetail)
      }
      
      const data = await res.json()
      if (data.success) {
        setRegisterName('')
        setRegisterPath('')
        setRegisterSubset('')
        setRegisterMaxSamples('')
        setDatasetInfo(null)
        setInfoFetched(false)
        await fetchDatasets()
        // Show success message
        if (onShowAlert) {
          onShowAlert(`Dataset "${registerName}" registered successfully.`, 'success', 'Registration Complete')
        }
      } else {
        setRegisterError(data.detail || data.error || 'Registration failed - please check dataset path')
      }
    } catch (e: any) {
      setRegisterError(formatUploadError(e, 'Registration'))
    } finally {
      setIsRegistering(false)
    }
  }

  const toggleSelection = (dataset: Dataset, forceAllow: boolean = false) => {
    // If trying to select (not deselect), check if types match
    if (!dataset.selected && !forceAllow) {
      const { allowed } = canSelectDataset(dataset)
      if (!allowed) {
        // Different type - deselect all others and select only this one
        const updated = datasets.map(d => ({
          ...d,
          selected: d.id === dataset.id
        }))
        setDatasets(updated)
        const paths = updated.filter(d => d.selected).map(d => d.path)
        onSelectionChange(paths)
        return
      }
    }
    
    const updated = datasets.map(d => 
      d.id === dataset.id ? { ...d, selected: !d.selected } : d
    )
    setDatasets(updated)
    const paths = updated.filter(d => d.selected).map(d => d.path)
    onSelectionChange(paths)
  }

  const confirmDelete = async () => {
    if (!deleteTarget || deleteConfirm !== deleteTarget.name) return
    setIsDeleting(true)
    try {
      // Pass the dataset name as confirmation (not "delete")
      const res = await fetch(`/api/datasets/unregister/${encodeURIComponent(deleteTarget.id)}?confirm=${encodeURIComponent(deleteTarget.name)}`, {
        method: 'DELETE'
      })
      if (res.ok) {
        await fetchDatasets()
        setDeleteTarget(null)
        setDeleteConfirm('')
      } else {
        const data = await res.json()
        if (onShowAlert) {
          onShowAlert(data.detail || 'Delete failed', 'error', 'Delete Failed')
        }
      }
    } catch (e) {
      console.error('Delete failed:', e)
      if (onShowAlert) {
        onShowAlert(`Delete failed: ${e instanceof Error ? e.message : String(e)}`, 'error', 'Delete Failed')
      }
    } finally {
      setIsDeleting(false)
    }
  }


  const selectedCount = datasets.filter(d => d.selected).length

  const getSourceIcon = (source: DatasetSource) => {
    switch (source) {
      case 'upload': return <Upload className="w-4 h-4" />
      case 'huggingface': return <span className="text-xs">🤗</span>
      case 'modelscope': return <Cloud className="w-4 h-4" />
      case 'local_path': return <FolderOpen className="w-4 h-4" />
    }
  }

  const getSourceColor = (source: DatasetSource) => {
    switch (source) {
      case 'upload': return 'bg-blue-100 text-blue-700'
      case 'huggingface': return 'bg-yellow-100 text-yellow-700'
      case 'modelscope': return 'bg-purple-100 text-purple-700'
      case 'local_path': return 'bg-green-100 text-green-700'
    }
  }

  const getSourceLabel = (source: DatasetSource) => {
    switch (source) {
      case 'upload': return 'Uploaded'
      case 'huggingface': return 'HuggingFace'
      case 'modelscope': return 'ModelScope'
      case 'local_path': return 'Local'
    }
  }

  const getDatasetTypeLabel = (type: string) => {
    return DATASET_TYPE_CONFIG[type]?.label || 'Unknown'
  }

  const getDatasetTypeColor = (type: string) => {
    const config = DATASET_TYPE_CONFIG[type] || DATASET_TYPE_CONFIG['unknown']
    return `${config.bgColor} ${config.color} ${config.borderColor}`
  }

  // Get the type of the first selected dataset (for enforcing same-type selection)
  const getSelectedDatasetType = (): string | null => {
    const selectedDatasets = datasets.filter(d => d.selected)
    if (selectedDatasets.length === 0) return null
    return selectedDatasets[0].dataset_type || null
  }

  // Check if a dataset can be selected with current selection (same type)
  // Returns allowed=false if types differ (will trigger auto-switch behavior)
  const canSelectDataset = (dataset: Dataset): { allowed: boolean; reason: string } => {
    const selectedType = getSelectedDatasetType()
    
    // If no datasets selected yet, or this dataset is already selected, allow
    if (!selectedType || dataset.selected) {
      return { allowed: true, reason: '' }
    }
    
    const datasetType = dataset.dataset_type || 'unknown'
    
    // STRICT: Unknown types are NOT allowed for training
    if (datasetType === 'unknown') {
      return { 
        allowed: false, 
        reason: 'This dataset has an unknown format and cannot be used for training. Please ensure your dataset follows a supported format.' 
      }
    }
    
    // Check if types match - if not, selection will auto-switch (deselect others)
    if (datasetType !== selectedType) {
      return {
        allowed: false,
        reason: '' // No warning needed - we auto-switch now
      }
    }
    
    return { allowed: true, reason: '' }
  }

  return (
    <div className="space-y-5">
      {/* Delete Confirmation Modal */}
      {deleteTarget && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl p-6 max-w-md w-full shadow-2xl">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center">
                <Trash2 className="w-5 h-5 text-red-600" />
              </div>
              <div>
                <h3 className="font-bold text-slate-900">Remove Dataset</h3>
                <p className="text-sm text-slate-500">This action cannot be undone</p>
              </div>
            </div>
            <div className="bg-slate-50 rounded-lg p-3 mb-4">
              <p className="text-sm text-slate-700"><strong>{deleteTarget.name}</strong></p>
              <p className="text-xs text-slate-500">{getSourceLabel(deleteTarget.source)} • {deleteTarget.path}</p>
            </div>
            <p className="text-sm text-slate-600 mb-2">
              To confirm deletion, type the dataset name:
            </p>
            <p className="text-sm font-mono bg-red-50 text-red-700 px-2 py-1 rounded mb-3 select-all">
              {deleteTarget.name}
            </p>
            <input type="text" value={deleteConfirm} onChange={(e) => setDeleteConfirm(e.target.value)}
              placeholder={`Type "${deleteTarget.name}" to confirm`} autoFocus
              className="w-full px-3 py-2 border border-slate-300 rounded-lg mb-4 focus:ring-2 focus:ring-red-500 focus:border-red-500" />
            <div className="flex gap-2">
              <button onClick={() => { setDeleteTarget(null); setDeleteConfirm('') }}
                className="flex-1 px-4 py-2 border border-slate-300 rounded-lg font-medium hover:bg-slate-50 transition-colors">Cancel</button>
              <button onClick={confirmDelete} disabled={deleteConfirm !== deleteTarget.name || isDeleting}
                className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-red-700 transition-colors flex items-center justify-center gap-2">
                {isDeleting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}Delete
              </button>
            </div>
          </div>
        </div>
      )}

      <div>
        <h2 className="text-xl font-bold text-slate-900 mb-1">Configure Dataset</h2>
        <p className="text-slate-600 text-sm">Add datasets from multiple sources for training</p>
      </div>

      {/* Sample Datasets Viewer Modal */}
      <SampleDatasetsViewer 
        isOpen={showSampleViewer} 
        onClose={() => setShowSampleViewer(false)} 
      />

      {/* Sample Datasets Button */}
      <button
        onClick={() => setShowSampleViewer(true)}
        className="w-full flex items-center justify-between p-3 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg hover:from-blue-100 hover:to-indigo-100 transition-all group"
      >
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-blue-600 rounded-lg flex items-center justify-center group-hover:scale-105 transition-transform">
            <BookOpen className="w-5 h-5 text-white" />
          </div>
          <div className="text-left">
            <span className="font-medium text-blue-900 block">Dataset Format Examples & Documentation</span>
            <span className="text-xs text-blue-700">
              Download examples, view format specifications, and read complete documentation
            </span>
          </div>
        </div>
        <ChevronDown className="w-5 h-5 text-blue-600 group-hover:translate-y-0.5 transition-transform" />
      </button>

      {/* Source Tabs */}
      <div className="flex flex-wrap gap-2 p-1 bg-slate-100 rounded-lg">
        {SOURCE_TABS.map(tab => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            className={`flex-1 min-w-[120px] flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-all ${
              activeTab === tab.id ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
            }`}>
            <tab.icon className="w-4 h-4" />
            <span className="hidden sm:inline">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Add Dataset Form */}
      <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
        <h4 className="font-medium text-slate-900 mb-3 flex items-center gap-2">
          <Plus className="w-4 h-4" />
          Add {SOURCE_TABS.find(t => t.id === activeTab)?.label} Dataset
        </h4>

        {/* Upload Form */}
        {activeTab === 'upload' && (
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Dataset Name *</label>
              <input type="text" value={uploadName} onChange={(e) => setUploadName(e.target.value)}
                placeholder="My Training Dataset"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Dataset File *</label>
              <input ref={fileInputRef} type="file" accept=".jsonl,.json,.csv,.txt" onChange={handleFileSelect}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg bg-white file:mr-3 file:px-3 file:py-1 file:border-0 file:bg-blue-100 file:text-blue-700 file:rounded file:font-medium file:text-sm" />
              <p className="text-xs text-slate-500 mt-1">
                <strong>Supported:</strong> .jsonl (recommended), .csv, .txt (unlimited size) | .json (max 2GB)
              </p>
              <p className="text-xs text-slate-400 mt-0.5">
                TSV, Parquet, Excel files are not supported. Use JSONL for large datasets.
              </p>
            </div>
            {/* File Info & Progress */}
            {uploadFile && (
              <div className="bg-white border border-slate-200 rounded-lg p-3">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                    <FileText className="w-5 h-5 text-blue-600" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-slate-900 truncate">{uploadFile.name}</p>
                    <p className="text-xs text-slate-500">
                      {(uploadFile.size / 1024 / 1024).toFixed(2)} MB
                      {uploadFile.name.endsWith('.jsonl') && ' • JSONL format'}
                      {uploadFile.name.endsWith('.json') && ' • JSON format'}
                      {uploadFile.name.endsWith('.csv') && ' • CSV format'}
                    </p>
                  </div>
                  {!isUploading && uploadStatus !== 'success' && (
                    <button 
                      onClick={() => {
                        setUploadFile(null)
                        setUploadStatus('idle')
                        setUploadProgress(0)
                        setUploadResult(null)
                        setUploadError('')
                        if (fileInputRef.current) fileInputRef.current.value = ''
                      }}
                      className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>
                
                {/* Upload Progress Bar - show during upload or on success */}
                {(isUploading || uploadStatus === 'success') && (
                  <div className="mt-3">
                    <div className="flex items-center justify-between text-xs text-slate-600 mb-1">
                      <span className={uploadStatus === 'success' ? 'text-green-600 font-medium' : ''}>
                        {uploadStatus === 'uploading' && 'Uploading...'}
                        {uploadStatus === 'processing' && 'Processing dataset...'}
                        {uploadStatus === 'success' && '✓ Upload complete!'}
                      </span>
                      <span className={uploadStatus === 'success' ? 'text-green-600' : ''}>
                        {Math.round(uploadProgress)}%
                      </span>
                    </div>
                    <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                      <div 
                        className={`h-full transition-all duration-300 rounded-full ${
                          uploadStatus === 'success' ? 'bg-green-500' : 'bg-blue-500'
                        }`}
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                    
                    {/* Success Result - show sample count and size */}
                    {uploadStatus === 'success' && uploadResult && (
                      <div className="mt-2 flex items-center gap-2 text-green-600">
                        <Check className="w-4 h-4" />
                        <span className="text-sm">
                          {uploadResult.samples.toLocaleString()} samples • {uploadResult.size}
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
            
            {uploadError && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                <div className="flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-red-700 whitespace-pre-line">{uploadError}</p>
                </div>
              </div>
            )}
            
            <button onClick={uploadDataset} disabled={!uploadFile || !uploadName.trim() || isUploading}
              className="w-full py-2.5 bg-blue-600 text-white rounded-lg font-medium disabled:opacity-50 hover:bg-blue-700 transition-colors flex items-center justify-center gap-2">
              {isUploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
              {isUploading ? (uploadStatus === 'processing' ? 'Processing...' : 'Uploading...') : 'Upload Dataset'}
            </button>
          </div>
        )}

        {/* HuggingFace / ModelScope / Local Path Form */}
        {activeTab !== 'upload' && (
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Dataset Name *</label>
              <input type="text" value={registerName} onChange={(e) => setRegisterName(e.target.value)}
                placeholder="My Dataset"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                {activeTab === 'local_path' ? 'Local Path *' : 'Dataset ID *'}
              </label>
              <div className="relative">
                <input type="text" value={registerPath} onChange={(e) => setRegisterPath(e.target.value)}
                  placeholder={activeTab === 'local_path' ? '/path/to/dataset.jsonl' : 'organization/dataset-name'}
                  className="w-full px-3 py-2 border border-slate-300 rounded-lg pr-10" />
                {isFetchingInfo && (
                  <div className="absolute right-3 top-1/2 -translate-y-1/2">
                    <Loader2 className="w-4 h-4 animate-spin text-slate-400" />
                  </div>
                )}
              </div>
              {activeTab === 'huggingface' && infoFetched && datasetInfo && (
                <p className={`text-xs mt-1 ${datasetInfo.isPrivate ? 'text-amber-600' : 'text-green-600'}`}>
                  {datasetInfo.isPrivate ? `⚠️ ${datasetInfo.error}` : `✓ Found ${datasetInfo.subsets.length} subset(s)`}
                </p>
              )}
            </div>

            {/* Subset & Split - Show dropdowns if API returned data, otherwise input fields */}
            {activeTab !== 'local_path' && (
              <>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Subset</label>
                    {datasetInfo && !datasetInfo.isPrivate && datasetInfo.subsets.length > 0 ? (
                      <select value={registerSubset} onChange={(e) => setRegisterSubset(e.target.value)}
                        className="w-full px-3 py-2 border border-slate-300 rounded-lg bg-white">
                        {datasetInfo.subsets.map(s => (
                          <option key={s} value={s === 'default' ? '' : s}>{s}</option>
                        ))}
                      </select>
                    ) : (
                      <input type="text" value={registerSubset} onChange={(e) => setRegisterSubset(e.target.value)}
                        placeholder="default (optional)"
                        className="w-full px-3 py-2 border border-slate-300 rounded-lg" />
                    )}
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">Split</label>
                    {datasetInfo && !datasetInfo.isPrivate ? (
                      <select value={registerSplit} onChange={(e) => setRegisterSplit(e.target.value)}
                        className="w-full px-3 py-2 border border-slate-300 rounded-lg bg-white">
                        {Object.keys(datasetInfo.splits[registerSubset || datasetInfo.subsets[0]] || {}).map(s => (
                          <option key={s} value={s}>{s}</option>
                        ))}
                        {Object.keys(datasetInfo.splits[registerSubset || datasetInfo.subsets[0]] || {}).length === 0 && (
                          <>
                            <option value="train">train</option>
                            <option value="validation">validation</option>
                            <option value="test">test</option>
                          </>
                        )}
                      </select>
                    ) : (
                      <select value={registerSplit} onChange={(e) => setRegisterSplit(e.target.value)}
                        className="w-full px-3 py-2 border border-slate-300 rounded-lg bg-white">
                        <option value="train">train</option>
                        <option value="validation">validation</option>
                        <option value="test">test</option>
                      </select>
                    )}
                  </div>
                </div>

                {/* Show sample count for selected split */}
                {datasetInfo && !datasetInfo.isPrivate && (
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-2">
                    <p className="text-xs text-blue-700">
                      📊 <strong>{registerSplit}</strong> split has{' '}
                      <strong>{(getSampleCount(registerSubset, registerSplit) || 0).toLocaleString()}</strong> samples
                    </p>
                  </div>
                )}
              </>
            )}

            {/* Max Samples - for all sources */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Max Samples <span className="text-slate-400 font-normal">(blank = all)</span>
              </label>
              <input type="number" value={registerMaxSamples} onChange={(e) => setRegisterMaxSamples(e.target.value)}
                placeholder="Leave empty to use all samples"
                min="0" step="1"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg" />
              <p className="text-xs text-slate-500 mt-1">
                Specify maximum number of samples to use, or leave blank for all
              </p>
            </div>

            {registerError && <p className="text-sm text-red-600 flex items-center gap-1"><AlertCircle className="w-4 h-4" />{registerError}</p>}
            <button onClick={registerDataset} disabled={!registerName.trim() || !registerPath.trim() || isRegistering}
              className="w-full py-2 bg-blue-600 text-white rounded-lg font-medium disabled:opacity-50 hover:bg-blue-700 transition-colors flex items-center justify-center gap-2">
              {isRegistering ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
              {isRegistering ? 'Registering...' : 'Add Dataset'}
            </button>

          </div>
        )}
      </div>

      {/* Dataset List */}
      <div>
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <h4 className="font-medium text-slate-900">
            Registered Datasets
            {selectedCount > 0 && <span className="text-blue-600 ml-2">({selectedCount} selected for training)</span>}
          </h4>
          <div className="flex items-center gap-2">
            {/* Type Filter */}
            <select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
              className="text-xs border border-slate-200 rounded-lg px-2 py-1.5 bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="all">All Types</option>
              <option value="sft">SFT</option>
              <option value="rlhf_offline">RLHF Offline</option>
              <option value="rlhf_online">RLHF Online</option>
              <option value="pt">Pre-Training</option>
              <option value="kto">KTO</option>
            </select>
            <button onClick={fetchDatasets} disabled={isLoading}
              className="text-sm text-slate-500 hover:text-slate-700 flex items-center gap-1">
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} /> Refresh
            </button>
          </div>
        </div>

        {isLoading ? (
          <div className="text-center py-8 text-slate-500">
            <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />Loading...
          </div>
        ) : datasets.length === 0 ? (
          <div className="text-center py-8 text-slate-500 bg-slate-50 rounded-lg border border-dashed border-slate-300">
            <Database className="w-10 h-10 mx-auto mb-2 opacity-50" />
            <p>No datasets registered yet</p>
            <p className="text-sm">Add a dataset using the form above</p>
          </div>
        ) : (() => {
          // STRICT: Filter out unknown datasets - they cannot be used for training
          const validDatasets = datasets.filter(dataset => {
            const datasetType = dataset.dataset_type || 'unknown'
            return datasetType !== 'unknown'
          })
          
          // Count unknown datasets for warning message
          const unknownCount = datasets.length - validDatasets.length
          
          const filteredDatasets = validDatasets.filter(dataset => 
            typeFilter === 'all' || dataset.dataset_type === typeFilter
          )
          
          if (filteredDatasets.length === 0) {
            const typeLabel = DATASET_TYPE_CONFIG[typeFilter]?.label || typeFilter
            return (
              <div className="text-center py-8 bg-amber-50 rounded-lg border border-dashed border-amber-200">
                <AlertCircle className="w-10 h-10 mx-auto mb-2 text-amber-400" />
                <p className="text-amber-700 font-medium">No {typeLabel} datasets available</p>
                <p className="text-sm text-amber-600 mt-1">
                  Upload a dataset with {typeLabel.toLowerCase()} format or select a different filter
                </p>
                <button 
                  onClick={() => setTypeFilter('all')}
                  className="mt-3 text-sm text-blue-600 hover:text-blue-700 font-medium"
                >
                  Show all datasets
                </button>
              </div>
            )
          }
          
          return (
          <div className="space-y-2">
            {/* Warning for unknown datasets that were filtered out */}
            {unknownCount > 0 && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg mb-2">
                <p className="text-sm text-red-700 flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  <span>
                    <strong>{unknownCount} dataset{unknownCount > 1 ? 's' : ''}</strong> with unrecognized format {unknownCount > 1 ? 'are' : 'is'} hidden. 
                    Please ensure datasets follow supported formats (SFT, RLHF, Pre-training, or KTO).
                  </span>
                </p>
              </div>
            )}
            <div className="max-h-72 overflow-y-auto space-y-2">
            {filteredDatasets.map(dataset => {
              const datasetType = dataset.dataset_type || 'unknown'
              const typeConfig = DATASET_TYPE_CONFIG[datasetType] || DATASET_TYPE_CONFIG['unknown']
              
              return (
                <div key={dataset.id} 
                  onClick={() => toggleSelection(dataset)}
                  className={`flex items-center gap-3 p-3 rounded-lg border transition-all ${
                    dataset.selected 
                      ? 'border-blue-500 bg-blue-50 cursor-pointer' 
                      : 'border-slate-200 hover:border-slate-300 bg-white cursor-pointer'
                  }`}>
                  <div className={`w-5 h-5 rounded border-2 flex items-center justify-center flex-shrink-0 ${
                    dataset.selected 
                      ? 'bg-blue-600 border-blue-600' 
                      : 'border-slate-300'
                  }`}>
                    {dataset.selected && <Check className="w-3 h-3 text-white" />}
                  </div>
                  
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${getSourceColor(dataset.source)}`}>
                    {getSourceIcon(dataset.source)}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="font-medium truncate text-slate-900">{dataset.name}</p>
                      {/* Dataset Type Badge */}
                      <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${typeConfig.bgColor} ${typeConfig.color} border ${typeConfig.borderColor}`}>
                        {typeConfig.label}
                      </span>
                    </div>
                    <p className="text-xs text-slate-500 truncate">
                      {dataset.path}
                      {dataset.subset && ` (${dataset.subset})`}
                      {dataset.split && ` [${dataset.split}]`}
                    </p>
                  </div>
                  
                  <div className="text-right flex-shrink-0">
                    <p className={`text-xs font-medium px-2 py-0.5 rounded-full inline-block ${getSourceColor(dataset.source)}`}>
                      {getSourceLabel(dataset.source)}
                    </p>
                    <p className="text-xs text-slate-400 mt-1 hidden sm:block">
                      {dataset.total_samples > 0 && `${dataset.total_samples.toLocaleString()} samples • `}
                      {dataset.size_human} • {dataset.format}
                    </p>
                    <p className="text-xs text-slate-400 mt-1 sm:hidden">
                      {dataset.total_samples > 0 ? `${dataset.total_samples.toLocaleString()}` : dataset.size_human}
                    </p>
                  </div>
                  
                  <button onClick={(e) => { e.stopPropagation(); setDeleteTarget(dataset) }}
                    className="p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg flex-shrink-0">
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              )
            })}
            </div>
          </div>
          )
        })()}

        {selectedCount === 0 && datasets.length > 0 && (
          <p className="text-sm text-amber-600 mt-2 flex items-center gap-1">
            <AlertCircle className="w-4 h-4" /> Please select at least one dataset for training
          </p>
        )}

        {/* Dataset Type Detection Info */}
        {selectedCount > 0 && detectedTypeInfo && detectedTypeInfo.dataset_type !== 'unknown' && (
          <div className={`mt-3 p-3 rounded-lg border ${getDatasetTypeColor(detectedTypeInfo.dataset_type)}`}>
            <div className="flex items-center gap-2">
              <Info className="w-4 h-4 flex-shrink-0" />
              <div>
                <p className="text-sm font-medium">
                  Dataset Format: {getDatasetTypeLabel(detectedTypeInfo.dataset_type)}
                </p>
                <p className="text-xs mt-0.5 opacity-80">
                  {detectedTypeInfo.message}
                </p>
              </div>
            </div>
          </div>
        )}

        {isDetectingType && (
          <div className="mt-3 p-3 rounded-lg border border-slate-200 bg-slate-50">
            <div className="flex items-center gap-2 text-slate-600">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm">Detecting dataset format...</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
