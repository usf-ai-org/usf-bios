'use client'

import { useState, useRef, useEffect } from 'react'
import { 
  Upload, X, FileText, Check, Trash2, RefreshCw, 
  Database, Cloud, FolderOpen, Loader2, AlertCircle,
  Plus, ExternalLink
} from 'lucide-react'

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
}

const ALL_SOURCE_TABS: { id: DatasetSource; label: string; icon: any; desc: string; sourceKey: string }[] = [
  { id: 'upload', label: 'Upload', icon: Upload, desc: 'Upload from your computer', sourceKey: 'local' },
  { id: 'huggingface', label: 'HuggingFace', icon: Cloud, desc: 'Use HuggingFace datasets', sourceKey: 'huggingface' },
  { id: 'modelscope', label: 'ModelScope', icon: Cloud, desc: 'Use ModelScope datasets', sourceKey: 'modelscope' },
  { id: 'local_path', label: 'Local Path', icon: FolderOpen, desc: 'Use server local path', sourceKey: 'local' },
]


export default function DatasetConfig({ selectedPaths, onSelectionChange, onShowAlert }: Props) {
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
        // Preserve selection state
        const newDatasets = data.datasets.map((ds: Dataset) => ({
          ...ds,
          selected: selectedPaths.includes(ds.path) || 
                    (datasets.find(d => d.id === ds.id)?.selected ?? true)
        }))
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
      if (!uploadName) setUploadName(file.name.replace(/\.(jsonl|json|csv)$/i, ''))
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

  const uploadDataset = async () => {
    if (!uploadFile || !uploadName.trim()) {
      setUploadError('Please provide both a file and a name')
      return
    }
    setIsUploading(true)
    setUploadError('')
    try {
      // Check name availability first
      const nameAvailable = await checkNameAvailable(uploadName.trim())
      if (!nameAvailable) {
        setUploadError(`Dataset name "${uploadName}" is already in use. Please choose a different name.`)
        setIsUploading(false)
        return
      }

      const formData = new FormData()
      formData.append('file', uploadFile)
      const res = await fetch(`/api/datasets/upload?dataset_name=${encodeURIComponent(uploadName.trim())}`, {
        method: 'POST', body: formData
      })
      
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || `Upload failed with status ${res.status}`)
      }
      
      const data = await res.json()
      if (data.success) {
        setUploadName('')
        setUploadFile(null)
        if (fileInputRef.current) fileInputRef.current.value = ''
        await fetchDatasets()
      } else {
        setUploadError(data.detail || 'Upload failed')
      }
    } catch (e: any) {
      setUploadError(e.message || `Upload failed: ${e}`)
    } finally {
      setIsUploading(false)
    }
  }

  const registerDataset = async () => {
    if (!registerName.trim() || !registerPath.trim()) {
      setRegisterError('Please provide a name and dataset ID/path')
      return
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
        const data = await res.json()
        throw new Error(data.detail || `Registration failed with status ${res.status}`)
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
      } else {
        setRegisterError(data.detail || 'Registration failed')
      }
    } catch (e: any) {
      setRegisterError(e.message || `Registration failed: ${e}`)
    } finally {
      setIsRegistering(false)
    }
  }

  const toggleSelection = (dataset: Dataset) => {
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
              <label className="block text-sm font-medium text-slate-700 mb-1">File (.jsonl, .json, .csv) *</label>
              <input ref={fileInputRef} type="file" accept=".jsonl,.json,.csv" onChange={handleFileSelect}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg bg-white file:mr-3 file:px-3 file:py-1 file:border-0 file:bg-blue-100 file:text-blue-700 file:rounded file:font-medium file:text-sm" />
            </div>
            {uploadFile && <p className="text-sm text-slate-600">Selected: <strong>{uploadFile.name}</strong></p>}
            {uploadError && <p className="text-sm text-red-600 flex items-center gap-1"><AlertCircle className="w-4 h-4" />{uploadError}</p>}
            <button onClick={uploadDataset} disabled={!uploadFile || !uploadName.trim() || isUploading}
              className="w-full py-2 bg-blue-600 text-white rounded-lg font-medium disabled:opacity-50 hover:bg-blue-700 transition-colors flex items-center justify-center gap-2">
              {isUploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
              {isUploading ? 'Uploading...' : 'Upload Dataset'}
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
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-medium text-slate-900">
            Registered Datasets
            {selectedCount > 0 && <span className="text-blue-600 ml-2">({selectedCount} selected for training)</span>}
          </h4>
          <button onClick={fetchDatasets} disabled={isLoading}
            className="text-sm text-slate-500 hover:text-slate-700 flex items-center gap-1">
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} /> Refresh
          </button>
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
        ) : (
          <div className="space-y-2 max-h-72 overflow-y-auto">
            {datasets.map(dataset => (
              <div key={dataset.id} onClick={() => toggleSelection(dataset)}
                className={`flex items-center gap-3 p-3 rounded-lg border transition-all cursor-pointer ${
                  dataset.selected ? 'border-blue-500 bg-blue-50' : 'border-slate-200 hover:border-slate-300 bg-white'
                }`}>
                <div className={`w-5 h-5 rounded border-2 flex items-center justify-center flex-shrink-0 ${
                  dataset.selected ? 'bg-blue-600 border-blue-600' : 'border-slate-300'
                }`}>
                  {dataset.selected && <Check className="w-3 h-3 text-white" />}
                </div>
                
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${getSourceColor(dataset.source)}`}>
                  {getSourceIcon(dataset.source)}
                </div>
                
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-slate-900 truncate">{dataset.name}</p>
                  <p className="text-xs text-slate-500 truncate">
                    {dataset.path}
                    {dataset.subset && ` (${dataset.subset})`}
                    {dataset.split && ` [${dataset.split}]`}
                  </p>
                </div>
                
                <div className="text-right flex-shrink-0 hidden sm:block">
                  <p className={`text-xs font-medium px-2 py-0.5 rounded-full inline-block ${getSourceColor(dataset.source)}`}>
                    {getSourceLabel(dataset.source)}
                  </p>
                  <p className="text-xs text-slate-400 mt-1">{dataset.size_human} • {dataset.format}</p>
                </div>
                
                <button onClick={(e) => { e.stopPropagation(); setDeleteTarget(dataset) }}
                  className="p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg flex-shrink-0">
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}

        {selectedCount === 0 && datasets.length > 0 && (
          <p className="text-sm text-amber-600 mt-2 flex items-center gap-1">
            <AlertCircle className="w-4 h-4" /> Please select at least one dataset for training
          </p>
        )}
      </div>
    </div>
  )
}
