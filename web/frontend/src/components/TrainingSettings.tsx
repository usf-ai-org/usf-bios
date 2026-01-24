'use client'

import { useState, useEffect } from 'react'
import { HelpCircle, Plus, Minus, ChevronDown, AlertCircle, Edit3 } from 'lucide-react'

// Type definitions
interface SelectConfig<T> {
  label: string
  options: T[]
  default: T
  tooltip: string
  effect?: { up: string; down: string }
}

interface NumberConfig {
  label: string
  min: number
  max: number
  step: number
  default: number
  tooltip: string
  effect?: { up: string; down: string }
}

interface TypeParamConfig {
  rank?: SelectConfig<number>
  alpha?: SelectConfig<number>
  dropout?: NumberConfig
  target_modules?: SelectConfig<string>
  quant_bits?: SelectConfig<number>
}

// Parameter configuration with tooltips, limits, and effects
export const PARAM_CONFIG: {
  lora: TypeParamConfig
  qlora: TypeParamConfig
  adalora: TypeParamConfig
  full: TypeParamConfig
  common: {
    epochs: NumberConfig
    learning_rate: SelectConfig<number>
    batch_size: SelectConfig<number>
    grad_accum: SelectConfig<number>
    max_length: SelectConfig<number>
    warmup_ratio: NumberConfig
  }
} = {
  lora: {
    rank: { label: 'LoRA Rank', options: [4, 8, 16, 32, 64, 128, 256], default: 8,
      tooltip: 'Controls LoRA expressiveness. Higher = more parameters, better learning but slower & more memory. Recommended: 8-32.',
      effect: { up: 'More expressive, more memory, slower', down: 'Less expressive, less memory, faster' }},
    alpha: { label: 'LoRA Alpha', options: [8, 16, 32, 64, 128, 256], default: 32,
      tooltip: 'Scaling factor for LoRA. Usually 2x the rank. Higher = stronger adaptation.',
      effect: { up: 'Stronger adaptation, may overfit', down: 'Weaker adaptation, more stable' }},
    dropout: { label: 'LoRA Dropout', min: 0, max: 0.5, step: 0.01, default: 0.05,
      tooltip: 'Regularization to prevent overfitting. 0.05-0.1 recommended.',
      effect: { up: 'More regularization, prevents overfitting', down: 'Less regularization, may overfit' }},
    target_modules: { label: 'Target Modules', options: ['all-linear', 'q_proj,v_proj', 'q_proj,k_proj,v_proj,o_proj', 'custom'], default: 'all-linear',
      tooltip: 'Layers to apply LoRA. "all-linear" applies to all linear layers. Select "custom" to specify your own module names for unsupported models.' }
  },
  qlora: {
    rank: { label: 'LoRA Rank', options: [4, 8, 16, 32, 64], default: 8,
      tooltip: 'For QLoRA, lower ranks work well due to quantization. 8-16 recommended.',
      effect: { up: 'More expressive, more memory', down: 'Less expressive, faster' }},
    alpha: { label: 'LoRA Alpha', options: [8, 16, 32, 64], default: 16,
      tooltip: 'For QLoRA, use equal to or 2x the rank.',
      effect: { up: 'Stronger adaptation', down: 'Weaker adaptation' }},
    dropout: { label: 'LoRA Dropout', min: 0, max: 0.3, step: 0.01, default: 0.05,
      tooltip: 'Keep low for QLoRA (0.05). Higher values can destabilize 4-bit training.',
      effect: { up: 'More regularization', down: 'Less regularization' }},
    quant_bits: { label: 'Quantization Bits', options: [4, 8], default: 4,
      tooltip: '4-bit uses ~75% less memory. 8-bit is a balance between memory and accuracy.' }
  },
  adalora: {
    rank: { label: 'Initial Rank', options: [8, 16, 32, 64, 128], default: 16,
      tooltip: 'AdaLoRA dynamically adjusts rank. Start higher, it will prune automatically.',
      effect: { up: 'More initial capacity', down: 'Less initial capacity' }},
    alpha: { label: 'LoRA Alpha', options: [16, 32, 64, 128], default: 32,
      tooltip: 'Scaling factor for AdaLoRA.',
      effect: { up: 'Stronger adaptation', down: 'Weaker adaptation' }},
    dropout: { label: 'LoRA Dropout', min: 0, max: 0.3, step: 0.01, default: 0.1,
      tooltip: 'Slightly higher dropout (0.1) helps AdaLoRA regularization.',
      effect: { up: 'More regularization', down: 'Less regularization' }}
  },
  full: {},
  common: {
    epochs: { label: 'Training Epochs', min: 1, max: 100, step: 1, default: 3,
      tooltip: 'Complete passes through the dataset. 1-5 epochs usually sufficient for fine-tuning.',
      effect: { up: 'More training, risk of overfitting', down: 'Less training, may underfit' }},
    learning_rate: { label: 'Learning Rate', options: [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005], default: 0.0001,
      tooltip: 'How fast the model learns. 1e-4 to 2e-4 for LoRA, lower for full fine-tuning.',
      effect: { up: 'Faster learning, may be unstable', down: 'Slower learning, more stable' }},
    batch_size: { label: 'Batch Size', options: [1, 2, 4, 8, 16, 32], default: 1,
      tooltip: 'Samples processed together. Higher = faster but more memory.',
      effect: { up: 'Faster training, more memory', down: 'Slower training, less memory' }},
    grad_accum: { label: 'Gradient Accumulation', options: [1, 2, 4, 8, 16, 32, 64], default: 16,
      tooltip: 'Simulates larger batch without memory increase. Effective batch = batch_size × grad_accum.',
      effect: { up: 'Larger effective batch, smoother', down: 'Smaller effective batch' }},
    max_length: { label: 'Max Sequence Length', options: [512, 1024, 2048, 4096, 8192], default: 2048,
      tooltip: 'Maximum tokens per sample. Longer = more context but more memory.',
      effect: { up: 'More context, more memory', down: 'Less context, less memory' }},
    warmup_ratio: { label: 'Warmup Ratio', min: 0, max: 0.3, step: 0.01, default: 0.03,
      tooltip: 'Portion of training for LR warmup. 0.03-0.1 prevents early instability.',
      effect: { up: 'Longer warmup, more stable', down: 'Shorter warmup, faster ramp-up' }}
  }
}

// Tooltip component
export function Tooltip({ text, effect }: { text: string, effect?: { up: string, down: string } }) {
  const [show, setShow] = useState(false)
  return (
    <div className="relative inline-block">
      <button type="button" className="p-1 text-slate-400 hover:text-blue-600 transition-colors"
        onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)} onClick={() => setShow(!show)}>
        <HelpCircle className="w-4 h-4" />
      </button>
      {show && (
        <div className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 sm:w-72 p-3 bg-slate-900 text-white text-xs rounded-lg shadow-xl">
          <p className="mb-2">{text}</p>
          {effect && (
            <div className="border-t border-slate-700 pt-2 mt-2 space-y-1">
              <p className="text-green-400"><Plus className="w-3 h-3 inline mr-1" />{effect.up}</p>
              <p className="text-amber-400"><Minus className="w-3 h-3 inline mr-1" />{effect.down}</p>
            </div>
          )}
          <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1 border-4 border-transparent border-t-slate-900" />
        </div>
      )}
    </div>
  )
}

// Number input with +/- buttons
export function NumberInput({ value, onChange, min, max, step, label, tooltip, effect, disabled }: { 
  value: number, onChange: (v: number) => void, min: number, max: number, step: number,
  label: string, tooltip?: string, effect?: { up: string, down: string }, disabled?: boolean
}) {
  const decrease = () => onChange(Number(Math.max(min, value - step).toFixed(6)))
  const increase = () => onChange(Number(Math.min(max, value + step).toFixed(6)))
  
  return (
    <div>
      <div className="flex items-center gap-1 mb-1">
        <label className="text-sm font-medium text-slate-700">{label}</label>
        {tooltip && <Tooltip text={tooltip} effect={effect} />}
      </div>
      <div className="flex items-center border border-slate-300 rounded-lg overflow-hidden bg-white">
        <button type="button" onClick={decrease} disabled={disabled || value <= min}
          className="px-3 py-2 bg-slate-50 hover:bg-slate-100 disabled:opacity-50 border-r border-slate-300">
          <Minus className="w-4 h-4 text-slate-600" />
        </button>
        <input type="number" value={value} min={min} max={max} step={step} disabled={disabled}
          onChange={(e) => { const v = parseFloat(e.target.value); if (!isNaN(v) && v >= min && v <= max) onChange(v) }}
          className="flex-1 px-3 py-2 text-center text-sm font-medium focus:outline-none disabled:bg-slate-50 w-full" />
        <button type="button" onClick={increase} disabled={disabled || value >= max}
          className="px-3 py-2 bg-slate-50 hover:bg-slate-100 disabled:opacity-50 border-l border-slate-300">
          <Plus className="w-4 h-4 text-slate-600" />
        </button>
      </div>
      <p className="text-xs text-slate-500 mt-1">Range: {min} - {max}</p>
    </div>
  )
}

// Select dropdown
export function SelectInput<T extends string | number>({ value, onChange, options, label, tooltip, effect, disabled, formatOption }: { 
  value: T, onChange: (v: T) => void, options: T[], label: string,
  tooltip?: string, effect?: { up: string, down: string }, disabled?: boolean, formatOption?: (v: T) => string
}) {
  return (
    <div>
      <div className="flex items-center gap-1 mb-1">
        <label className="text-sm font-medium text-slate-700">{label}</label>
        {tooltip && <Tooltip text={tooltip} effect={effect} />}
      </div>
      <div className="relative">
        <select value={value} disabled={disabled}
          onChange={(e) => onChange((typeof options[0] === 'number' ? Number(e.target.value) : e.target.value) as T)}
          className="w-full px-3 py-2 pr-10 border border-slate-300 rounded-lg bg-white text-sm font-medium appearance-none focus:ring-2 focus:ring-blue-500 disabled:bg-slate-50">
          {options.map((opt) => <option key={String(opt)} value={opt}>{formatOption ? formatOption(opt) : String(opt)}</option>)}
        </select>
        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
      </div>
    </div>
  )
}

interface TrainingConfig {
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
  // API Tokens for private models/datasets (optional)
  hf_token: string | null
  ms_token: string | null
  // Model source for conditional token display
  model_source?: 'huggingface' | 'modelscope' | 'local'
}

// Training method configuration
const TRAINING_METHOD_CONFIG = {
  sft: { label: 'SFT (Supervised Fine-Tuning)', desc: 'Train model on instruction-response pairs', tooltip: 'Standard fine-tuning method. Best for teaching models specific behaviors or knowledge.' },
  pt: { label: 'PT (Pre-Training)', desc: 'Continue pre-training on raw text', tooltip: 'Extend pre-training on domain-specific data. Good for adapting to new domains.' },
  rlhf: { label: 'RLHF (Reinforcement Learning)', desc: 'Align model with human preferences', tooltip: 'Use preference data to align model behavior. Includes DPO, PPO, GRPO, etc.' },
}

// RLHF algorithm configuration  
const RLHF_ALGORITHM_CONFIG = {
  dpo: { label: 'DPO', desc: 'Direct Preference Optimization', tooltip: 'Simple and stable. No reward model needed. Good default choice.' },
  orpo: { label: 'ORPO', desc: 'Odds Ratio Preference Optimization', tooltip: 'Combines SFT and preference optimization. No reference model needed.' },
  simpo: { label: 'SimPO', desc: 'Simple Preference Optimization', tooltip: 'Simplified DPO without reference model. Uses length-normalized rewards.' },
  kto: { label: 'KTO', desc: 'Kahneman-Tversky Optimization', tooltip: 'Works with binary feedback (good/bad). Good when you only have thumbs up/down.' },
  cpo: { label: 'CPO', desc: 'Contrastive Preference Optimization', tooltip: 'Contrastive learning approach. No reference model needed.' },
  rm: { label: 'RM', desc: 'Reward Model Training', tooltip: 'Train a reward model for scoring responses. Used with PPO.' },
  ppo: { label: 'PPO', desc: 'Proximal Policy Optimization', tooltip: 'Classic RLHF. Requires reward model. More complex but flexible.' },
  grpo: { label: 'GRPO', desc: 'Group Relative Policy Optimization', tooltip: 'Generates multiple responses and learns from relative rankings. No reward model needed.' },
  gkd: { label: 'GKD', desc: 'Generalized Knowledge Distillation', tooltip: 'Distill knowledge from teacher model. Good for model compression.' },
}

// Optimization configuration
const OPTIMIZATION_CONFIG = {
  attn_impl: {
    label: 'Attention Implementation',
    options: [
      { value: null, label: 'Auto (Recommended)', desc: 'Automatically selects best available' },
      { value: 'flash_attention_2', label: 'Flash Attention 2', desc: 'Fast, memory efficient (Ampere+ GPU)' },
      { value: 'flash_attention_3', label: 'Flash Attention 3', desc: 'Latest, fastest (Hopper GPU only)' },
      { value: 'sdpa', label: 'SDPA (PyTorch)', desc: 'PyTorch native, good compatibility' },
      { value: 'eager', label: 'Eager', desc: 'Standard attention, most compatible' },
    ],
    tooltip: 'Flash Attention provides 2-4x speedup and 5-20x memory reduction. SDPA is built into PyTorch. Eager is the fallback.'
  },
  deepspeed: {
    label: 'DeepSpeed ZeRO',
    options: [
      { value: null, label: 'Disabled', desc: 'No distributed optimization' },
      { value: 'zero0', label: 'ZeRO-0', desc: 'DDP only, no memory optimization' },
      { value: 'zero1', label: 'ZeRO-1', desc: 'Optimizer state partitioning' },
      { value: 'zero2', label: 'ZeRO-2', desc: '+ Gradient partitioning (recommended)' },
      { value: 'zero2_offload', label: 'ZeRO-2 + Offload', desc: '+ CPU offload for large models' },
      { value: 'zero3', label: 'ZeRO-3', desc: '+ Parameter partitioning (70B+ models)' },
      { value: 'zero3_offload', label: 'ZeRO-3 + Offload', desc: 'Maximum memory savings' },
    ],
    tooltip: 'DeepSpeed ZeRO reduces memory by partitioning optimizer states, gradients, and parameters across GPUs. Cannot be used with FSDP.'
  },
  fsdp: {
    label: 'FSDP (PyTorch)',
    options: [
      { value: null, label: 'Disabled', desc: 'No FSDP' },
      { value: 'full_shard', label: 'Full Shard', desc: 'Shard parameters, gradients, optimizer' },
      { value: 'shard_grad_op', label: 'Shard Grad/Op', desc: 'Shard gradients and optimizer only' },
      { value: 'fsdp2', label: 'FSDP2', desc: 'PyTorch FSDP2 (newer, recommended)' },
    ],
    tooltip: 'Fully Sharded Data Parallel - PyTorch native distributed training. Cannot be used with DeepSpeed.'
  }
}

// Incompatible combination warnings
const INCOMPATIBLE_WARNINGS = {
  deepspeed_fsdp: 'DeepSpeed and FSDP cannot be used together',
  packing_no_flash: 'Packing requires Flash Attention',
  liger_packing: 'Liger Kernel is incompatible with Packing',
}

interface GPUInfo {
  id: number
  name: string
  memory_total_gb: number | null
  memory_free_gb: number | null
  utilization: number | null
}

// Training capabilities from backend (dynamic based on installed packages + GPU)
interface TrainingCapabilities {
  gpu_architecture: string
  gpu_name: string | null
  compute_capability: string | null
  is_hopper: boolean
  is_ampere_or_newer: boolean
  attention_implementations: Array<{ value: string | null; label: string; desc: string; available: boolean }>
  default_attention: string
  deepspeed_available: boolean
  deepspeed_options: Array<{ value: string | null; label: string; desc: string }>
  fsdp_available: boolean
  fsdp_options: Array<{ value: string | null; label: string; desc: string }>
  liger_kernel_available: boolean
  bitsandbytes_available: boolean
  xformers_available: boolean
  incompatible_combinations: Array<{ id: string; condition: string; message: string; severity: string }>
}

// Default capabilities (fallback if API fails)
const DEFAULT_CAPABILITIES: TrainingCapabilities = {
  gpu_architecture: 'unknown',
  gpu_name: null,
  compute_capability: null,
  is_hopper: false,
  is_ampere_or_newer: false,
  attention_implementations: [
    { value: null, label: 'Auto (Recommended)', desc: 'Automatically selects best available', available: true },
    { value: 'flash_attention_2', label: 'Flash Attention 2', desc: 'Fast, memory efficient (Ampere+ GPU)', available: true },
    { value: 'sdpa', label: 'SDPA (PyTorch)', desc: 'PyTorch native, good compatibility', available: true },
    { value: 'eager', label: 'Eager', desc: 'Standard attention, most compatible', available: true },
  ],
  default_attention: 'sdpa',
  deepspeed_available: true,
  deepspeed_options: OPTIMIZATION_CONFIG.deepspeed.options,
  fsdp_available: true,
  fsdp_options: OPTIMIZATION_CONFIG.fsdp.options,
  liger_kernel_available: false,
  bitsandbytes_available: false,
  xformers_available: false,
  incompatible_combinations: []
}

interface Props {
  config: TrainingConfig
  setConfig: (fn: (prev: TrainingConfig) => TrainingConfig) => void
  availableGpus?: GPUInfo[]
  modelContextLength?: number  // Dynamic context length from selected model
}

// Generate dynamic max_length options based on model context length
function generateMaxLengthOptions(contextLength: number): number[] {
  const options: number[] = []
  const standardOptions = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
  
  for (const opt of standardOptions) {
    if (opt <= contextLength) {
      options.push(opt)
    }
  }
  
  // Always include at least some options
  if (options.length === 0) {
    options.push(512)
  }
  
  // Add the context length itself if it's not already included and is a reasonable value
  if (!options.includes(contextLength) && contextLength > 0) {
    options.push(contextLength)
    options.sort((a, b) => a - b)
  }
  
  return options
}

export default function TrainingSettingsStep({ config, setConfig, availableGpus = [], modelContextLength = 4096 }: Props) {
  const typeConfig = PARAM_CONFIG[config.train_type] || {}
  const commonConfig = PARAM_CONFIG.common
  
  // State for custom target modules input
  const [isCustomModules, setIsCustomModules] = useState(false)
  const [customModulesInput, setCustomModulesInput] = useState('')
  
  // Training capabilities from backend (dynamic based on GPU + installed packages)
  const [capabilities, setCapabilities] = useState<TrainingCapabilities>(DEFAULT_CAPABILITIES)
  const [capabilitiesLoaded, setCapabilitiesLoaded] = useState(false)
  
  // GPU selection mode: 'auto' | 'select'
  const [gpuMode, setGpuMode] = useState<'auto' | 'select'>(
    config.gpu_ids !== null ? 'select' : 'auto'
  )
  
  // Fetch training capabilities from backend on mount
  useEffect(() => {
    const fetchCapabilities = async () => {
      try {
        const res = await fetch('/api/system/training-capabilities')
        if (res.ok) {
          const data = await res.json()
          setCapabilities(data)
        }
      } catch (e) {
        console.error('Failed to fetch training capabilities:', e)
      } finally {
        setCapabilitiesLoaded(true)
      }
    }
    fetchCapabilities()
  }, [])
  
  // Set default attention based on capabilities when loaded (only once)
  useEffect(() => {
    if (capabilitiesLoaded && !config.attn_impl && capabilities.default_attention) {
      const defaultAttn = capabilities.default_attention === 'sdpa' ? null : capabilities.default_attention
      setConfig(p => ({ ...p, attn_impl: defaultAttn }))
    }
  }, [capabilitiesLoaded, capabilities.default_attention])
  
  // Handle GPU checkbox toggle
  const handleGpuToggle = (gpuId: number) => {
    const currentIds = config.gpu_ids || []
    const newIds = currentIds.includes(gpuId)
      ? currentIds.filter(id => id !== gpuId)
      : [...currentIds, gpuId].sort((a, b) => a - b)
    setConfig(p => ({ ...p, gpu_ids: newIds.length > 0 ? newIds : null, num_gpus: null }))
  }
  
  // Handle GPU mode change
  const handleGpuModeChange = (mode: 'auto' | 'select') => {
    setGpuMode(mode)
    if (mode === 'auto') {
      setConfig(p => ({ ...p, gpu_ids: null, num_gpus: null }))
    } else {
      // Default to first GPU when switching to select mode
      setConfig(p => ({ ...p, gpu_ids: availableGpus.length > 0 ? [0] : null, num_gpus: null }))
    }
  }
  
  // Check if current target_modules is a custom value (not in predefined options)
  useEffect(() => {
    const predefinedOptions = ['all-linear', 'q_proj,v_proj', 'q_proj,k_proj,v_proj,o_proj', 'custom']
    if (config.target_modules && !predefinedOptions.includes(config.target_modules)) {
      setIsCustomModules(true)
      setCustomModulesInput(config.target_modules)
    }
  }, [])

  const handleTargetModulesChange = (value: string) => {
    if (value === 'custom') {
      setIsCustomModules(true)
      // Keep current value if it was already custom, otherwise clear
      if (!customModulesInput) {
        setCustomModulesInput('')
      }
    } else {
      setIsCustomModules(false)
      setConfig(p => ({ ...p, target_modules: value }))
    }
  }

  const handleCustomModulesChange = (value: string) => {
    setCustomModulesInput(value)
    // Update config with the custom value
    if (value.trim()) {
      setConfig(p => ({ ...p, target_modules: value.trim() }))
    }
  }

  const applyDefaults = (type: TrainingConfig['train_type']) => {
    const tc = PARAM_CONFIG[type] || {}
    setConfig(prev => ({
      ...prev, train_type: type,
      lora_rank: tc.rank?.default ?? prev.lora_rank,
      lora_alpha: tc.alpha?.default ?? prev.lora_alpha,
      lora_dropout: tc.dropout?.default ?? prev.lora_dropout,
      quant_bits: type === 'qlora' ? 4 : null,
      learning_rate: type === 'full' ? 0.00002 : commonConfig.learning_rate.default,
    }))
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-slate-900 mb-1">Training Settings</h2>
        <p className="text-slate-600 text-sm">Configure hyperparameters for optimal training</p>
      </div>

      {/* Training Method */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4 border border-blue-200">
        <div className="flex items-center gap-1 mb-3">
          <label className="text-sm font-medium text-slate-700">Training Method</label>
          <Tooltip text="SFT: Standard supervised fine-tuning. PT: Continue pre-training. RLHF: Reinforcement learning for alignment." />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {Object.entries(TRAINING_METHOD_CONFIG).map(([id, cfg]) => (
            <button key={id} 
              onClick={() => setConfig(p => ({ 
                ...p, 
                training_method: id as any, 
                rlhf_type: id === 'rlhf' ? 'dpo' : null,
                // PT (Pre-Training) requires Full parameter training - auto-set train_type
                train_type: id === 'pt' ? 'full' : p.train_type
              }))}
              className={`p-4 rounded-lg border-2 text-left transition-all ${config.training_method === id ? 'border-blue-500 bg-white shadow-sm' : 'border-slate-200 bg-white/50 hover:border-slate-300'}`}>
              <span className="font-semibold text-sm block text-slate-900">{cfg.label}</span>
              <span className="text-xs text-slate-500 mt-1 block">{cfg.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* RLHF Algorithm Selection - Only show when RLHF is selected */}
      {config.training_method === 'rlhf' && (
        <div className="bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg p-4 border border-amber-200">
          <div className="flex items-center gap-1 mb-3">
            <label className="text-sm font-medium text-slate-700">RLHF Algorithm</label>
            <Tooltip text="Choose the alignment algorithm. DPO is simple and stable. GRPO generates multiple responses. PPO is classic but complex." />
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
            {Object.entries(RLHF_ALGORITHM_CONFIG).map(([id, cfg]) => (
              <button key={id}
                onClick={() => setConfig(p => ({ ...p, rlhf_type: id as any }))}
                className={`p-3 rounded-lg border-2 text-center transition-all ${config.rlhf_type === id ? 'border-amber-500 bg-white shadow-sm' : 'border-slate-200 bg-white/50 hover:border-slate-300'}`}>
                <span className="font-semibold text-xs block text-slate-900">{cfg.label}</span>
                <span className="text-[10px] text-slate-500 mt-0.5 block leading-tight">{cfg.desc}</span>
              </button>
            ))}
          </div>
          
          {/* RLHF Algorithm-Specific Parameters */}
          <div className="mt-4 pt-4 border-t border-amber-200">
            <h5 className="text-sm font-medium text-slate-700 mb-3">{config.rlhf_type?.toUpperCase()} Parameters</h5>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Common RLHF params */}
              <NumberInput value={config.beta ?? 0.1} onChange={(v) => setConfig(p => ({ ...p, beta: v }))}
                min={0} max={10} step={0.01} label="Beta" tooltip="Controls deviation from reference model. Higher = less deviation." />
              <NumberInput value={config.max_completion_length} onChange={(v) => setConfig(p => ({ ...p, max_completion_length: v }))}
                min={64} max={modelContextLength} step={64} label="Max Completion Length" 
                tooltip={`Maximum tokens to generate for GRPO/PPO/GKD. Model supports up to ${modelContextLength.toLocaleString()} tokens.`} />
              
              {/* DPO specific */}
              {config.rlhf_type === 'dpo' && (
                <>
                  <NumberInput value={config.label_smoothing} onChange={(v) => setConfig(p => ({ ...p, label_smoothing: v }))}
                    min={0} max={0.5} step={0.01} label="Label Smoothing" tooltip="Smoothing for DPO loss. 0 = disabled." />
                </>
              )}
              
              {/* SimPO specific */}
              {config.rlhf_type === 'simpo' && (
                <NumberInput value={config.simpo_gamma} onChange={(v) => setConfig(p => ({ ...p, simpo_gamma: v }))}
                  min={0} max={3} step={0.1} label="SimPO Gamma" tooltip="Reward margin. Paper suggests 0.5-1.5." />
              )}
              
              {/* KTO specific */}
              {config.rlhf_type === 'kto' && (
                <>
                  <NumberInput value={config.desirable_weight} onChange={(v) => setConfig(p => ({ ...p, desirable_weight: v }))}
                    min={0} max={5} step={0.1} label="Desirable Weight" tooltip="Weight for positive examples." />
                  <NumberInput value={config.undesirable_weight} onChange={(v) => setConfig(p => ({ ...p, undesirable_weight: v }))}
                    min={0} max={5} step={0.1} label="Undesirable Weight" tooltip="Weight for negative examples." />
                </>
              )}
              
              {/* PPO specific */}
              {config.rlhf_type === 'ppo' && (
                <>
                  <NumberInput value={config.num_ppo_epochs} onChange={(v) => setConfig(p => ({ ...p, num_ppo_epochs: v }))}
                    min={1} max={10} step={1} label="PPO Epochs" tooltip="Number of PPO update epochs per batch." />
                  <NumberInput value={config.kl_coef} onChange={(v) => setConfig(p => ({ ...p, kl_coef: v }))}
                    min={0} max={1} step={0.01} label="KL Coefficient" tooltip="KL divergence penalty coefficient." />
                  <NumberInput value={config.cliprange} onChange={(v) => setConfig(p => ({ ...p, cliprange: v }))}
                    min={0} max={1} step={0.05} label="Clip Range" tooltip="PPO clipping range for policy updates." />
                </>
              )}
              
              {/* GRPO specific */}
              {config.rlhf_type === 'grpo' && (
                <NumberInput value={config.num_generations} onChange={(v) => setConfig(p => ({ ...p, num_generations: v }))}
                  min={2} max={32} step={2} label="Num Generations" tooltip="Number of responses to generate per prompt (G in paper)." />
              )}
            </div>
          </div>
        </div>
      )}
      
      {/* Training Type (Parameter Efficient) - PT only supports Full */}
      <div>
        <div className="flex items-center gap-1 mb-2">
          <label className="text-sm font-medium text-slate-700">Parameter Efficiency</label>
          <Tooltip text={config.training_method === 'pt' 
            ? "Pre-Training requires Full parameter training to create a new base model. LoRA/QLoRA create adapters, not base models."
            : "LoRA: Efficient adapters. QLoRA: 4-bit quantized for less memory. AdaLoRA: Adaptive rank. Full: All parameters (most memory)."} />
        </div>
        {config.training_method === 'pt' ? (
          /* PT (Pre-Training) - Full only */
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-lg border-2 border-blue-500 bg-blue-50 text-center flex-shrink-0">
                <span className="font-medium text-sm block">Full</span>
                <span className="text-xs text-slate-500">Required</span>
              </div>
              <div className="text-sm text-amber-800">
                <p className="font-medium">Pre-Training requires Full parameter training</p>
                <p className="text-xs mt-1 text-amber-600">
                  Continuous pre-training creates a new base model. LoRA/QLoRA create adapters that sit on top of a base model - not suitable for pre-training.
                </p>
              </div>
            </div>
          </div>
        ) : (
          /* SFT and RLHF - All options available */
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {[
              { id: 'lora', name: 'LoRA', desc: 'Balanced' },
              { id: 'qlora', name: 'QLoRA', desc: 'Low memory' },
              { id: 'adalora', name: 'AdaLoRA', desc: 'Adaptive' },
              { id: 'full', name: 'Full', desc: 'Best quality' },
            ].map((t) => (
              <button key={t.id} onClick={() => applyDefaults(t.id as any)}
                className={`p-3 rounded-lg border-2 text-center transition-all ${config.train_type === t.id ? 'border-blue-500 bg-blue-50' : 'border-slate-200 hover:border-slate-300'}`}>
                <span className="font-medium text-sm block">{t.name}</span>
                <span className="text-xs text-slate-500">{t.desc}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* General Settings */}
      <div className="bg-slate-50 rounded-lg p-4">
        <h4 className="font-medium text-slate-900 mb-4">General Settings</h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <NumberInput value={config.num_train_epochs} onChange={(v) => setConfig(p => ({ ...p, num_train_epochs: v }))}
            min={commonConfig.epochs.min} max={commonConfig.epochs.max} step={commonConfig.epochs.step}
            label={commonConfig.epochs.label} tooltip={commonConfig.epochs.tooltip} effect={commonConfig.epochs.effect} />
          <SelectInput value={config.learning_rate} onChange={(v) => setConfig(p => ({ ...p, learning_rate: v }))}
            options={commonConfig.learning_rate.options} label={commonConfig.learning_rate.label}
            tooltip={commonConfig.learning_rate.tooltip} effect={commonConfig.learning_rate.effect}
            formatOption={(v) => v.toExponential(0)} />
          <SelectInput value={config.per_device_train_batch_size} onChange={(v) => setConfig(p => ({ ...p, per_device_train_batch_size: v }))}
            options={commonConfig.batch_size.options} label={commonConfig.batch_size.label}
            tooltip={commonConfig.batch_size.tooltip} effect={commonConfig.batch_size.effect} />
          <SelectInput value={config.gradient_accumulation_steps} onChange={(v) => setConfig(p => ({ ...p, gradient_accumulation_steps: v }))}
            options={commonConfig.grad_accum.options} label={commonConfig.grad_accum.label}
            tooltip={commonConfig.grad_accum.tooltip} effect={commonConfig.grad_accum.effect} />
          <SelectInput value={config.max_length} onChange={(v) => setConfig(p => ({ ...p, max_length: v }))}
            options={generateMaxLengthOptions(modelContextLength)} label={commonConfig.max_length.label}
            tooltip={`${commonConfig.max_length.tooltip} Model context: ${modelContextLength.toLocaleString()} tokens.`} effect={commonConfig.max_length.effect} />
          <NumberInput value={config.warmup_ratio} onChange={(v) => setConfig(p => ({ ...p, warmup_ratio: v }))}
            min={commonConfig.warmup_ratio.min} max={commonConfig.warmup_ratio.max} step={commonConfig.warmup_ratio.step}
            label={commonConfig.warmup_ratio.label} tooltip={commonConfig.warmup_ratio.tooltip} effect={commonConfig.warmup_ratio.effect} />
        </div>
        <div className="mt-3 p-2 bg-blue-50 rounded-lg">
          <p className="text-xs text-blue-700">
            <strong>Effective batch:</strong> {config.per_device_train_batch_size} × {config.gradient_accumulation_steps} = <strong>{config.per_device_train_batch_size * config.gradient_accumulation_steps}</strong>
          </p>
        </div>
      </div>

      {/* LoRA Settings */}
      {['lora', 'qlora', 'adalora'].includes(config.train_type) && (
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 rounded-lg p-4 border border-purple-200">
          <h4 className="font-medium text-slate-900 mb-4 flex items-center gap-2">
            <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded text-xs font-semibold">{config.train_type.toUpperCase()}</span>
            Adapter Settings
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {typeConfig.rank && (
              <SelectInput value={config.lora_rank} onChange={(v) => setConfig(p => ({ ...p, lora_rank: v }))}
                options={typeConfig.rank.options} label={typeConfig.rank.label}
                tooltip={typeConfig.rank.tooltip} effect={typeConfig.rank.effect} />
            )}
            {typeConfig.alpha && (
              <SelectInput value={config.lora_alpha} onChange={(v) => setConfig(p => ({ ...p, lora_alpha: v }))}
                options={typeConfig.alpha.options} label={typeConfig.alpha.label}
                tooltip={typeConfig.alpha.tooltip} effect={typeConfig.alpha.effect} />
            )}
            {typeConfig.dropout && (
              <NumberInput value={config.lora_dropout} onChange={(v) => setConfig(p => ({ ...p, lora_dropout: v }))}
                min={typeConfig.dropout.min} max={typeConfig.dropout.max} step={typeConfig.dropout.step}
                label={typeConfig.dropout.label} tooltip={typeConfig.dropout.tooltip} effect={typeConfig.dropout.effect} />
            )}
            {config.train_type === 'qlora' && typeConfig.quant_bits && (
              <SelectInput value={config.quant_bits || 4} onChange={(v) => setConfig(p => ({ ...p, quant_bits: v }))}
                options={typeConfig.quant_bits.options} label={typeConfig.quant_bits.label}
                tooltip={typeConfig.quant_bits.tooltip} formatOption={(v) => `${v}-bit`} />
            )}
            {typeConfig.target_modules && (
              <div className="sm:col-span-2 lg:col-span-3 space-y-2">
                <div className="flex items-center gap-1 mb-1">
                  <label className="text-sm font-medium text-slate-700">{typeConfig.target_modules.label}</label>
                  <Tooltip text={typeConfig.target_modules.tooltip} />
                </div>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <select 
                      value={isCustomModules ? 'custom' : config.target_modules}
                      onChange={(e) => handleTargetModulesChange(e.target.value)}
                      className="w-full px-3 py-2 pr-10 border border-slate-300 rounded-lg bg-white text-sm font-medium appearance-none focus:ring-2 focus:ring-primary-500">
                      <option value="all-linear">all-linear</option>
                      <option value="q_proj,v_proj">q_proj,v_proj</option>
                      <option value="q_proj,k_proj,v_proj,o_proj">q_proj,k_proj,v_proj,o_proj</option>
                      <option value="custom">✏️ Custom...</option>
                    </select>
                    <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
                  </div>
                </div>
                {isCustomModules && (
                  <div className="mt-2">
                    <div className="flex items-center gap-2 mb-1">
                      <Edit3 className="w-4 h-4 text-purple-600" />
                      <label className="text-sm font-medium text-purple-700">Custom Module Names</label>
                    </div>
                    <input
                      type="text"
                      value={customModulesInput}
                      onChange={(e) => handleCustomModulesChange(e.target.value)}
                      placeholder="e.g., q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
                      className="w-full px-3 py-2 border border-purple-300 rounded-lg bg-white text-sm focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                    />
                    <p className="text-xs text-slate-500 mt-1">
                      Enter comma-separated module names. Common modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, fc1, fc2
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Full fine-tuning warning */}
      {config.train_type === 'full' && (
        <div className="bg-amber-50 rounded-lg p-4 border border-amber-200">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
            <div>
              <h4 className="font-medium text-amber-800">Full Fine-tuning</h4>
              <p className="text-sm text-amber-700 mt-1">
                Trains all parameters. Requires 4-8x more GPU memory than LoRA. 
                Use only for small models or with ample resources.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Performance Optimization */}
      <div className="bg-gradient-to-r from-emerald-50 to-teal-50 rounded-lg p-4 border border-emerald-200">
        <h4 className="font-medium text-slate-900 mb-4 flex items-center gap-2">
          <span className="px-2 py-1 bg-emerald-100 text-emerald-700 rounded text-xs font-semibold">⚡</span>
          Performance Optimization
        </h4>
        
        {/* GPU Architecture Info Banner */}
        {capabilitiesLoaded && capabilities.gpu_name && (
          <div className="col-span-full mb-2 p-3 bg-slate-100 rounded-lg border border-slate-200">
            <div className="flex items-center gap-2 text-sm">
              <span className="font-medium text-slate-700">Detected GPU:</span>
              <span className="text-slate-900">{capabilities.gpu_name}</span>
              <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                capabilities.is_hopper ? 'bg-purple-100 text-purple-700' : 
                capabilities.is_ampere_or_newer ? 'bg-blue-100 text-blue-700' : 'bg-slate-200 text-slate-600'
              }`}>
                {capabilities.gpu_architecture.toUpperCase()}
              </span>
              {capabilities.is_hopper && (
                <span className="px-2 py-0.5 bg-emerald-100 text-emerald-700 rounded text-xs font-medium">
                  FA3 Supported
                </span>
              )}
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Attention Implementation - DYNAMIC based on capabilities */}
          <div>
            <div className="flex items-center gap-1 mb-2">
              <label className="text-sm font-medium text-slate-700">{OPTIMIZATION_CONFIG.attn_impl.label}</label>
              <Tooltip text={OPTIMIZATION_CONFIG.attn_impl.tooltip} />
            </div>
            <div className="space-y-2">
              {capabilities.attention_implementations.map((opt) => (
                <label key={opt.value ?? 'auto'} 
                  className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                    config.attn_impl === opt.value 
                      ? 'border-emerald-500 bg-emerald-50' 
                      : 'border-slate-200 hover:border-emerald-300 bg-white'
                  }`}>
                  <input type="radio" name="attn_impl" 
                    checked={config.attn_impl === opt.value}
                    onChange={() => setConfig(p => ({ ...p, attn_impl: opt.value }))}
                    className="mt-1 text-emerald-600 focus:ring-emerald-500" />
                  <div>
                    <span className="font-medium text-sm text-slate-900">{opt.label}</span>
                    <p className="text-xs text-slate-500">{opt.desc}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* DeepSpeed - DYNAMIC based on capabilities */}
          <div>
            <div className="flex items-center gap-1 mb-2">
              <label className="text-sm font-medium text-slate-700">{OPTIMIZATION_CONFIG.deepspeed.label}</label>
              <Tooltip text={OPTIMIZATION_CONFIG.deepspeed.tooltip} />
            </div>
            <div className="space-y-2">
              {capabilities.deepspeed_options.map((opt) => (
                <label key={opt.value ?? 'disabled'} 
                  className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                    config.deepspeed === opt.value 
                      ? 'border-emerald-500 bg-emerald-50' 
                      : 'border-slate-200 hover:border-emerald-300 bg-white'
                  } ${config.fsdp ? 'opacity-50 cursor-not-allowed' : ''}`}>
                  <input type="radio" name="deepspeed" 
                    checked={config.deepspeed === opt.value}
                    disabled={!!config.fsdp && opt.value !== null}
                    onChange={() => setConfig(p => ({ ...p, deepspeed: opt.value, fsdp: null }))}
                    className="mt-1 text-emerald-600 focus:ring-emerald-500" />
                  <div>
                    <span className="font-medium text-sm text-slate-900">{opt.label}</span>
                    <p className="text-xs text-slate-500">{opt.desc}</p>
                  </div>
                </label>
              ))}
            </div>
            {config.fsdp && (
              <p className="text-xs text-amber-600 mt-2">⚠️ Disabled: FSDP is active. DeepSpeed and FSDP cannot be used together.</p>
            )}
          </div>

          {/* FSDP Selection - DYNAMIC based on capabilities */}
          <div>
            <div className="flex items-center gap-1 mb-2">
              <label className="text-sm font-medium text-slate-700">{OPTIMIZATION_CONFIG.fsdp.label}</label>
              <Tooltip text={OPTIMIZATION_CONFIG.fsdp.tooltip} />
            </div>
            <div className="space-y-2">
              {capabilities.fsdp_options.map((opt) => (
                <label key={opt.value ?? 'disabled'} 
                  className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                    config.fsdp === opt.value 
                      ? 'border-emerald-500 bg-emerald-50' 
                      : 'border-slate-200 hover:border-emerald-300 bg-white'
                  } ${config.deepspeed ? 'opacity-50 cursor-not-allowed' : ''}`}>
                  <input type="radio" name="fsdp" 
                    checked={config.fsdp === opt.value}
                    disabled={!!config.deepspeed && opt.value !== null}
                    onChange={() => setConfig(p => ({ ...p, fsdp: opt.value, deepspeed: null }))}
                    className="mt-1 text-emerald-600 focus:ring-emerald-500" />
                  <div>
                    <span className="font-medium text-sm text-slate-900">{opt.label}</span>
                    <p className="text-xs text-slate-500">{opt.desc}</p>
                  </div>
                </label>
              ))}
            </div>
            {config.deepspeed && (
              <p className="text-xs text-amber-600 mt-2">⚠️ Disabled: DeepSpeed is active. DeepSpeed and FSDP cannot be used together.</p>
            )}
          </div>
        </div>

        {/* Gradient Checkpointing Toggle */}
        <div className="mt-4 pt-4 border-t border-emerald-200">
          <label className="flex items-center gap-3 cursor-pointer">
            <input type="checkbox" 
              checked={config.gradient_checkpointing}
              onChange={(e) => setConfig(p => ({ ...p, gradient_checkpointing: e.target.checked }))}
              className="w-5 h-5 text-emerald-600 rounded focus:ring-emerald-500" />
            <div className="flex items-center gap-2">
              <span className="font-medium text-sm text-slate-700">Gradient Checkpointing</span>
              <Tooltip text="Trades compute for memory. Re-computes activations during backward pass instead of storing them. Recommended for large models." />
            </div>
          </label>
          <p className="text-xs text-slate-500 ml-8 mt-1">Saves ~30-50% GPU memory at ~20% speed cost. Recommended for most trainings.</p>
        </div>

        {/* Advanced Optimizations */}
        <div className="mt-4 pt-4 border-t border-emerald-200">
          <div className="flex items-center gap-2 mb-3">
            <span className="font-medium text-sm text-slate-700">Advanced Optimizations</span>
            <Tooltip text="Additional optimizations for faster training and better memory efficiency." />
          </div>
          <div className="space-y-3">
            {/* Liger Kernel - only show if available */}
            {capabilities.liger_kernel_available && (
              <label className={`flex items-center gap-3 cursor-pointer ${config.packing ? 'opacity-50' : ''}`}>
                <input type="checkbox" 
                  checked={config.use_liger_kernel}
                  disabled={config.packing}
                  onChange={(e) => setConfig(p => ({ ...p, use_liger_kernel: e.target.checked }))}
                  className="w-5 h-5 text-emerald-600 rounded focus:ring-emerald-500" />
                <div>
                  <span className="font-medium text-sm text-slate-700">Liger Kernel</span>
                  <p className="text-xs text-slate-500">Triton-based optimizations for faster forward/backward pass (up to 20% speedup).</p>
                  {config.packing && <p className="text-xs text-amber-600">⚠️ Disabled: Incompatible with Packing</p>}
                </div>
              </label>
            )}
            {/* Sequence Packing - only show if Flash Attention is available */}
            {(capabilities.attention_implementations.some(a => a.value === 'flash_attention_2' || a.value === 'flash_attention_3')) && (
              <label className="flex items-center gap-3 cursor-pointer">
                <input type="checkbox" 
                  checked={config.packing}
                  onChange={(e) => {
                    const newPacking = e.target.checked
                    // Find best available flash attention
                    const bestFA = capabilities.attention_implementations.find(a => a.value === 'flash_attention_3')?.value 
                      || capabilities.attention_implementations.find(a => a.value === 'flash_attention_2')?.value
                      || 'flash_attention_2'
                    setConfig(p => ({ 
                      ...p, 
                      packing: newPacking,
                      // Auto-set Flash Attention if packing is enabled and no flash attn selected
                      attn_impl: newPacking && !['flash_attn', 'flash_attention_2', 'flash_attention_3'].includes(p.attn_impl || '') 
                        ? bestFA : p.attn_impl,
                      // Disable Liger kernel if packing is enabled (incompatible)
                      use_liger_kernel: newPacking ? false : p.use_liger_kernel
                    }))
                  }}
                  className="w-5 h-5 text-emerald-600 rounded focus:ring-emerald-500" />
                <div>
                  <span className="font-medium text-sm text-slate-700">Sequence Packing</span>
                  <p className="text-xs text-slate-500">Combine multiple short sequences to reduce padding waste. Requires Flash Attention.</p>
                  {config.packing && !['flash_attn', 'flash_attention_2', 'flash_attention_3'].includes(config.attn_impl || '') && (
                    <p className="text-xs text-amber-600">⚠️ Flash Attention will be auto-selected</p>
                  )}
                </div>
              </label>
            )}
            {/* Show message if no advanced optimizations available */}
            {!capabilities.liger_kernel_available && !capabilities.attention_implementations.some(a => a.value === 'flash_attention_2' || a.value === 'flash_attention_3') && (
              <p className="text-sm text-slate-500 italic">No advanced optimizations available. Install Flash Attention or Liger Kernel for additional options.</p>
            )}
          </div>
        </div>

        {/* Learning Rate Scheduler */}
        <div className="mt-4 pt-4 border-t border-emerald-200">
          <div className="flex items-center gap-2 mb-3">
            <span className="font-medium text-sm text-slate-700">Learning Rate Scheduler</span>
            <Tooltip text="Controls how the learning rate changes during training." />
          </div>
          <select 
            value={config.lr_scheduler_type}
            onChange={(e) => setConfig(p => ({ ...p, lr_scheduler_type: e.target.value }))}
            className="w-full px-3 py-2 border border-slate-300 rounded-lg bg-white text-sm focus:ring-2 focus:ring-emerald-500">
            <option value="cosine">Cosine (Recommended)</option>
            <option value="linear">Linear</option>
            <option value="constant">Constant</option>
            <option value="constant_with_warmup">Constant with Warmup</option>
            <option value="cosine_with_restarts">Cosine with Restarts</option>
            <option value="polynomial">Polynomial</option>
            <option value="cosine_with_min_lr">Cosine with Min LR</option>
          </select>
        </div>

        {/* GPU Selection */}
        <div className="mt-4 pt-4 border-t border-emerald-200">
          <div className="flex items-center gap-2 mb-3">
            <span className="font-medium text-sm text-slate-700">GPU Selection</span>
            <Tooltip text="Select which GPUs to use for training. You can use all GPUs or select specific ones." />
            {availableGpus.length > 0 && (
              <span className="text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded-full">
                {availableGpus.length} GPU{availableGpus.length > 1 ? 's' : ''} available
              </span>
            )}
          </div>
          
          {availableGpus.length === 0 ? (
            <div className="p-4 bg-amber-50 rounded-lg border border-amber-200">
              <p className="text-sm text-amber-700">No GPUs detected. Training requires at least one GPU.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {/* Auto mode */}
              <label className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                gpuMode === 'auto' ? 'border-emerald-500 bg-emerald-50' : 'border-slate-200 hover:border-emerald-300 bg-white'
              }`}>
                <input type="radio" name="gpu_mode" 
                  checked={gpuMode === 'auto'}
                  onChange={() => handleGpuModeChange('auto')}
                  className="mt-1 text-emerald-600 focus:ring-emerald-500" />
                <div>
                  <span className="font-medium text-sm text-slate-900">Auto (Use All {availableGpus.length} GPUs)</span>
                  <p className="text-xs text-slate-500">Use all available GPUs for maximum training speed.</p>
                </div>
              </label>
              
              {/* Select specific GPUs mode */}
              <label className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                gpuMode === 'select' ? 'border-emerald-500 bg-emerald-50' : 'border-slate-200 hover:border-emerald-300 bg-white'
              }`}>
                <input type="radio" name="gpu_mode" 
                  checked={gpuMode === 'select'}
                  onChange={() => handleGpuModeChange('select')}
                  className="mt-1 text-emerald-600 focus:ring-emerald-500" />
                <div className="flex-1">
                  <span className="font-medium text-sm text-slate-900">Select Specific GPUs</span>
                  <p className="text-xs text-slate-500 mb-3">Choose which GPUs to use for training.</p>
                  
                  {gpuMode === 'select' && (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {availableGpus.map((gpu) => (
                        <label key={gpu.id} 
                          className={`flex items-center gap-3 p-2 rounded-lg border cursor-pointer transition-all ${
                            config.gpu_ids?.includes(gpu.id) 
                              ? 'border-emerald-400 bg-emerald-50' 
                              : 'border-slate-200 hover:border-emerald-300 bg-white'
                          }`}>
                          <input 
                            type="checkbox"
                            checked={config.gpu_ids?.includes(gpu.id) || false}
                            onChange={() => handleGpuToggle(gpu.id)}
                            className="w-4 h-4 text-emerald-600 rounded focus:ring-emerald-500"
                          />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-sm text-slate-900">GPU {gpu.id}</span>
                              {gpu.memory_total_gb && (
                                <span className="text-xs text-slate-500">{gpu.memory_total_gb}GB</span>
                              )}
                            </div>
                            <p className="text-xs text-slate-500 truncate">{gpu.name}</p>
                          </div>
                        </label>
                      ))}
                    </div>
                  )}
                </div>
              </label>
              
              {/* Show selected GPUs summary */}
              {gpuMode === 'select' && config.gpu_ids && config.gpu_ids.length > 0 && (
                <div className="p-2 bg-emerald-50 rounded-lg">
                  <p className="text-xs text-emerald-700">
                    <strong>Selected:</strong> GPU {config.gpu_ids.join(', ')} ({config.gpu_ids.length} GPU{config.gpu_ids.length > 1 ? 's' : ''})
                  </p>
                </div>
              )}
            </div>
          )}
          
          <p className="text-xs text-slate-500 mt-2">💡 Multi-GPU training is enabled automatically when more than 1 GPU is selected.</p>
        </div>

      </div>
    </div>
  )
}
