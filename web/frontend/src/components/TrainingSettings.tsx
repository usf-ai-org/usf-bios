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
  train_type: 'full' | 'lora' | 'qlora' | 'adalora'
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
}

interface Props {
  config: TrainingConfig
  setConfig: (fn: (prev: TrainingConfig) => TrainingConfig) => void
}

export default function TrainingSettingsStep({ config, setConfig }: Props) {
  const typeConfig = PARAM_CONFIG[config.train_type] || {}
  const commonConfig = PARAM_CONFIG.common
  
  // State for custom target modules input
  const [isCustomModules, setIsCustomModules] = useState(false)
  const [customModulesInput, setCustomModulesInput] = useState('')
  
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
      
      {/* Training Type */}
      <div>
        <div className="flex items-center gap-1 mb-2">
          <label className="text-sm font-medium text-slate-700">Training Type</label>
          <Tooltip text="LoRA: Efficient adapters. QLoRA: 4-bit quantized for less memory. AdaLoRA: Adaptive rank. Full: All parameters (most memory)." />
        </div>
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
            options={commonConfig.max_length.options} label={commonConfig.max_length.label}
            tooltip={commonConfig.max_length.tooltip} effect={commonConfig.max_length.effect} />
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
    </div>
  )
}
