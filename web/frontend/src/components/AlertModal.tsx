'use client'

import { AlertCircle, CheckCircle, Info, XCircle, AlertTriangle, X } from 'lucide-react'

export type AlertType = 'error' | 'success' | 'warning' | 'info'

export interface AlertModalProps {
  isOpen: boolean
  onClose: () => void
  title?: string
  message: string
  type?: AlertType
  confirmText?: string
}

const alertConfig = {
  error: {
    icon: XCircle,
    iconBg: 'bg-red-100',
    iconColor: 'text-red-600',
    buttonBg: 'bg-red-600 hover:bg-red-700',
    titleColor: 'text-red-900',
  },
  success: {
    icon: CheckCircle,
    iconBg: 'bg-emerald-100',
    iconColor: 'text-emerald-600',
    buttonBg: 'bg-emerald-600 hover:bg-emerald-700',
    titleColor: 'text-emerald-900',
  },
  warning: {
    icon: AlertTriangle,
    iconBg: 'bg-amber-100',
    iconColor: 'text-amber-600',
    buttonBg: 'bg-amber-600 hover:bg-amber-700',
    titleColor: 'text-amber-900',
  },
  info: {
    icon: Info,
    iconBg: 'bg-blue-100',
    iconColor: 'text-blue-600',
    buttonBg: 'bg-blue-600 hover:bg-blue-700',
    titleColor: 'text-blue-900',
  },
}

const defaultTitles = {
  error: 'Error',
  success: 'Success',
  warning: 'Warning',
  info: 'Information',
}

export default function AlertModal({ 
  isOpen, 
  onClose, 
  title, 
  message, 
  type = 'info',
  confirmText = 'OK'
}: AlertModalProps) {
  if (!isOpen) return null

  const config = alertConfig[type]
  const Icon = config.icon
  const displayTitle = title || defaultTitles[type]

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[100] p-4">
      <div 
        className="bg-white rounded-xl shadow-2xl w-full max-w-md transform transition-all"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start gap-4 p-5 border-b border-slate-100">
          <div className={`w-10 h-10 ${config.iconBg} rounded-full flex items-center justify-center flex-shrink-0`}>
            <Icon className={`w-5 h-5 ${config.iconColor}`} />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className={`font-semibold text-lg ${config.titleColor}`}>{displayTitle}</h3>
          </div>
          <button 
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        
        {/* Body */}
        <div className="p-5">
          <p className="text-slate-700 whitespace-pre-wrap">{message}</p>
        </div>
        
        {/* Footer */}
        <div className="px-5 pb-5">
          <button
            onClick={onClose}
            className={`w-full px-4 py-2.5 ${config.buttonBg} text-white rounded-lg font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  )
}

// Confirm Modal Component
export interface ConfirmModalProps {
  isOpen: boolean
  onClose: () => void
  onConfirm: () => void
  title?: string
  message: string
  type?: 'danger' | 'warning' | 'info'
  confirmText?: string
  cancelText?: string
  isLoading?: boolean
}

const confirmConfig = {
  danger: {
    icon: AlertTriangle,
    iconBg: 'bg-red-100',
    iconColor: 'text-red-600',
    buttonBg: 'bg-red-600 hover:bg-red-700',
  },
  warning: {
    icon: AlertTriangle,
    iconBg: 'bg-amber-100',
    iconColor: 'text-amber-600',
    buttonBg: 'bg-amber-600 hover:bg-amber-700',
  },
  info: {
    icon: Info,
    iconBg: 'bg-blue-100',
    iconColor: 'text-blue-600',
    buttonBg: 'bg-blue-600 hover:bg-blue-700',
  },
}

export function ConfirmModal({
  isOpen,
  onClose,
  onConfirm,
  title = 'Confirm Action',
  message,
  type = 'info',
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  isLoading = false,
}: ConfirmModalProps) {
  if (!isOpen) return null

  const config = confirmConfig[type]
  const Icon = config.icon

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[100] p-4">
      <div 
        className="bg-white rounded-xl shadow-2xl w-full max-w-md"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start gap-4 p-5 border-b border-slate-100">
          <div className={`w-10 h-10 ${config.iconBg} rounded-full flex items-center justify-center flex-shrink-0`}>
            <Icon className={`w-5 h-5 ${config.iconColor}`} />
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-lg text-slate-900">{title}</h3>
          </div>
          <button 
            onClick={onClose}
            disabled={isLoading}
            className="text-slate-400 hover:text-slate-600 transition-colors disabled:opacity-50"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        
        {/* Body */}
        <div className="p-5">
          <p className="text-slate-700 whitespace-pre-wrap">{message}</p>
        </div>
        
        {/* Footer */}
        <div className="flex gap-3 px-5 pb-5">
          <button
            onClick={onClose}
            disabled={isLoading}
            className="flex-1 px-4 py-2.5 border border-slate-300 text-slate-700 rounded-lg font-medium hover:bg-slate-50 transition-colors disabled:opacity-50"
          >
            {cancelText}
          </button>
          <button
            onClick={onConfirm}
            disabled={isLoading}
            className={`flex-1 px-4 py-2.5 ${config.buttonBg} text-white rounded-lg font-medium transition-colors disabled:opacity-50 flex items-center justify-center gap-2`}
          >
            {isLoading && (
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            )}
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  )
}
