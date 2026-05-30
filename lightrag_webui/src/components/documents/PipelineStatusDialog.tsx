import { useState, useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { AlignLeft, AlignCenter, AlignRight } from 'lucide-react'

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription
} from '@/components/ui/Dialog'
import Button from '@/components/ui/Button'
import {
  getPipelineStatus,
  cancelPipeline,
  stopIngestionSafely,
  startOverIngestion,
  PipelineStatusResponse
} from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { cn } from '@/lib/utils'

type DialogPosition = 'left' | 'center' | 'right'

interface PipelineStatusDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export default function PipelineStatusDialog({
  open,
  onOpenChange
}: PipelineStatusDialogProps) {
  const { t } = useTranslation()
  const [status, setStatus] = useState<PipelineStatusResponse | null>(null)
  const [position, setPosition] = useState<DialogPosition>('center')
  const [isUserScrolled, setIsUserScrolled] = useState(false)
  const [showCancelConfirm, setShowCancelConfirm] = useState(false)
  const [showPauseConfirm, setShowPauseConfirm] = useState(false)
  const historyRef = useRef<HTMLDivElement>(null)

  const activeJobId =
    status?.stopped_job_id ??
    (status?.active_track_ids?.length
      ? status.active_track_ids[status.active_track_ids.length - 1]
      : null)

  // Reset UI state whenever the controlling open prop changes.
  useEffect(() => {
    if (open) {
      // Resetting local dialog UI when the controlling prop changes is intentional.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setPosition('center')
      setIsUserScrolled(false)
      return
    }

    setShowCancelConfirm(false)
    setShowPauseConfirm(false)
  }, [open])

  // Handle scroll position
  useEffect(() => {
    const container = historyRef.current
    if (!container || isUserScrolled) return

    container.scrollTop = container.scrollHeight
  }, [status?.history_messages, isUserScrolled])

  const handleScroll = () => {
    const container = historyRef.current
    if (!container) return

    const isAtBottom = Math.abs(
      (container.scrollHeight - container.scrollTop) - container.clientHeight
    ) < 1

    if (isAtBottom) {
      setIsUserScrolled(false)
    } else {
      setIsUserScrolled(true)
    }
  }

  // Refresh status every 2 seconds
  useEffect(() => {
    if (!open) return

    const fetchStatus = async () => {
      try {
        const data = await getPipelineStatus()
        setStatus(data)
      } catch (err) {
        toast.error(t('documentPanel.pipelineStatus.errors.fetchFailed', { error: errorMessage(err) }))
      }
    }

    fetchStatus()
    const interval = setInterval(fetchStatus, 2000)
    return () => clearInterval(interval)
  }, [open, t])

  // Handle cancel pipeline confirmation
  const handleConfirmCancel = async () => {
    setShowCancelConfirm(false)
    try {
      const result = await cancelPipeline()
      if (result.status === 'cancellation_requested') {
        toast.success(t('documentPanel.pipelineStatus.cancelSuccess'))
      } else if (result.status === 'not_busy') {
        toast.info(t('documentPanel.pipelineStatus.cancelNotBusy'))
      }
    } catch (err) {
      toast.error(t('documentPanel.pipelineStatus.cancelFailed', { error: errorMessage(err) }))
    }
  }

  const handleConfirmStop = async () => {
    setShowPauseConfirm(false)
    if (!activeJobId) {
      toast.error(t('documentPanel.pipelineStatus.pauseNoJob'))
      return
    }
    try {
      const result = await stopIngestionSafely(activeJobId)
      if (result.status === 'stop_requested' || result.status === 'stopped') {
        toast.success(t('documentPanel.pipelineStatus.pauseSuccess'))
      } else if (result.status === 'already_stopped') {
        toast.info(t('documentPanel.pipelineStatus.pauseAlready'))
      } else {
        toast.info(result.message)
      }
    } catch (err) {
      toast.error(t('documentPanel.pipelineStatus.pauseFailed', { error: errorMessage(err) }))
    }
  }

  const handleStartOver = async () => {
    if (!activeJobId) {
      toast.error(t('documentPanel.pipelineStatus.pauseNoJob'))
      return
    }
    try {
      const result = await startOverIngestion(activeJobId)
      if (result.status === 'start_over_started') {
        toast.success(t('documentPanel.pipelineStatus.resumeSuccess'))
      } else if (result.status === 'already_running') {
        toast.info(t('documentPanel.pipelineStatus.resumeQueued'))
      } else {
        toast.info(result.message)
      }
    } catch (err) {
      toast.error(t('documentPanel.pipelineStatus.resumeFailed', { error: errorMessage(err) }))
    }
  }

  const canCancel = status?.busy === true && !status?.cancellation_requested
  const canStopSafely =
    Boolean(activeJobId) &&
    status?.busy === true &&
    !status?.cancellation_requested &&
    !status?.stop_requested &&
    !status?.stopped
  const canStartOver =
    Boolean(activeJobId) &&
    (status?.stopped === true || status?.stop_requested === true)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className={cn(
          'sm:max-w-[800px] transition-all duration-200 fixed',
          position === 'left' && '!left-[25%] !translate-x-[-50%] !mx-4',
          position === 'center' && '!left-1/2 !-translate-x-1/2',
          position === 'right' && '!left-[75%] !translate-x-[-50%] !mx-4'
        )}
      >
        <DialogDescription className="sr-only">
          {status?.job_name
            ? `${t('documentPanel.pipelineStatus.jobName')}: ${status.job_name}, ${t('documentPanel.pipelineStatus.progress')}: ${status.cur_batch}/${status.batchs}`
            : t('documentPanel.pipelineStatus.noActiveJob')
          }
        </DialogDescription>
        <DialogHeader className="flex flex-row items-center">
          <DialogTitle className="flex-1">
            {t('documentPanel.pipelineStatus.title')}
          </DialogTitle>

          {/* Position control buttons */}
          <div className="flex items-center gap-2 mr-8">
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                'h-6 w-6',
                position === 'left' && 'bg-zinc-200 text-zinc-800 hover:bg-zinc-300 dark:bg-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-600'
              )}
              onClick={() => setPosition('left')}
            >
              <AlignLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                'h-6 w-6',
                position === 'center' && 'bg-zinc-200 text-zinc-800 hover:bg-zinc-300 dark:bg-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-600'
              )}
              onClick={() => setPosition('center')}
            >
              <AlignCenter className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                'h-6 w-6',
                position === 'right' && 'bg-zinc-200 text-zinc-800 hover:bg-zinc-300 dark:bg-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-600'
              )}
              onClick={() => setPosition('right')}
            >
              <AlignRight className="h-4 w-4" />
            </Button>
          </div>
        </DialogHeader>

        {/* Status Content */}
        <div className="space-y-4 pt-4">
          {/* Pipeline Status - with cancel button */}
          <div className="flex flex-wrap items-center justify-between gap-4">
            {/* Left side: Status indicators */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="text-sm font-medium">{t('documentPanel.pipelineStatus.busy')}:</div>
                <div className={`h-2 w-2 rounded-full ${status?.busy ? 'bg-green-500' : 'bg-gray-300'}`} />
              </div>
              {/* Only show cancellation status when it's requested */}
              {status?.cancellation_requested && (
                <div className="flex items-center gap-2">
                  <div className="text-sm font-medium">{t('documentPanel.pipelineStatus.cancellationRequested')}:</div>
                  <div className="h-2 w-2 rounded-full bg-red-500" />
                </div>
              )}
              {status?.stop_requested && (
                <div className="flex items-center gap-2">
                  <div className="text-sm font-medium">{t('documentPanel.pipelineStatus.pauseRequested')}:</div>
                  <div className="h-2 w-2 rounded-full bg-amber-500" />
                </div>
              )}
              {status?.stopped && (
                <div className="flex items-center gap-2">
                  <div className="text-sm font-medium">{t('documentPanel.pipelineStatus.paused')}:</div>
                  <div className="h-2 w-2 rounded-full bg-amber-500" />
                </div>
              )}
            </div>

            <div className="flex flex-wrap gap-2">
              {status?.busy && (
                <Button
                  variant="outline"
                  size="sm"
                  disabled={!canStopSafely}
                  onClick={() => setShowPauseConfirm(true)}
                  title={t('documentPanel.pipelineStatus.pauseTooltip')}
                >
                  {t('documentPanel.pipelineStatus.pauseButton')}
                </Button>
              )}
              {(status?.busy || status?.stopped) && (
                <Button
                  variant="outline"
                  size="sm"
                  disabled={!canStartOver}
                  onClick={handleStartOver}
                  title={t('documentPanel.pipelineStatus.resumeTooltip')}
                >
                  {t('documentPanel.pipelineStatus.resumeButton')}
                </Button>
              )}
              {status?.busy && (
                <Button
                  variant="destructive"
                  size="sm"
                  disabled={!canCancel}
                  onClick={() => setShowCancelConfirm(true)}
                  title={
                    status?.cancellation_requested
                      ? t('documentPanel.pipelineStatus.cancelInProgress')
                      : t('documentPanel.pipelineStatus.cancelTooltip')
                  }
                >
                  {t('documentPanel.pipelineStatus.cancelButton')}
                </Button>
              )}
            </div>
          </div>
          {(status?.stopped || status?.stop_requested) && (
            <div className="text-xs text-muted-foreground">
              {t('documentPanel.pipelineStatus.resumeHint')}
            </div>
          )}

          {/* Job Information */}
          <div className="rounded-md border p-3 space-y-2">
            <div>{t('documentPanel.pipelineStatus.jobName')}: {status?.job_name || '-'}</div>
            <div className="flex justify-between">
              <span>{t('documentPanel.pipelineStatus.startTime')}: {status?.job_start
                ? new Date(status.job_start).toLocaleString(undefined, {
                  year: 'numeric',
                  month: 'numeric',
                  day: 'numeric',
                  hour: 'numeric',
                  minute: 'numeric',
                  second: 'numeric'
                })
                : '-'}</span>
              <span>{t('documentPanel.pipelineStatus.progress')}: {status ? `${status.cur_batch}/${status.batchs} ${t('documentPanel.pipelineStatus.unit')}` : '-'}</span>
            </div>
          </div>

          {/* History Messages */}
          <div className="space-y-2">
            <div className="text-sm font-medium">{t('documentPanel.pipelineStatus.pipelineMessages')}:</div>
            <div
              ref={historyRef}
              onScroll={handleScroll}
              className="font-mono text-xs rounded-md bg-zinc-800 text-zinc-100 p-3 overflow-y-auto overflow-x-hidden min-h-[7.5em] max-h-[40vh]"
            >
              {status?.history_messages?.length ? (
                status.history_messages.map((msg, idx) => (
                  <div key={idx} className="whitespace-pre-wrap break-all">{msg}</div>
                ))
              ) : '-'}
            </div>
          </div>
        </div>
      </DialogContent>

      <Dialog open={showPauseConfirm} onOpenChange={setShowPauseConfirm}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>{t('documentPanel.pipelineStatus.pauseConfirmTitle')}</DialogTitle>
            <DialogDescription>
              {t('documentPanel.pipelineStatus.pauseConfirmDescription')}
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end gap-3 mt-4">
            <Button variant="outline" onClick={() => setShowPauseConfirm(false)}>
              {t('common.cancel')}
            </Button>
            <Button onClick={handleConfirmStop}>
              {t('documentPanel.pipelineStatus.pauseConfirmButton')}
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Cancel Confirmation Dialog */}
      <Dialog open={showCancelConfirm} onOpenChange={setShowCancelConfirm}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>{t('documentPanel.pipelineStatus.cancelConfirmTitle')}</DialogTitle>
            <DialogDescription>
              {t('documentPanel.pipelineStatus.cancelConfirmDescription')}
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end gap-3 mt-4">
            <Button
              variant="outline"
              onClick={() => setShowCancelConfirm(false)}
            >
              {t('common.cancel')}
            </Button>
            <Button
              variant="destructive"
              onClick={handleConfirmCancel}
            >
              {t('documentPanel.pipelineStatus.cancelConfirmButton')}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </Dialog>
  )
}
