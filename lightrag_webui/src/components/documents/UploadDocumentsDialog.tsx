import { useState, useCallback, useEffect, useRef } from 'react'
import { FileRejection } from 'react-dropzone'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter
} from '@/components/ui/Dialog'
import FileUploader from '@/components/ui/FileUploader'
import { toast } from 'sonner'
import { supportedFileTypes } from '@/lib/constants'
import {
  deriveUploaderInputs,
  flattenAcceptExtensions,
  formatFileTypesLabel,
  normalizeSupportedFileTypes,
  type FileTypesState
} from '@/lib/fileTypes'
import { errorMessage } from '@/lib/utils'
import { getSupportedFileTypes, uploadDocument } from '@/api/lightrag'

import { UploadIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

type LargeIngestionGuardDetail = {
  estimated_chunks?: number
  max_chunks?: number
  confirm_large_ingestion_required?: boolean
}

function parseLargeIngestionGuard(err: unknown): LargeIngestionGuardDetail | null {
  if (!err || typeof err !== 'object' || !('response' in err)) {
    return null
  }
  const response = (err as { response?: { status?: number; data?: { detail?: unknown } } }).response
  if (response?.status !== 400 || !response.data?.detail || typeof response.data.detail !== 'object') {
    return null
  }
  const detail = response.data.detail as LargeIngestionGuardDetail
  return detail.confirm_large_ingestion_required ? detail : null
}

interface UploadDocumentsDialogProps {
  onDocumentsUploaded?: () => Promise<void>
  /**
   * Fired once per batch as soon as the first file is accepted by the server.
   * Lets the parent start its activity probe as early as possible (rather
   * than waiting for the whole sequential batch to finish).
   */
  onUploadBatchAccepted?: () => void
}

export default function UploadDocumentsDialog({
  onDocumentsUploaded,
  onUploadBatchAccepted
}: UploadDocumentsDialogProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [progresses, setProgresses] = useState<Record<string, number>>({})
  const [fileErrors, setFileErrors] = useState<Record<string, string>>({})
  const [fileTypes, setFileTypes] = useState<FileTypesState>({ status: 'idle' })
  const [showLargeIngestionConfirm, setShowLargeIngestionConfirm] = useState(false)
  const [largeIngestionPrompt, setLargeIngestionPrompt] = useState({
    estimated: '?',
    max: '?'
  })
  const largeIngestionResolverRef = useRef<((proceed: boolean) => void) | null>(null)

  // Fetch the live allowlist + engine capability matrix while the dialog is
  // open. `loading` is entered synchronously in onOpenChange (not here) so
  // the very first open render already has the uploader disabled.
  useEffect(() => {
    if (!open) return
    const controller = new AbortController()
    getSupportedFileTypes(controller.signal)
      .then((res) => {
        if (controller.signal.aborted) return
        const data = normalizeSupportedFileTypes(res)
        setFileTypes(data ? { status: 'ready', data } : { status: 'fallback' })
      })
      .catch((err) => {
        if (controller.signal.aborted) return
        // Old backend (404) or transient failure: fall back to the static
        // allowlist and let the server judge hinted filenames.
        console.warn('Failed to fetch supported file types:', errorMessage(err))
        setFileTypes({ status: 'fallback' })
      })
    return () => controller.abort()
  }, [open])

  const requestLargeIngestionConfirm = useCallback((estimated: string, max: string): Promise<boolean> => {
    setLargeIngestionPrompt({ estimated, max })
    setShowLargeIngestionConfirm(true)
    return new Promise((resolve) => {
      largeIngestionResolverRef.current = resolve
    })
  }, [])

  const resolveLargeIngestionConfirm = useCallback((proceed: boolean) => {
    setShowLargeIngestionConfirm(false)
    const resolver = largeIngestionResolverRef.current
    largeIngestionResolverRef.current = null
    resolver?.(proceed)
  }, [])

  const handleRejectedFiles = useCallback(
    (rejectedFiles: FileRejection[]) => {
      // Process rejected files and add them to fileErrors
      rejectedFiles.forEach(({ file, errors }) => {
        // Get the first error message
        let errorMsg = errors[0]?.message || t('documentPanel.uploadDocuments.fileUploader.fileRejected', { name: file.name })

        // Simplify error message for unsupported file types
        if (errorMsg.includes('file-invalid-type')) {
          errorMsg = t('documentPanel.uploadDocuments.fileUploader.unsupportedType')
        }

        // Set progress to 100% to display error message
        setProgresses((pre) => ({
          ...pre,
          [file.name]: 100
        }))

        // Add error message to fileErrors
        setFileErrors(prev => ({
          ...prev,
          [file.name]: errorMsg
        }))
      })
    },
    [setProgresses, setFileErrors, t]
  )

  const handleDocumentsUpload = useCallback(
    async (filesToUpload: File[]) => {
      setIsUploading(true)
      let hasSuccessfulUpload = false

      // Only clear errors for files that are being uploaded, keep errors for rejected files
      setFileErrors(prev => {
        const newErrors = { ...prev };
        filesToUpload.forEach(file => {
          delete newErrors[file.name];
        });
        return newErrors;
      });

      // Show uploading toast
      const toastId = toast.loading(t('documentPanel.uploadDocuments.batch.uploading'))

      try {
        // Track errors locally to ensure we have the final state
        const uploadErrors: Record<string, string> = {}
        let batchProbeTriggered = false

        // Create a collator that supports Chinese sorting
        const collator = new Intl.Collator(['zh-CN', 'en'], {
          sensitivity: 'accent',  // consider basic characters, accents, and case
          numeric: true           // enable numeric sorting, e.g., "File 10" will be after "File 2"
        });
        const sortedFiles = [...filesToUpload].sort((a, b) =>
          collator.compare(a.name, b.name)
        );

        // Upload files in sequence, not parallel
        for (const file of sortedFiles) {
          try {
            // Initialize upload progress
            setProgresses((pre) => ({
              ...pre,
              [file.name]: 0
            }))

            const uploadWithProgress = (confirmLargeIngestion: boolean) =>
              uploadDocument(
                file,
                (percentCompleted: number) => {
                  console.debug(
                    t('documentPanel.uploadDocuments.single.uploading', {
                      name: file.name,
                      percent: percentCompleted
                    })
                  )
                  setProgresses((pre) => ({
                    ...pre,
                    [file.name]: percentCompleted
                  }))
                },
                { confirmLargeIngestion }
              )

            let result
            try {
              result = await uploadWithProgress(false)
            } catch (firstErr) {
              const guard = parseLargeIngestionGuard(firstErr)
              const shouldProceed = guard
                ? await requestLargeIngestionConfirm(
                  String(guard.estimated_chunks ?? '?'),
                  String(guard.max_chunks ?? '?')
                )
                : false

              if (shouldProceed) {
                result = await uploadWithProgress(true)
              } else {
                throw firstErr
              }
            }

            if (result.status !== 'success') {
              uploadErrors[file.name] = result.message
              setFileErrors(prev => ({
                ...prev,
                [file.name]: result.message
              }))
            } else {
              // Mark that we had at least one successful upload
              hasSuccessfulUpload = true
              if (!batchProbeTriggered) {
                batchProbeTriggered = true
                onUploadBatchAccepted?.()
              }
            }
          } catch (err) {
            console.error(`Upload failed for ${file.name}:`, err)

            // Handle HTTP errors, including 400 errors
            let errorMsg = errorMessage(err)
            const duplicateFileMsg = t('documentPanel.uploadDocuments.fileUploader.duplicateFile')

            // If it's an axios error with response data, try to extract more detailed error info
            if (err && typeof err === 'object' && 'response' in err) {
              const axiosError = err as { response?: { status: number, data?: { detail?: string } } }
              const status = axiosError.response?.status
              const detail = axiosError.response?.data?.detail
              if (status === 409) {
                // Server now rejects same-name uploads with HTTP 409 instead of
                // returning a 200 ``status="duplicated"`` payload.  Map the most
                // common cases (existing record / file in INPUT dir) back to the
                // dedicated "duplicate file" UI affordance, and surface other
                // 409 reasons (pipeline busy / scanning) verbatim from the
                // server detail so users can tell why they were rejected.
                if (
                  typeof detail === 'string' &&
                  (/already contains/i.test(detail) || /Status:/i.test(detail))
                ) {
                  errorMsg = duplicateFileMsg
                } else {
                  errorMsg = detail || errorMsg
                }
              } else if (status === 400) {
                if (typeof detail === 'object' && detail !== null) {
                  const guard = detail as LargeIngestionGuardDetail
                  if (guard.confirm_large_ingestion_required) {
                    errorMsg = t(
                      'documentPanel.uploadDocuments.fileUploader.largeIngestionConfirm',
                      {
                        estimated: guard.estimated_chunks ?? '?',
                        max: guard.max_chunks ?? '?'
                      }
                    )
                  } else {
                    errorMsg =
                      (detail as { message?: string }).message || errorMsg
                  }
                } else {
                  errorMsg = (detail as string) || errorMsg
                }
              } else if (status === 413 || status === 429 || status === 503) {
                // 413 body/file too large, 429 pipeline at capacity
                // (MAX_PENDING_DOCUMENTS — the detail carries how many
                // documents are active, how many were requested, the capacity and
                // a retry hint), 503 document storage unavailable. Each detail is
                // written to be shown to a user verbatim.
                errorMsg = detail || errorMsg
              }

              // Set progress to 100% to display error message
              setProgresses((pre) => ({
                ...pre,
                [file.name]: 100
              }))
            }

            // Record error message in both local tracking and state
            uploadErrors[file.name] = errorMsg
            setFileErrors(prev => ({
              ...prev,
              [file.name]: errorMsg
            }))
          }
        }

        // Check if any files failed to upload using our local tracking
        const hasErrors = Object.keys(uploadErrors).length > 0

        // Update toast status
        if (hasErrors) {
          toast.error(t('documentPanel.uploadDocuments.batch.error'), { id: toastId })
        } else {
          toast.success(t('documentPanel.uploadDocuments.batch.success'), { id: toastId })
        }

        // Only update if at least one file was uploaded successfully
        if (hasSuccessfulUpload) {
          // Refresh document list
          if (onDocumentsUploaded) {
            onDocumentsUploaded().catch(err => {
              console.error('Error refreshing documents:', err)
            })
          }
        }
      } catch (err) {
        console.error('Unexpected error during upload:', err)
        toast.error(t('documentPanel.uploadDocuments.generalError', { error: errorMessage(err) }), { id: toastId })
      } finally {
        setIsUploading(false)
      }
    },
    [setIsUploading, setProgresses, setFileErrors, t, onDocumentsUploaded, onUploadBatchAccepted, requestLargeIngestionConfirm]
  )

  const uploaderInputs = deriveUploaderInputs(fileTypes)

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (isUploading) {
          return
        }
        if (nextOpen) {
          // Enter loading synchronously so the first open render already has
          // the uploader disabled — no window where a hinted file could start
          // uploading before the capability matrix arrives.
          setFileTypes({ status: 'loading' })
        } else {
          setProgresses({})
          setFileErrors({})
          setFileTypes({ status: 'idle' })
        }
        setOpen(nextOpen)
      }}
    >
      <DialogTrigger asChild>
        <Button variant="default" side="bottom" tooltip={t('documentPanel.uploadDocuments.tooltip')} size="sm">
          <UploadIcon /> {t('documentPanel.uploadDocuments.button')}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>{t('documentPanel.uploadDocuments.title')}</DialogTitle>
          <DialogDescription>
            {t('documentPanel.uploadDocuments.description')}
          </DialogDescription>
        </DialogHeader>
        <FileUploader
          maxFileCount={Infinity}
          maxSize={200 * 1024 * 1024}
          description={t('documentPanel.uploadDocuments.fileTypes', {
            types: formatFileTypesLabel(
              uploaderInputs.acceptedExtensions ?? flattenAcceptExtensions(supportedFileTypes)
            )
          })}
          onUpload={handleDocumentsUpload}
          onReject={handleRejectedFiles}
          progresses={progresses}
          fileErrors={fileErrors}
          disabled={isUploading || uploaderInputs.disabled}
          acceptedExtensions={uploaderInputs.acceptedExtensions}
          engineCapabilities={uploaderInputs.engineCapabilities}
        />
      </DialogContent>

      <Dialog open={showLargeIngestionConfirm} onOpenChange={(open) => {
        if (!open) {
          resolveLargeIngestionConfirm(false)
        }
      }}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{t('documentPanel.uploadDocuments.largeIngestionTitle')}</DialogTitle>
            <DialogDescription>
              {t('documentPanel.uploadDocuments.fileUploader.largeIngestionConfirm', {
                estimated: largeIngestionPrompt.estimated,
                max: largeIngestionPrompt.max
              })}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => resolveLargeIngestionConfirm(false)}>
              {t('common.cancel')}
            </Button>
            <Button onClick={() => resolveLargeIngestionConfirm(true)}>
              {t('common.confirm')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Dialog>
  )
}
