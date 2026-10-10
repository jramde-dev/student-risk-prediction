import type { BatchResponse, Prediction, Student } from './types'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/** FastAPI renvoie `detail` en chaîne (503, 409…) ou en liste d'erreurs de validation (422). */
function formatDetail(detail: unknown): string {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail
      .map((d: { loc?: Array<string | number>; msg?: string }) => `${d.loc?.slice(1).join('.') ?? ''} : ${d.msg ?? ''}`)
      .join(' ; ')
  }
  return 'Erreur inattendue du serveur.'
}

export async function predictBatch(
  students: Student[],
  threshold: number,
  signal?: AbortSignal,
): Promise<Prediction[]> {
  let res: Response
  try {
    res = await fetch(`${API_URL}/predict/batch?threshold=${threshold}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ students }),
      signal,
    })
  } catch {
    throw new ApiError(0, `API injoignable (${API_URL}). Le backend est-il démarré ?`)
  }
  if (!res.ok) {
    const body = (await res.json().catch(() => null)) as { detail?: unknown } | null
    throw new ApiError(res.status, formatDetail(body?.detail))
  }
  return ((await res.json()) as BatchResponse).predictions
}
