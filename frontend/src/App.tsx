import { useState, type SubmitEvent } from 'react'
import { predictBatch } from './api'
import { ResultCard } from './components/ResultCard'
import { StudentFields } from './components/StudentFields'
import { EMPTY_DRAFT, toStudent, validateStudent, type StudentDraft } from './studentDraft'
import type { Prediction } from './types'

interface Entry {
  id: string
  draft: StudentDraft
}

const newEntry = (): Entry => ({ id: crypto.randomUUID(), draft: { ...EMPTY_DRAFT } })

export default function App() {
  const [entries, setEntries] = useState<Entry[]>([newEntry()])
  const [threshold, setThreshold] = useState('0.5')
  const [results, setResults] = useState<Record<string, Prediction>>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Toute modification invalide les résultats affichés (ils ne correspondraient plus à la saisie).
  const edit = (update: (prev: Entry[]) => Entry[]) => {
    setEntries(update)
    setResults({})
  }

  async function handleSubmit(e: SubmitEvent<HTMLFormElement>) {
    e.preventDefault()
    setError(null)

    const students = entries.map((en) => toStudent(en.draft))
    for (const [i, s] of students.entries()) {
      const problem = validateStudent(s)
      if (problem) return setError(`Étudiant ${i + 1} : ${problem}`)
    }

    setLoading(true)
    try {
      const predictions = await predictBatch(students, Number(threshold))
      setResults(Object.fromEntries(entries.map((en, i) => [en.id, predictions[i]])))
    } catch (err) {
      setResults({})
      setError(err instanceof Error ? err.message : 'Erreur inconnue.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main>
      <h1>Repérage des étudiants à risque</h1>
      <p className="hint">Outil d'aide au repérage, pas un diagnostic.</p>

      <form onSubmit={handleSubmit}>
        {entries.map((en, i) => (
          <StudentFields
            key={en.id}
            index={i}
            draft={en.draft}
            onChange={(draft) => edit((prev) => prev.map((x) => (x.id === en.id ? { ...x, draft } : x)))}
            onRemove={entries.length > 1 ? () => edit((prev) => prev.filter((x) => x.id !== en.id)) : undefined}
          >
            {results[en.id] && <ResultCard prediction={results[en.id]} />}
          </StudentFields>
        ))}

        <div className="toolbar">
          <button type="button" onClick={() => edit((prev) => [...prev, newEntry()])} disabled={entries.length >= 50}>
            + Ajouter un étudiant
          </button>

          <label className="inline">
            Seuil de décision
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              required
              value={threshold}
              onChange={(e) => setThreshold(e.target.value)}
            />
          </label>

          <button type="submit" className="primary" disabled={loading}>
            {loading ? 'Calcul…' : 'Prédire'}
          </button>
        </div>

        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </form>
    </main>
  )
}
