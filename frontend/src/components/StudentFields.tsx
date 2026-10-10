import { EDUCATION_LEVELS, GENDERS, NUMERIC_FIELDS } from '../constants'
import type { StudentDraft } from '../studentDraft'
import type { EducationLevel, Gender } from '../types'

interface Props {
  index: number
  draft: StudentDraft
  onChange: (draft: StudentDraft) => void
  onRemove?: () => void
  children?: React.ReactNode
}

export function StudentFields({ index, draft, onChange, onRemove, children }: Props) {
  return (
    <fieldset className="card">
      <legend>Étudiant {index + 1}</legend>

      <div className="grid">
        {NUMERIC_FIELDS.map((f) => (
          <label key={f.name}>
            {f.label}
            <input
              type="number"
              inputMode="decimal"
              required
              min={f.min}
              max={f.max}
              step={f.step}
              value={draft[f.name]}
              onChange={(e) => onChange({ ...draft, [f.name]: e.target.value })}
            />
          </label>
        ))}

        <label>
          Genre
          <select value={draft.Gender} onChange={(e) => onChange({ ...draft, Gender: e.target.value as Gender })}>
            {GENDERS.map((g) => (
              <option key={g.value} value={g.value}>
                {g.label}
              </option>
            ))}
          </select>
        </label>

        <label>
          Niveau d'éducation
          <select
            value={draft.Education_Level}
            onChange={(e) => onChange({ ...draft, Education_Level: e.target.value as EducationLevel })}
          >
            {EDUCATION_LEVELS.map((l) => (
              <option key={l.value} value={l.value}>
                {l.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      {children}

      {onRemove && (
        <button type="button" className="link" onClick={onRemove}>
          Retirer cet étudiant
        </button>
      )}
    </fieldset>
  )
}
