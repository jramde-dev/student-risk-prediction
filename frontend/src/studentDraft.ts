import type { NumericKey } from './constants'
import type { EducationLevel, Gender, Student } from './types'

/** État du formulaire : les champs numériques restent des chaînes (saisie vide possible). */
export type StudentDraft = Record<NumericKey, string> & {
  Gender: Gender
  Education_Level: EducationLevel
}

export const EMPTY_DRAFT: StudentDraft = {
  Age: '',
  Gender: 'Female',
  Education_Level: 'High School',
  Daily_Social_Media_Hours: '',
  Daily_AI_Tool_Usage_Hours: '',
  Sleep_Hours: '',
  Physical_Activity_Hours: '',
  Physical_Health_Score: '',
}

export function toStudent(d: StudentDraft): Student {
  return {
    Age: Number(d.Age),
    Gender: d.Gender,
    Education_Level: d.Education_Level,
    Daily_Social_Media_Hours: Number(d.Daily_Social_Media_Hours),
    Daily_AI_Tool_Usage_Hours: Number(d.Daily_AI_Tool_Usage_Hours),
    Sleep_Hours: Number(d.Sleep_Hours),
    Physical_Activity_Hours: Number(d.Physical_Activity_Hours),
    Physical_Health_Score: Number(d.Physical_Health_Score),
  }
}

/** Même règle que le backend : la somme des heures quotidiennes ne dépasse pas 24 h. */
export function validateStudent(s: Student): string | null {
  const total =
    s.Sleep_Hours + s.Daily_Social_Media_Hours + s.Daily_AI_Tool_Usage_Hours + s.Physical_Activity_Hours
  return total > 24 ? `La somme des heures (${total.toFixed(1)} h) dépasse 24 h.` : null
}
