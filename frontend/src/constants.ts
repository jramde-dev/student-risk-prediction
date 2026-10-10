import type { EducationLevel, Gender, Student } from './types'

export const NUMERIC_FIELDS = [
  { name: 'Age', label: 'Âge', min: 13, max: 25, step: 1 },
  { name: 'Daily_Social_Media_Hours', label: 'Réseaux sociaux (h/jour)', min: 0, max: 24, step: 0.1 },
  { name: 'Daily_AI_Tool_Usage_Hours', label: 'Outils IA (h/jour)', min: 0, max: 24, step: 0.1 },
  { name: 'Sleep_Hours', label: 'Sommeil (h/nuit)', min: 0, max: 24, step: 0.1 },
  { name: 'Physical_Activity_Hours', label: 'Activité physique (h/jour)', min: 0, max: 24, step: 0.1 },
  { name: 'Physical_Health_Score', label: 'Score de santé physique (0-100)', min: 0, max: 100, step: 0.1 },
] as const satisfies ReadonlyArray<{
  name: keyof Student
  label: string
  min: number
  max: number
  step: number
}>

export type NumericKey = (typeof NUMERIC_FIELDS)[number]['name']

export const GENDERS: ReadonlyArray<{ value: Gender; label: string }> = [
  { value: 'Female', label: 'Femme' },
  { value: 'Male', label: 'Homme' },
  { value: 'Non-binary', label: 'Non-binaire' },
]

export const EDUCATION_LEVELS: ReadonlyArray<{ value: EducationLevel; label: string }> = [
  { value: 'High School', label: 'High School (lycée)' },
  { value: 'College', label: 'College' },
  { value: 'University', label: 'University (université)' },
]
