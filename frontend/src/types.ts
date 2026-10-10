// Miroir des schémas Pydantic du backend (app/schemas.py)
export type Gender = 'Male' | 'Female' | 'Non-binary'
export type EducationLevel = 'High School' | 'College' | 'University'

export interface Student {
  Age: number
  Gender: Gender
  Education_Level: EducationLevel
  Daily_Social_Media_Hours: number
  Daily_AI_Tool_Usage_Hours: number
  Sleep_Hours: number
  Physical_Activity_Hours: number
  Physical_Health_Score: number
}

export interface Prediction {
  mental_health_score: number | null
  at_risk_probability: number | null
  at_risk: boolean | null
  threshold: number
}

export interface BatchResponse {
  predictions: Prediction[]
}
