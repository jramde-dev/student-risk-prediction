import type { Prediction } from '../types'

export function ResultCard({ prediction }: { prediction: Prediction }) {
  const { at_risk, at_risk_probability, mental_health_score, threshold } = prediction
  const tone = at_risk === null ? '' : at_risk ? 'risk' : 'ok'

  return (
    <div className={`result ${tone}`} role="status">
      {at_risk !== null && <strong>{at_risk ? 'À risque' : 'Pas à risque'}</strong>}
      {at_risk_probability !== null && (
        <span>
          Probabilité de risque : {(at_risk_probability * 100).toFixed(1)} % (seuil {threshold})
        </span>
      )}
      {mental_health_score !== null && <span>Score de santé mentale prédit : {mental_health_score.toFixed(1)} / 100</span>}
    </div>
  )
}
