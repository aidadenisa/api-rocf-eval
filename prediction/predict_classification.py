from prediction.model_storage import retrieveModel

def getLabelTextFromNumber(labelNumber):
  if labelNumber == 0:
    return 'normal'
  elif labelNumber == 1:
    return 'MCI'
  else:
    return 'dementia'

def predictDiagnosis(scores):
  classifierNormalPathology = retrieveModel('svm_model_normal_to_pathology.joblib')
  scalerNormalPathology = retrieveModel('svm_scaler_normal_to_pathology.joblib')
  # classifierMCIDementia = retrieveModel('svm_model_MCI_to_dementia.joblib')
  # scalerMCIDementia = retrieveModel('svm_scaler_MCI_to_dementia.joblib')
  classifierMCIDementia = retrieveModel('svm_model_unbalanced_MCI_to_dementia.joblib')
  scalerMCIDementia = retrieveModel('svm_scaler_unbalanced_MCI_to_dementia.joblib')

  scoresLabels = [pattern['label'] for pattern in scores]

  scoresScaledNormalPathology = scalerNormalPathology.transform([scoresLabels])
  predictionDiagnosisNormalPathology = classifierNormalPathology.predict(scoresScaledNormalPathology)[0]
  predictionProbabilitiesNormalPathology = classifierNormalPathology.predict_proba(scoresScaledNormalPathology)[0]

  scoresScaledMCIDementia = scalerMCIDementia.transform([scoresLabels])
  predictionDiagnosisMCIDementia = classifierMCIDementia.predict(scoresScaledMCIDementia)[0]
  predictionProbabilitiesMCIDementia = classifierMCIDementia.predict_proba(scoresScaledMCIDementia)[0]
  # predictionProbabilitiesMCIDementia = classifierMCIDementia.predict_proba(scoresScaledMCIDementia)[0]

  # if the predicted diagnosis is Normal 
  if predictionDiagnosisNormalPathology == 0 :
    predictionDiagnosis = 0
  elif predictionDiagnosisMCIDementia == 0: 
    predictionDiagnosis = 1
  else:
    predictionDiagnosis = 2

  predictionProbabilities = [
    predictionProbabilitiesNormalPathology[0], 
    predictionProbabilitiesNormalPathology[1] * predictionProbabilitiesMCIDementia[0],
    predictionProbabilitiesNormalPathology[1] * predictionProbabilitiesMCIDementia[1]
  ]  

  return {
    'labelNumber': int(predictionDiagnosis),
    'labelText': getLabelTextFromNumber(predictionDiagnosis),
    'probabilities': list(predictionProbabilities)
  }

