import usd_try_ai as ut

model=ut.UsdToTRY()
model.checkData()
model.dataPreprocessing()
model.showOutputDataPreprocessing()
model.buildAndTrainTheModel()
model.evaluateTheModel()
model.predict(1,7,2025)
model.predictMonth()