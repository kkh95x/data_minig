import usd_try_ai as ut



usd_try = ut.UsdToTRY()
usd_try.checkData()
usd_try.dataPreprocessing()
usd_try.showOutputDataPreprocessing()

# Linear Regression
usd_try.buildAndTrainTheLinearModel()
usd_try.evaluateTheLinearModel()

# Decision Tree
usd_try.buildAndTrainTheDecisionTreeModel()
usd_try.evaluateTheDecisionTreeModel()

# KNN
usd_try.buildAndTrainTheKNNModel()
usd_try.evaluateTheKNNModel()


usd_try.predict(15, 10, 2024, model_type='linear')
usd_try.predict(15, 10, 2024, model_type='decision_tree')
usd_try.predict(15, 10, 2024, model_type='knn')
