from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

def print_metrics(self, y_test, y_pred) -> None:
    self.r2 = r2_score(y_test, y_pred)
    print('R2 Score: ' + str(self.r2))

    self.mqe = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error: ' + str(self.mqe))

    self.rmse = root_mean_squared_error(y_test, y_pred)
    print('Root Mean Squared Error: ' + str(self.rmse))
    self.nrmse = self.rmse/(y_test.max() - y_test.min())
    print('Normalized Root Mean Squared Error: ' + str(self.nrmse))

    self.mape = mean_absolute_percentage_error(y_test, y_pred)
    print("Mean Absolute Percentage Error: " + str(self.mape))