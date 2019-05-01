import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


X_full = pd.read_csv('Data/train.csv', index_col = 'Id')
X_test_full = pd.read_csv('Data/test.csv', index_col = 'Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Define models
a = 1
b = 1
c = 2

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

# Find optimal model
lowest_mae = float('inf')
best_model = 0

while c <= 25:
    if a <= 100:
        model = RandomForestRegressor(n_estimators=a, max_depth=b, min_samples_split=c, random_state=0)
        mae = score_model(model)
        print("{a}, {b}, {c}, {mae}".format(a=a, b=b, c=c, mae=mae))
        if mae < lowest_mae:
            lowest_mae = mae
            best_model = model
        a += 1

    elif a > 100 and b <= 25:
        a = 1
        b += 1

    elif a > 100 and b > 25:
        a = 1
        b = 1
        c += 1

model = best_model

model.fit(X, y)
preds_test = model.predict(X_test)

output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    pass
