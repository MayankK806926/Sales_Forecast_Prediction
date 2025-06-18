import xgboost as xgb

def train_xgboost_model(X_train, y_train, params):
    """Train an XGBoost regression model."""
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model