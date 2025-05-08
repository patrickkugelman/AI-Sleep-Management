import numpy as np
import pandas as pd
import shap
import joblib
import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna

class SleepPerformancePredictor:
    """Advanced AI models for predicting sleep quality and performance in harsh environments.
    Provides comprehensive analysis and actionable insights.
    """
    """
    AI models for predicting sleep quality and performance in harsh environments.
    """
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.X = None
        self.y_sleep = None
        self.y_cognitive = None
        self.y_physical = None
        self.models = {}
        self.standard_scaler = None
        self.minmax_scaler = None
        
    def preprocess_data(self, test_size=0.2, random_state=42):
        """Advanced data preprocessing with multiple scaling and feature engineering techniques"""
        features = ['noise_level', 'temperature', 'light_intensity', 'air_quality_index']
        self.X = self.data[features]
        self.y_sleep = self.data['sleep_quality_score']
        self.y_cognitive = self.data['cognitive_performance']
        self.y_physical = self.data['physical_performance']
        
        # Feature interaction and polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        self.X_poly = poly.fit_transform(self.X)
        
        # Multiple scaling techniques
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
        # Scaled features
        self.X_standard_scaled = self.standard_scaler.fit_transform(self.X)
        self.X_minmax_scaled = self.minmax_scaler.fit_transform(self.X)
        self.X_poly_scaled = self.standard_scaler.fit_transform(self.X_poly)
        
        # Train-test split
        self.X_train_std, self.X_test_std, self.y_train, self.y_test = train_test_split(
            self.X_standard_scaled, self.y_sleep, test_size=test_size, random_state=random_state
        )
        
        return self.X_train_std, self.X_test_std, self.y_train, self.y_test
        
    def train_models(self, advanced_techniques=True):
        """Advanced ensemble and stacking model training with meta-learning"""
        # Advanced base models with more complex configurations
        rf_model = RandomForestRegressor(
            n_estimators=300, 
            max_depth=None, 
            min_samples_split=5, 
            min_samples_leaf=2, 
            random_state=42
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=250, 
            learning_rate=0.05, 
            max_depth=5, 
            random_state=42
        )
        
        xgb_model = XGBRegressor(
            n_estimators=250, 
            learning_rate=0.05, 
            max_depth=6, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42
        )
        
        lgbm_model = LGBMRegressor(
            n_estimators=250, 
            learning_rate=0.05, 
            max_depth=5, 
            num_leaves=31, 
            random_state=42
        )
        
        if advanced_techniques:
            # Meta-learning and advanced hyperparameter optimization
            def objective(trial):
                # Hyperparameter search space
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
                }
                
                # Cross-validated performance
                model = GradientBoostingRegressor(**params, random_state=42)
                scores = cross_val_score(model, self.X_train_std, self.y_train, 
                                         cv=5, scoring='neg_mean_squared_error')
                return np.mean(scores)
            
            # Optuna optimization
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)
            
            # Best model configuration
            best_params = study.best_params
            gb_model = GradientBoostingRegressor(**best_params, random_state=42)
        
        # Train base models
        rf_model.fit(self.X_train_std, self.y_train)
        gb_model.fit(self.X_train_std, self.y_train)
        xgb_model.fit(self.X_train_std, self.y_train)
        lgbm_model.fit(self.X_train_std, self.y_train)
        
        # Neural Network with advanced architecture
        nn_model, _ = self.train_neural_network(self.X_train_std, self.y_train, self.X_test_std, self.y_test)
        
        # Advanced Stacking Ensemble with meta-learner
        estimators = [
            ('rf', rf_model),
            ('gb', gb_model),
            ('xgb', xgb_model),
            ('lgbm', lgbm_model)
        ]
        
        # Meta-learner with regularization
        meta_learner = Ridge(alpha=1.0)
        
        stacking_model = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,
            passthrough=True  # Include original features
        )
        stacking_model.fit(self.X_train_std, self.y_train)
        
        # Store models with performance metrics
        self.models = {
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'xgboost': xgb_model,
            'lightgbm': lgbm_model,
            'neural_network': nn_model,
            'stacking_ensemble': stacking_model
        }
        
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Advanced neural network with multiple architectures and regularization"""
        def create_model(hp):
            model = Sequential()
            model.add(Dense(units=hp.Int('units_1', min_value=32, max_value=512, step=32),
                            activation='relu', input_shape=(X_train.shape[1],)))
            model.add(BatchNormalization())
            model.add(Dropout(hp.Float('dropout_1', 0, 0.5, step=0.1)))
            
            # Dynamic number of hidden layers
            for i in range(hp.Int('num_layers', 1, 4)):
                model.add(Dense(units=hp.Int(f'units_{i+2}', min_value=32, max_value=512, step=32),
                                activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(hp.Float(f'dropout_{i+2}', 0, 0.5, step=0.1)))
            
            model.add(Dense(1))
            
            # Dynamic learning rate
            hp_learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
            model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mse')
            return model
        
        # Hyperparameter tuning
        tuner = tf.keras.tuners.Hyperband(
            create_model,
            objective='val_loss',
            max_epochs=100,
            factor=3,
            directory='hyperparameter_tuning',
            project_name='sleep_performance'
        )
        
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        
        tuner.search(X_train, y_train, 
                     validation_split=0.2, 
                     epochs=100, 
                     batch_size=32,
                     callbacks=[early_stopping, lr_scheduler])
        
        # Get best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        # Retrain model with best hyperparameters
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(
            X_train, y_train, 
            validation_split=0.2, 
            epochs=100, 
            batch_size=32,
            callbacks=[early_stopping, lr_scheduler]
        )
        
        return model, history   
        
    def feature_importance(self, model):
        """Analyze feature importance in predicting sleep performance"""
        feature_names = ['noise_level', 'temperature', 'light_intensity', 'air_quality_index']
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nFeature Importance:")
        for f in range(self.X.shape[1]):
            print(f"{feature_names[indices[f]]}: {importances[indices[f]]}")

if __name__ == "__main__":
    predictor = SleepPerformancePredictor(
        'c:\\Users\\patri\\OneDrive\\Desktop\\Proiect IRA\\sleep_monitoring_ai\\sleep_environment_data.csv'
    )
    predictor.preprocess_data()
    
    rf_model = predictor.random_forest_model()
    predictor.feature_importance(rf_model)
    
    nn_model = predictor.neural_network_model()
