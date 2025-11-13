import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
import joblib
import json
from pathlib import Path

try:
    from catboost import CatBoostRegressor, Pool
except ImportError:
    logger.warning("CatBoost not installed. Install with: pip install catboost")
    CatBoostRegressor = None
    Pool = None
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    # Model type
    model_type: str = "catboost"
    # CatBoost parameters
    iterations: int = 1000
    learning_rate: float = 0.03
    depth: int = 6
    l2_leaf_reg: float = 3.0
    random_strength: float = 1.0
    bagging_temperature: float = 1.0
    border_count: int = 254
    # Loss function
    loss_function: str = "RMSE"  # RMSE for energy recovery prediction
    eval_metric: str = "MAE"
    # Training
    random_seed: int = 42
    verbose: int = 100
    early_stopping_rounds: int = 50
    use_best_model: bool = True    
    # Data split
    train_size: float = 0.8
    validation_size: float = 0.1
    test_size: float = 0.1
    stratify_by: Optional[str] = None    
    # Feature engineering
    handle_missing: str = "mean"  # "mean", "median", "drop"
    scale_features: bool = False    
    # Model persistence
    model_name: str = "energy_recovery_regressor"
    save_path: str = "models/saved_models"


@dataclass
class TrainingResult:
    train_mae: float
    train_rmse: float
    train_r2: float
    val_mae: float
    val_rmse: float
    val_r2: float
    test_mae: Optional[float] = None
    test_rmse: Optional[float] = None
    test_r2: Optional[float] = None
    n_iterations: int = 0
    training_time_seconds: float = 0.0
    feature_importance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        return {
            'train_metrics': {
                'mae': self.train_mae,
                'rmse': self.train_rmse,
                'r2': self.train_r2
            },
            'validation_metrics': {
                'mae': self.val_mae,
                'rmse': self.val_rmse,
                'r2': self.val_r2
            },
            'test_metrics': {
                'mae': self.test_mae,
                'rmse': self.test_rmse,
                'r2': self.test_r2
            } if self.test_mae is not None else None,
            'n_iterations': self.n_iterations,
            'training_time_seconds': self.training_time_seconds
        }


class EnergyRecoveryRegressor:    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.feature_names = None
        self.categorical_features = None
        self.training_result = None
        self.scaler = None        
        # Initialize model
        if CatBoostRegressor is not None:
            self._initialize_model()
        else:
            raise ImportError("CatBoost is required. Install with: pip install catboost")
    
    def _initialize_model(self):
        self.model = CatBoostRegressor(
            iterations=self.config.iterations,
            learning_rate=self.config.learning_rate,
            depth=self.config.depth,
            l2_leaf_reg=self.config.l2_leaf_reg,
            random_strength=self.config.random_strength,
            bagging_temperature=self.config.bagging_temperature,
            border_count=self.config.border_count,
            loss_function=self.config.loss_function,
            eval_metric=self.config.eval_metric,
            random_seed=self.config.random_seed,
            verbose=self.config.verbose,
            early_stopping_rounds=self.config.early_stopping_rounds,
            use_best_model=self.config.use_best_model
        )        
        logger.info("CatBoost model initialized")    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_features: Optional[List[str]] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None
    ) -> TrainingResult:
        import time
        start_time = time.time()        
        logger.info("Starting model training...")        
        # Store feature information
        self.feature_names = X.columns.tolist()
        self.categorical_features = categorical_features or []        
        # Handle missing values
        X = self._handle_missing_values(X)        
        # Create validation set if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.config.validation_size / (1.0 - self.config.test_size),
                random_state=self.config.random_seed
            )
        else:
            X_train = X
            y_train = y
            X_val = self._handle_missing_values(X_val)        
        # Scale features if configured
        if self.config.scale_features:
            X_train, X_val = self._scale_features(X_train, X_val)
            if X_test is not None:
                X_test = self._scale_features(X_test, fit=False)[0]        
        # Get categorical feature indices
        cat_features_idx = [
            self.feature_names.index(feat) 
            for feat in self.categorical_features 
            if feat in self.feature_names
        ]        
        # Create CatBoost Pool objects
        train_pool = Pool(
            data=X_train,
            label=y_train,
            cat_features=cat_features_idx
        )        
        val_pool = Pool(
            data=X_val,
            label=y_val,
            cat_features=cat_features_idx
        )        
        # Train model
        logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")        
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            plot=False
        )        
        training_time = time.time() - start_time        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_r2 = r2_score(y_train, train_pred)        
        val_pred = self.model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_r2 = r2_score(y_val, val_pred)        
        # Test metrics if provided
        test_mae, test_rmse, test_r2 = None, None, None
        if X_test is not None and y_test is not None:
            X_test = self._handle_missing_values(X_test)
            test_pred = self.model.predict(X_test)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_r2 = r2_score(y_test, test_pred)        
        # Get feature importance
        feature_importance = self.get_feature_importance()        
        # Store results
        self.training_result = TrainingResult(
            train_mae=train_mae,
            train_rmse=train_rmse,
            train_r2=train_r2,
            val_mae=val_mae,
            val_rmse=val_rmse,
            val_r2=val_r2,
            test_mae=test_mae,
            test_rmse=test_rmse,
            test_r2=test_r2,
            n_iterations=self.model.tree_count_,
            training_time_seconds=training_time,
            feature_importance=feature_importance
        )        
        logger.info(
            f"Training complete in {training_time:.2f}s. "
            f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}"
        )        
        return self.training_result    
    def predict(
        self,
        X: pd.DataFrame,
        return_confidence: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if self.model is None:
            raise ValueError("Model must be trained first")        
        # Handle missing values
        X = self._handle_missing_values(X)        
        # Scale if configured
        if self.config.scale_features and self.scaler is not None:
            X = self.scaler.transform(X)        
        # Make predictions
        predictions = self.model.predict(X)        
        if return_confidence:
            # Use virtual ensembles for uncertainty estimation
            try:
                virtual_pred = self.model.virtual_ensembles_predict(
                    X,
                    prediction_type='TotalUncertainty',
                    virtual_ensembles_count=10
                )
                # Convert uncertainty to confidence (inverse relationship)
                uncertainty = virtual_pred[:, 1]
                confidence = 1.0 / (1.0 + uncertainty)
                return predictions, confidence
            except Exception as e:
                logger.warning(f"Could not compute confidence: {e}")
                # Return dummy confidence
                return predictions, np.ones(len(predictions)) * 0.5        
        return predictions
    
    def predict_proba_class(
        self,
        X: pd.DataFrame,
        threshold_kwh: float = 0.5
    ) -> np.ndarray:
        predictions = self.predict(X)        
        # Convert to probabilities using sigmoid
        # Higher energy recovery = higher probability of NTL
        probabilities = 1.0 / (1.0 + np.exp(-predictions / 10.0))        
        return probabilities    
    def get_feature_importance(
        self,
        importance_type: str = 'FeatureImportance'
    ) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model must be trained first")        
        try:
            importance_values = self.model.get_feature_importance(type=importance_type)            
            # Create dictionary
            importance_dict = {
                feat: float(imp)
                for feat, imp in zip(self.feature_names, importance_values)
            }            
            # Sort by importance
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )            
            return importance_dict            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}    
    def get_feature_importance_df(self) -> pd.DataFrame:
        """Get feature importance as DataFrame"""
        importance = self.get_feature_importance()        
        df = pd.DataFrame([
            {'feature': feat, 'importance': imp}
            for feat, imp in importance.items()
        ])        
        return df.sort_values('importance', ascending=False).reset_index(drop=True)    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        X = X.copy()        
        if self.config.handle_missing == "mean":
            X = X.fillna(X.mean())
        elif self.config.handle_missing == "median":
            X = X.fillna(X.median())
        elif self.config.handle_missing == "drop":
            X = X.dropna()        
        return X    
    def _scale_features(
        self,
        X_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        fit: bool = True
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        from sklearn.preprocessing import StandardScaler
        
        if fit:
            self.scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )            
            if X_val is not None:
                X_val_scaled = pd.DataFrame(
                    self.scaler.transform(X_val),
                    columns=X_val.columns,
                    index=X_val.index
                )
                return X_train_scaled, X_val_scaled
            
            return X_train_scaled
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted")
            
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            return X_scaled
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prefix: str = "eval"
    ) -> Dict[str, float]:
        predictions = self.predict(X)        
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        mask = y != 0
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(y[mask], predictions[mask])
        else:
            mape = np.nan        
        metrics = {
            f'{prefix}_mae': mae,
            f'{prefix}_rmse': rmse,
            f'{prefix}_r2': r2,
            f'{prefix}_mape': mape
        }        
        logger.info(f"{prefix.upper()} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")        
        return metrics
    
    def save(self, filepath: Optional[str] = None):
        if self.model is None:
            raise ValueError("No model to save")        
        if filepath is None:
            save_dir = Path(self.config.save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            filepath = save_dir / f"{self.config.model_name}.cbm"        
        filepath = Path(filepath)        
        # Save CatBoost model
        self.model.save_model(str(filepath))        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'config': {
                'iterations': self.config.iterations,
                'learning_rate': self.config.learning_rate,
                'depth': self.config.depth,
                'loss_function': self.config.loss_function
            },
            'training_result': self.training_result.to_dict() if self.training_result else None
        }        
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)        
        # Save scaler if exists
        if self.scaler is not None:
            scaler_path = filepath.with_suffix('.scaler')
            joblib.dump(self.scaler, scaler_path)        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        filepath = Path(filepath)        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")        
        # Load CatBoost model
        self.model = CatBoostRegressor()
        self.model.load_model(str(filepath))        
        # Load metadata
        metadata_path = filepath.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)            
            self.feature_names = metadata.get('feature_names')
            self.categorical_features = metadata.get('categorical_features')        
        # Load scaler if exists
        scaler_path = filepath.with_suffix('.scaler')
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)        
        logger.info(f"Model loaded from {filepath}")    
    def get_training_summary(self) -> str:
        if self.training_result is None:
            return "Model not trained yet"        
        result = self.training_result        
        summary = f"""
{'='*80}
TRAINING SUMMARY - {self.config.model_name}
{'='*80}

MODEL CONFIGURATION:
  Type:                   {self.config.model_type}
  Loss Function:          {self.config.loss_function}
  Iterations:             {result.n_iterations}/{self.config.iterations}
  Learning Rate:          {self.config.learning_rate}
  Depth:                  {self.config.depth}

TRAINING METRICS:
  MAE:                    {result.train_mae:.4f}
  RMSE:                   {result.train_rmse:.4f}
  R²:                     {result.train_r2:.4f}

VALIDATION METRICS:
  MAE:                    {result.val_mae:.4f}
  RMSE:                   {result.val_rmse:.4f}
  R²:                     {result.val_r2:.4f}
"""
        
        if result.test_mae is not None:
            summary += f"""
TEST METRICS:
  MAE:                    {result.test_mae:.4f}
  RMSE:                   {result.test_rmse:.4f}
  R²:                     {result.test_r2:.4f}
"""
        
        summary += f"""
TRAINING TIME:            {result.training_time_seconds:.2f} seconds

TOP 10 IMPORTANT FEATURES:
"""
        if result.feature_importance:
            for i, (feat, imp) in enumerate(list(result.feature_importance.items())[:10], 1):
                summary += f"  {i:2d}. {feat:40s} {imp:10.4f}\n"        
        summary += f"\n{'='*80}\n"        
        return summary
# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("Energy Recovery Regressor - ML Model")
    print("="*80)
    print("\nFeatures:")
    print("  ✓ CatBoost regression for energy recovery prediction")
    print("  ✓ Automatic feature importance calculation")
    print("  ✓ Virtual ensembles for confidence estimation")
    print("  ✓ Model persistence and loading")
    print("  ✓ Comprehensive evaluation metrics")
    print("\nUsage:")
    print("  from models.ml_models.regression_model import EnergyRecoveryRegressor, ModelConfig")
    print("  config = ModelConfig()")
    print("  model = EnergyRecoveryRegressor(config)")
    print("  model.train(X_train, y_train)")
    print("  predictions = model.predict(X_test)")
    print("="*80)