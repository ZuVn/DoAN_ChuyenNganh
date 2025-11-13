import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("SHAP not installed. Install with: pip install shap")
logger = logging.getLogger(__name__)

@dataclass
class SHAPConfig:
    # Explainer settings
    explainer_type: str = "tree"  # "tree", "kernel", "linear"
    n_samples: int = 100  # Number of samples for explanation
    check_additivity: bool = False  # Check SHAP additivity property    
    # Visualization settings
    max_display: int = 20  # Max features to display
    plot_type: str = "bar"  # "bar", "beeswarm", "waterfall", "force"    
    # Feature selection
    feature_selection_method: str = "shap"  # "shap", "importance"
    top_k_features: int = 50    
    # Output settings
    save_plots: bool = True
    output_dir: str = "reports/explainability"    
    # Analysis thresholds
    reliable_pattern_threshold: float = 0.1  # Min SHAP value for reliable pattern
    consumption_feature_weight: float = 0.6  # Expected weight of consumption features

@dataclass
class ExplanationResult:
    instance_id: int
    prediction: float
    base_value: float
    shap_values: Dict[str, float]
    top_features: List[str]
    top_feature_values: Dict[str, float]
    explanation_quality: float  # 0-1 score
        
    def to_dict(self) -> Dict:
        return {
            'instance_id': self.instance_id,
            'prediction': self.prediction,
            'base_value': self.base_value,
            'shap_values': self.shap_values,
            'top_features': self.top_features,
            'top_feature_values': self.top_feature_values,
            'explanation_quality': self.explanation_quality
        }

class SHAPExplainer:    
    def __init__(self, model, config: SHAPConfig):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")        
        self.model = model
        self.config = config
        self.explainer = None
        self.feature_names = None
        self.base_value = None        
        # Storage for explanations
        self.global_shap_values = None
        self.global_data = None        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
                
    def setup_explainer(self, X: pd.DataFrame, sample_size: Optional[int] = None):
        self.feature_names = X.columns.tolist()        
        # Sample data if needed
        if sample_size is not None and len(X) > sample_size:
            X_background = X.sample(n=sample_size, random_state=42)
        else:
            X_background = X        
        logger.info(f"Setting up {self.config.explainer_type} explainer with {len(X_background)} background samples")        
        # Create explainer based on type
        if self.config.explainer_type == "tree":
            # For tree-based models (CatBoost, XGBoost, etc.)
            self.explainer = shap.TreeExplainer(
                self.model.model if hasattr(self.model, 'model') else self.model,
                check_additivity=self.config.check_additivity
            )
        elif self.config.explainer_type == "kernel":
            # For any model (model-agnostic)
            def predict_fn(X):
                if hasattr(self.model, 'predict'):
                    return self.model.predict(X)
                else:
                    return self.model(X)            
            self.explainer = shap.KernelExplainer(predict_fn, X_background)
        elif self.config.explainer_type == "linear":
            # For linear models
            self.explainer = shap.LinearExplainer(
                self.model.model if hasattr(self.model, 'model') else self.model,
                X_background
            )
        else:
            raise ValueError(f"Unknown explainer type: {self.config.explainer_type}")
        
        logger.info("SHAP explainer setup complete")
    
    def explain_global(
        self,
        X: pd.DataFrame,
        save_plots: Optional[bool] = None
    ) -> Dict[str, float]:
        if self.explainer is None:
            raise ValueError("Explainer not setup. Call setup_explainer() first")        
        logger.info(f"Computing SHAP values for {len(X)} instances...")        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X)        
        # Store for later use
        self.global_shap_values = shap_values
        self.global_data = X        
        # Calculate mean absolute SHAP values (global importance)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)        
        # Create importance dictionary
        importance_dict = {
            feat: float(imp)
            for feat, imp in zip(self.feature_names, mean_abs_shap)
        }        
        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )        
        logger.info(f"Top 5 features: {list(importance_dict.keys())[:5]}")        
        # Generate and save plots
        if save_plots or (save_plots is None and self.config.save_plots):
            self._plot_global_importance(importance_dict)
            self._plot_summary(shap_values, X)        
        return importance_dict
    
    def explain_instance(
        self,
        X: pd.DataFrame,
        instance_idx: int = 0,
        save_plot: Optional[bool] = None
    ) -> ExplanationResult:
        if self.explainer is None:
            raise ValueError("Explainer not setup. Call setup_explainer() first")        
        # Get single instance
        instance = X.iloc[instance_idx:instance_idx+1]        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(instance)[0]        
        # Get prediction
        if hasattr(self.model, 'predict'):
            prediction = self.model.predict(instance)[0]
        else:
            prediction = self.model(instance)[0]        
        # Get base value
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
        else:
            base_value = 0.0        
        # Create SHAP value dictionary
        shap_dict = {
            feat: float(val)
            for feat, val in zip(self.feature_names, shap_values)
        }        
        # Sort by absolute SHAP value
        sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)        
        # Get top features
        top_k = min(self.config.max_display, len(sorted_shap))
        top_features = [feat for feat, _ in sorted_shap[:top_k]]        
        # Get feature values
        feature_values = {
            feat: float(instance[feat].values[0])
            for feat in top_features
        }        
        # Assess explanation quality
        quality = self._assess_explanation_quality(shap_dict, feature_values)        
        result = ExplanationResult(
            instance_id=instance_idx,
            prediction=float(prediction),
            base_value=float(base_value),
            shap_values=shap_dict,
            top_features=top_features,
            top_feature_values=feature_values,
            explanation_quality=quality
        )        
        # Generate plot
        if save_plot or (save_plot is None and self.config.save_plots):
            self._plot_waterfall(shap_values, instance, instance_idx)        
        return result
    
    def explain_batch(
        self,
        X: pd.DataFrame,
        indices: Optional[List[int]] = None
    ) -> List[ExplanationResult]:
        if indices is None:
            indices = list(range(len(X)))
        
        results = []
        for idx in indices:
            try:
                result = self.explain_instance(X, idx, save_plot=False)
                results.append(result)
            except Exception as e:
                logger.error(f"Error explaining instance {idx}: {e}")
        
        logger.info(f"Generated explanations for {len(results)}/{len(indices)} instances")        
        return results
    
    def validate_model_patterns(
        self,
        X: pd.DataFrame,
        consumption_features: List[str],
        visit_features: List[str]
    ) -> Dict:
        if self.global_shap_values is None:
            logger.info("Computing global SHAP values for validation...")
            self.explain_global(X, save_plots=False)        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.global_shap_values).mean(axis=0)
        total_importance = mean_abs_shap.sum()        
        # Calculate contribution of different feature groups
        consumption_importance = 0.0
        visit_importance = 0.0        
        for i, feat in enumerate(self.feature_names):
            importance = mean_abs_shap[i]
            if feat in consumption_features:
                consumption_importance += importance
            elif feat in visit_features:
                visit_importance += importance        
        # Calculate percentages
        consumption_pct = (consumption_importance / total_importance) * 100
        visit_pct = (visit_importance / total_importance) * 100
        other_pct = 100 - consumption_pct - visit_pct        
        # Validate pattern quality
        is_reliable = consumption_pct >= (self.config.consumption_feature_weight * 100)        
        # Identify top features by type
        feature_importance = {
            feat: mean_abs_shap[i]
            for i, feat in enumerate(self.feature_names)
        }        
        top_consumption = [
            feat for feat in sorted(
                [f for f in consumption_features if f in feature_importance],
                key=lambda x: feature_importance[x],
                reverse=True
            )[:5]
        ]        
        top_visit = [
            feat for feat in sorted(
                [f for f in visit_features if f in feature_importance],
                key=lambda x: feature_importance[x],
                reverse=True
            )[:5]
        ]        
        validation_result = {
            'is_reliable': is_reliable,
            'consumption_feature_percentage': consumption_pct,
            'visit_feature_percentage': visit_pct,
            'other_feature_percentage': other_pct,
            'top_consumption_features': top_consumption,
            'top_visit_features': top_visit,
            'recommendation': self._generate_recommendation(
                consumption_pct, visit_pct, is_reliable
            )
        }        
        logger.info(
            f"Pattern validation: Consumption={consumption_pct:.1f}%, "
            f"Visit={visit_pct:.1f}%, Reliable={is_reliable}"
        )        
        return validation_result
    
    def select_features(
        self,
        X: pd.DataFrame,
        top_k: Optional[int] = None
    ) -> List[str]:
        if top_k is None:
            top_k = self.config.top_k_features        
        # Get global importance
        importance = self.explain_global(X, save_plots=False)        
        # Select top K
        selected_features = list(importance.keys())[:top_k]        
        logger.info(f"Selected {len(selected_features)} features using SHAP values")        
        return selected_features
    
    def _assess_explanation_quality(
        self,
        shap_values: Dict[str, float],
        feature_values: Dict[str, float]
    ) -> float:
        # Sort by absolute SHAP value
        sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)        
        # Concentration score (top 5 features)
        top_5_sum = sum(abs(val) for _, val in sorted_shap[:5])
        total_sum = sum(abs(val) for val in shap_values.values())
        concentration = top_5_sum / (total_sum + 1e-10)        
        # Magnitude score (normalized by max possible)
        max_shap = max(abs(val) for val in shap_values.values())
        magnitude = min(1.0, max_shap / 10.0)  # Assume 10 is high        
        # Combined quality score
        quality = 0.6 * concentration + 0.4 * magnitude        
        return float(np.clip(quality, 0.0, 1.0))    
    def _generate_recommendation(
        self,
        consumption_pct: float,
        visit_pct: float,
        is_reliable: bool
    ) -> str:
        if is_reliable:
            return (
                "✓ Model is reliable. Consumption features dominate predictions, "
                "indicating the model learns valid patterns."
            )
        else:
            if visit_pct > consumption_pct:
                return (
                    "⚠ Model may be over-relying on visit history. Consider:\n"
                    "  1. Re-balancing training data\n"
                    "  2. Reducing weight of visit features\n"
                    "  3. Adding more consumption-related features"
                )
            else:
                return (
                    "⚠ Model patterns are unclear. Investigate:\n"
                    "  1. Feature engineering quality\n"
                    "  2. Data quality issues\n"
                    "  3. Model hyperparameters"
                )    
    def _plot_global_importance(self, importance: Dict[str, float]):
        # Convert to DataFrame
        df = pd.DataFrame([
            {'feature': feat, 'importance': imp}
            for feat, imp in list(importance.items())[:self.config.max_display]
        ])        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.barh(df['feature'], df['importance'])
        plt.xlabel('Mean |SHAP value|')
        plt.title('Global Feature Importance (SHAP)')
        plt.tight_layout()        
        # Save
        output_path = Path(self.config.output_dir) / 'global_importance.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()        
        logger.info(f"Global importance plot saved to {output_path}")
    
    def _plot_summary(self, shap_values: np.ndarray, X: pd.DataFrame):
        plt.figure(figsize=(12, 8))
        
        shap.summary_plot(
            shap_values,
            X,
            max_display=self.config.max_display,
            show=False
        )        
        # Save
        output_path = Path(self.config.output_dir) / 'summary_plot.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary plot saved to {output_path}")
    
    def _plot_waterfall(
        self,
        shap_values: np.ndarray,
        instance: pd.DataFrame,
        instance_idx: int
    ):
        plt.figure(figsize=(10, 8))        
        # Create Explanation object
        explanation = shap.Explanation(
            values=shap_values,
            base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            data=instance.values[0],
            feature_names=self.feature_names
        )        
        shap.waterfall_plot(explanation, max_display=self.config.max_display, show=False)        
        # Save
        output_path = Path(self.config.output_dir) / f'waterfall_instance_{instance_idx}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Waterfall plot saved to {output_path}")
    
    def generate_explanation_report(
        self,
        X: pd.DataFrame,
        consumption_features: List[str],
        visit_features: List[str],
        sample_instances: int = 5
    ) -> str:
        # Global explanation
        global_importance = self.explain_global(X)        
        # Pattern validation
        validation = self.validate_model_patterns(X, consumption_features, visit_features)        
        # Sample instance explanations
        sample_indices = np.random.choice(len(X), min(sample_instances, len(X)), replace=False)
        instance_explanations = self.explain_batch(X, sample_indices.tolist())        
        # Generate report
        report = f"""
{'='*80}
SHAP EXPLAINABILITY REPORT
{'='*80}

GLOBAL FEATURE IMPORTANCE (Top 10):
"""
        for i, (feat, imp) in enumerate(list(global_importance.items())[:10], 1):
            report += f"  {i:2d}. {feat:40s} {imp:10.6f}\n"
        
        report += f"""
PATTERN VALIDATION:
  Reliability:                  {'✓ RELIABLE' if validation['is_reliable'] else '✗ UNRELIABLE'}
  Consumption Features:         {validation['consumption_feature_percentage']:.1f}%
  Visit Features:               {validation['visit_feature_percentage']:.1f}%
  Other Features:               {validation['other_feature_percentage']:.1f}%

Top Consumption Features:
"""
        for feat in validation['top_consumption_features']:
            report += f"  - {feat}\n"
        
        report += f"""
Top Visit Features:
"""
        for feat in validation['top_visit_features']:
            report += f"  - {feat}\n"
        
        report += f"""
RECOMMENDATION:
{validation['recommendation']}

SAMPLE INSTANCE EXPLANATIONS:
"""
        for expl in instance_explanations[:3]:
            report += f"""
Instance #{expl.instance_id}:
  Prediction:                   {expl.prediction:.2f} kWh
  Base Value:                   {expl.base_value:.2f}
  Explanation Quality:          {expl.explanation_quality:.2%}
  Top 3 Features:
"""
            for feat in expl.top_features[:3]:
                shap_val = expl.shap_values[feat]
                feat_val = expl.top_feature_values[feat]
                report += f"    - {feat:30s} = {feat_val:8.2f}  (SHAP: {shap_val:+.4f})\n"        
        report += f"\n{'='*80}\n"        
        return report
# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("SHAP Explainability Module")
    print("="*80)
    print("\nFeatures:")
    print("  ✓ Global and local explanations")
    print("  ✓ Pattern validation (consumption vs visit features)")
    print("  ✓ Feature selection based on SHAP values")
    print("  ✓ Comprehensive visualization")
    print("  ✓ Explanation quality assessment")
    print("\nUsage:")
    print("  from explainability.shap_explainer import SHAPExplainer, SHAPConfig")
    print("  config = SHAPConfig()")
    print("  explainer = SHAPExplainer(model, config)")
    print("  explainer.setup_explainer(X_train)")
    print("  importance = explainer.explain_global(X_test)")
    print("="*80)