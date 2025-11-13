import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
import json
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer, FeatureConfig
from models.ml_models.regression_model import EnergyRecoveryRegressor, ModelConfig
from models.physics_models.grid_simulator import GridSimulator, GridConfig, GridTopology
from detection.anomaly_detector import (
    AnomalyDetectionConfig,
    StreakSignalDetector,
    PowerBalanceDetector,
    ConsumptionPatternDetector,
    AnomalyAggregator
)
from detection.ntl_localizer import (
    NTLLocalizer,
    LocalizationConfig,
    LocalizationStrategy
)
from detection.energy_estimator import EnergyEstimator, EstimationMethod
from models.hybrid.fusion_model import HybridFusionModel, FusionConfig
from explainability.shap_explainer import SHAPExplainer, SHAPConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hybrid_detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HybridNTLDetectionPipeline:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.results = {}
        
        self.data_loader = None
        self.feature_engineer = None
        self.ml_model = None
        self.physics_localizer = None
        self.fusion_model = None
        self.explainer = None
        
        logger.info("Hybrid NTL Detection Pipeline initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    def run_pipeline(
        self,
        mode: str = "full",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        logger.info(f"Starting pipeline in '{mode}' mode")
        logger.info(f"Period: {start_date} to {end_date}")
        
        try:
            logger.info("="*80)
            logger.info("STEP 1: Loading Data")
            logger.info("="*80)
            self._load_data(start_date, end_date)
            
            logger.info("\n" + "="*80)
            logger.info("STEP 2: Feature Engineering")
            logger.info("="*80)
            self._engineer_features()
            
            if mode in ["full", "ml_only", "evaluation"]:
                logger.info("\n" + "="*80)
                logger.info("STEP 3: Training ML Model")
                logger.info("="*80)
                self._train_ml_model()
              
                logger.info("\n" + "="*80)
                logger.info("STEP 4: Generating ML Predictions")
                logger.info("="*80)
                ml_predictions = self._generate_ml_predictions()
            else:
                ml_predictions = None
            
            if mode in ["full", "physics_only", "evaluation"]:
                logger.info("\n" + "="*80)
                logger.info("STEP 5: Anomaly Detection")
                logger.info("="*80)
                anomalies = self._detect_anomalies()
                
                logger.info("\n" + "="*80)
                logger.info("STEP 6: Physics-based Localization")
                logger.info("="*80)
                physics_results = self._run_physics_localization(
                    ml_predictions if ml_predictions is not None else None
                )
            else:
                anomalies = None
                physics_results = None
            
            if mode == "full":
                logger.info("\n" + "="*80)
                logger.info("STEP 7: Hybrid Fusion")
                logger.info("="*80)
                fusion_results = self._fuse_predictions(ml_predictions, physics_results)
                
                logger.info("\n" + "="*80)
                logger.info("STEP 8: Generating Reports")
                logger.info("="*80)
                self._generate_reports(fusion_results)
            
            elif mode == "evaluation":
                logger.info("\n" + "="*80)
                logger.info("STEP 7: Model Evaluation & Comparison")
                logger.info("="*80)
                self._evaluate_models(ml_predictions, physics_results)
            
            if self.config.get('explainability', {}).get('enabled', True):
                logger.info("\n" + "="*80)
                logger.info("STEP 9: Explainability Analysis")
                logger.info("="*80)
                self._run_explainability_analysis()
            
            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def _load_data(self, start_date: Optional[str], end_date: Optional[str]):
        data_config = self.config['data']
        
        logger.info(f"Loading consumption data from {data_config['consumption_path']}")
        consumption_df = pd.read_csv(data_config['consumption_path'])
        consumption_df['timestamp'] = pd.to_datetime(consumption_df['timestamp'])
        
        if start_date:
            consumption_df = consumption_df[consumption_df['timestamp'] >= start_date]
        if end_date:
            consumption_df = consumption_df[consumption_df['timestamp'] <= end_date]
        
        logger.info(f"Loaded {len(consumption_df)} consumption records")
        
        if data_config.get('transformer_path'):
            logger.info(f"Loading transformer data from {data_config['transformer_path']}")
            transformer_df = pd.read_csv(data_config['transformer_path'])
            transformer_df['timestamp'] = pd.to_datetime(transformer_df['timestamp'])
            
            if start_date:
                transformer_df = transformer_df[transformer_df['timestamp'] >= start_date]
            if end_date:
                transformer_df = transformer_df[transformer_df['timestamp'] <= end_date]
            
            logger.info(f"Loaded {len(transformer_df)} transformer records")
        else:
            transformer_df = None
        
        logger.info(f"Loading grid topology from {data_config['topology_path']}")
        with open(data_config['topology_path'], 'r') as f:
            grid_topology = json.load(f)
        
        logger.info(f"Grid: {len(grid_topology['buses'])} buses, {len(grid_topology['lines'])} lines")
        
        if data_config.get('visits_path'):
            logger.info(f"Loading visit history from {data_config['visits_path']}")
            visits_df = pd.read_csv(data_config['visits_path'])
            visits_df['visit_date'] = pd.to_datetime(visits_df['visit_date'])
            logger.info(f"Loaded {len(visits_df)} visit records")
        else:
            visits_df = None
        
        self.results['consumption_data'] = consumption_df
        self.results['transformer_data'] = transformer_df
        self.results['grid_topology'] = grid_topology
        self.results['visits_data'] = visits_df
        
        logger.info("Data loading complete")
    
    def _engineer_features(self):
        feature_config = FeatureConfig(
            lookback_months=self.config['features']['lookback_months'],
            zone_comparison_window=self.config['features']['zone_comparison_window'],
            statistical_features=self.config['features']['statistical_features'],
            physics_features=self.config['features']['physics_features']
        )
        
        self.feature_engineer = FeatureEngineer(feature_config)
        
        logger.info("Engineering features...")
        features_df = self.feature_engineer.generate_all_features(
            meter_data=self.results['consumption_data'],
            transformer_data=self.results['transformer_data'],
            grid_topology=self.results['grid_topology'],
            historical_visits=self.results['visits_data']
        )
        
        self.results['features'] = features_df
        logger.info(f"Generated {len(features_df.columns)} features for {len(features_df)} customers")
        
        output_path = Path(self.config['output']['features_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(output_path, index=True)
        logger.info(f"Features saved to {output_path}")
    
    def _train_ml_model(self):
        ml_config = ModelConfig(**self.config['ml_model'])
        
        self.ml_model = EnergyRecoveryRegressor(ml_config)
        
        X = self.results['features']
        
        if self.results['visits_data'] is not None:
            visits = self.results['visits_data']
            y = visits.set_index('customer_id')['energy_recovered_kwh']
            
            common_indices = X.index.intersection(y.index)
            X = X.loc[common_indices]
            y = y.loc[common_indices]
        else:
            logger.warning("No visit data available for training. Using simulated targets.")
            y = pd.Series(
                np.random.gamma(2, 5, size=len(X)),
                index=X.index
            )
        
        logger.info(f"Training on {len(X)} samples")
        categorical_features = [
            col for col in X.columns
            if X[col].dtype == 'object' or col in ['town', 'tariff', 'meter_location']
        ]
        
        training_result = self.ml_model.train(
            X=X,
            y=y,
            categorical_features=categorical_features
        )
        
        logger.info(training_result.to_dict())
        
        # Save model
        model_path = Path(self.config['output']['model_path'])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.ml_model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        self.results['ml_training'] = training_result
    
    def _generate_ml_predictions(self) -> pd.DataFrame:
        X = self.results['features']
        
        logger.info(f"Generating predictions for {len(X)} customers")
        
        predictions, confidence = self.ml_model.predict(X, return_confidence=True)
        
        ml_results = pd.DataFrame({
            'customer_id': X.index,
            'ml_prediction_kwh': predictions,
            'ml_confidence': confidence,
            'ml_probability': self.ml_model.predict_proba_class(X)
        })
        
        ml_results['ml_rank'] = ml_results['ml_prediction_kwh'].rank(ascending=False)
        
        logger.info(f"ML predictions generated. Mean: {predictions.mean():.2f} kWh")
        
        self.results['ml_predictions'] = ml_results
        
        return ml_results
    
    def _detect_anomalies(self) -> pd.DataFrame:
        anomaly_config = AnomalyDetectionConfig(**self.config['anomaly_detection'])
        
        streak_detector = StreakSignalDetector(anomaly_config)
        power_detector = PowerBalanceDetector(anomaly_config)
        pattern_detector = ConsumptionPatternDetector(anomaly_config)
        aggregator = AnomalyAggregator(anomaly_config)
        
        all_anomalies = []
        if self.results['transformer_data'] is not None:
            logger.info("Detecting power balance anomalies...")
            
            consumption = self.results['consumption_data']
            sm_totals = consumption.groupby('timestamp').agg({
                'active_power_kw': 'sum',
                'reactive_power_kvar': 'sum'
            })
            
            transformer = self.results['transformer_data'].set_index('timestamp')
            
            technical_losses = transformer['active_power_kw'] * 0.02
            
            power_anomalies = power_detector.detect_imbalance(
                transformer_power=transformer['active_power_kw'],
                sm_total_power=sm_totals['active_power_kw'],
                technical_losses=technical_losses
            )
            
            all_anomalies.extend(power_anomalies)
            logger.info(f"Found {len(power_anomalies)} power balance anomalies")
        
        logger.info("Detecting consumption pattern anomalies...")
        
        zero_anomalies = pattern_detector.detect_zero_consumption(
            self.results['consumption_data']
        )
        all_anomalies.extend(zero_anomalies)
        logger.info(f"Found {len(zero_anomalies)} zero consumption anomalies")
        
        drop_anomalies = pattern_detector.detect_abrupt_drop(
            self.results['consumption_data']
        )
        all_anomalies.extend(drop_anomalies)
        logger.info(f"Found {len(drop_anomalies)} abrupt drop anomalies")
        
        logger.info("Aggregating anomalies...")
        aggregated = aggregator.aggregate_anomalies(all_anomalies)
        
        priority_meters = aggregator.prioritize_meters(
            aggregated,
            top_k=self.config['detection']['max_suspects']
        )
        
        logger.info(f"Prioritized {len(priority_meters)} meters for investigation")
        
        self.results['anomalies'] = aggregated
        self.results['priority_meters'] = priority_meters
        
        return aggregated
    
    def _run_physics_localization(
        self,
        ml_predictions: Optional[pd.DataFrame]
    ) -> List:
        localization_config = LocalizationConfig(**self.config['physics_localization'])
        
        self.physics_localizer = NTLLocalizer(localization_config)
        
        logger.info(f"Using approach: {localization_config.approach}")
        
        timestamp = self.results['consumption_data']['timestamp'].iloc[0]
        
        if localization_config.approach == "ml_prioritization" and ml_predictions is not None:
            ml_scores = pd.DataFrame({
                'meter_id': ml_predictions['customer_id'],
                'score': ml_predictions['ml_probability']
            })
            
            suspect_meters = LocalizationStrategy.approach_d_ml_prioritization(
                consumption_data=self.results['consumption_data'],
                timestamp=timestamp,
                ml_scores=ml_scores,
                top_k=self.config['detection']['max_suspects']
            )
        
        elif localization_config.approach == "zero_consumption":
            suspect_meters = LocalizationStrategy.approach_a_zero_consumption(
                consumption_data=self.results['consumption_data'],
                timestamp=timestamp
            )
        
        elif localization_config.approach == "random":
            suspect_meters = LocalizationStrategy.approach_c_random_sampling(
                consumption_data=self.results['consumption_data'],
                timestamp=timestamp,
                sample_size=self.config['detection']['max_suspects']
            )
        
        else:
            # Combined approach
            suspect_meters = LocalizationStrategy.combined_approach(
                consumption_data=self.results['consumption_data'],
                timestamp=timestamp,
                grid_topology=self.results['grid_topology'],
                ml_scores=ml_scores if ml_predictions is not None else None
            )
        
        logger.info(f"Selected {len(suspect_meters)} meters for physics-based localization")
        # Run localization
        localization_results = self.physics_localizer.localize_multiple_meters(
            suspect_meters=suspect_meters,
            timestamp=timestamp,
            grid_topology=self.results['grid_topology'],
            consumption_data=self.results['consumption_data'],
            transformer_data=self.results['transformer_data']
        )
        
        logger.info(f"Completed localization for {len(localization_results)} meters")
        # Calculate energy estimates
        estimator = EnergyEstimator(
            electricity_price_per_kwh=self.config['energy_estimation']['price_per_kwh']
        )
        
        energy_estimates = []
        for result in localization_results:
            estimate = estimator.estimate_single_meter(
                meter_id=result.meter_id,
                localization_results=[result],
                consumption_data=self.results['consumption_data'],
                method=EstimationMethod.WEIGHTED_AVERAGE
            )
            energy_estimates.append(estimate)
        
        self.results['physics_localizations'] = localization_results
        self.results['energy_estimates'] = energy_estimates
        
        return localization_results
    
    def _fuse_predictions(
        self,
        ml_predictions: pd.DataFrame,
        physics_results: List
    ) -> pd.DataFrame:
        fusion_config = FusionConfig(**self.config['fusion'])
        
        self.fusion_model = HybridFusionModel(fusion_config)
        ml_scores = ml_predictions.set_index('customer_id')['ml_prediction_kwh'].values
        ml_confidence = ml_predictions.set_index('customer_id')['ml_confidence'].values
        # Physics scores (map to same customers)
        physics_dict = {r.meter_id: r.stolen_active_power_kw for r in physics_results}
        physics_conf_dict = {r.meter_id: r.confidence for r in physics_results}
        # Align
        customer_ids = ml_predictions['customer_id'].values
        physics_scores = np.array([physics_dict.get(cid, 0.0) for cid in customer_ids])
        physics_confidence = np.array([physics_conf_dict.get(cid, 0.5) for cid in customer_ids])
        
        logger.info("Fusing ML and Physics predictions...")
        # Fuse
        fused_scores, confidence, metadata = self.fusion_model.fuse_predictions(
            ml_scores=ml_scores,
            physics_anomalies=physics_scores,
            ml_confidence=ml_confidence,
            physics_confidence=physics_confidence
        )
        
        logger.info(f"Fusion metadata: {metadata}")
        # Apply decision rules
        decisions = self.fusion_model.apply_decision_rules(
            fused_scores=fused_scores,
            ml_scores=ml_scores,
            physics_scores=physics_scores,
            confidence=confidence
        )
        # Add customer IDs
        decisions['customer_id'] = customer_ids
        # Reorder columns
        decisions = decisions[[
            'customer_id', 'fused_score', 'ml_score', 'physics_score',
            'confidence', 'priority', 'alert', 'explanation'
        ]]
        
        logger.info(f"Fusion complete. Alerts: {decisions['alert'].sum()}")
        
        self.results['fusion_results'] = decisions
        
        return decisions
    
    def _evaluate_models(
        self,
        ml_predictions: pd.DataFrame,
        physics_results: List
    ):
        logger.info("Evaluating model performance...")
        if self.results['visits_data'] is not None:
            visits = self.results['visits_data']
            ground_truth = visits.set_index('customer_id')['is_ntl'].to_dict()
        else:
            logger.warning("No ground truth available. Skipping evaluation.")
            return
        # Evaluate ML
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

        ml_ids = ml_predictions['customer_id'].values
        ml_probs = ml_predictions['ml_probability'].values
        ml_preds = (ml_probs > 0.5).astype(int)
        ml_true = np.array([ground_truth.get(cid, 0) for cid in ml_ids])
        
        ml_metrics = {
            'precision': precision_score(ml_true, ml_preds),
            'recall': recall_score(ml_true, ml_preds),
            'f1': f1_score(ml_true, ml_preds),
            'auc': roc_auc_score(ml_true, ml_probs)
        }
        # Evaluate Physics
        physics_ids = [r.meter_id for r in physics_results]
        physics_preds = [1 if r.is_theft_detected else 0 for r in physics_results]
        physics_true = [ground_truth.get(mid, 0) for mid in physics_ids]
        
        if len(physics_preds) > 0:
            physics_metrics = {
                'precision': precision_score(physics_true, physics_preds),
                'recall': recall_score(physics_true, physics_preds),
                'f1': f1_score(physics_true, physics_preds)
            }
        else:
            physics_metrics = {}
        # Log results
        logger.info("\nML MODEL METRICS:")
        for metric, value in ml_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("\nPHYSICS MODEL METRICS:")
        for metric, value in physics_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        self.results['evaluation'] = {
            'ml_metrics': ml_metrics,
            'physics_metrics': physics_metrics
        }
    
    def _run_explainability_analysis(self):
        if self.ml_model is None:
            logger.warning("ML model not trained. Skipping explainability.")
            return
        
        shap_config = SHAPConfig(**self.config['explainability'])
        
        self.explainer = SHAPExplainer(self.ml_model, shap_config)
        
        X = self.results['features']
        
        logger.info("Setting up SHAP explainer...")
        self.explainer.setup_explainer(X, sample_size=100)
        
        logger.info("Generating global explanations...")
        importance = self.explainer.explain_global(X)
        # Identify feature types
        consumption_features = [
            col for col in X.columns
            if any(keyword in col.lower() for keyword in [
                'consumption', 'kwh', 'bill', 'zone', 'diff'
            ])
        ]        
        visit_features = [
            col for col in X.columns
            if any(keyword in col.lower() for keyword in [
                'visit', 'ntl', 'fraud', 'threat'
            ])
        ]
        logger.info("Validating model patterns...")
        validation = self.explainer.validate_model_patterns(
            X=X,
            consumption_features=consumption_features,
            visit_features=visit_features
        )        
        logger.info(f"Pattern validation: {validation['recommendation']}")        
        # Generate report
        report = self.explainer.generate_explanation_report(
            X=X,
            consumption_features=consumption_features,
            visit_features=visit_features,
            sample_instances=5
        )        
        # Save report
        report_path = Path(self.config['output']['explainability_report'])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)        
        logger.info(f"Explainability report saved to {report_path}")
        
        self.results['explainability'] = {
            'importance': importance,
            'validation': validation
        }
    
    def _generate_reports(self, fusion_results: pd.DataFrame):
        output_dir = Path(self.config['output']['reports_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)        
        # 1. Campaign list (high priority alerts)
        campaign = fusion_results[fusion_results['alert'] == True].copy()
        campaign = campaign.sort_values('fused_score', ascending=False)
        
        campaign_path = output_dir / 'campaign_list.csv'
        campaign.to_csv(campaign_path, index=False)
        logger.info(f"Campaign list saved: {len(campaign)} customers ({campaign_path})")        
        # 2. Detailed results
        detailed_path = output_dir / 'detailed_results.csv'
        fusion_results.to_csv(detailed_path, index=False)
        logger.info(f"Detailed results saved ({detailed_path})")        
        # 3. Summary statistics
        summary = {
            'total_customers': len(fusion_results),
            'high_priority': (fusion_results['priority'] == 'HIGH_PRIORITY').sum(),
            'medium_priority': (fusion_results['priority'] == 'MEDIUM_PRIORITY').sum(),
            'alerts_generated': fusion_results['alert'].sum(),
            'mean_fused_score': fusion_results['fused_score'].mean(),
            'mean_confidence': fusion_results['confidence'].mean()
        }        
        summary_path = output_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary statistics saved ({summary_path})")
        logger.info(f"  Total customers: {summary['total_customers']}")
        logger.info(f"  High priority: {summary['high_priority']}")
        logger.info(f"  Alerts: {summary['alerts_generated']}")        
        # 4. Energy estimates report
        if 'energy_estimates' in self.results:
            estimator = EnergyEstimator()
            
            estimates_report = []
            for estimate in self.results['energy_estimates']:
                estimates_report.append(estimate.to_dict())
            
            estimates_df = pd.DataFrame(estimates_report)
            estimates_path = output_dir / 'energy_estimates.csv'
            estimates_df.to_csv(estimates_path, index=False)
            logger.info(f"Energy estimates saved ({estimates_path})")
        
        self.results['reports_generated'] = True


def main():
    parser = argparse.ArgumentParser(description='Run Hybrid NTL Detection System')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'ml_only', 'physics_only', 'evaluation'],
        default='full',
        help='Pipeline mode'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )    
    args = parser.parse_args()    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)    
    # Initialize pipeline
    pipeline = HybridNTLDetectionPipeline(args.config)    
    # Run pipeline
    try:
        pipeline.run_pipeline(
            mode=args.mode,
            start_date=args.start_date,
            end_date=args.end_date
        )        
        logger.info("\n" + "="*80)
        logger.info("SUCCESS: Pipeline completed successfully!")
        logger.info("="*80)        
        return 0
        
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error(f"FAILURE: Pipeline failed with error:")
        logger.error(f"{str(e)}")
        logger.error("="*80)
        return 1
if __name__ == "__main__":
    sys.exit(main())