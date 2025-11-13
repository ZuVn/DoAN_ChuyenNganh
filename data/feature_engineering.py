import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    lookback_months: int = 12
    zone_comparison_window: int = 30  # days
    statistical_features: bool = True
    physics_features: bool = True


class FeatureEngineer:
    """
    Extract features from raw smart meter and grid data
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.feature_names = []
        
    def generate_all_features(
        self, 
        meter_data: pd.DataFrame,
        transformer_data: pd.DataFrame,
        grid_topology: Dict,
        historical_visits: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive feature set
        
        Args:
            meter_data: Smart meter readings (timestamp, customer_id, P, Q, V)
            transformer_data: Transformer measurements
            grid_topology: Grid structure and parameters
            historical_visits: Past inspection results
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        
        features = pd.DataFrame()
        
        # 1. Consumption-based features (File 1 approach)
        consumption_features = self._extract_consumption_features(meter_data)
        features = pd.concat([features, consumption_features], axis=1)
        
        # 2. Physics-based features (File 2 approach)
        physics_features = self._extract_physics_features(
            meter_data, transformer_data, grid_topology
        )
        features = pd.concat([features, physics_features], axis=1)
        
        # 3. Temporal features
        temporal_features = self._extract_temporal_features(meter_data)
        features = pd.concat([features, temporal_features], axis=1)
        
        # 4. Visit history features
        if historical_visits is not None:
            visit_features = self._extract_visit_features(historical_visits)
            features = pd.concat([features, visit_features], axis=1)
        
        # 5. Statistical aggregation features
        if self.config.statistical_features:
            stat_features = self._extract_statistical_features(meter_data)
            features = pd.concat([features, stat_features], axis=1)
        
        self.feature_names = features.columns.tolist()
        logger.info(f"Generated {len(self.feature_names)} features")
        
        return features
    
    def _extract_consumption_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract consumption pattern features (File 1)
        """
        features = pd.DataFrame(index=df['customer_id'].unique())
        
        # Raw consumption
        features['consumption_last_3m'] = df.groupby('customer_id')['active_power'].tail(90).sum()
        features['consumption_last_12m'] = df.groupby('customer_id')['active_power'].tail(365).sum()
        features['consumption_penultimate_year'] = df.groupby('customer_id')['active_power'].apply(
            lambda x: x.iloc[-730:-365].sum() if len(x) >= 730 else 0
        )
        
        # Consumption changes
        def calc_diff_6months(group):
            if len(group) < 180:
                return 0
            current_6m = group.iloc[-180:].sum()
            previous_6m = group.iloc[-360:-180].sum()
            return previous_6m - current_6m
        
        features['diff_consumption_6m'] = df.groupby('customer_id')['active_power'].apply(
            calc_diff_6months
        )
        
        # Zero consumption periods
        features['months_zero_consumption'] = df.groupby('customer_id')['active_power'].apply(
            lambda x: (x.tail(30) == 0).sum()
        )
        
        # Min/Max ratio
        features['min_max_bill_ratio'] = df.groupby('customer_id')['active_power'].apply(
            lambda x: x.tail(365).min() / (x.tail(365).max() + 1e-6)
        )
        
        # Comparison with zone average
        zone_avg = df.groupby(['zone', 'tariff'])['active_power'].mean()
        features['consumption_vs_zone'] = df.apply(
            lambda row: row['active_power'] / zone_avg.get((row['zone'], row['tariff']), 1),
            axis=1
        ).groupby(df['customer_id']).mean()
        
        return features
    
    def _extract_physics_features(
        self, 
        meter_data: pd.DataFrame,
        transformer_data: pd.DataFrame,
        grid_topology: Dict
    ) -> pd.DataFrame:
        """
        Extract physics-based features (File 2)
        """
        features = pd.DataFrame(index=meter_data['customer_id'].unique())
        
        # Voltage deviation features
        features['voltage_mean'] = meter_data.groupby('customer_id')['voltage'].mean()
        features['voltage_std'] = meter_data.groupby('customer_id')['voltage'].std()
        features['voltage_min'] = meter_data.groupby('customer_id')['voltage'].min()
        features['voltage_max'] = meter_data.groupby('customer_id')['voltage'].max()
        
        # Voltage deviation from nominal (e.g., 230V or 1.0 p.u.)
        nominal_voltage = 1.0  # per unit
        features['voltage_deviation'] = (
            features['voltage_mean'] - nominal_voltage
        ).abs()
        
        # Power factor
        features['power_factor'] = meter_data.apply(
            lambda row: row['active_power'] / np.sqrt(
                row['active_power']**2 + row['reactive_power']**2 + 1e-6
            ),
            axis=1
        ).groupby(meter_data['customer_id']).mean()
        
        # Power balance check
        total_sm_power = meter_data.groupby('timestamp')['active_power'].sum()
        transformer_power = transformer_data.set_index('timestamp')['active_power']
        
        # Align timestamps
        common_timestamps = total_sm_power.index.intersection(transformer_power.index)
        power_diff = transformer_power.loc[common_timestamps] - total_sm_power.loc[common_timestamps]
        
        # Assign power balance deviation to each customer
        # (simplified: use mean deviation as a grid-level feature)
        features['grid_power_imbalance'] = power_diff.mean()
        features['grid_power_imbalance_std'] = power_diff.std()
        
        # Distance from transformer (from topology)
        if 'distance_from_transformer' in grid_topology:
            features['distance_from_transformer'] = meter_data['customer_id'].map(
                grid_topology['distance_from_transformer']
            )
        
        # Reading absences (suspicious meter behavior)
        features['reading_absences'] = meter_data.groupby('customer_id').apply(
            lambda g: (g['timestamp'].diff() > pd.Timedelta('1 hour')).sum()
        )
        
        return features
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features
        """
        features = pd.DataFrame(index=df['customer_id'].unique())
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Last reading time
        features['days_since_last_reading'] = (
            pd.Timestamp.now() - df.groupby('customer_id')['timestamp'].max()
        ).dt.days
        
        # Consumption patterns by time of day
        df['hour'] = df['timestamp'].dt.hour
        features['peak_hour_consumption'] = df[df['hour'].isin([18, 19, 20])].groupby(
            'customer_id'
        )['active_power'].mean()
        
        features['offpeak_hour_consumption'] = df[df['hour'].isin([2, 3, 4])].groupby(
            'customer_id'
        )['active_power'].mean()
        
        # Day of week patterns
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        features['weekend_consumption'] = df[df['dayofweek'] >= 5].groupby(
            'customer_id'
        )['active_power'].mean()
        
        features['weekday_consumption'] = df[df['dayofweek'] < 5].groupby(
            'customer_id'
        )['active_power'].mean()
        
        return features
    
    def _extract_visit_features(self, visits_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from historical inspection visits
        """
        features = pd.DataFrame(index=visits_df['customer_id'].unique())
        
        # NTL history
        features['ntl_count'] = visits_df[visits_df['result'] == 'NTL'].groupby(
            'customer_id'
        ).size()
        
        features['non_ntl_count'] = visits_df[visits_df['result'] == 'non_NTL'].groupby(
            'customer_id'
        ).size()
        
        # Last visit type
        last_visit = visits_df.sort_values('visit_date').groupby('customer_id').last()
        features['last_visit_result'] = last_visit['result']
        
        # Days since last visit
        features['days_since_last_visit'] = (
            pd.Timestamp.now() - last_visit['visit_date']
        ).dt.days
        
        # Impossible visits
        features['impossible_visit_count'] = visits_df[
            visits_df['result'] == 'impossible'
        ].groupby('customer_id').size()
        
        # Energy recovered in past
        features['total_energy_recovered'] = visits_df.groupby('customer_id')[
            'energy_recovered_kwh'
        ].sum()
        
        # Threat incidents
        features['threat_count'] = visits_df[visits_df['threat'] == True].groupby(
            'customer_id'
        ).size()
        
        return features.fillna(0)
    
    def _extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract statistical aggregation features
        """
        features = pd.DataFrame(index=df['customer_id'].unique())
        
        # Statistical moments for active power
        features['ap_mean'] = df.groupby('customer_id')['active_power'].mean()
        features['ap_std'] = df.groupby('customer_id')['active_power'].std()
        features['ap_skew'] = df.groupby('customer_id')['active_power'].skew()
        features['ap_kurtosis'] = df.groupby('customer_id')['active_power'].apply(
            lambda x: x.kurtosis()
        )
        
        # Statistical moments for reactive power
        features['rp_mean'] = df.groupby('customer_id')['reactive_power'].mean()
        features['rp_std'] = df.groupby('customer_id')['reactive_power'].std()
        
        # Percentiles
        features['ap_25percentile'] = df.groupby('customer_id')['active_power'].quantile(0.25)
        features['ap_50percentile'] = df.groupby('customer_id')['active_power'].quantile(0.50)
        features['ap_75percentile'] = df.groupby('customer_id')['active_power'].quantile(0.75)
        
        # Coefficient of variation
        features['ap_cv'] = features['ap_std'] / (features['ap_mean'] + 1e-6)
        
        return features


class FeatureSelector:
    """
    Select important features using multiple methods
    """
    
    def __init__(self, method: str = 'shap'):
        """
        Args:
            method: 'shap', 'permutation', 'correlation', or 'recursive'
        """
        self.method = method
        self.selected_features = []
        self.feature_importance = {}
    
    def select(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        model = None,
        top_k: int = 50
    ) -> List[str]:
        """
        Select top K important features
        """
        if self.method == 'shap':
            return self._shap_selection(X, y, model, top_k)
        elif self.method == 'correlation':
            return self._correlation_selection(X, y, top_k)
        elif self.method == 'recursive':
            return self._recursive_elimination(X, y, top_k)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _shap_selection(self, X, y, model, top_k):
        """Use SHAP values for feature selection"""
        import shap
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Get top K features
        top_indices = np.argsort(mean_shap)[-top_k:]
        self.selected_features = X.columns[top_indices].tolist()
        self.feature_importance = dict(zip(
            X.columns[top_indices],
            mean_shap[top_indices]
        ))
        
        return self.selected_features
    
    def _correlation_selection(self, X, y, top_k):
        """Select features based on correlation with target"""
        correlations = X.corrwith(y).abs()
        top_features = correlations.nlargest(top_k).index.tolist()
        self.selected_features = top_features
        return top_features
    
    def _recursive_elimination(self, X, y, top_k):
        """Recursive Feature Elimination"""
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import GradientBoostingRegressor
        
        estimator = GradientBoostingRegressor(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=top_k, step=1)
        selector.fit(X, y)
        
        self.selected_features = X.columns[selector.support_].tolist()
        return self.selected_features


# Example usage
if __name__ == "__main__":
    # Load data
    meter_data = pd.read_csv("data/smart_meter_readings.csv")
    transformer_data = pd.read_csv("data/transformer_readings.csv")
    grid_topology = pd.read_json("data/grid_topology.json")
    historical_visits = pd.read_csv("data/inspection_visits.csv")
    
    # Initialize feature engineer
    config = FeatureConfig(
        lookback_months=12,
        zone_comparison_window=30,
        statistical_features=True,
        physics_features=True
    )
    
    engineer = FeatureEngineer(config)
    
    # Generate features
    features = engineer.generate_all_features(
        meter_data=meter_data,
        transformer_data=transformer_data,
        grid_topology=grid_topology,
        historical_visits=historical_visits
    )
    
    print(f"Generated features shape: {features.shape}")
    print(f"Feature names: {engineer.feature_names[:10]}...")
    
    # Feature selection
    selector = FeatureSelector(method='shap')
    # (would need trained model for SHAP selection)