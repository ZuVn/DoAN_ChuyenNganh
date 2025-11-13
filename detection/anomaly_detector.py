import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    POWER_IMBALANCE = "power_imbalance"
    VOLTAGE_DEVIATION = "voltage_deviation"
    ZERO_CONSUMPTION = "zero_consumption"
    ABRUPT_DROP = "abrupt_drop"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    STREAK_SIGNAL = "streak_signal"


@dataclass
class AnomalyDetectionConfig:
    # Streak signal parameters
    time_window_hours: int = 24
    k_sigma: float = 2.5  # Threshold multiplier for statistical threshold
    min_streak_count: int = 3  # Minimum consecutive anomalies
    
    # Power imbalance thresholds
    power_imbalance_threshold_kw: float = 5.0
    power_imbalance_percentage: float = 0.10  
    voltage_deviation_threshold_pu: float = 0.05  
    zero_consumption_hours: int = 48  # Hours of zero consumption to flag
    abrupt_drop_percentage: float = 0.5  # 50% drop
    abrupt_drop_window_days: int = 7
    
    # Moving statistics window
    baseline_window_days: int = 30


@dataclass
class Anomaly:
    timestamp: pd.Timestamp
    anomaly_type: AnomalyType
    affected_meters: List[int]
    severity: float  # 0-1 scale
    details: Dict
    confidence: float = 1.0


class StreakSignalDetector:
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.historical_imbalances = deque(maxlen=1000)
        self.baseline_mean = 0.0
        self.baseline_std = 1.0
        self.threshold = 0.0
        
    def update_baseline(self, power_imbalances: List[float]):
      
        if len(power_imbalances) == 0:
            logger.warning("No data to update baseline")
            return
        
        self.baseline_mean = np.mean(power_imbalances)
        self.baseline_std = np.std(power_imbalances)
        self.threshold = self.baseline_mean + self.config.k_sigma * self.baseline_std
        
        logger.info(
            f"Baseline updated: μ={self.baseline_mean:.4f}, "
            f"σ={self.baseline_std:.4f}, threshold={self.threshold:.4f}"
        )
    
    def detect_streak(
        self,
        power_imbalances: pd.Series,
        technical_losses: pd.Series
    ) -> Tuple[bool, Dict]:
        deviations = np.abs(power_imbalances - technical_losses)
        
        # Check if cumulative deviation exceeds threshold
        cumulative_deviation = deviations.sum()
        streak_detected = cumulative_deviation > self.threshold
        # Count consecutive anomalies
        consecutive_count = 0
        max_consecutive = 0
        for dev in deviations:
            if dev > self.baseline_mean + self.config.k_sigma * self.baseline_std:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 0
        # Require minimum consecutive anomalies
        if max_consecutive < self.config.min_streak_count:
            streak_detected = False
        
        details = {
            'cumulative_deviation': cumulative_deviation,
            'threshold': self.threshold,
            'max_consecutive_anomalies': max_consecutive,
            'mean_deviation': deviations.mean(),
            'std_deviation': deviations.std(),
            'n_samples': len(deviations)
        }
        
        if streak_detected:
            logger.warning(
                f"Streak signal detected! Cumulative deviation: {cumulative_deviation:.2f}, "
                f"Threshold: {self.threshold:.2f}"
            )
        
        return streak_detected, details
    
    def compute_anomaly_score(self, deviation: float) -> float:
        
        if self.baseline_std == 0:
            return 0.0        
        # Z-score
        z_score = abs((deviation - self.baseline_mean) / self.baseline_std)
        # Convert to 0-1 score using sigmoid
        score = 1.0 / (1.0 + np.exp(-0.5 * (z_score - 3.0)))
        
        return min(1.0, score)

class PowerBalanceDetector:
    
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
    
    def detect_imbalance(
        self,
        transformer_power: pd.Series,
        sm_total_power: pd.Series,
        technical_losses: pd.Series
    ) -> List[Anomaly]:
      
        anomalies = []
        # Calculate power imbalance
        power_imbalance = transformer_power - sm_total_power - technical_losses
        
        for idx, timestamp in enumerate(transformer_power.index):
            imbalance = power_imbalance.iloc[idx]
            transformer_val = transformer_power.iloc[idx]
            # Check absolute threshold
            if abs(imbalance) > self.config.power_imbalance_threshold_kw:
                # Check percentage threshold
                if abs(imbalance) / (transformer_val + 1e-6) > self.config.power_imbalance_percentage:
                    severity = min(1.0, abs(imbalance) / (2 * self.config.power_imbalance_threshold_kw))
                    
                    anomaly = Anomaly(
                        timestamp=timestamp,
                        anomaly_type=AnomalyType.POWER_IMBALANCE,
                        affected_meters=[],  # Grid-wide
                        severity=severity,
                        details={
                            'power_imbalance_kw': imbalance,
                            'transformer_power_kw': transformer_val,
                            'sm_total_power_kw': sm_total_power.iloc[idx],
                            'technical_losses_kw': technical_losses.iloc[idx],
                            'imbalance_percentage': abs(imbalance) / (transformer_val + 1e-6)
                        }
                    )
                    anomalies.append(anomaly)
        
        if anomalies:
            logger.info(f"Detected {len(anomalies)} power imbalance anomalies")
        
        return anomalies


class VoltageDeviationDetector:
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.nominal_voltage = 1.0  # per unit
    
    def detect_voltage_anomalies(
        self,
        voltage_data: pd.DataFrame  # columns: timestamp, meter_id, voltage_pu
    ) -> List[Anomaly]:
        anomalies = []
        
        for timestamp in voltage_data['timestamp'].unique():
            data_t = voltage_data[voltage_data['timestamp'] == timestamp]    
            for _, row in data_t.iterrows():
                voltage = row['voltage_pu']
                deviation = abs(voltage - self.nominal_voltage)
                
                if deviation > self.config.voltage_deviation_threshold_pu:
                    severity = min(1.0, deviation / (2 * self.config.voltage_deviation_threshold_pu))
                    
                    anomaly = Anomaly(
                        timestamp=timestamp,
                        anomaly_type=AnomalyType.VOLTAGE_DEVIATION,
                        affected_meters=[int(row['meter_id'])],
                        severity=severity,
                        details={
                            'voltage_pu': voltage,
                            'nominal_voltage_pu': self.nominal_voltage,
                            'deviation_pu': deviation,
                            'deviation_percentage': deviation / self.nominal_voltage
                        }
                    )
                    anomalies.append(anomaly)
        
        if anomalies:
            logger.info(f"Detected {len(anomalies)} voltage deviation anomalies")
        
        return anomalies


class ConsumptionPatternDetector:
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
    
    def detect_zero_consumption(
        self,
        consumption_data: pd.DataFrame  # columns: timestamp, meter_id, active_power_kw
    ) -> List[Anomaly]:
        anomalies = []
        
        for meter_id in consumption_data['meter_id'].unique():
            meter_data = consumption_data[consumption_data['meter_id'] == meter_id].sort_values('timestamp')
            
            # Find consecutive zero consumption
            zero_mask = meter_data['active_power_kw'] <= 0.01  # Near zero
            
            # Count consecutive zeros
            consecutive_zeros = 0
            start_timestamp = None
            
            for idx, is_zero in enumerate(zero_mask):
                if is_zero:
                    if consecutive_zeros == 0:
                        start_timestamp = meter_data.iloc[idx]['timestamp']
                    consecutive_zeros += 1
                else:
                    if consecutive_zeros > 0:
                        # Check if exceeded threshold
                        hours = consecutive_zeros * 0.5  # Assuming 30-min intervals
                        if hours >= self.config.zero_consumption_hours:
                            severity = min(1.0, hours / (2 * self.config.zero_consumption_hours))
                            
                            anomaly = Anomaly(
                                timestamp=start_timestamp,
                                anomaly_type=AnomalyType.ZERO_CONSUMPTION,
                                affected_meters=[int(meter_id)],
                                severity=severity,
                                details={
                                    'consecutive_hours': hours,
                                    'consecutive_readings': consecutive_zeros
                                }
                            )
                            anomalies.append(anomaly)
                    
                    consecutive_zeros = 0
        
        if anomalies:
            logger.info(f"Detected {len(anomalies)} zero consumption anomalies")
        
        return anomalies
    
    def detect_abrupt_drop(
        self,
        consumption_data: pd.DataFrame
    ) -> List[Anomaly]:
        anomalies = []
        
        for meter_id in consumption_data['meter_id'].unique():
            meter_data = consumption_data[consumption_data['meter_id'] == meter_id].sort_values('timestamp')
            
            if len(meter_data) < self.config.abrupt_drop_window_days * 48:  # 48 readings per day
                continue
            # Calculate moving average
            meter_data['ma_7d'] = meter_data['active_power_kw'].rolling(
                window=self.config.abrupt_drop_window_days * 48,
                min_periods=1
            ).mean() 
            # Compare recent vs historical average
            recent_avg = meter_data['ma_7d'].iloc[-7*48:].mean()  # Last week
            historical_avg = meter_data['ma_7d'].iloc[:-7*48].mean()  # Before last week
            
            if historical_avg > 0:
                drop_ratio = (historical_avg - recent_avg) / historical_avg
                
                if drop_ratio > self.config.abrupt_drop_percentage:
                    severity = min(1.0, drop_ratio / self.config.abrupt_drop_percentage)
                    
                    anomaly = Anomaly(
                        timestamp=meter_data.iloc[-1]['timestamp'],
                        anomaly_type=AnomalyType.ABRUPT_DROP,
                        affected_meters=[int(meter_id)],
                        severity=severity,
                        details={
                            'recent_avg_kw': recent_avg,
                            'historical_avg_kw': historical_avg,
                            'drop_percentage': drop_ratio,
                            'drop_kw': historical_avg - recent_avg
                        }
                    )
                    anomalies.append(anomaly)
        
        if anomalies:
            logger.info(f"Detected {len(anomalies)} abrupt drop anomalies")
        
        return anomalies
    
    def detect_suspicious_patterns(
        self,
        consumption_data: pd.DataFrame,
        zone_avg_data: Optional[pd.DataFrame] = None
    ) -> List[Anomaly]:
        if zone_avg_data is None:
            logger.info("No zone average data provided, skipping pattern detection")
            return []
        
        anomalies = []
        
        # Merge with zone averages
        merged = consumption_data.merge(
            zone_avg_data,
            on=['timestamp', 'zone'],
            how='left',
            suffixes=('', '_zone_avg')
        )
        # Calculate ratio
        merged['consumption_ratio'] = merged['active_power_kw'] / (merged['active_power_kw_zone_avg'] + 1e-6)
        # Detect anomalous ratios (too low)
        anomalous = merged[merged['consumption_ratio'] < 0.3]  # Less than 30% of zone average
        for _, row in anomalous.iterrows():
            severity = 1.0 - row['consumption_ratio']
            
            anomaly = Anomaly(
                timestamp=row['timestamp'],
                anomaly_type=AnomalyType.SUSPICIOUS_PATTERN,
                affected_meters=[int(row['meter_id'])],
                severity=severity,
                details={
                    'consumption_kw': row['active_power_kw'],
                    'zone_avg_kw': row['active_power_kw_zone_avg'],
                    'consumption_ratio': row['consumption_ratio']
                }
            )
            anomalies.append(anomaly)
        if anomalies:
            logger.info(f"Detected {len(anomalies)} suspicious pattern anomalies")
        return anomalies
class AnomalyAggregator:
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
    
    def aggregate_anomalies(
        self,
        anomalies: List[Anomaly],
        time_window_hours: int = 24
    ) -> pd.DataFrame:
        if not anomalies:
            return pd.DataFrame()
        # Convert to DataFrame
        records = []
        for anomaly in anomalies:
            for meter_id in anomaly.affected_meters:
                records.append({
                    'timestamp': anomaly.timestamp,
                    'meter_id': meter_id,
                    'anomaly_type': anomaly.anomaly_type.value,
                    'severity': anomaly.severity,
                    'confidence': anomaly.confidence
                })
        df = pd.DataFrame(records)
        if df.empty:
            return df
        # Group by meter and time window
        df['time_window'] = df['timestamp'].dt.floor(f'{time_window_hours}H')
        
        aggregated = df.groupby(['meter_id', 'time_window']).agg({
            'severity': ['mean', 'max', 'count'],
            'confidence': 'mean',
            'anomaly_type': lambda x: list(x.unique())
        }).reset_index()
        aggregated.columns = [
            'meter_id', 'time_window', 
            'avg_severity', 'max_severity', 'anomaly_count',
            'avg_confidence', 'anomaly_types'
        ]
        # Calculate composite score
        aggregated['composite_score'] = (
            0.4 * aggregated['avg_severity'] +
            0.3 * aggregated['max_severity'] +
            0.2 * np.log1p(aggregated['anomaly_count']) / np.log(10) +
            0.1 * aggregated['avg_confidence']
        )
        # Sort by composite score
        aggregated = aggregated.sort_values('composite_score', ascending=False)
        
        logger.info(f"Aggregated {len(anomalies)} anomalies into {len(aggregated)} groups")
        
        return aggregated
    
    def prioritize_meters(
        self,
        aggregated_anomalies: pd.DataFrame,
        top_k: Optional[int] = None
    ) -> List[int]:
        if aggregated_anomalies.empty:
            return []
        
        # Sum composite scores by meter
        meter_scores = aggregated_anomalies.groupby('meter_id')['composite_score'].sum()
        meter_scores = meter_scores.sort_values(ascending=False)
        
        if top_k is not None:
            meter_scores = meter_scores.head(top_k)
        
        prioritized_meters = meter_scores.index.tolist()
        
        logger.info(f"Prioritized {len(prioritized_meters)} meters for investigation")
        
        return prioritized_meters
# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Create sample data
    timestamps = pd.date_range('2024-01-01', periods=100, freq='30T')
    n_meters = 10
    consumption_data = []
    for ts in timestamps:
        for meter_id in range(n_meters):
            # Normal consumption
            consumption = np.random.normal(1.5, 0.3)
            # Inject anomaly in meter 5
            if meter_id == 5 and ts.hour >= 12:
                consumption *= 0.2  # 80% reduction (fraud)
            
            consumption_data.append({
                'timestamp': ts,
                'meter_id': meter_id,
                'active_power_kw': max(0, consumption),
                'reactive_power_kvar': max(0, consumption * 0.3),
                'voltage_pu': np.random.normal(1.0, 0.02),
                'zone': 'zone_A'
            })
    consumption_df = pd.DataFrame(consumption_data)
    # Configure detectors
    config = AnomalyDetectionConfig(
        time_window_hours=24,
        k_sigma=2.5,
        min_streak_count=3
    )
    # Run detectors
    pattern_detector = ConsumptionPatternDetector(config)
    voltage_detector = VoltageDeviationDetector(config)
    
    anomalies = []
    anomalies.extend(pattern_detector.detect_abrupt_drop(consumption_df))
    anomalies.extend(voltage_detector.detect_voltage_anomalies(consumption_df))
    
    print(f"\nDetected {len(anomalies)} anomalies")
    # Aggregate
    aggregator = AnomalyAggregator(config)
    aggregated = aggregator.aggregate_anomalies(anomalies)
    
    print(f"\nAggregated anomalies:")
    print(aggregated.head())
    # Prioritize
    priority_meters = aggregator.prioritize_meters(aggregated, top_k=5)
    print(f"\nTop priority meters: {priority_meters}")