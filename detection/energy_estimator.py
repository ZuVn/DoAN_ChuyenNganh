import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import logging
from datetime import timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class EstimationMethod(Enum):
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    REGRESSION_BASED = "regression_based"
    INTERPOLATION = "interpolation"
    
@dataclass
class EnergyTheftEstimate:
    meter_id: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    
    total_stolen_kwh: float
    average_theft_rate_kw: float
    peak_theft_rate_kw: float
    
    reported_consumption_kwh: float
    estimated_actual_consumption_kwh: float
    theft_percentage: float
    
    estimated_financial_loss: float
    currency: str = "USD"
    electricity_price_per_kwh: float = 0.15
    
    average_confidence: float
    min_confidence: float
    max_confidence: float
    n_samples: int
    
    first_detection: pd.Timestamp
    last_detection: pd.Timestamp
    detection_frequency_hours: float
    
    severity_level: str  # "low", "medium", "high", "critical"
    priority_score: float  # 0-100
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class GridLossEstimate:
    
    zone_id: str
    timestamp_start: pd.Timestamp
    timestamp_end: pd.Timestamp
    
    total_meters: int
    meters_with_theft: int
    theft_rate_percentage: float
    
    total_stolen_kwh: float
    total_reported_kwh: float
    total_actual_kwh: float
    
    total_financial_loss: float
    average_loss_per_meter: float
    
    top_offenders: List[int]  # meter_ids
    top_offenders_contribution_percentage: float
    
    average_confidence: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class EnergyEstimator:
    def __init__(
        self,
        electricity_price_per_kwh: float = 0.15,
        currency: str = "USD"
    ):
        self.electricity_price_per_kwh = electricity_price_per_kwh
        self.currency = currency
        
    def estimate_single_meter(
        self,
        meter_id: int,
        localization_results: List,  # List of LocalizationResult
        consumption_data: pd.DataFrame,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        method: EstimationMethod = EstimationMethod.WEIGHTED_AVERAGE
    ) -> EnergyTheftEstimate:
        meter_results = [r for r in localization_results if r.meter_id == meter_id]
        
        if not meter_results:
            logger.warning(f"No localization results found for meter {meter_id}")
            return self._create_empty_estimate(meter_id, start_date, end_date)
        # Determine time period
        if start_date is None:
            start_date = min(r.timestamp for r in meter_results)
        if end_date is None:
            end_date = max(r.timestamp for r in meter_results)
        # Filter results within time period
        period_results = [
            r for r in meter_results
            if start_date <= r.timestamp <= end_date
        ]
        
        if not period_results:
            logger.warning(f"No results in specified period for meter {meter_id}")
            return self._create_empty_estimate(meter_id, start_date, end_date)
        
        # Get consumption data for this period
        meter_consumption = consumption_data[
            (consumption_data['meter_id'] == meter_id) &
            (consumption_data['timestamp'] >= start_date) &
            (consumption_data['timestamp'] <= end_date)
        ]
        # Calculate energy theft based on method
        if method == EstimationMethod.SIMPLE_AVERAGE:
            energy_estimate = self._estimate_simple_average(
                period_results, meter_consumption
            )
        elif method == EstimationMethod.WEIGHTED_AVERAGE:
            energy_estimate = self._estimate_weighted_average(
                period_results, meter_consumption
            )
        elif method == EstimationMethod.INTERPOLATION:
            energy_estimate = self._estimate_interpolation(
                period_results, meter_consumption, start_date, end_date
            )
        else:
            # Default to weighted average
            energy_estimate = self._estimate_weighted_average(
                period_results, meter_consumption
            )
        # Calculate reported consumption
        reported_kwh = meter_consumption['active_power_kw'].sum() * 0.5  # 30-min intervals
        # Calculate estimated actual consumption
        estimated_actual_kwh = reported_kwh + energy_estimate['total_stolen_kwh']
        # Calculate theft percentage
        if estimated_actual_kwh > 0:
            theft_percentage = (energy_estimate['total_stolen_kwh'] / estimated_actual_kwh) * 100
        else:
            theft_percentage = 0.0
        # Calculate financial loss
        financial_loss = energy_estimate['total_stolen_kwh'] * self.electricity_price_per_kwh
        # Confidence metrics
        confidences = [r.confidence for r in period_results]
        avg_confidence = np.mean(confidences)
        min_confidence = np.min(confidences)
        max_confidence = np.max(confidences)
        # Detection timeline
        first_detection = min(r.timestamp for r in period_results)
        last_detection = max(r.timestamp for r in period_results)
        detection_span = (last_detection - first_detection).total_seconds() / 3600
        detection_frequency = detection_span / len(period_results) if len(period_results) > 1 else 0
        # Severity classification
        severity_level = self._classify_severity(
            energy_estimate['total_stolen_kwh'],
            theft_percentage,
            (end_date - start_date).days
        )       
        # Priority score (0-100)
        priority_score = self._calculate_priority_score(
            energy_estimate['total_stolen_kwh'],
            theft_percentage,
            avg_confidence,
            len(period_results)
        )
        
        return EnergyTheftEstimate(
            meter_id=meter_id,
            start_date=start_date,
            end_date=end_date,
            total_stolen_kwh=energy_estimate['total_stolen_kwh'],
            average_theft_rate_kw=energy_estimate['average_theft_rate_kw'],
            peak_theft_rate_kw=energy_estimate['peak_theft_rate_kw'],
            reported_consumption_kwh=reported_kwh,
            estimated_actual_consumption_kwh=estimated_actual_kwh,
            theft_percentage=theft_percentage,
            estimated_financial_loss=financial_loss,
            currency=self.currency,
            electricity_price_per_kwh=self.electricity_price_per_kwh,
            average_confidence=avg_confidence,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            n_samples=len(period_results),
            first_detection=first_detection,
            last_detection=last_detection,
            detection_frequency_hours=detection_frequency,
            severity_level=severity_level,
            priority_score=priority_score
        )
    
    def estimate_grid_losses(
        self,
        localization_results: List,
        consumption_data: pd.DataFrame,
        zone_id: str = "entire_grid",
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        top_n_offenders: int = 10
    ) -> GridLossEstimate:
        # Determine time period
        if not localization_results:
            logger.warning("No localization results provided")
            return self._create_empty_grid_estimate(zone_id, start_date, end_date)
        if start_date is None:
            start_date = min(r.timestamp for r in localization_results)
        if end_date is None:
            end_date = max(r.timestamp for r in localization_results)
        # Filter results within period
        period_results = [
            r for r in localization_results
            if start_date <= r.timestamp <= end_date
        ]
        
        if not period_results:
            logger.warning("No results in specified period")
            return self._create_empty_grid_estimate(zone_id, start_date, end_date)
        # Get unique meters
        unique_meters = set(r.meter_id for r in period_results)
        total_meters = len(unique_meters)
        
        # Calculate per-meter estimates
        meter_estimates = {}
        for meter_id in unique_meters:
            estimate = self.estimate_single_meter(
                meter_id,
                period_results,
                consumption_data,
                start_date,
                end_date
            )
            meter_estimates[meter_id] = estimate
        # Count meters with theft
        meters_with_theft = sum(
            1 for est in meter_estimates.values()
            if est.total_stolen_kwh > 0.5
        )
        theft_rate = (meters_with_theft / total_meters * 100) if total_meters > 0 else 0
        # Aggregate energy statistics
        total_stolen = sum(est.total_stolen_kwh for est in meter_estimates.values())
        total_reported = sum(est.reported_consumption_kwh for est in meter_estimates.values())
        total_actual = sum(est.estimated_actual_consumption_kwh for est in meter_estimates.values())
        # Financial impact
        total_loss = sum(est.estimated_financial_loss for est in meter_estimates.values())
        avg_loss_per_meter = total_loss / total_meters if total_meters > 0 else 0 
        # Identify top offenders
        sorted_estimates = sorted(
            meter_estimates.items(),
            key=lambda x: x[1].total_stolen_kwh,
            reverse=True
        )
        top_offenders = [meter_id for meter_id, _ in sorted_estimates[:top_n_offenders]]        
        # Calculate contribution of top offenders
        top_offenders_stolen = sum(
            meter_estimates[meter_id].total_stolen_kwh
            for meter_id in top_offenders
        )
        top_contribution = (top_offenders_stolen / total_stolen * 100) if total_stolen > 0 else 0       
        # Average confidence
        avg_confidence = np.mean([est.average_confidence for est in meter_estimates.values()])
        
        return GridLossEstimate(
            zone_id=zone_id,
            timestamp_start=start_date,
            timestamp_end=end_date,
            total_meters=total_meters,
            meters_with_theft=meters_with_theft,
            theft_rate_percentage=theft_rate,
            total_stolen_kwh=total_stolen,
            total_reported_kwh=total_reported,
            total_actual_kwh=total_actual,
            total_financial_loss=total_loss,
            average_loss_per_meter=avg_loss_per_meter,
            top_offenders=top_offenders,
            top_offenders_contribution_percentage=top_contribution,
            average_confidence=avg_confidence
        )
    
    def _estimate_simple_average(
        self,
        results: List,
        consumption_data: pd.DataFrame
    ) -> Dict:
        avg_theft_kw = np.mean([r.stolen_active_power_kw for r in results])
        peak_theft_kw = np.max([r.stolen_active_power_kw for r in results])
        
        # Calculate total time period
        timestamps = sorted([r.timestamp for r in results])
        if len(timestamps) > 1:
            start = timestamps[0]
            end = timestamps[-1]
            hours = (end - start).total_seconds() / 3600
        else:
            hours = 0.5  # Single measurement      
        # Total stolen energy
        total_stolen_kwh = avg_theft_kw * hours
        
        return {
            'total_stolen_kwh': total_stolen_kwh,
            'average_theft_rate_kw': avg_theft_kw,
            'peak_theft_rate_kw': peak_theft_kw
        }   
    def _estimate_weighted_average(
        self,
        results: List,
        consumption_data: pd.DataFrame
    ) -> Dict:
        weights = np.array([r.confidence for r in results])
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        
        theft_values = np.array([r.stolen_active_power_kw for r in results])
        avg_theft_kw = np.average(theft_values, weights=weights)
        peak_theft_kw = np.max(theft_values)       
        # Calculate time period
        timestamps = sorted([r.timestamp for r in results])
        if len(timestamps) > 1:
            start = timestamps[0]
            end = timestamps[-1]
            hours = (end - start).total_seconds() / 3600
        else:
            hours = 0.5       
        # Total stolen energy
        total_stolen_kwh = avg_theft_kw * hours
        
        return {
            'total_stolen_kwh': total_stolen_kwh,
            'average_theft_rate_kw': avg_theft_kw,
            'peak_theft_rate_kw': peak_theft_kw
        }
    
    def _estimate_interpolation(
        self,
        results: List,
        consumption_data: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> Dict:
        df = pd.DataFrame([
            {
                'timestamp': r.timestamp,
                'stolen_kw': r.stolen_active_power_kw,
                'confidence': r.confidence
            }
            for r in results
        ]).sort_values('timestamp')      
        # Create complete time index (30-min intervals)
        full_index = pd.date_range(start=start_date, end=end_date, freq='30T')       
        # Reindex and interpolate
        df = df.set_index('timestamp')
        df = df.reindex(full_index)
        df['stolen_kw'] = df['stolen_kw'].interpolate(method='linear', limit_direction='both')
        df['stolen_kw'] = df['stolen_kw'].fillna(0)       
        # Calculate energy (power * time)
        interval_hours = 0.5
        df['stolen_kwh'] = df['stolen_kw'] * interval_hours       
        # Total stolen energy
        total_stolen_kwh = df['stolen_kwh'].sum()
        avg_theft_kw = df['stolen_kw'].mean()
        peak_theft_kw = df['stolen_kw'].max()       
        return {
            'total_stolen_kwh': total_stolen_kwh,
            'average_theft_rate_kw': avg_theft_kw,
            'peak_theft_rate_kw': peak_theft_kw
        }    
    def _classify_severity(
        self,
        total_stolen_kwh: float,
        theft_percentage: float,
        period_days: int
    ) -> str:
        daily_avg = total_stolen_kwh / max(period_days, 1)
        
        # Classification logic
        if daily_avg < 5 and theft_percentage < 30:
            return "low"
        elif daily_avg < 15 and theft_percentage < 60:
            return "medium"
        elif daily_avg < 30 or theft_percentage < 80:
            return "high"
        else:
            return "critical"
    
    def _calculate_priority_score(
        self,
        total_stolen_kwh: float,
        theft_percentage: float,
        confidence: float,
        n_samples: int
    ) -> float:
        energy_score = min(1.0, total_stolen_kwh / 1000.0)  # Normalize by 1000 kWh
        percentage_score = min(1.0, theft_percentage / 100.0)
        confidence_score = confidence
        sample_score = min(1.0, n_samples / 100.0)       
        # Weighted combination
        priority = (
            0.4 * energy_score +
            0.3 * percentage_score +
            0.2 * confidence_score +
            0.1 * sample_score
        )        
        return priority * 100.0   
    def _create_empty_estimate(
        self,
        meter_id: int,
        start_date: Optional[pd.Timestamp],
        end_date: Optional[pd.Timestamp]
    ) -> EnergyTheftEstimate:
        now = pd.Timestamp.now()
        
        return EnergyTheftEstimate(
            meter_id=meter_id,
            start_date=start_date or now,
            end_date=end_date or now,
            total_stolen_kwh=0.0,
            average_theft_rate_kw=0.0,
            peak_theft_rate_kw=0.0,
            reported_consumption_kwh=0.0,
            estimated_actual_consumption_kwh=0.0,
            theft_percentage=0.0,
            estimated_financial_loss=0.0,
            currency=self.currency,
            electricity_price_per_kwh=self.electricity_price_per_kwh,
            average_confidence=0.0,
            min_confidence=0.0,
            max_confidence=0.0,
            n_samples=0,
            first_detection=now,
            last_detection=now,
            detection_frequency_hours=0.0,
            severity_level="low",
            priority_score=0.0
        )
    
    def _create_empty_grid_estimate(
        self,
        zone_id: str,
        start_date: Optional[pd.Timestamp],
        end_date: Optional[pd.Timestamp]
    ) -> GridLossEstimate:
        now = pd.Timestamp.now()
        
        return GridLossEstimate(
            zone_id=zone_id,
            timestamp_start=start_date or now,
            timestamp_end=end_date or now,
            total_meters=0,
            meters_with_theft=0,
            theft_rate_percentage=0.0,
            total_stolen_kwh=0.0,
            total_reported_kwh=0.0,
            total_actual_kwh=0.0,
            total_financial_loss=0.0,
            average_loss_per_meter=0.0,
            top_offenders=[],
            top_offenders_contribution_percentage=0.0,
            average_confidence=0.0
        )
    
    def generate_report(
        self,
        estimates: Union[EnergyTheftEstimate, List[EnergyTheftEstimate], GridLossEstimate]
    ) -> str:
        if isinstance(estimates, EnergyTheftEstimate):
            return self._generate_single_meter_report(estimates)
        elif isinstance(estimates, list):
            return self._generate_multiple_meters_report(estimates)
        elif isinstance(estimates, GridLossEstimate):
            return self._generate_grid_report(estimates)
        else:
            return "Unknown estimate type"
    
    def _generate_single_meter_report(self, estimate: EnergyTheftEstimate) -> str:
        report = f"""
{'='*80}
ENERGY THEFT ESTIMATE REPORT - METER {estimate.meter_id}
{'='*80}

PERIOD: {estimate.start_date.strftime('%Y-%m-%d %H:%M')} to {estimate.end_date.strftime('%Y-%m-%d %H:%M')}
DURATION: {(estimate.end_date - estimate.start_date).days} days

ENERGY THEFT:
  Total Stolen:           {estimate.total_stolen_kwh:,.2f} kWh
  Average Theft Rate:     {estimate.average_theft_rate_kw:.2f} kW
  Peak Theft Rate:        {estimate.peak_theft_rate_kw:.2f} kW
  Theft Percentage:       {estimate.theft_percentage:.1f}%

CONSUMPTION:
  Reported:               {estimate.reported_consumption_kwh:,.2f} kWh
  Estimated Actual:       {estimate.estimated_actual_consumption_kwh:,.2f} kWh

FINANCIAL IMPACT:
  Estimated Loss:         {estimate.currency} {estimate.estimated_financial_loss:,.2f}
  Price per kWh:          {estimate.currency} {estimate.electricity_price_per_kwh:.4f}

DETECTION CONFIDENCE:
  Average:                {estimate.average_confidence:.2%}
  Range:                  {estimate.min_confidence:.2%} - {estimate.max_confidence:.2%}
  Samples:                {estimate.n_samples}

CLASSIFICATION:
  Severity:               {estimate.severity_level.upper()}
  Priority Score:         {estimate.priority_score:.1f}/100

TIMELINE:
  First Detection:        {estimate.first_detection.strftime('%Y-%m-%d %H:%M')}
  Last Detection:         {estimate.last_detection.strftime('%Y-%m-%d %H:%M')}
  Detection Frequency:    Every {estimate.detection_frequency_hours:.1f} hours

{'='*80}
"""
        return report
    
    def _generate_multiple_meters_report(self, estimates: List[EnergyTheftEstimate]) -> str:
        """Generate summary report for multiple meters"""
        total_stolen = sum(e.total_stolen_kwh for e in estimates)
        total_loss = sum(e.estimated_financial_loss for e in estimates)
        avg_confidence = np.mean([e.average_confidence for e in estimates])
        
        # Sort by stolen energy
        top_10 = sorted(estimates, key=lambda e: e.total_stolen_kwh, reverse=True)[:10]
        
        report = f"""
{'='*80}
MULTI-METER ENERGY THEFT REPORT
{'='*80}

SUMMARY:
  Total Meters:           {len(estimates)}
  Total Stolen Energy:    {total_stolen:,.2f} kWh
  Total Financial Loss:   USD {total_loss:,.2f}
  Average Confidence:     {avg_confidence:.2%}

TOP 10 OFFENDERS:
"""
        for i, est in enumerate(top_10, 1):
            report += f"  {i:2d}. Meter {est.meter_id}: {est.total_stolen_kwh:,.2f} kWh " \
                     f"({est.theft_percentage:.1f}% theft, {est.severity_level})\n"
        
        report += f"\n{'='*80}\n"
        return report
    
    def _generate_grid_report(self, estimate: GridLossEstimate) -> str:
        report = f"""
{'='*80}
GRID-LEVEL LOSS ESTIMATE REPORT - {estimate.zone_id}
{'='*80}

PERIOD: {estimate.timestamp_start.strftime('%Y-%m-%d')} to {estimate.timestamp_end.strftime('%Y-%m-%d')}

GRID OVERVIEW:
  Total Meters:           {estimate.total_meters}
  Meters with Theft:      {estimate.meters_with_theft} ({estimate.theft_rate_percentage:.1f}%)

ENERGY STATISTICS:
  Total Stolen:           {estimate.total_stolen_kwh:,.2f} kWh
  Total Reported:         {estimate.total_reported_kwh:,.2f} kWh
  Total Actual:           {estimate.total_actual_kwh:,.2f} kWh
  Loss Rate:              {(estimate.total_stolen_kwh/estimate.total_actual_kwh*100):.2f}%

FINANCIAL IMPACT:
  Total Loss:             USD {estimate.total_financial_loss:,.2f}
  Average per Meter:      USD {estimate.average_loss_per_meter:,.2f}

TOP OFFENDERS:
  Count:                  {len(estimate.top_offenders)}
  Contribution:           {estimate.top_offenders_contribution_percentage:.1f}%
  Meter IDs:              {', '.join(map(str, estimate.top_offenders[:5]))}...

DETECTION QUALITY:
  Average Confidence:     {estimate.average_confidence:.2%}

{'='*80}
"""
        return report


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("Energy Theft Estimator Module")
    print("="*80)
    print("\nFeatures:")
    print("  ✓ Multiple estimation methods (simple, weighted, interpolation)")
    print("  ✓ Single meter and grid-level estimates")
    print("  ✓ Financial impact calculation")
    print("  ✓ Severity classification and priority scoring")
    print("  ✓ Comprehensive reporting")
    print("\nUsage:")
    print("  from detection.energy_estimator import EnergyEstimator")
    print("  estimator = EnergyEstimator(electricity_price_per_kwh=0.15)")
    print("  estimate = estimator.estimate_single_meter(...)")
    print("  report = estimator.generate_report(estimate)")
    print("="*80)