# DoAN_ChuyenNganh
Do an chuyen nganh - Model AI du doan that thoat dien(NTL - Non-technical losses)

# Hybrid NTL Detection System
## AI-Powered Non-Technical Loss Detection for Smart Grids

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)

---

## 🎯 Overview

This is a **state-of-the-art hybrid system** for detecting Non-Technical Losses (NTL) in power distribution grids, combining:

- **Machine Learning** (CatBoost Regression) for large-scale screening
- **Physics-based modeling** (PSO + Power Flow) for precise localization
- **SHAP explainability** for transparent decision-making
- **Hybrid fusion** for optimal accuracy

### Key Features

✅ **Dual-Approach Detection**
- ML regression predicts energy recovery (File 1 approach)
- Physics-based PSO+PowerFlow estimates actual consumption (File 2 approach)
- Intelligent fusion combines both for superior accuracy

✅ **Comprehensive Feature Engineering**
- 150+ features including consumption, visit history, and physics-based metrics
- Automatic feature selection using SHAP values

✅ **Explainable AI**
- SHAP values for model interpretation
- Pattern validation (consumption vs visit features)
- Local and global explanations

✅ **Production-Ready**
- Edge computing compatible
- Parallel processing support
- Comprehensive error handling
- Full logging and monitoring

✅ **Energy Theft Estimation**
- Historical energy theft calculation
- Financial impact assessment
- Severity classification
- Priority scoring

---

## 📁 Project Structure
```
ntl-detection-system/
│
├── config/                          # Configuration files
│   ├── config.yaml                  # Main system configuration
│   ├── model_params.yaml            # ML model parameters
│   └── grid_params.yaml             # Grid topology parameters
│
├── data/                            # Data loading and processing
│   ├── data_loader.py               # Load SM and transformer data
│   ├── data_validator.py            # Data quality checks
│   ├── feature_engineering.py       # Feature generation (150+ features)
│   └── preprocessor.py              # Data cleaning
│
├── models/
│   ├── ml_models/
│   │   ├── regression_model.py      # CatBoost regression (File 1)
│   │   └── model_trainer.py         # Training pipeline
│   │
│   ├── physics_models/
│   │   ├── pso_optimizer.py         # PSO implementation (File 2)
│   │   ├── power_flow.py            # Newton-Raphson power flow
│   │   └── grid_simulator.py        # Grid topology simulator
│   │
│   └── hybrid/
│       ├── fusion_model.py          # Hybrid ML+Physics fusion
│       ├── meta_learner.py          # Meta-learning layer
│       └── uncertainty_estimator.py # Confidence scoring
│
├── detection/
│   ├── anomaly_detector.py          # Streak signal & anomaly detection
│   ├── ntl_localizer.py             # Physics-based localization (PSO+PF)
│   ├── energy_estimator.py          # Energy theft estimation
│   └── false_positive_filter.py     # Reduce false positives
│
├── explainability/
│   ├── shap_explainer.py            # SHAP-based explanations
│   ├── physics_interpreter.py       # Physical law interpretation
│   ├── report_generator.py          # Generate explanations
│   └── visualization.py             # Visualize results
│
├── validation/
│   ├── cross_validator.py           # Model validation
│   ├── benchmark.py                 # Performance metrics (MAE, MAPE, NDCG)
│   └── ab_testing.py                # A/B testing framework
│
├── deployment/
│   ├── edge_runtime.py              # Edge device deployment
│   ├── api_server.py                # REST API
│   ├── batch_processor.py           # Batch inference
│   └── monitoring.py                # System monitoring
│
├── scripts/
│   ├── train_ml_model.py            # Train ML model
│   ├── calibrate_physics_model.py   # Calibrate physics parameters
│   ├── run_hybrid_detection.py      # 🚀 MAIN SCRIPT
│   └── generate_reports.py          # Generate reports
│
├── tests/                           # Unit tests
├── notebooks/                       # Jupyter notebooks for analysis
├── docs/                            # Documentation
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/your-org/ntl-detection-system.git
cd ntl-detection-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config/config.yaml`:
```yaml
data:
  consumption_path: "data/smart_meter_readings.csv"
  transformer_path: "data/transformer_readings.csv"
  topology_path: "data/grid_topology.json"
  visits_path: "data/inspection_visits.csv"

ml_model:
  iterations: 1000
  learning_rate: 0.03
  depth: 6
  loss_function: "RMSE"

physics_localization:
  pso_n_particles: 200
  pso_n_iterations: 500
  approach: "ml_prioritization"  # or "zero_consumption", "random"

fusion:
  ml_weight: 0.6
  physics_weight: 0.4
  confidence_threshold: 0.75
```

### 3. Run Detection Pipeline
```bash
# Full pipeline (ML + Physics + Fusion)
python scripts/run_hybrid_detection.py --config config/config.yaml --mode full

# ML only
python scripts/run_hybrid_detection.py --mode ml_only

# Physics only
python scripts/run_hybrid_detection.py --mode physics_only

# With date range
python scripts/run_hybrid_detection.py \
    --start-date 2024-01-01 \
    --end-date 2024-01-31
```

---

## 📊 Data Format

### Smart Meter Data
```csv
timestamp,meter_id,bus_id,active_power_kw,reactive_power_kvar,voltage_pu
2024-01-01 00:00:00,1,1,1.5,0.3,1.02
2024-01-01 00:30:00,1,1,1.8,0.4,1.01
...
```

### Grid Topology
```json
{
  "buses": [
    {
      "bus_id": 0,
      "bus_type": "slack",
      "voltage_magnitude": 1.0,
      "voltage_angle": 0.0
    },
    ...
  ],
  "lines": [
    {
      "from_bus": 0,
      "to_bus": 1,
      "resistance": 0.01,
      "reactance": 0.1
    },
    ...
  ],
  "smart_meters": {
    "0": 1,
    "1": 2,
    ...
  }
}
```

### Inspection Visits (Optional, for training)
```csv
customer_id,visit_date,result,energy_recovered_kwh
12345,2023-12-15,NTL,2540
12346,2023-12-16,non_NTL,0
...
```

---

## 🔬 Methodology

### 1. Feature Engineering

**Consumption Features (File 1):**
- Last 3/12 months consumption
- Min/Max bill ratio
- Consumption vs zone average
- Diff consumption 6 months
- Months with zero consumption

**Physics Features (File 2):**
- Voltage magnitude and angle
- Power factor
- Grid power imbalance
- Distance from transformer
- Reading absences

**Visit History Features:**
- NTL count, last visit result
- Impossible visits, threats
- Total energy recovered

### 2. ML Regression Model (File 1 Approach)
```python
from models.ml_models.regression_model import EnergyRecoveryRegressor, ModelConfig

# Configure
config = ModelConfig(
    iterations=1000,
    learning_rate=0.03,
    loss_function="RMSE"  # Focus on high-energy cases
)

# Train
model = EnergyRecoveryRegressor(config)
model.train(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

**Key Innovation:** Uses **energy recovered (kWh)** as target instead of binary NTL/non-NTL, automatically prioritizing high-value cases.

### 3. Physics-Based Localization (File 2 Approach)
```python
from detection.ntl_localizer import NTLLocalizer, LocalizationConfig

# Configure
config = LocalizationConfig(
    pso_n_particles=200,
    pso_n_iterations=500
)

# Localize
localizer = NTLLocalizer(config)
result = localizer.localize_single_meter(
    meter_id=12345,
    bus_id=5,
    timestamp=pd.Timestamp('2024-01-15 12:00:00'),
    grid_topology=grid_topology,
    consumption_data=consumption_df
)

print(f"Estimated: {result.estimated_active_power_kw:.2f} kW")
print(f"Reported: {result.reported_active_power_kw:.2f} kW")
print(f"Stolen: {result.stolen_active_power_kw:.2f} kW")
print(f"Confidence: {result.confidence:.2%}")
```

**How it works:**
1. Run power flow with reported consumption → get measured voltages
2. Use PSO to find P, Q that minimize voltage deviations
3. Compare estimated vs reported → detect theft

### 4. Hybrid Fusion
```python
from models.hybrid.fusion_model import HybridFusionModel, FusionConfig

config = FusionConfig(
    ml_weight=0.6,
    physics_weight=0.4,
    adaptive_weights=True
)

fusion = HybridFusionModel(config)

fused_scores, confidence, metadata = fusion.fuse_predictions(
    ml_scores=ml_predictions,
    physics_anomalies=physics_scores,
    ml_confidence=ml_conf,
    physics_confidence=physics_conf
)

# Apply decision rules
decisions = fusion.apply_decision_rules(
    fused_scores, ml_scores, physics_scores, confidence
)
```

---

## 📈 Performance Metrics

### File 1 Results (Regression vs Classification)

| Campaign Size | Energy Recovery Improvement |
|--------------|----------------------------|
| n=42 (small) | **+93%** vs classification |
| n=106        | **+69%** vs classification |
| n=211        | **+36%** vs classification |

**NDCG scores:** Regression consistently outperforms classification

### File 2 Results (Physics-based Localization)

| Scenario | MAE (Active Power) | MAE (Reactive Power) |
|----------|-------------------|---------------------|
| Simulated | **0.1391 kW** | **0.0175 kVAr** |
| Real data | **0.1028 kW** | **0.1385 kVAr** |
| Zero consumption | **0.8095 W** | **0.0944 VAr** |

**MAPE:** Typically < 10% for active power

---

## 🧠 Explainability

### SHAP Analysis
```python
from explainability.shap_explainer import SHAPExplainer, SHAPConfig

config = SHAPConfig(max_display=20)
explainer = SHAPExplainer(model, config)

# Setup
explainer.setup_explainer(X_train)

# Global explanation
importance = explainer.explain_global(X_test)

# Local explanation
result = explainer.explain_instance(X_test, instance_idx=0)

# Pattern validation
validation = explainer.validate_model_patterns(
    X_test,
    consumption_features=consumption_cols,
    visit_features=visit_cols
)

print(validation['recommendation'])
```

### Pattern Validation (File 1)

**Regression Model:**
- ✅ 4/8 top features are consumption-related
- ✅ Clear patterns: low consumption → high suspicion
- ✅ Easier to validate by stakeholders

**Classification Model:**
- ❌ Only 1/8 top features are consumption-related
- ❌ Over-reliance on visit history
- ❌ Patterns difficult to interpret

---

## 🎛️ Configuration Options

### Localization Strategies (File 2, Approaches A-D)
```python
# Approach A: Zero consumption
LocalizationStrategy.approach_a_zero_consumption(consumption_data, timestamp)

# Approach B: State estimation
LocalizationStrategy.approach_b_state_estimation(
    consumption_data, timestamp, grid_topology
)

# Approach C: Random sampling
LocalizationStrategy.approach_c_random_sampling(
    consumption_data, timestamp, sample_size=3
)

# Approach D: ML prioritization (RECOMMENDED)
LocalizationStrategy.approach_d_ml_prioritization(
    consumption_data, timestamp, ml_scores, top_k=10
)

# Combined approach
LocalizationStrategy.combined_approach(
    consumption_data, timestamp, grid_topology, ml_scores
)
```

### Fusion Weights

Adjust based on your needs:
```yaml
fusion:
  ml_weight: 0.6          # Higher = trust ML more
  physics_weight: 0.4     # Higher = trust physics more
  adaptive_weights: true  # Auto-adjust based on confidence
```

---

## 📊 Output Reports

### 1. Campaign List
`reports/campaign_list.csv` - High-priority customers for inspection

### 2. Detailed Results
`reports/detailed_results.csv` - All customers with scores

### 3. Energy Estimates
`reports/energy_estimates.csv` - Estimated stolen energy per meter

### 4. Explainability Report
`reports/explainability/shap_report.txt` - Model interpretation

### 5. Summary Statistics
`reports/summary.json` - Aggregate statistics

---

## 🔧 Advanced Usage

### Custom Feature Engineering
```python
from data.feature_engineering import FeatureEngineer, FeatureConfig

config = FeatureConfig(
    lookback_months=12,
    zone_comparison_window=30,
    statistical_features=True,
    physics_features=True
)

engineer = FeatureEngineer(config)
features = engineer.generate_all_features(
    meter_data=consumption_df,
    transformer_data=transformer_df,
    grid_topology=grid_topology,
    historical_visits=visits_df
)
```

### Grid Simulation (Testing)
```python
from models.physics_models.grid_simulator import GridSimulator, GridConfig

config = GridConfig(
    n_feeders=2,
    n_smart_meters=20,
    topology_type=GridTopology.RADIAL
)

simulator = GridSimulator(config)
grid = simulator.generate_grid(seed=42)

# Generate consumption
consumption = simulator.generate_consumption_data(n_timestamps=48)

# Inject fraud
consumption_with_fraud = simulator.inject_fraud(
    consumption,
    fraud_meter_ids=[5, 15],
    fraud_factor=0.2  # Report only 20% of actual
)
```

### Energy Estimation
```python
from detection.energy_estimator import EnergyEstimator, EstimationMethod

estimator = EnergyEstimator(electricity_price_per_kwh=0.15)

estimate = estimator.estimate_single_meter(
    meter_id=12345,
    localization_results=results,
    consumption_data=consumption_df,
    method=EstimationMethod.WEIGHTED_AVERAGE
)

print(estimator.generate_report(estimate))
```

---

## 🧪 Testing
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_models.py

# With coverage
pytest --cov=. tests/
```

---

## 📚 References

### Academic Papers

1. **File 1:** Coma-Puig, B., & Carmona, J. (2022). "Non-technical losses detection in energy consumption focusing on energy recovery and explainability." *Machine Learning*, 111, 487-517.

2. **File 2:** Livanos, N.-A., et al. (2026). "Expert system for non-technical loss detection in power distribution grids using particle swarm optimization and nested power flow integration." *Expert Systems With Applications*, 296, 128997.

### Key Concepts

- **Non-Technical Losses (NTL):** Energy losses due to theft, meter tampering, or billing errors
- **Shapley Values:** Game-theoretic approach to feature attribution
- **PSO (Particle Swarm Optimization):** Metaheuristic optimization inspired by swarm behavior
- **Power Flow Analysis:** Calculation of voltages/currents in electrical networks
- **NDCG (Normalized Discounted Cumulative Gain):** Ranking quality metric

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## 📧 Contact

For questions or support:
- **Email:** support@ntl-detection.com
- **Issues:** [GitHub Issues](https://github.com/your-org/ntl-detection-system/issues)
- **Documentation:** [Full Docs](https://docs.ntl-detection.com)

---

## 🙏 Acknowledgments

- Based on research by Coma-Puig & Carmona (2022) and Livanos et al. (2026)
- CatBoost library by Yandex
- SHAP library by Scott Lundberg
- Pandapower for power flow calculations

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star!

---

**Built with ❤️ for a sustainable energy future** 🌍⚡