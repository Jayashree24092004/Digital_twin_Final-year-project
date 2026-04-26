import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11

# Color palette
COLORS = {'baseline': '#E74C3C', 'forecast': '#3498DB', 'optimized': '#2ECC71', 
          'anomaly': '#E74C3C', 'primary': '#2C3E50'}

np.random.seed(42)
print('✅ All imports successful!')

class AstronautDataGenerator:
    """Generate synthetic astronaut health data with realistic microgravity effects."""
    
    def __init__(self, mission_days=365):
        self.days = mission_days
        self.t = np.arange(mission_days)
        
    def _add_noise(self, signal, noise_level=0.02):
        return signal + np.random.normal(0, noise_level * np.abs(signal).mean(), len(signal))
    
    def _circadian_disruption(self):
        """Circadian rhythm disruption factor (increases over time)."""
        return 1 + 0.15 * (1 - np.exp(-self.t / 120))
    
    def generate_heart_rate(self):
        """HR: Slight increase due to cardiovascular deconditioning."""
        baseline = 65
        trend = 8 * (1 - np.exp(-self.t / 180))
        circadian = 3 * np.sin(2 * np.pi * self.t / 1)
        return self._add_noise(baseline + trend + circadian, 0.03)
    
    def generate_hrv(self):
        """HRV: Decreases due to autonomic dysfunction."""
        baseline = 55
        decline = -12 * (1 - np.exp(-self.t / 150)) * self._circadian_disruption()
        return np.clip(self._add_noise(baseline + decline, 0.05), 25, 70)

        
    def generate_muscle_mass(self):
        """Muscle mass %: ~1-2% loss per month initially, then slows."""
        baseline = 42
        decline = -8 * (1 - np.exp(-self.t / 200))
        return np.clip(self._add_noise(baseline + decline, 0.01), 30, 45)
    
    def generate_bone_density(self):
        """Bone density %: ~1% loss per month."""
        baseline = 100
        decline = -12 * (self.t / 365)
        return np.clip(self._add_noise(baseline + decline, 0.008), 80, 102)
    
    def generate_vo2max(self):
        """VO2 max: Declines due to cardiovascular deconditioning."""
        baseline = 48
        decline = -10 * (1 - np.exp(-self.t / 180))
        return np.clip(self._add_noise(baseline + decline, 0.02), 32, 52)
    
    def generate_sleep_quality(self):
        """Sleep quality index (0-100): Disrupted by microgravity."""
        baseline = 85
        decline = -18 * (1 - np.exp(-self.t / 100))
        weekly_var = 5 * np.sin(2 * np.pi * self.t / 7)
        return np.clip(self._add_noise(baseline + decline + weekly_var, 0.04), 40, 95)
        
    def generate_radiation(self):
        """Cumulative radiation exposure (mSv)."""
        daily_dose = 0.5 + 0.3 * np.random.random(self.days)
        solar_events = np.zeros(self.days)
        event_days = np.random.choice(self.days, 8, replace=False)
        solar_events[event_days] = np.random.uniform(5, 20, 8)
        return np.cumsum(daily_dose + solar_events)
    
    def generate_cognitive_score(self, radiation):
        """Cognitive performance: Affected by radiation, sleep, stress."""
        baseline = 95
        rad_effect = -0.08 * radiation
        time_decline = -5 * (1 - np.exp(-self.t / 250))
        return np.clip(self._add_noise(baseline + rad_effect + time_decline, 0.03), 60, 100)
    
    def generate_stress_index(self):
        """Stress index (0-100): Increases with mission duration."""
        baseline = 25
        trend = 20 * (1 - np.exp(-self.t / 200))
        events = np.zeros(self.days)
        stress_days = np.random.choice(self.days, 15, replace=False)
        events[stress_days] = np.random.uniform(10, 25, 15)
        return np.clip(self._add_noise(baseline + trend + events, 0.05), 10, 90)
    
    def generate_workload(self):
        """Daily workload intensity (0-100)."""
        base = 60 + 10 * np.sin(2 * np.pi * self.t / 30)
        return np.clip(self._add_noise(base, 0.1), 30, 95)
    
    def generate_complete_dataset(self):
        """Generate complete astronaut health dataset."""
        radiation = self.generate_radiation()
        data = pd.DataFrame({
            'day': self.t,
            'heart_rate': self.generate_heart_rate(),
            'hrv': self.generate_hrv(),
            'muscle_mass': self.generate_muscle_mass(),
            'bone_density': self.generate_bone_density(),
            'vo2_max': self.generate_vo2max(),
            'sleep_quality': self.generate_sleep_quality(),
            'radiation_cumulative': radiation,
            'cognitive_score': self.generate_cognitive_score(radiation),
            'stress_index': self.generate_stress_index(),
            'workload': self.generate_workload()
        })
        return data

# Generate baseline data
generator = AstronautDataGenerator(365)
baseline_data = generator.generate_complete_dataset()
print(f'✅ Generated {len(baseline_data)} days of astronaut health data')
baseline_data.head(10)


class DigitalTwinPredictor:
    """Digital Twin model for health trajectory prediction."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.target_cols = ['muscle_mass', 'bone_density', 'hrv', 'vo2_max', 
                           'cognitive_score', 'sleep_quality', 'stress_index']
        
    def prepare_features(self, data, lookback=7):
        """Create lagged features for time series prediction."""
        df = data.copy()
        for col in self.target_cols:
            for lag in range(1, lookback + 1):
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
        return df.dropna()
    
    def train(self, data):
        """Train prediction models for each health metric."""
        df = self.prepare_features(data)
        feature_cols = [c for c in df.columns if 'lag' in c or c in ['day', 'day_sin', 'day_cos', 'workload', 'radiation_cumulative']]
        
        for target in self.target_cols:
            X = df[feature_cols]
            y = df[target]
            
            self.scalers[target] = StandardScaler()
            X_scaled = self.scalers[target].fit_transform(X)
            
            self.models[target] = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            self.models[target].fit(X_scaled, y)
            self.feature_importance[target] = dict(zip(feature_cols, self.models[target].feature_importances_))
        
        print('✅ Digital Twin models trained successfully')
        return self
    
    def predict_trajectory(self, data, horizon=30):
        """Predict future health trajectories."""
        df = self.prepare_features(data)
        predictions = {target: [] for target in self.target_cols}
        
        feature_cols = [c for c in df.columns if 'lag' in c or c in ['day', 'day_sin', 'day_cos', 'workload', 'radiation_cumulative']]
        last_row = df.iloc[-1:].copy()
        
        for day_ahead in range(1, horizon + 1):
            new_day = last_row['day'].values[0] + day_ahead
            last_row['day'] = new_day
            last_row['day_sin'] = np.sin(2 * np.pi * new_day / 365)
            last_row['day_cos'] = np.cos(2 * np.pi * new_day / 365)
            
            for target in self.target_cols:
                X = last_row[feature_cols]
                X_scaled = self.scalers[target].transform(X)
                pred = self.models[target].predict(X_scaled)[0]
                predictions[target].append(pred)
        
        return pd.DataFrame(predictions)

# Train Digital Twin
dt_predictor = DigitalTwinPredictor()
dt_predictor.train(baseline_data)

# Generate predictions
pred_7d = dt_predictor.predict_trajectory(baseline_data[:200], horizon=7)
pred_30d = dt_predictor.predict_trajectory(baseline_data[:200], horizon=30)
pred_180d = dt_predictor.predict_trajectory(baseline_data[:180], horizon=180)
print(f'✅ Generated predictions: 7-day, 30-day, 180-day trajectories')


class AnomalyDetector:
    """Detect physiological anomalies in astronaut health data."""
    
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        
    def fit_detect(self, data):
        """Fit model and detect anomalies."""
        features = data[['heart_rate', 'hrv', 'muscle_mass', 'vo2_max', 
                        'cognitive_score', 'stress_index', 'sleep_quality']]
        X_scaled = self.scaler.fit_transform(features)
        anomalies = self.model.fit_predict(X_scaled)
        return anomalies == -1

# Detect anomalies
anomaly_detector = AnomalyDetector()
baseline_data['is_anomaly'] = anomaly_detector.fit_detect(baseline_data)
n_anomalies = baseline_data['is_anomaly'].sum()
print(f'✅ Detected {n_anomalies} anomalous health events')


class InterventionSimulator:
    """Simulate Digital Twin-guided countermeasures."""
    
    def __init__(self, baseline_data):
        self.baseline = baseline_data.copy()
        self.days = len(baseline_data)
        
    def apply_exercise_protocol(self, intensity=0.7):
        """High-intensity exercise countermeasure."""
        effect = {
            'muscle_mass': 0.6 * intensity,
            'bone_density': 0.4 * intensity,
            'vo2_max': 0.7 * intensity,
            'hrv': 0.3 * intensity
        }
        return effect
    
    def apply_sleep_correction(self, intensity=0.8):
        """Sleep optimization protocol."""
        return {
            'sleep_quality': 0.6 * intensity,
            'cognitive_score': 0.4 * intensity,
            'stress_index': -0.3 * intensity,
            'hrv': 0.25 * intensity
        }
    
    def apply_workload_management(self, intensity=0.6):
        """Workload optimization."""
        return {
            'stress_index': -0.5 * intensity,
            'cognitive_score': 0.3 * intensity,
            'sleep_quality': 0.2 * intensity
        }
    
    def simulate_optimized(self, plan='balanced'):
        """Simulate optimized health trajectory with interventions."""
        optimized = self.baseline.copy()
        
        if plan == 'A':  # Exercise-focused
            effects = {**self.apply_exercise_protocol(0.9), **self.apply_sleep_correction(0.3)}
        elif plan == 'B':  # Balanced
            effects = {**self.apply_exercise_protocol(0.6), **self.apply_sleep_correction(0.6), 
                      **self.apply_workload_management(0.5)}
        else:  # plan C - Recovery focused
            effects = {**self.apply_exercise_protocol(0.4), **self.apply_sleep_correction(0.9),
                      **self.apply_workload_management(0.8)}
        
        for metric, effect in effects.items():
            if metric in optimized.columns:
                intervention_curve = effect * (1 - np.exp(-np.arange(self.days) / 60))
                if metric == 'stress_index':
                    optimized[metric] = np.clip(self.baseline[metric] + intervention_curve * 15, 10, 80)
                elif metric in ['muscle_mass', 'bone_density']:
                    optimized[metric] = np.clip(self.baseline[metric] - intervention_curve * 5 + effect * 8, 
                                               self.baseline[metric].min(), self.baseline[metric].max() * 1.02)
                else:
                    delta = (self.baseline[metric].iloc[0] - self.baseline[metric]) * intervention_curve
                    optimized[metric] = self.baseline[metric] + delta * 0.7
        
        return optimized

# Generate intervention scenarios
simulator = InterventionSimulator(baseline_data)
plan_a = simulator.simulate_optimized('A')
plan_b = simulator.simulate_optimized('B')
plan_c = simulator.simulate_optimized('C')
print('✅ Generated 3 intervention scenarios (A: Exercise, B: Balanced, C: Recovery)')

def generate_intervention_heatmap(data, days=30):
    """Generate daily intervention recommendations."""
    recommendations = np.zeros((4, days))
    labels = ['Exercise', 'Nutrition', 'Sleep', 'Workload']
    
    for d in range(days):
        idx = min(d, len(data) - 1)
        # Exercise need based on muscle/VO2
        recommendations[0, d] = max(0, (42 - data['muscle_mass'].iloc[idx]) / 10)
        # Nutrition need
        recommendations[1, d] = max(0, (100 - data['bone_density'].iloc[idx]) / 15)
        # Sleep correction need
        recommendations[2, d] = max(0, (85 - data['sleep_quality'].iloc[idx]) / 30)
        # Workload reduction need
        recommendations[3, d] = max(0, (data['stress_index'].iloc[idx] - 30) / 40)
    
    return np.clip(recommendations, 0, 1), labels

intervention_matrix, intervention_labels = generate_intervention_heatmap(baseline_data, 60)
print('✅ Generated intervention recommendation matrix')

# A. Multi-parameter Health Degradation Chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = [('muscle_mass', 'Muscle Mass (%)', 'tab:blue'),
           ('bone_density', 'Bone Density (%)', 'tab:orange'),
           ('hrv', 'HRV (ms)', 'tab:green'),
           ('vo2_max', 'VO2 Max (mL/kg/min)', 'tab:red')]

for ax, (col, title, color) in zip(axes.flat, metrics):
    ax.plot(baseline_data['day'], baseline_data[col], color=COLORS['baseline'], 
            label='Baseline Decline', linewidth=2, alpha=0.8)
    ax.plot(plan_b['day'], plan_b[col], color=COLORS['optimized'], 
            label='DT-Optimized', linewidth=2, linestyle='--')
    
    # Add forecast line
    forecast_start = 200
    if col in pred_30d.columns:
        forecast_days = np.arange(forecast_start, forecast_start + len(pred_30d))
        ax.plot(forecast_days, pred_30d[col], color=COLORS['forecast'], 
                label='DT Forecast (30d)', linewidth=2, linestyle=':')
    
    ax.set_xlabel('Mission Day')
    ax.set_ylabel(title)
    ax.set_title(f'{title} Over 12-Month Mission', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.set_xlim(0, 365)

plt.suptitle('Multi-Parameter Health Degradation Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('health_degradation.png', dpi=150, bbox_inches='tight')
plt.show()

# B. Radiation Exposure vs Cognitive Performance
fig, ax1 = plt.subplots(figsize=(12, 5))

color1 = '#E74C3C'
ax1.set_xlabel('Mission Day', fontsize=12)
ax1.set_ylabel('Cumulative Radiation (mSv)', color=color1, fontsize=12)
ax1.plot(baseline_data['day'], baseline_data['radiation_cumulative'], color=color1, linewidth=2.5)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.fill_between(baseline_data['day'], 0, baseline_data['radiation_cumulative'], alpha=0.2, color=color1)

ax2 = ax1.twinx()
color2 = '#3498DB'
ax2.set_ylabel('Cognitive Performance Score', color=color2, fontsize=12)
ax2.plot(baseline_data['day'], baseline_data['cognitive_score'], color=color2, linewidth=2.5)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Radiation Exposure vs Cognitive Performance', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig('radiation_cognitive.png', dpi=150, bbox_inches='tight')
plt.show()

# C. Anomaly Detection Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Heart Rate with anomalies
ax1 = axes[0]
ax1.plot(baseline_data['day'], baseline_data['heart_rate'], color='#3498DB', linewidth=1.5, label='Heart Rate')
anomaly_mask = baseline_data['is_anomaly']
ax1.scatter(baseline_data.loc[anomaly_mask, 'day'], baseline_data.loc[anomaly_mask, 'heart_rate'],
           color=COLORS['anomaly'], s=80, zorder=5, label='Anomaly Detected', marker='o', edgecolors='black')
ax1.set_ylabel('Heart Rate (bpm)')
ax1.set_title('Heart Rate with Anomaly Detection', fontweight='bold')
ax1.legend()

# Stress Index with anomalies
ax2 = axes[1]
ax2.plot(baseline_data['day'], baseline_data['stress_index'], color='#9B59B6', linewidth=1.5, label='Stress Index')
ax2.scatter(baseline_data.loc[anomaly_mask, 'day'], baseline_data.loc[anomaly_mask, 'stress_index'],
           color=COLORS['anomaly'], s=80, zorder=5, label='Anomaly Detected', marker='o', edgecolors='black')
ax2.set_xlabel('Mission Day')
ax2.set_ylabel('Stress Index')
ax2.set_title('Stress Index with Anomaly Detection', fontweight='bold')
ax2.legend()

plt.suptitle('Physiological Anomaly Detection', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('anomaly_detection.png', dpi=150, bbox_inches='tight')
plt.show()

# D. Feature Importance (Explainable AI)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
key_metrics = ['muscle_mass', 'cognitive_score', 'vo2_max', 'stress_index']

for ax, metric in zip(axes.flat, key_metrics):
    importance = dt_predictor.feature_importance.get(metric, {})
    # Get top 10 features
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    features, values = zip(*sorted_imp) if sorted_imp else ([], [])
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))
    bars = ax.barh(range(len(features)), values, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([f.replace('_', ' ').title()[:20] for f in features], fontsize=9)
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'{metric.replace("_", " ").title()} - Key Predictors', fontweight='bold')
    ax.invert_yaxis()

plt.suptitle('Explainable AI: Feature Importance Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# E. Scenario Analysis Comparison
def calculate_health_score(data):
    """Calculate overall health score at end of mission."""
    end_idx = -1
    return {
        'Muscle Mass': data['muscle_mass'].iloc[end_idx],
        'Bone Density': data['bone_density'].iloc[end_idx],
        'VO2 Max': data['vo2_max'].iloc[end_idx],
        'Cognitive': data['cognitive_score'].iloc[end_idx],
        'Sleep Quality': data['sleep_quality'].iloc[end_idx]
    }

scores_baseline = calculate_health_score(baseline_data)
scores_a = calculate_health_score(plan_a)
scores_b = calculate_health_score(plan_b)
scores_c = calculate_health_score(plan_c)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(scores_baseline))
width = 0.2

bars1 = ax.bar(x - 1.5*width, list(scores_baseline.values()), width, label='Baseline (No Intervention)', color='#E74C3C', alpha=0.8)
bars2 = ax.bar(x - 0.5*width, list(scores_a.values()), width, label='Plan A (Exercise Focus)', color='#3498DB', alpha=0.8)
bars3 = ax.bar(x + 0.5*width, list(scores_b.values()), width, label='Plan B (Balanced)', color='#2ECC71', alpha=0.8)
bars4 = ax.bar(x + 1.5*width, list(scores_c.values()), width, label='Plan C (Recovery Focus)', color='#9B59B6', alpha=0.8)

ax.set_ylabel('End-of-Mission Value')
ax.set_title('Scenario Analysis: Countermeasure Plan Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(list(scores_baseline.keys()))
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('scenario_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# F. Digital Twin Health Radar Chart
def create_radar_chart(ax, data_dict, title):
    categories = list(data_dict.keys())
    n = len(categories)
    angles = [i / float(n) * 2 * np.pi for i in range(n)]
    angles += angles[:1]
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    
    return angles

# Normalize metrics to 0-100 scale
def normalize_health(data, day_idx):
    return {
        'Physical\nFitness': (data['muscle_mass'].iloc[day_idx] / 42) * 100,
        'Cognitive\nHealth': data['cognitive_score'].iloc[day_idx],
        'Stress\nManagement': 100 - data['stress_index'].iloc[day_idx],
        'Cardio-\nvascular': (data['vo2_max'].iloc[day_idx] / 48) * 100,
        'Sleep\nQuality': data['sleep_quality'].iloc[day_idx]
    }

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Baseline at start
baseline_start = normalize_health(baseline_data, 0)
# Current (day 180)
current = normalize_health(baseline_data, 180)
# Optimized (day 180)
optimized = normalize_health(plan_b, 180)

angles = create_radar_chart(ax, baseline_start, 'Health Radar')

for data, label, color, alpha in [
    (baseline_start, 'Baseline (Day 0)', '#2ECC71', 0.3),
    (current, 'Current (Day 180)', '#E74C3C', 0.3),
    (optimized, 'DT-Optimized', '#3498DB', 0.3)
]:
    values = list(data.values())
    values += values[:1]
    ax.plot(angles, values, linewidth=2.5, label=label, color=color)
    ax.fill(angles, values, alpha=alpha, color=color)

ax.set_ylim(0, 110)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title('Digital Twin Health Radar Chart', fontsize=14, fontweight='bold', y=1.08)
plt.tight_layout()
plt.savefig('health_radar.png', dpi=150, bbox_inches='tight')
plt.show()

# G. Intervention Recommendation Heatmap
fig, ax = plt.subplots(figsize=(14, 4))

im = ax.imshow(intervention_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax.set_yticks(range(4))
ax.set_yticklabels(intervention_labels)
ax.set_xlabel('Mission Day')
ax.set_title('Daily Intervention Recommendation Intensity', fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Intervention Intensity', rotation=270, labelpad=15)

# Add day markers
ax.set_xticks(np.arange(0, 60, 10))
ax.set_xticklabels(np.arange(0, 60, 10))

plt.tight_layout()
plt.savefig('intervention_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()