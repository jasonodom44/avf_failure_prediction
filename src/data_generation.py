import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

#Set random seed for reproducing
np.random.seed(42)

class AVFPatientGenerator:

    def __init__(self, n_patients=1000):
        self.n_patients = n_patients
        self.patients_df = None

    def generate_baseline_characteristics(self):

        age = np.random.gamma(shape = 7, scale = 9, size = self.n_patients)
        age = np.clip(age, 25,90).astype(int)

# ~55% male, 45% female
        sex = np.random.choice(['M', 'F'], size = self.n_patients, p = [0.55, 0.45])
#Diabetes: ~40% prevelance
        diabetes = np.random.choice([0, 1], size = self.n_patients, p = [0.60, 0.40])
# Hypertension: ~80% prevalence
        hypertension = np.random.choice([0, 1], size = self.n_patients, p = [0.20, 0.80])
# Coronary Artery Disease: ~30% prevalence
        cad = np.random.choice([0, 1], size = self.n_patients, p = [0.70, 0.30])
# Peripheral Vascular Disease: ~25% prevalance
        pvd = np.random.choice([0,1], size = self.n_patients, p = [0.75, 0.25])

#Prior Intervention: Most have 0-2, few have many
        prior_interventions = np.random.negative_binomial(n=1, p=0.5, size = self.n_patients)
        prior_interventions = np.clip(prior_interventions, 0, 8)

        #History of CVC use: ~35% have had a catheter
        history_cvc = np.random.choice([0,1], size = self.n_patients, p=[0.65, 0.35])

        #Create baseline risk score (we'll use this to influence outcomes later)
        # This is a "hidden" variable that represents underlying vascular health
        baseline_risk = self._calculate_baseline_risk(
            age, sex, diabetes, hypertension, cad, pvd,
            prior_interventions, history_cvc
        )

        #Create Dataframe
        self.patients_df = pd.DataFrame({
            'patient_id': [f'PT_{i:05d}' for i in range (self.n_patients)],
            'age': age,
            'sex': sex,
            'diabetes': diabetes,
            'hypertension': hypertension,
            'cad': cad,
            'pvd': pvd,
            'prior_interventions': prior_interventions,
            'history_cvc': history_cvc,
            'baseline_risk_score': baseline_risk
        })

        return self.patients_df

    def _calculate_baseline_risk(self, age, sex, diabetes, hypertension, cad, pvd, prior_interventions, history_cvc):

        risk = np.zeros(len(age))

    #Age effect (normalized to 0-1 scale)
        risk += (age - 25) / 65 * 0.2

    #Sex effect (female = higher risk)
        risk += np.where(sex == 'F', 0.15, 0)

    #Comorbidity effects
        risk += diabetes * 0.15
        risk += hypertension * 0.05
        risk += cad * 0.10
        risk += pvd * 0.15

    #Prior interventions (each one increases risk)
        risk += prior_interventions * 0.05

    #History of CVC (major risk factor for central stenosis)
        risk += history_cvc * 0.12

    #Add some random variation (biological variability)
        risk += np.random.normal(0, 0.1, size = len(age))

    #Clip to reasonable range
        risk = np.clip(risk, 0, 1)

        return risk

    def generate_treatment_timeseries(self, n_treatments = 156, treatment_interval_days=2):

        if self.patients_df is None:
            raise ValueError("Must call generate_baseline_characteristics() first")

        all_treatments = []

        for idx, patient in self.patients_df.iterrows():
            patient_id = patient['patient_id']
            baseline_risk = patient['baseline_risk_score']

            #Generate Treatment Data
            start_date = datetime(2024, 1, 1) #Arbitrary Start
            treatment_dates = [start_date + timedelta(days=i*treatment_interval_days)
                               for i in range(n_treatments)]

            #Generate time varying features
            for treatment_num, treatment_date in enumerate(treatment_dates):

                #Calculate progression factor (how far along are wem 0 to 1)
                progression = treatment_num / n_treatments

                #Generate Access Blood Flow (Qa)
                #High_risk patients start lowe and decline faster
                qa_baseline = np.random.normal(900, 150) #healthy baseline
                qa_baseline -= baseline_risk * 300 #High-risk patients start to lower

                #Add progressive decline for high_risk patients
                qa_decline = baseline_risk * progression * 400

                #Add random session-to-session variation
                qa = qa_baseline - qa_decline + np.random.normal(0,50)
                qa = np.clip(qa, 200, 1500) #Physiological limits

                #Generate Arterial Pressure (More negative = harder to draw blood)
                #Normal range: -150 to -200 mmHg
                arterial_pressure = np.random.normal(-175, 25)
                #High-risk and declining Qa makes it worse
                arterial_pressure -= (baseline_risk * 50 + (800-qa) / 20)
                arterial_pressure = np.clip(arterial_pressure, -350, -100)

                #Generate Venous Pressure (higher - outflow obstruction)
                #normal range: 100-180 mmHg
                venous_pressure = np.random.normal(140, 20)
                #Increase with risk and progression
                venous_pressure += baseline_risk * progression * 100
                venous_pressure = np.clip(venous_pressure, 80, 400)

                # Calculate Static Venous Pressure Ratio
                # Assume MAP around 85-95 mmHg
                map_value = np.random.normal(90, 5)
                svpr = (venous_pressure * 0.75) / map_value  # 0.75 converts dynamic to static approximation
                svpr = np.clip(svpr, 0.1, 1.2)

                if baseline_risk < 0.4:
                    svpr = np.clip(svpr, 0.1, 0.5)

                # Generate alarm counts
                # High venous pressure alarms increase with risk and progression
                high_vp_alarms = np.random.poisson(baseline_risk * progression * 5)
                low_ap_alarms = np.random.poisson(baseline_risk * progression * 3)

                # Access Recirculation (should be <10%, spikes when access failing)
                ar_base = np.random.uniform(0, 5)  # Normal baseline

                if qa < 500 or svpr > 0.5:  # Failing access
                    ar_base += np.random.uniform(5, 20)
                access_recirculation = np.clip(ar_base, 0, 40)

                # Kt/V (adequacy) - declines as access fails
                ktv_baseline = np.random.normal(1.4, 0.2)
                ktv_decline = (baseline_risk * progression * 0.4) + ((800 - qa) / 2000)
                ktv = ktv_baseline - ktv_decline
                ktv = np.clip(ktv, 0.6, 2.0)

                # Store treatment data
                treatment_data = {
                    'patient_id': patient_id,
                    'treatment_number': treatment_num + 1,
                    'treatment_date': treatment_date,
                    'access_blood_flow_qa': round(qa, 1),
                    'arterial_pressure_mean': round(arterial_pressure, 1),
                    'venous_pressure_mean': round(venous_pressure, 1),
                    'map': round(map_value, 1),
                    'svpr': round(svpr, 3),
                    'high_vp_alarms': high_vp_alarms,
                    'low_ap_alarms': low_ap_alarms,
                    'access_recirculation_pct': round(access_recirculation, 1),
                    'ktv': round(ktv, 2)
                }

                all_treatments.append(treatment_data)

        self.treatments_df = pd.DataFrame(all_treatments)
        return self.treatments_df

    def generate_failure_outcomes(self, failure_rate=0.30):
        if self.treatments_df is None:
            raise ValueError("Must call generate_treatment_timeseries() first")

            # Merge patient baseline data with treatment data
        full_data = self.treatments_df.merge(
            self.patients_df,
            on='patient_id',
            how='left'
        )

        failure_outcomes = []

        for patient_id in full_data['patient_id'].unique():
            # Get all treatments for this patient
            patient_treatments = full_data[full_data['patient_id'] == patient_id].sort_values('treatment_number')

            # Calculate dynamic risk score for each treatment
            patient_treatments = patient_treatments.copy()

            # Risk factors that accumulate over time
            patient_treatments['qa_risk'] = np.where(
                patient_treatments['access_blood_flow_qa'] < 600,
                (600 - patient_treatments['access_blood_flow_qa']) / 400,  # Normalized 0-1
                0
            )

            patient_treatments['svpr_risk'] = np.where(
                patient_treatments['svpr'] > 0.5,
                (patient_treatments['svpr'] - 0.5) / 0.7,  # Normalized 0-1
                0
            )

            patient_treatments['recirculation_risk'] = np.where(
                patient_treatments['access_recirculation_pct'] > 10,
                (patient_treatments['access_recirculation_pct'] - 10) / 30,  # Normalized 0-1
                0
            )

            patient_treatments['ktv_risk'] = np.where(
                patient_treatments['ktv'] < 1.2,
                (1.2 - patient_treatments['ktv']) / 0.6,  # Normalized 0-1
                0
            )

            patient_treatments['alarm_risk'] = (
                                                       patient_treatments['high_vp_alarms'] + patient_treatments[
                                                   'low_ap_alarms']
                                               ) / 10  # Normalize by expected max
            patient_treatments['alarm_risk'] = patient_treatments['alarm_risk'].clip(0, 1)

            # Calculate cumulative risk (weighted combination)
            patient_treatments['treatment_risk_score'] = (
                    patient_treatments['baseline_risk_score'] * 0.25 +
                    patient_treatments['qa_risk'] * 0.30 +
                    patient_treatments['svpr_risk'] * 0.20 +
                    patient_treatments['recirculation_risk'] * 0.10 +
                    patient_treatments['ktv_risk'] * 0.10 +
                    patient_treatments['alarm_risk'] * 0.05
            )

            # Add temporal acceleration (risk increases faster as time goes on)
            progression_factor = patient_treatments['treatment_number'] / patient_treatments['treatment_number'].max()
            patient_treatments['treatment_risk_score'] *= (1 + progression_factor * 0.5)

            # Clip to 0-1 range
            patient_treatments['treatment_risk_score'] = patient_treatments['treatment_risk_score'].clip(0, 1)

            # Calculate cumulative probability of failure by this treatment
            # Uses logistic function to convert risk score to probability
            patient_treatments['failure_probability'] = 1 / (
                        1 + np.exp(-3 * (patient_treatments['treatment_risk_score'] - 0.85)))

            # Determine if/when failure occurs
            # Draw random number for each treatment; if it exceeds failure probability, access fails
            failure_occurred = False
            failure_treatment = None

            for idx, row in patient_treatments.iterrows():
                if not failure_occurred and row['treatment_number'] > 20:
                    adjusted_prob = row['failure_probability'] * np.random.uniform(0.8,1.2)
                    # Random draw against failure probability
                    if adjusted_prob > 0.5  and np.random.random() < (adjusted_prob - 0.3):
                        failure_occurred = True
                        failure_treatment = row['treatment_number']
                        break

            # Get patient baseline data
            patient_baseline = self.patients_df[self.patients_df['patient_id'] == patient_id].iloc[0]

            # Store outcome
            outcome = {
                'patient_id': patient_id,
                'failed': 1 if failure_occurred else 0,
                'failure_treatment_number': failure_treatment if failure_occurred else None,
                'baseline_risk_score': patient_baseline['baseline_risk_score'],
                'mean_qa': patient_treatments['access_blood_flow_qa'].mean(),
                'min_qa': patient_treatments['access_blood_flow_qa'].min(),
                'final_qa': patient_treatments['access_blood_flow_qa'].iloc[-1],
                'mean_svpr': patient_treatments['svpr'].mean(),
                'max_svpr': patient_treatments['svpr'].max(),
                'mean_recirculation': patient_treatments['access_recirculation_pct'].mean(),
                'total_alarms': (patient_treatments['high_vp_alarms'] + patient_treatments['low_ap_alarms']).sum()
            }

            failure_outcomes.append(outcome)

        self.outcomes_df = pd.DataFrame(failure_outcomes)

        # Adjust failure rate if needed (probabilistic, so might not match exactly)
        actual_failure_rate = self.outcomes_df['failed'].mean()
        print(f"\nTarget failure rate: {failure_rate:.1%}")
        print(f"Actual failure rate: {actual_failure_rate:.1%}")

        return self.outcomes_df

#Test the generator
# Test the generator
if __name__ == '__main__':
    # Generate Patients
    generator = AVFPatientGenerator(n_patients=1000)  # Use 100 for faster testing
    patients = generator.generate_baseline_characteristics()

    # Inspect baseline
    print("=" * 60)
    print("BASELINE PATIENT CHARACTERISTICS")
    print("=" * 60)
    print(f"\nTotal patients generated: {len(patients)}")
    print(f"\nBaseline risk score distribution:")
    print(patients['baseline_risk_score'].describe())

    # Generate treatment timeseries
    print("\n" + "=" * 60)
    print("GENERATING TREATMENT TIMESERIES...")
    print("=" * 60)

    treatments = generator.generate_treatment_timeseries(n_treatments=156)
    print(f"\nTotal treatment records: {len(treatments)}")

    # Generate failure outcomes
    print("\n" + "=" * 60)
    print("GENERATING FAILURE OUTCOMES...")
    print("=" * 60)

    outcomes = generator.generate_failure_outcomes(failure_rate=0.30)

    print(f"\nTotal patients: {len(outcomes)}")
    print(f"\nPatients who failed: {outcomes['failed'].sum()}")
    print(f"Failure rate: {outcomes['failed'].mean():.1%}")

    print(f"\nFailure timing distribution:")
    print(outcomes[outcomes['failed'] == 1]['failure_treatment_number'].describe())

    print(f"\nFirst 10 outcomes:")
    print(outcomes.head(10))

    print(f"\nComparison of failed vs non-failed patients:")
    print(outcomes.groupby('failed')[['baseline_risk_score', 'mean_qa', 'mean_svpr', 'mean_recirculation']].mean())

    #SAVE the Data
    print("\n" + "=" * 60)
    print("Saving Data to files")
    print("=" * 60)

    #Save to CSV file
    patients.to_csv('../data/raw/patients_baseline.csv', index=False)
    treatments.to_csv('../data/raw/treatments_baseline.csv', index=False)
    outcomes.to_csv('../data/raw/outcomes_baseline.csv', index=False)


    print("✓ Saved patients_baseline.csv")
    print("✓ Saved treatments_timeseries.csv")
    print("✓ Saved failure_outcomes.csv")
    print(f"\nTotal file size: ~{(len(patients) + len(treatments) + len(outcomes)) / 1000:.1f}K rows")