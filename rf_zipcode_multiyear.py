import argparse
import os

import numpy as np
import pandas as pd

# Use a non-interactive backend (Render is headless).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RF county forecast + zip conversion (demo-friendly).")
    ap.add_argument("--input", required=True, help="County general data CSV path")
    ap.add_argument("--zip-map", required=True, help="zip_to_county.csv mapping CSV path")
    ap.add_argument("--output", required=True, help="Output CSV path for county forecasts")
    ap.add_argument(
        "--zip-output-dir",
        required=True,
        help="Directory to write zipcode prediction CSVs into (one CSV per prediction year)",
    )

    ap.add_argument("--train-start-year", type=int, default=2011)
    ap.add_argument("--train-end-year", type=int, default=2023)
    ap.add_argument("--prediction-years", type=str, default="2024,2025,2026")
    ap.add_argument("--county-to-graph", type=str, default="Allen County")

    # Speed knobs for demos (reduces the number of county models trained).
    ap.add_argument("--max-counties", type=int, default=10, help="Max counties to process (limit runtime)")
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=6)

    ap.add_argument("--no-plots", action="store_true", help="Do not call plt.show() (still saves plots)")
    ap.add_argument("--no-zip", action="store_true", help="Skip converting county predictions to zip predictions")
    return ap.parse_args()


args = parse_args()

# Keep the original variable names used throughout this script, but source them from CLI args.
input_path = args.input
output_path = args.output
zip_to_county_path = args.zip_map
zipcode_output_dir = args.zip_output_dir

TRAIN_START_YEAR = args.train_start_year
TRAIN_END_YEAR = args.train_end_year
PREDICTION_YEARS = [int(x.strip()) for x in args.prediction_years.split(",") if x.strip()]
COUNTY_TO_GRAPH = args.county_to_graph

MAX_COUNTIES = args.max_counties
RF_N_ESTIMATORS = args.n_estimators
RF_MAX_DEPTH = args.max_depth
NO_PLOTS = args.no_plots
NO_ZIP = args.no_zip

output_dir = os.path.dirname(output_path) or "."
os.makedirs(output_dir, exist_ok=True)
os.makedirs(zipcode_output_dir, exist_ok=True)

# 1. Load Data
print("Loading data...")
df_original = pd.read_csv(input_path)  # Keep original unfiltered data for graphing

print(f"Loaded {len(df_original)} rows")
print(f"Columns: {list(df_original.columns)}")
print(f"Years in dataset: {sorted(df_original['year'].unique())}")
print(f"Counties in dataset: {df_original['county'].nunique()}")
print(f"\nAvailable counties: {sorted(df_original['county'].unique())}")

# 2. Data Validation and Cleaning
print("\nCleaning data...")

# Define feature columns (everything except county and year)
feature_columns = ['population', 'housing_per_zip_county', 'income_per_zip_county', 
                   'employment', 'gross product']

# Clean numeric columns in original data
for col in feature_columns:
    df_original[col] = pd.to_numeric(df_original[col], errors='coerce').fillna(0)

# Ensure year is integer
df_original['year'] = df_original['year'].astype(int)

# Create a copy for training
df = df_original.copy()

# 3. Filter data by training year range
print("\nFiltering training data by year range...")
if TRAIN_START_YEAR:
    df = df[df['year'] >= TRAIN_START_YEAR]
if TRAIN_END_YEAR:
    df = df[df['year'] <= TRAIN_END_YEAR]

print(f"Training on years: {df['year'].min()} to {df['year'].max()}")
print(f"Data points used for training: {len(df)} rows")
print(f"Full dataset years available: {df_original['year'].min()} to {df_original['year'].max()}")

# 4. Calculate Population Growth Rate for Each County (on training data)
print("\nCalculating population growth rates on training data...")

# Sort by county and year
df = df.sort_values(['county', 'year']).reset_index(drop=True)

# Calculate growth rate: (current - previous) / previous * 100
df['population_growth_rate'] = np.nan

for county in df['county'].unique():
    county_mask = df['county'] == county
    county_data = df[county_mask].copy()
    
    # Calculate growth rate for each year (except the first)
    for i in range(1, len(county_data)):
        current_idx = county_data.index[i]
        previous_idx = county_data.index[i-1]
        
        current_pop = df.loc[current_idx, 'population']
        previous_pop = df.loc[previous_idx, 'population']
        
        if previous_pop != 0:
            growth_rate = ((current_pop - previous_pop) / previous_pop) * 100
            df.loc[current_idx, 'population_growth_rate'] = growth_rate

if not NO_PLOTS:
    # Also calculate growth rates for original data (for comparison on graph)
    print("\nCalculating population growth rates on full dataset (for graphing)...")
    df_original = df_original.sort_values(['county', 'year']).reset_index(drop=True)
    df_original['population_growth_rate'] = np.nan

    for county in df_original['county'].unique():
        county_mask = df_original['county'] == county
        county_data = df_original[county_mask].copy()
        
        for i in range(1, len(county_data)):
            current_idx = county_data.index[i]
            previous_idx = county_data.index[i-1]
            
            current_pop = df_original.loc[current_idx, 'population']
            previous_pop = df_original.loc[previous_idx, 'population']
            
            if previous_pop != 0:
                growth_rate = ((current_pop - previous_pop) / previous_pop) * 100
                df_original.loc[current_idx, 'population_growth_rate'] = growth_rate

# 5. Add Lagged Growth Rate Features (t-1 and t-2) on training data
print("\nAdding lagged growth rate features (t-1 and t-2)...")

df['growth_rate_lag1'] = np.nan  # Previous year's growth rate
df['growth_rate_lag2'] = np.nan  # Two years ago growth rate

for county in df['county'].unique():
    county_mask = df['county'] == county
    county_indices = df[county_mask].index
    
    for i, idx in enumerate(county_indices):
        current_growth = df.loc[idx, 'population_growth_rate']
        
        # Set lag1 (t-1): previous year's growth rate
        if i >= 1:
            df.loc[idx, 'growth_rate_lag1'] = df.loc[county_indices[i-1], 'population_growth_rate']
        else:
            # For first year with growth rate, use current year's value
            df.loc[idx, 'growth_rate_lag1'] = current_growth
        
        # Set lag2 (t-2): two years ago growth rate
        if i >= 2:
            df.loc[idx, 'growth_rate_lag2'] = df.loc[county_indices[i-2], 'population_growth_rate']
        else:
            # For first or second year, use current year's value
            df.loc[idx, 'growth_rate_lag2'] = current_growth

print("Sample data with lagged features:")
print(df[['county', 'year', 'population_growth_rate', 'growth_rate_lag1', 'growth_rate_lag2']].head(20))

# 6. Organize Data by County
county_timeseries = {}
county_timeseries_full = {}  # For full dataset

for county in df['county'].unique():
    county_data = df[df['county'] == county].sort_values('year')
    # Only include rows where growth rate is available
    county_data_with_growth = county_data[county_data['population_growth_rate'].notna()]
    county_timeseries[county] = county_data_with_growth

if not NO_PLOTS:
    # Organize full original data
    for county in df_original['county'].unique():
        county_data_full = df_original[df_original['county'] == county].sort_values('year')
        county_timeseries_full[county] = county_data_full

print(f"\nTotal unique counties in training: {len(county_timeseries)}")
print(f"Total unique counties in full dataset: {len(county_timeseries_full)}")

# 7. Runtime Estimation
print(f"\nRuntime Estimate:")
print(f"Counties: {len(county_timeseries)}")
print(f"Features per model: {len(feature_columns) + 3} (year + all parameters + 2 lagged growth rates)")
estimated_seconds = len(county_timeseries) * 0.05
print(f"Estimated runtime: {estimated_seconds:.1f} seconds")

# 8. Generate Predictions for Custom Years
num_future_years = len(PREDICTION_YEARS)
results = []

print(f"\nGenerating Random Forest predictions for population growth rate...")
print(f"Predicting {num_future_years} future years: {PREDICTION_YEARS}")
print(f"Using features: {feature_columns} + growth_rate_lag1 + growth_rate_lag2")

# Progress bar for counties
processed_count = 0
for county, county_data in tqdm(county_timeseries.items(), desc="Processing Counties", unit="county"):
    
    if processed_count >= MAX_COUNTIES:
        break

    # Only process if we have enough years of data
    if len(county_data) >= 5:
        
        # Prepare feature matrix: [year, population, housing, income, employment, gross_product, lag1, lag2]
        X_train_list = []
        y_train_list = []
        
        for idx, row in county_data.iterrows():
            # Features: year + all parameter values + lagged growth rates
            feature_row = ([row['year']] + 
                          [row[col] for col in feature_columns] + 
                          [row['growth_rate_lag1'], row['growth_rate_lag2']])
            X_train_list.append(feature_row)
            
            # Target: population growth rate (%)
            y_train_list.append(row['population_growth_rate'])
        
        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        
        # Train Random Forest model to predict growth rate
        rf_model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            random_state=42,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=2,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Make iterative predictions for custom years
        # Start with the last known values FROM TRAINING DATA
        last_row = county_data.iloc[-1]
        second_last_row = county_data.iloc[-2] if len(county_data) >= 2 else last_row
        
        # Get the most recent actual population from the TRAINING dataframe
        original_county_data = df[df['county'] == county].sort_values('year')
        last_actual_year = original_county_data.iloc[-1]['year']
        last_actual_population = original_county_data.iloc[-1]['population']
        
        # Initialize current values
        current_values = {col: last_row[col] for col in feature_columns}
        current_population = last_actual_population
        
        # Initialize lagged growth rates
        lag1_growth = last_row['population_growth_rate']
        lag2_growth = second_last_row['population_growth_rate']
        
        predictions_growth = []
        predictions_population = []
        
        for future_year in PREDICTION_YEARS:
            # Create feature vector: [year, pop, housing, income, employment, gdp, lag1, lag2]
            feature_vector = ([future_year] + 
                            [current_values[col] for col in feature_columns] + 
                            [lag1_growth, lag2_growth])
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Predict growth rate (%)
            growth_rate_prediction = rf_model.predict(feature_vector)[0]
            predictions_growth.append(round(growth_rate_prediction, 3))
            
            # Convert growth rate to actual population
            # Formula: new_population = current_population * (1 + growth_rate/100)
            new_population = current_population * (1 + growth_rate_prediction / 100)
            predictions_population.append(round(new_population, 3))
            
            # Update values for next iteration
            current_population = new_population
            current_values['population'] = new_population
            
            # Update lagged growth rates for next prediction
            # Shift the lags: current prediction becomes lag1, lag1 becomes lag2
            lag2_growth = lag1_growth
            lag1_growth = growth_rate_prediction
        
        # Store results dynamically based on PREDICTION_YEARS
        result_row = {
            'county': county,
            f'population_{last_actual_year}': round(last_actual_population, 3),
        }
        
        # Add predictions for each year
        for i, pred_year in enumerate(PREDICTION_YEARS):
            result_row[f'growth_rate_{pred_year}'] = predictions_growth[i]
            result_row[f'population_{pred_year}'] = predictions_population[i]
        
        results.append(result_row)
        processed_count += 1

print(f"\nGenerated predictions for {len(results)} counties")

# 9. Save Results
if len(results) > 0:
    predictions_df = pd.DataFrame(results)
    
    # Round to 3 decimal places
    numeric_cols = predictions_df.select_dtypes(include=[np.number]).columns
    predictions_df[numeric_cols] = predictions_df[numeric_cols].round(3)
    
    # Save
    predictions_df.to_csv(output_path, index=False, float_format='%.3f')
    
    print(f"\nPredictions successfully saved to '{output_path}'")
    print(f"Shape: {predictions_df.shape}")
    print(f"\nResults:")
    print(predictions_df.to_string(index=False))
else:
    print("\nNo predictions generated - no counties had enough data!")

if not NO_ZIP:
    # 10. Convert County Predictions to Zip Code Predictions
    print("\n" + "="*60)
    print("CONVERTING TO ZIP CODE PREDICTIONS")
    print("="*60)

    # Load zip to county mapping
    print(f"\nLoading zip to county mapping from: {zip_to_county_path}")
    zip_to_county_df = pd.read_csv(zip_to_county_path, dtype={'ZIP': str})

    print(f"Loaded {len(zip_to_county_df)} zip-county mappings")
    print(f"Columns: {list(zip_to_county_df.columns)}")
    print("\nSample data:")
    print(zip_to_county_df.head(5))

    # Strip " County" suffix from the zip CSV county column (e.g. "Madison County" -> "Madison")
    zip_to_county_df['county_name'] = zip_to_county_df['County'].str.replace(r'\s+County$', '', regex=True).str.strip()

    # Also strip " County" from predictions_df in case source data included it
    predictions_df['county_clean'] = predictions_df['county'].str.replace(r'\s+County$', '', regex=True).str.strip()

    # Derive each ZIP's share of its county's total population.
    zip_to_county_df['zip_pop'] = pd.to_numeric(zip_to_county_df['ZIP Code Population'], errors='coerce')
    zip_to_county_df['zip_pop_in_county'] = (
        zip_to_county_df['zip_pop'] * zip_to_county_df['% of ZIP Residents in County']
    )

    county_totals = (
        zip_to_county_df.groupby('county_name')['zip_pop_in_county']
        .sum()
        .rename('county_total_from_zips')
    )
    zip_to_county_df = zip_to_county_df.join(county_totals, on='county_name')
    zip_to_county_df['pct_of_county'] = (
        zip_to_county_df['zip_pop_in_county'] / zip_to_county_df['county_total_from_zips']
    )

    # Verify the county name matching
    csv_counties = set(zip_to_county_df['county_name'].unique())
    pred_counties = set(predictions_df['county_clean'].unique())
    matched = csv_counties & pred_counties
    unmatched_csv = csv_counties - pred_counties
    unmatched_pred = pred_counties - csv_counties
    print(f"\nCounty matching: {len(matched)} matched")
    if unmatched_csv:
        print(f"In zip CSV but not in predictions: {sorted(unmatched_csv)}")
    if unmatched_pred:
        print(f"In predictions but not in zip CSV: {sorted(unmatched_pred)}")

    # Generate zip code predictions for each prediction year
    for pred_year in PREDICTION_YEARS:
        print(f"\nGenerating zip code predictions for {pred_year}...")

        pop_col = f'population_{pred_year}'

        # Build lookup: cleaned county name -> predicted population
        county_pop_lookup = dict(zip(predictions_df['county_clean'], predictions_df[pop_col]))

        zipcode_predictions = []

        for zip_code, zip_group in zip_to_county_df.groupby('ZIP'):
            total_population = 0.0

            for _, row in zip_group.iterrows():
                county_name = row['county_name']
                pct = row['pct_of_county']  # this ZIP's share of the county population

                if county_name in county_pop_lookup and pd.notna(pct):
                    county_population = county_pop_lookup[county_name]
                    total_population += county_population * pct

            zipcode_predictions.append({
                'zip_code': zip_code,
                'predicted_population': round(total_population, 3)
            })

        zipcode_df = pd.DataFrame(zipcode_predictions).sort_values('zip_code').reset_index(drop=True)

        output_filename = os.path.join(zipcode_output_dir, f'{pred_year}_zipcode_predictions_randforest.csv')
        zipcode_df.to_csv(output_filename, index=False, float_format='%.3f')

        print(f"Saved zip code predictions to: {output_filename}")
        print(f"Total zip codes: {len(zipcode_df)}")
        print(f"\nSample predictions for {pred_year}:")
        print(zipcode_df.head(10))

    print("\nZip code prediction conversion complete!")

if not NO_PLOTS:
    # 11. Model Evaluation
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)

# If we're in demo mode, stop after producing the requested CSV outputs.
if NO_PLOTS:
    print("\nSkipping model evaluation and graphing (demo mode).")
    raise SystemExit(0)

# Evaluate on the selected county or first available
if COUNTY_TO_GRAPH and COUNTY_TO_GRAPH in county_timeseries:
    eval_county = COUNTY_TO_GRAPH
elif len(county_timeseries) > 0:
    eval_county = list(county_timeseries.keys())[0]
else:
    eval_county = None

if eval_county:
    sample_data = county_timeseries[eval_county]
    
    if len(sample_data) >= 5:
        # Prepare features and target
        X_train_list = []
        y_train_list = []
        
        for idx, row in sample_data.iterrows():
            feature_row = ([row['year']] + 
                          [row[col] for col in feature_columns] + 
                          [row['growth_rate_lag1'], row['growth_rate_lag2']])
            X_train_list.append(feature_row)
            y_train_list.append(row['population_growth_rate'])
        
        X = np.array(X_train_list)
        y = np.array(y_train_list)
        
        rf_model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            max_depth=6,
            n_jobs=-1
        )
        rf_model.fit(X, y)
        
        # Predict on training data to see fit
        y_pred = rf_model.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Feature importance
        feature_names = ['year'] + feature_columns + ['growth_rate_lag1', 'growth_rate_lag2']
        importances = rf_model.feature_importances_
        
        print(f"Evaluation County: {eval_county}")
        print(f"Target: Population Growth Rate (%)")
        print(f"Training years: {sample_data['year'].min()}-{sample_data['year'].max()}")
        print(f"Number of training samples: {len(sample_data)}")
        print(f"Mean Squared Error: {mse:.3f}%")
        print(f"R² Score: {r2:.3f}")
        print(f"\nFeature Importance (which features most influence growth rate prediction):")
        # Sort by importance
        feature_importance_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        for feat, imp in feature_importance_pairs:
            print(f"  {feat}: {imp:.3f}")

# 12. Graph ALL ACTUAL DATA + PREDICTIONS with Visual Distinction
print("\n" + "="*60)
print("GRAPHING SECTION")
print("="*60)

# Determine which counties to plot
if COUNTY_TO_GRAPH:
    if COUNTY_TO_GRAPH in county_timeseries_full:
        counties_to_plot = [COUNTY_TO_GRAPH]
        print(f"\nGraphing selected county: {COUNTY_TO_GRAPH}")
    else:
        print(f"\nWARNING: County '{COUNTY_TO_GRAPH}' not found in dataset!")
        print(f"Available counties: {sorted(county_timeseries_full.keys())}")
        counties_to_plot = []
else:
    # Plot first 10 counties
    num_counties_to_plot = min(10, len(county_timeseries_full))
    counties_to_plot = list(county_timeseries_full.keys())[:num_counties_to_plot]
    print(f"\nGraphing {len(counties_to_plot)} counties")

print(f"Plotting ALL actual years ({df_original['year'].min()}-{df_original['year'].max()}) + predictions ({PREDICTION_YEARS[0]}-{PREDICTION_YEARS[-1]})")

if len(counties_to_plot) > 0:
    # Create two subplots: one for population, one for growth rate
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))

    # Plot each county
    for county in counties_to_plot:
        # Get ALL actual data for this county (full original dataset)
        county_full_data = county_timeseries_full[county]
        
        if len(county_full_data) >= 1:
            # Extract ALL years and population values from original data
            all_years = county_full_data['year'].tolist()
            all_populations = county_full_data['population'].tolist()
            
            # Separate into training years and non-training years
            training_years = []
            training_populations = []
            non_training_years = []
            non_training_populations = []
            
            for year, pop in zip(all_years, all_populations):
                if (TRAIN_START_YEAR is None or year >= TRAIN_START_YEAR) and \
                   (TRAIN_END_YEAR is None or year <= TRAIN_END_YEAR):
                    training_years.append(year)
                    training_populations.append(pop)
                else:
                    non_training_years.append(year)
                    non_training_populations.append(pop)
            
            # Get predictions for this county
            county_pred = predictions_df[predictions_df['county'] == county]
            
            if len(county_pred) > 0:
                # Extract predictions dynamically
                pred_populations = []
                pred_growths = []
                
                for pred_year in PREDICTION_YEARS:
                    pop_col = f'population_{pred_year}'
                    growth_col = f'growth_rate_{pred_year}'
                    
                    if pop_col in county_pred.columns:
                        pred_populations.append(county_pred[pop_col].values[0])
                    if growth_col in county_pred.columns:
                        pred_growths.append(county_pred[growth_col].values[0])
                
                # PLOT 1: Population
                # Plot training data (solid, bold)
                if training_years:
                    ax1.plot(training_years, training_populations, marker='o', 
                            label=f'{county} County (Training Data)', linewidth=3, color='blue')
                
                # Plot non-training data (dotted, thinner)
                if non_training_years:
                    ax1.plot(non_training_years, non_training_populations, marker='x', 
                            linestyle=':', linewidth=2, alpha=0.6, color='gray',
                            label=f'{county} County (Not Used in Training)')
                
                # Plot predictions (dashed, different color)
                last_training_year = training_years[-1] if training_years else all_years[-1]
                last_training_pop = training_populations[-1] if training_populations else all_populations[-1]
                
                prediction_years_plot = [last_training_year] + PREDICTION_YEARS
                prediction_values_plot = [last_training_pop] + pred_populations
                ax1.plot(prediction_years_plot, prediction_values_plot, 
                        linestyle='--', marker='s', linewidth=3, alpha=0.8, color='red',
                        label=f'{county} County (Predictions)')
                
                # PLOT 2: Growth Rate
                # Get ALL growth rates from full data
                county_growth_full = county_full_data[county_full_data['population_growth_rate'].notna()]
                all_growth_years = county_growth_full['year'].tolist()
                all_growth_rates = county_growth_full['population_growth_rate'].tolist()
                
                # Separate into training and non-training
                training_growth_years = []
                training_growth_rates = []
                non_training_growth_years = []
                non_training_growth_rates = []
                
                for year, growth in zip(all_growth_years, all_growth_rates):
                    if (TRAIN_START_YEAR is None or year >= TRAIN_START_YEAR) and \
                       (TRAIN_END_YEAR is None or year <= TRAIN_END_YEAR):
                        training_growth_years.append(year)
                        training_growth_rates.append(growth)
                    else:
                        non_training_growth_years.append(year)
                        non_training_growth_rates.append(growth)
                
                # Plot training growth rates (solid, bold)
                if training_growth_years:
                    ax2.plot(training_growth_years, training_growth_rates, marker='o', 
                            label=f'{county} County (Training Data)', linewidth=3, color='blue')
                
                # Plot non-training growth rates (dotted, thinner)
                if non_training_growth_years:
                    ax2.plot(non_training_growth_years, non_training_growth_rates, marker='x', 
                            linestyle=':', linewidth=2, alpha=0.6, color='gray',
                            label=f'{county} County (Not Used in Training)')
                
                # Plot predicted growth rates (dashed, different color)
                last_training_growth_year = training_growth_years[-1] if training_growth_years else all_growth_years[-1]
                last_training_growth = training_growth_rates[-1] if training_growth_rates else all_growth_rates[-1]
                
                growth_pred_years_plot = [last_training_growth_year] + PREDICTION_YEARS
                growth_pred_values_plot = [last_training_growth] + pred_growths
                ax2.plot(growth_pred_years_plot, growth_pred_values_plot, 
                        linestyle='--', marker='s', linewidth=3, alpha=0.8, color='red',
                        label=f'{county} County (Predictions)')

    # Format first subplot (Population)
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Population', fontsize=12, fontweight='bold')
    
    pred_years_str = f"{PREDICTION_YEARS[0]}-{PREDICTION_YEARS[-1]}" if len(PREDICTION_YEARS) > 1 else str(PREDICTION_YEARS[0])
    train_years_str = f"{TRAIN_START_YEAR if TRAIN_START_YEAR else 'start'}-{TRAIN_END_YEAR if TRAIN_END_YEAR else 'end'}"
    
    ax1.set_title(f'Population: Actual vs Predicted for {COUNTY_TO_GRAPH} County\n(Blue=Training Data [{train_years_str}], Gray=Actual but Not Trained, Red=Predictions [{pred_years_str}])', 
              fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Format second subplot (Growth Rate)
    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Population Growth Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Growth Rate: Actual vs Predicted for {COUNTY_TO_GRAPH} County\n(Model trained on [{train_years_str}] using lagged growth rates)', 
              fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)  # Zero line

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, "county_population_growth_forecast.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nGraph saved to: {plot_path}")

    # Display the plot
    if not NO_PLOTS:
        plt.show()
else:
    print("\nNo counties to plot!")

print("\nProcessing complete!")