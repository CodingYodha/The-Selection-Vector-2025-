
import os
import pandas as pd
import joblib
import argparse


from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline

# This line is CRITICAL. It imports all approved custom classes.
from custom_definitions import *

# --- Main Competition Configuration ---
ALL_DAYS_INFO = {
    1: {'task': 'classification', 'metric': 'Accuracy'},
    2: {'task': 'regression', 'metric': 'R2-Score'},
    3: {'task': 'classification', 'metric': 'Accuracy'},
    4: {'task': 'regression', 'metric': 'R2-Score'}
}

def get_rank_points(rank):
    """Assigns points based on rank."""
    if rank == 1: return 100
    if rank == 2: return 90
    if rank == 3: return 80
    if rank == 4: return 75
    if rank == 5: return 70
    if 6 <= rank <= 10: return 60
    return 25

def validate_day(day_num, task_type):
    """Validates all submissions for a specific day and extracts the model name."""
    print(f"--- Starting Validation for Day {day_num} ({task_type}) ---")
    models_dir = f"day{day_num}_submissions"
    validation_dir = f"day{day_num}_validation"
    output_scores_path = f"day{day_num}_scores.csv"

    if not os.path.exists(models_dir) or not os.path.exists(validation_dir):
        print(f"Error: Required directories for Day {day_num} not found. Aborting.")
        return

    try:
        X_val = pd.read_csv(os.path.join(validation_dir, 'X_val.csv'))
        y_val = pd.read_csv(os.path.join(validation_dir, 'y_val.csv')).squeeze()
    except FileNotFoundError as e:
        print(f"Error loading validation data: {e}. Aborting.")
        return

    daily_results = []
    for file_name in os.listdir(models_dir):
        if file_name.endswith(".pkl"):
            participant_name = os.path.splitext(file_name)[0]
            model_path = os.path.join(models_dir, file_name)
            score = -999.99
            model_name = "Load Failed"

            try:
                pipeline = joblib.load(model_path)
                
                # --- NEW: Robustly extract the model name ---
                try:
                    if isinstance(pipeline, Pipeline):
                        # Standard scikit-learn pipeline
                        final_model_object = pipeline.steps[-1][1]
                        model_name = type(final_model_object).__name__
                    elif hasattr(pipeline, 'pipeline'): # Check for a nested .pipeline attribute
                        final_model_object = pipeline.pipeline.steps[-1][1]
                        model_name = type(final_model_object).__name__
                    elif hasattr(pipeline, 'model'): # Check for a .model attribute
                        model_name = type(pipeline.model).__name__
                    else:
                        model_name = "Custom Wrapper"
                except Exception:
                    model_name = "Inspect Error"

                predictions = pipeline.predict(X_val)

                if task_type == 'classification':
                    score = accuracy_score(y_val, predictions)
                elif task_type == 'regression':
                    score = r2_score(y_val, predictions)
                
                print(f"  [SUCCESS] Evaluated: {participant_name:<25} | Model: {model_name:<20} | Score: {score:.4f}")

            except Exception as e:
                print(f"  [ FAILED] Evaluated: {participant_name:<25} | Reason: {e}")
            
            daily_results.append({
                "Participant": participant_name,
                ALL_DAYS_INFO[day_num]['metric']: score,
                "Model": model_name
            })

    if daily_results:
        scores_df = pd.DataFrame(daily_results)
        scores_df.to_csv(output_scores_path, index=False)
        print(f"\nDay {day_num} scores saved to {output_scores_path}")

def update_leaderboard():
    """Recalculates the entire leaderboard, including raw scores and model names for each day."""
    print("\n--- Updating Main Leaderboard ---")
    master_leaderboard = pd.DataFrame()

    for day_num, info in ALL_DAYS_INFO.items():
        scores_file = f"day{day_num}_scores.csv"
        if os.path.exists(scores_file):
            print(f"Processing scores from {scores_file}...")
            daily_df = pd.read_csv(scores_file)
            metric_col = info['metric']
            
            if 'Model' not in daily_df.columns:
                daily_df['Model'] = 'Legacy'
            
            daily_df['Rank'] = daily_df[metric_col].rank(method='min', ascending=False)
            daily_df[f'Day_{day_num}_Points'] = daily_df['Rank'].apply(get_rank_points)
            
            daily_df.rename(columns={
                metric_col: f'Day_{day_num}_Score',
                'Model': f'Day_{day_num}_Model'
            }, inplace=True)
            
            daily_summary = daily_df[['Participant', f'Day_{day_num}_Score', f'Day_{day_num}_Points', f'Day_{day_num}_Model']]

            if master_leaderboard.empty:
                master_leaderboard = daily_summary
            else:
                master_leaderboard = pd.merge(master_leaderboard, daily_summary, on='Participant', how='outer')

    if master_leaderboard.empty:
        print("No score files found. Leaderboard not updated.")
        return

    score_cols = [col for col in master_leaderboard.columns if '_Score' in col]
    point_cols = [col for col in master_leaderboard.columns if '_Points' in col]
    model_cols = [col for col in master_leaderboard.columns if '_Model' in col]
    
    master_leaderboard.loc[:, score_cols] = master_leaderboard.loc[:, score_cols].fillna(0)
    master_leaderboard.loc[:, point_cols] = master_leaderboard.loc[:, point_cols].fillna(0).astype(int)
    master_leaderboard.loc[:, model_cols] = master_leaderboard.loc[:, model_cols].fillna("N/A")
    
    master_leaderboard['Total_Points'] = master_leaderboard[point_cols].sum(axis=1)

    master_leaderboard = master_leaderboard.sort_values(by='Total_Points', ascending=False).reset_index(drop=True)
    master_leaderboard['Overall_Rank'] = master_leaderboard.index + 1

    final_cols = ['Overall_Rank', 'Participant', 'Total_Points']
    for day in sorted(ALL_DAYS_INFO.keys()):
        score_col_name = f"Day_{day}_Score"
        points_col_name = f"Day_{day}_Points"
        model_col_name = f"Day_{day}_Model"
        
        if score_col_name in master_leaderboard.columns:
            final_cols.extend([score_col_name, points_col_name, model_col_name])
            
    master_leaderboard = master_leaderboard[final_cols]

    master_leaderboard.to_csv('leaderboard.csv', index=False)
    print("\nLeaderboard successfully updated and saved to leaderboard.csv.")
    print("--- CURRENT LEADERBOARD ---")
    print(master_leaderboard.to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated evaluation script for the AIRAC Challenge.")
    parser.add_argument("--day", type=int, required=True, help="The challenge day number to validate (1, 2, 3, or 4).")
    args = parser.parse_args()

    day_to_validate = args.day
    if day_to_validate not in ALL_DAYS_INFO:
        print(f"Error: Day {day_to_validate} is not valid. Please choose from {list(ALL_DAYS_INFO.keys())}.")
    else:
        task = ALL_DAYS_INFO[day_to_validate]['task']
        validate_day(day_to_validate, task)
        update_leaderboard()
