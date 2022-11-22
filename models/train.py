import lightgbm as lgb
from utils import read_and_merge_dataset, clean_df, convert_to_unix_timedelta, evaluate, generate_dataloaders
from constants import RELEVANT_COLUMNS, CATEGORICAL_COLLUMNS, LIGHTGBM_TRAINING_PARAMS, NUM_TRAINING_ROUNDS

# Data transformations
print("Reading dataset, applying transformations...")
df = read_and_merge_dataset()
df = clean_df(df[RELEVANT_COLUMNS].copy())  # Clean and drop duplicate columns
df["date_recorded"] = convert_to_unix_timedelta(df)  # Transform into unix timedeltas
print("Dataset ready")

# Generate dataloaders
train_loader, val_loader, X_test, y_test = generate_dataloaders(df, CATEGORICAL_COLLUMNS)
print("Dataloaders generated, training...")

model = lgb.train(
    LIGHTGBM_TRAINING_PARAMS,
    train_loader,
    NUM_TRAINING_ROUNDS,
    valid_sets=[val_loader],
)

test_report = evaluate(model, X_test, y_test)

print(f"Test partition metrics report: \n {test_report}")

model.save_model("models/model_checkpoints/lgb_gbdt_model.pkl")
print("Model saved")
