import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle


def train_and_save_model():
    # Read the dataset
    df = pd.read_csv("D:/python projects/windmill_project/file/windmill_data.csv")

    # Separate features and target variable
    X = df[["wind_speed"]]
    y = df["power_output"]

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Save the model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved as 'model.pkl'!")


# Run this script locally once to train and save the model
if __name__ == "__main__":
    train_and_save_model()
