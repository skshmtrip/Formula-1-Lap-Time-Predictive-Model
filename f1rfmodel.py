#imports (make sure the library is installed on your machine)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#assigning the dataset to an variable
df = pd.read_csv("laps_dataset.csv")

#event encoding because rf models can't process strings
event_to_id = {}
for event in df["Event"].unique():
    event_to_id[event] = len(event_to_id)

df["EventID"] = df["Event"].map(event_to_id)

#input and outputs
features = ["Driver", "AirTemp", "TrackTemp", "EventID"]
target = "LapTime"

df = df[features + [target]].dropna()

split_index = int(0.8 * len(df))


#training and testing
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"training's done, twin. here's mean absolute error = {mae}.")


#user input
print("YOUR FAVORITE LAP TIME PREDICTOR")
#list the events
print("known event (copy and paste the special characters):")
for e in event_to_id:
    print("-", e)

try:
    driver = int(input("hey twin, what's the driver number? "))
    air_temp = float(input("lmk the air temperature twin? "))
    track_temp = float(input("twin, what's the track temperature. "))
    event_name = input("what event was it twin? (case sensitive)? ")

    if event_name not in event_to_id:
        print("this event not in the training data bruh.")
    else:
        event_id = event_to_id[event_name]
        sample = [[driver, air_temp, track_temp, event_id]]
        prediction = model.predict(sample)
        print("heres the lap time:", round(prediction[0], 3), "seconds")

except ValueError:

    print("son... wrong input type.")
