import pandas as pd

# Load your dataset
file_path = "flight_data.csv"
data = pd.read_csv(file_path)

# Ensure the FL_DATE is in datetime format
data["FL_DATE"] = pd.to_datetime(data["FL_DATE"], errors="coerce")

# Add a column for the weekday (Sunday-Saturday)
data["WEEKDAY"] = data["FL_DATE"].dt.day_name()


# Function to determine the part of the day
def get_part_of_day(time):
    if pd.isna(time):
        return None
    time = int(time)
    if 600 <= time < 1200:
        return "Morning"
    elif 1200 <= time < 1800:
        return "Afternoon"
    elif 1800 <= time < 2400:
        return "Evening"
    else:
        return "Night"


# Add columns for the departure and arrival part of the day
data["DEPARTURE PART OF THE DAY"] = data["DEP_TIME"].apply(get_part_of_day)
data["ARRIVAL PART OF THE DAY"] = data["ARR_TIME"].apply(get_part_of_day)


# Function to determine the season
def get_season(date):
    if pd.isna(date):
        return None
    month = date.month
    day = date.day
    if (month == 12 and day >= 21) or (month in [1, 2]) or (month == 3 and day < 20):
        return "Winter"
    elif (month == 3 and day >= 20) or (month in [4, 5]) or (month == 6 and day < 21):
        return "Spring"
    elif (month == 6 and day >= 21) or (month in [7, 8]) or (month == 9 and day < 22):
        return "Summer"
    elif (
        (month == 9 and day >= 22) or (month in [10, 11]) or (month == 12 and day < 21)
    ):
        return "Fall"


# Add the SEASON column
data["SEASON"] = data["FL_DATE"].apply(get_season)

# Save the new dataset with all columns
output_path = "flight_data_latest.csv"
data.to_csv(output_path, index=False)

# Print the path to the updated dataset
print(f"Updated dataset saved to: {output_path}")
