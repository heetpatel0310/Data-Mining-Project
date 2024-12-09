import sys
import time
import warnings
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class FlightAnalyzer:
    def __init__(self, data_path, sample_size=None, random_state=42):
        """
        Initialize the FlightAnalyzer with data verification and optional sampling.

        Parameters:
            data_path (str): Path to the CSV data file.
            sample_size (int, optional): Number of samples to draw from the dataset.
                                         If None, use the entire dataset.
            random_state (int, optional): Seed for random number generator for reproducibility.
        """
        self.sample_size = sample_size
        self.random_state = random_state
        try:
            self.df = pd.read_csv(data_path)
            self.verify_data()
            self.scaler = StandardScaler()
        except FileNotFoundError:
            sys.exit(f"Error: File '{data_path}' not found.")
        except pd.errors.EmptyDataError:
            sys.exit(f"Error: File '{data_path}' is empty.")
        except Exception as e:
            sys.exit(f"Error loading data: {str(e)}")

    def verify_data(self):
        """Verify data integrity."""
        required_columns = [
            "FL_DATE",
            "PRICE",
            "DISTANCE",
            "REMAINING_SEATS",
            "WEEKDAY",
            "DEPARTURE PART OF THE DAY",
            "SEASON",
            "DEP_TIME",
            "AIRLINE",
        ]

        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        blank_rows = self.df.isna().all(axis=1).sum()
        if blank_rows > 0:
            print(f"Warning: Found {blank_rows} blank rows. Removing them...")
            self.df = self.df.dropna(how="all")

        print("\nData Summary:")
        print(f"Total rows: {len(self.df)}")
        print(f"Seasons: {self.df['SEASON'].unique()}")
        print(f"Departure times: {self.df['DEPARTURE PART OF THE DAY'].unique()}")
        print(f"Airlines: {self.df['AIRLINE'].unique()}")
        print("=" * 50)

    def preprocess_data(self):
        """Preprocess flight dates, times, utility, and identify holidays."""
        try:
            # Convert FL_DATE to datetime
            self.df["FL_DATE"] = pd.to_datetime(self.df["FL_DATE"], format="%m/%d/%Y")

            # Calculate DAY_OF_YEAR and MONTH for additional insights
            self.df["DAY_OF_YEAR"] = self.df["FL_DATE"].dt.dayofyear
            self.df["MONTH"] = self.df["FL_DATE"].dt.month

            # Convert DEP_TIME to decimal hours
            def convert_time_to_decimal(time_str):
                try:
                    if pd.isna(time_str) or time_str == "":
                        return np.nan
                    time_str = str(int(time_str)).zfill(4)

                    if time_str == "2400":
                        return 24.0

                    hours = int(time_str[:2])
                    minutes = int(time_str[2:])
                    if hours > 24 or (hours == 24 and minutes > 0) or minutes > 59:
                        return np.nan
                    return hours + minutes / 60
                except:
                    return np.nan

            self.df["DEP_TIME_DECIMAL"] = self.df["DEP_TIME"].apply(
                convert_time_to_decimal
            )
            median_dep_time = self.df["DEP_TIME_DECIMAL"].median()
            self.df["DEP_TIME_DECIMAL"] = self.df["DEP_TIME_DECIMAL"].fillna(
                median_dep_time
            )

            # Define ordered categories for better visualization
            WEEKDAY_ORDER = [
                "Sunday",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
            ]
            SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
            DEPARTURE_PART_ORDER = ["Morning", "Afternoon", "Evening", "Night"]

            self.df["WEEKDAY"] = pd.Categorical(
                self.df["WEEKDAY"], categories=WEEKDAY_ORDER, ordered=True
            )
            self.df["SEASON"] = pd.Categorical(
                self.df["SEASON"], categories=SEASON_ORDER, ordered=True
            )
            self.df["DEPARTURE PART OF THE DAY"] = pd.Categorical(
                self.df["DEPARTURE PART OF THE DAY"],
                categories=DEPARTURE_PART_ORDER,
                ordered=True,
            )

            # Calculate Utility = Price / Distance
            self.df["UTILITY"] = self.df["PRICE"] / self.df["DISTANCE"]

            # Identify Holidays
            self.df["HOLIDAY"] = self.identify_holidays(self.df["FL_DATE"])

            # Sampling (if specified)
            if self.sample_size is not None:
                if self.sample_size > len(self.df):
                    print(
                        f"Requested sample size {self.sample_size} exceeds dataset size. Using entire dataset."
                    )
                else:
                    print(
                        f"\nSampling {self.sample_size} rows from the dataset for analysis..."
                    )
                    self.df = self.df.sample(
                        n=self.sample_size, random_state=self.random_state
                    ).reset_index(drop=True)

            return self.df

        except Exception as e:
            sys.exit(f"Error in preprocessing: {str(e)}")

    def identify_holidays(self, flight_dates):
        """
        Identify whether each flight date falls within a holiday period.

        Parameters:
            flight_dates (pd.Series): Series of flight dates.

        Returns:
            pd.Series: Boolean series indicating holiday status.
        """
        # Define holiday dates for each year
        holiday_dict = {
            2019: {
                "New Year's Day": pd.Timestamp("2019-01-01"),
                "Thanksgiving Day": pd.Timestamp("2019-11-28"),
                "Christmas Day": pd.Timestamp("2019-12-25"),
            },
            2020: {
                "New Year's Day": pd.Timestamp("2020-01-01"),
                "Thanksgiving Day": pd.Timestamp("2020-11-26"),
                "Christmas Day": pd.Timestamp("2020-12-25"),
            },
            2021: {
                "New Year's Day": pd.Timestamp("2021-01-01"),
                "Thanksgiving Day": pd.Timestamp("2021-11-25"),
                "Christmas Day": pd.Timestamp("2021-12-25"),
            },
            2022: {
                "New Year's Day": pd.Timestamp("2022-01-01"),
                "Thanksgiving Day": pd.Timestamp("2022-11-24"),
                "Christmas Day": pd.Timestamp("2022-12-25"),
            },
            2023: {
                "New Year's Day": pd.Timestamp("2023-01-01"),
                "Thanksgiving Day": pd.Timestamp("2023-11-23"),
                "Christmas Day": pd.Timestamp("2023-12-25"),
            },
        }

        # Define the range around each holiday (e.g., 3 days before and after)
        holiday_buffer = 3  # days

        # Initialize an empty set to store all holiday dates
        all_holiday_dates = set()

        for year, holidays in holiday_dict.items():
            for holiday_name, holiday_date in holidays.items():
                # Adjust for observed holidays (e.g., if holiday falls on weekend)
                if holiday_date.weekday() == 5:  # Saturday
                    observed_date = holiday_date - timedelta(days=1)
                elif holiday_date.weekday() == 6:  # Sunday
                    observed_date = holiday_date + timedelta(days=1)
                else:
                    observed_date = holiday_date

                # Define holiday period range
                start_date = observed_date - timedelta(days=holiday_buffer)
                end_date = observed_date + timedelta(days=holiday_buffer)

                # Add all dates in the range to the set
                date_range = pd.date_range(start=start_date, end=end_date)
                all_holiday_dates.update(date_range)

        # Create a boolean series indicating holiday status
        holiday_status = flight_dates.isin(all_holiday_dates)

        return holiday_status

    def plot_price_vs_holiday(self, per_airline=False):
        """
        Plot Price vs. Holiday and Non-Holiday Dates.

        Parameters:
            per_airline (bool): If True, generate separate plots for each airline.
                                If False, generate a combined plot for all airlines.
        """
        try:
            if per_airline:
                airlines = self.df["AIRLINE"].unique()
                for airline in airlines:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(
                        data=self.df[self.df["AIRLINE"] == airline],
                        x="HOLIDAY",
                        y="PRICE",
                        palette="Set2",
                    )
                    plt.title(f"Price Distribution for {airline} Flights", fontsize=16)
                    plt.xlabel("Holiday Period", fontsize=12)
                    plt.ylabel("Price ($)", fontsize=12)
                    plt.xticks([0, 1], ["Non-Holiday", "Holiday"])
                    plt.tight_layout()
                    plt.show()
            else:
                plt.figure(figsize=(10, 6))
                sns.boxplot(
                    data=self.df,
                    x="HOLIDAY",
                    y="PRICE",
                    palette="Set3",
                )
                plt.title(
                    "Price Distribution: Holiday vs. Non-Holiday Flights", fontsize=16
                )
                plt.xlabel("Holiday Period", fontsize=12)
                plt.ylabel("Price ($)", fontsize=12)
                plt.xticks([0, 1], ["Non-Holiday", "Holiday"])
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Error plotting Price vs. Holiday: {str(e)}")

    def plot_remaining_seats_vs_holiday(self, per_airline=False):
        """
        Plot Remaining Seats vs. Holiday and Non-Holiday Dates.

        Parameters:
            per_airline (bool): If True, generate separate plots for each airline.
                                If False, generate a combined plot for all airlines.
        """
        try:
            if per_airline:
                airlines = self.df["AIRLINE"].unique()
                for airline in airlines:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(
                        data=self.df[self.df["AIRLINE"] == airline],
                        x="HOLIDAY",
                        y="REMAINING_SEATS",
                        palette="Set1",
                    )
                    plt.title(f"Remaining Seats for {airline} Flights", fontsize=16)
                    plt.xlabel("Holiday Period", fontsize=12)
                    plt.ylabel("Remaining Seats", fontsize=12)
                    plt.xticks([0, 1], ["Non-Holiday", "Holiday"])
                    plt.tight_layout()
                    plt.show()
            else:
                plt.figure(figsize=(10, 6))
                sns.boxplot(
                    data=self.df,
                    x="HOLIDAY",
                    y="REMAINING_SEATS",
                    palette="Set3",
                )
                plt.title(
                    "Remaining Seats Distribution: Holiday vs. Non-Holiday Flights",
                    fontsize=16,
                )
                plt.xlabel("Holiday Period", fontsize=12)
                plt.ylabel("Remaining Seats", fontsize=12)
                plt.xticks([0, 1], ["Non-Holiday", "Holiday"])
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Error plotting Remaining Seats vs. Holiday: {str(e)}")

    def plot_utility_distribution_by_airline(self):
        """
        Plot Utility Distribution by Airline.
        """
        try:
            plt.figure(figsize=(12, 6))
            airlines = self.df["AIRLINE"].unique()
            for airline in airlines:
                sns.kdeplot(
                    self.df[self.df["AIRLINE"] == airline]["UTILITY"],
                    label=airline,
                    shade=True,
                )
            plt.title("Utility Distribution by Airline", fontsize=16)
            plt.xlabel("Utility (Price / Distance)", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.legend(title="Airline")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting Utility distribution by airline: {str(e)}")

    def plot_utility_over_time_holiday(self, per_airline=False):
        """
        Plot Utility over Flight Dates, differentiating between Holiday and Non-Holiday flights.

        Parameters:
            per_airline (bool): If True, generate separate plots for each airline.
                                If False, generate a combined plot for all airlines.
        """
        try:
            if per_airline:
                airlines = self.df["AIRLINE"].unique()
                for airline in airlines:
                    plt.figure(figsize=(14, 7))
                    sns.lineplot(
                        data=self.df[self.df["AIRLINE"] == airline],
                        x="FL_DATE",
                        y="UTILITY",
                        hue="HOLIDAY",
                        palette={False: "green", True: "orange"},
                        alpha=0.5,
                    )
                    plt.title(f"Utility Over Time for {airline} Flights", fontsize=16)
                    plt.xlabel("Flight Date", fontsize=12)
                    plt.ylabel("Utility (Price / Distance)", fontsize=12)
                    plt.legend(title="Holiday", labels=["Non-Holiday", "Holiday"])
                    plt.tight_layout()
                    plt.show()
            else:
                plt.figure(figsize=(14, 7))
                sns.lineplot(
                    data=self.df,
                    x="FL_DATE",
                    y="UTILITY",
                    hue="HOLIDAY",
                    palette={False: "green", True: "orange"},
                    alpha=0.3,
                )
                plt.title(
                    "Utility Over Time: Holiday vs. Non-Holiday Flights", fontsize=16
                )
                plt.xlabel("Flight Date", fontsize=12)
                plt.ylabel("Utility (Price / Distance)", fontsize=12)
                plt.legend(title="Holiday", labels=["Non-Holiday", "Holiday"])
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Error plotting Utility over Time: {str(e)}")

    def plot_utility_over_time_all_airlines(self):
        """
        Plot Utility Over Time for All Airlines.
        """
        try:
            plt.figure(figsize=(14, 7))
            sns.lineplot(
                data=self.df,
                x="FL_DATE",
                y="UTILITY",
                hue="AIRLINE",
                palette="tab10",
                alpha=0.5,
            )
            plt.title("Utility Over Time for All Airlines", fontsize=16)
            plt.xlabel("Flight Date", fontsize=12)
            plt.ylabel("Utility (Price / Distance)", fontsize=12)
            plt.legend(title="Airline", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting Utility over Time for All Airlines: {str(e)}")

    def plot_utility_distribution(self, per_airline=False):
        """
        Plot the distribution of Utility.

        Parameters:
            per_airline (bool): If True, generate separate histograms for each airline.
                                If False, generate a combined histogram for all airlines.
        """
        try:
            plt.figure(figsize=(12, 6))
            if per_airline:
                airlines = self.df["AIRLINE"].unique()
                for airline in airlines:
                    sns.kdeplot(
                        self.df[self.df["AIRLINE"] == airline]["UTILITY"],
                        label=airline,
                        shade=True,
                    )
                plt.title("Utility Distribution by Airline", fontsize=16)
            else:
                sns.histplot(self.df["UTILITY"], kde=True, color="skyblue", bins=50)
                plt.title("Overall Utility Distribution", fontsize=16)
            plt.xlabel("Utility (Price / Distance)", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.legend(title="Airline")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting Utility distribution: {str(e)}")

    def print_analysis(self):
        """Print statistical analysis."""
        try:
            factors = ["SEASON", "DEPARTURE PART OF THE DAY", "WEEKDAY", "HOLIDAY"]
            for factor in factors:
                print(f"\n{factor} Analysis:")
                if factor == "HOLIDAY":
                    stats = (
                        self.df.groupby(factor, observed=False)
                        .agg(
                            {
                                "PRICE": ["mean", "std", "min", "max"],
                                "REMAINING_SEATS": ["mean", "std", "min", "max"],
                                "UTILITY": ["mean", "std", "min", "max"],
                            }
                        )
                        .round(2)
                    )
                else:
                    stats = (
                        self.df.groupby(factor, observed=False)
                        .agg(
                            {
                                "PRICE": ["mean", "std", "min", "max"],
                                "REMAINING_SEATS": ["mean", "std", "min", "max"],
                                "UTILITY": ["mean", "std", "min", "max"],
                            }
                        )
                        .round(2)
                    )
                print(stats)
                print("\n" + "=" * 50)
        except Exception as e:
            print(f"Error in statistical analysis: {str(e)}")


def main():
    start_time = time.time()

    try:
        # Set parameters
        DATA_PATH = "flight_data.csv"  # Update this path if necessary
        SAMPLE_SIZE = 20000  # Set to a number (e.g., 2000) to sample, or None to use entire dataset
        RANDOM_STATE = 42  # For reproducibility

        print("Initializing analyzer...")
        analyzer = FlightAnalyzer(
            data_path=DATA_PATH, sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE
        )

        print("\nPreprocessing data...")
        analyzer.preprocess_data()

        print("\nGenerating Price vs. Holiday/Non-Holiday Plots...")
        # Plotting Price vs. Holiday (Combined)
        analyzer.plot_price_vs_holiday(per_airline=False)

        # Plotting Price vs. Holiday (Per Airline)
        analyzer.plot_price_vs_holiday(per_airline=True)

        print("\nGenerating Remaining Seats vs. Holiday/Non-Holiday Plots...")
        # Plotting Remaining Seats vs. Holiday (Combined)
        analyzer.plot_remaining_seats_vs_holiday(per_airline=False)

        # Plotting Remaining Seats vs. Holiday (Per Airline)
        analyzer.plot_remaining_seats_vs_holiday(per_airline=True)

        print("\nGenerating Utility Distribution by Airline...")
        # Plotting Utility Distribution by Airline
        analyzer.plot_utility_distribution_by_airline()

        print("\nGenerating Utility Over Time: Holiday vs. Non-Holiday Flights...")
        # Plotting Utility over Time by Holiday (Combined)
        analyzer.plot_utility_over_time_holiday(per_airline=False)

        # Plotting Utility over Time by Holiday (Per Airline)
        analyzer.plot_utility_over_time_holiday(per_airline=True)

        print("\nGenerating Utility Over Time for All Airlines...")
        # Plotting Utility over Time for All Airlines
        analyzer.plot_utility_over_time_all_airlines()

        print("\nStatistical Analysis:")
        analyzer.print_analysis()

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nExecution time: {execution_time:.2f} seconds")

        memory_usage = analyzer.df.memory_usage(deep=True).sum() / 1024**2
        print(f"Memory usage: {memory_usage:.2f} MB")

    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
