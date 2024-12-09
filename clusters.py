import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
            "REMAINING_SEATS",
            "WEEKDAY",
            "DEPARTURE PART OF THE DAY",
            "SEASON",
            "DEP_TIME",
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
        print("=" * 50)

    def preprocess_data(self):
        """Preprocess flight dates and times with proper handling of 2400."""
        try:
            self.df = self.df.assign(FL_DATE=pd.to_datetime(self.df["FL_DATE"]))
            self.df = self.df.assign(DAY_OF_YEAR=self.df["FL_DATE"].dt.dayofyear)
            self.df = self.df.assign(MONTH=self.df["FL_DATE"].dt.month)

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

            dep_time_decimal = self.df["DEP_TIME"].apply(convert_time_to_decimal)
            median_dep_time = dep_time_decimal.median()
            self.df = self.df.assign(
                DEP_TIME_DECIMAL=dep_time_decimal.fillna(median_dep_time)
            )

            # Define ordered categories
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

    def create_box_plots(self):
        """Create individual box plots for better visualization."""
        try:
            # 1. Price by Season
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=self.df,
                x="SEASON",
                y="PRICE",
                order=["Winter", "Spring", "Summer", "Fall"],
                palette="Set3",
            )
            plt.title("Price Distribution by Season", fontsize=14, pad=20)
            plt.xlabel("Season", fontsize=12)
            plt.ylabel("Price ($)", fontsize=12)
            plt.xticks(rotation=45)
            # Removed plt.legend() as hue is not used
            plt.tight_layout()
            plt.show()

            # 2. Price by Departure Time
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=self.df,
                x="DEPARTURE PART OF THE DAY",
                y="PRICE",
                order=["Morning", "Afternoon", "Evening", "Night"],
                palette="Set2",
            )
            plt.title("Price Distribution by Departure Time", fontsize=14, pad=20)
            plt.xlabel("Departure Time", fontsize=12)
            plt.ylabel("Price ($)", fontsize=12)
            plt.xticks(rotation=45)
            # Removed plt.legend() as hue is not used
            plt.tight_layout()
            plt.show()

            # 3. Price by Weekday
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=self.df,
                x="WEEKDAY",
                y="PRICE",
                order=[
                    "Sunday",
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                ],
                palette="Set1",
            )
            plt.title("Price Distribution by Day of Week", fontsize=14, pad=20)
            plt.xlabel("Day of Week", fontsize=12)
            plt.ylabel("Price ($)", fontsize=12)
            plt.xticks(rotation=45)
            # Removed plt.legend() as hue is not used
            plt.tight_layout()
            plt.show()

            # 4. Remaining Seats by Season
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=self.df,
                x="SEASON",
                y="REMAINING_SEATS",
                order=["Winter", "Spring", "Summer", "Fall"],
                palette="Pastel1",
            )
            plt.title("Remaining Seats Distribution by Season", fontsize=14, pad=20)
            plt.xlabel("Season", fontsize=12)
            plt.ylabel("Remaining Seats", fontsize=12)
            plt.xticks(rotation=45)
            # Removed plt.legend() as hue is not used
            plt.tight_layout()
            plt.show()

            # 5. Remaining Seats by Departure Time
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=self.df,
                x="DEPARTURE PART OF THE DAY",
                y="REMAINING_SEATS",
                order=["Morning", "Afternoon", "Evening", "Night"],
                palette="Pastel2",
            )
            plt.title(
                "Remaining Seats Distribution by Departure Time", fontsize=14, pad=20
            )
            plt.xlabel("Departure Time", fontsize=12)
            plt.ylabel("Remaining Seats", fontsize=12)
            plt.xticks(rotation=45)
            # Removed plt.legend() as hue is not used
            plt.tight_layout()
            plt.show()

            # 6. Remaining Seats by Weekday
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=self.df,
                x="WEEKDAY",
                y="REMAINING_SEATS",
                order=[
                    "Sunday",
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                ],
                palette="Set3",
            )
            plt.title(
                "Remaining Seats Distribution by Day of Week", fontsize=14, pad=20
            )
            plt.xlabel("Day of Week", fontsize=12)
            plt.ylabel("Remaining Seats", fontsize=12)
            plt.xticks(rotation=45)
            # Removed plt.legend() as hue is not used
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error creating box plots: {str(e)}")

    def determine_optimal_k(self, X, max_k=10):
        """
        Determine the optimal number of clusters using Elbow Method and Silhouette Score.

        Parameters:
            X (numpy.ndarray): The input data for clustering.
            max_k (int): Maximum number of clusters to test.

        Returns:
            optimal_k (int): The determined optimal number of clusters.
        """
        wcss = []
        silhouette_scores = []
        K = range(2, max_k + 1)

        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
            cluster_labels = kmeans.labels_
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        # Plot Elbow Method
        plt.figure(figsize=(12, 6))
        plt.plot(K, wcss, "bx-")
        plt.xlabel("Number of clusters (K)", fontsize=12)
        plt.ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=12)
        plt.title("Elbow Method For Optimal K", fontsize=14)
        plt.xticks(K)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot Silhouette Scores
        plt.figure(figsize=(12, 6))
        plt.plot(K, silhouette_scores, "bx-")
        plt.xlabel("Number of clusters (K)", fontsize=12)
        plt.ylabel("Silhouette Score", fontsize=12)
        plt.title("Silhouette Score For Optimal K", fontsize=14)
        plt.xticks(K)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Determine the optimal K as the one with the highest silhouette score
        optimal_k = K[np.argmax(silhouette_scores)]
        print(
            f"Optimal number of clusters determined to be: {optimal_k} based on Silhouette Score."
        )

        return optimal_k

    def create_clustering_plots(self):
        """Create cluster visualizations with proper handling of time data."""
        try:
            clustering_data = self.df.copy()
            for col in ["DAY_OF_YEAR", "PRICE", "DEP_TIME_DECIMAL", "REMAINING_SEATS"]:
                clustering_data[col] = clustering_data[col].fillna(
                    clustering_data[col].median()
                )

            # 1. Price vs Date Clustering
            print("\n--- Clustering: Price vs Flight Date ---")
            X_date = np.column_stack(
                [clustering_data["DAY_OF_YEAR"], clustering_data["PRICE"]]
            )
            X_date_scaled = self.scaler.fit_transform(X_date)

            optimal_k_date = self.determine_optimal_k(X_date_scaled, max_k=10)
            kmeans_date = KMeans(
                n_clusters=optimal_k_date, random_state=self.random_state, n_init=10
            )
            date_clusters = kmeans_date.fit_predict(X_date_scaled)

            plt.figure(figsize=(15, 8))
            scatter = plt.scatter(
                clustering_data["FL_DATE"],
                clustering_data["PRICE"],
                c=date_clusters,
                cmap="viridis",
                alpha=0.6,
                s=100,
            )
            plt.xlabel("Flight Date", fontsize=12)
            plt.ylabel("Price ($)", fontsize=12)
            plt.title(
                f"Price Clusters by Flight Date (K={optimal_k_date})",
                fontsize=16,
                pad=20,
            )
            cbar = plt.colorbar(scatter)
            cbar.set_label("Cluster")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            # 2. Remaining Seats vs Date Clustering
            print("\n--- Clustering: Remaining Seats vs Flight Date ---")
            X_seats_date = np.column_stack(
                [clustering_data["DAY_OF_YEAR"], clustering_data["REMAINING_SEATS"]]
            )
            X_seats_date_scaled = self.scaler.fit_transform(X_seats_date)

            optimal_k_seats_date = self.determine_optimal_k(
                X_seats_date_scaled, max_k=10
            )
            kmeans_seats_date = KMeans(
                n_clusters=optimal_k_seats_date,
                random_state=self.random_state,
                n_init=10,
            )
            seats_date_clusters = kmeans_seats_date.fit_predict(X_seats_date_scaled)

            plt.figure(figsize=(15, 8))
            scatter = plt.scatter(
                clustering_data["FL_DATE"],
                clustering_data["REMAINING_SEATS"],
                c=seats_date_clusters,
                cmap="viridis",
                alpha=0.6,
                s=100,
            )
            plt.xlabel("Flight Date", fontsize=12)
            plt.ylabel("Remaining Seats", fontsize=12)
            plt.title(
                f"Remaining Seats Clusters by Flight Date (K={optimal_k_seats_date})",
                fontsize=16,
                pad=20,
            )
            cbar = plt.colorbar(scatter)
            cbar.set_label("Cluster")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            # 3. Price vs Departure Time Clustering
            print("\n--- Clustering: Price vs Departure Time ---")
            X_time = np.column_stack(
                [clustering_data["DEP_TIME_DECIMAL"], clustering_data["PRICE"]]
            )
            X_time_scaled = self.scaler.fit_transform(X_time)

            optimal_k_time = self.determine_optimal_k(X_time_scaled, max_k=10)
            kmeans_time = KMeans(
                n_clusters=optimal_k_time, random_state=self.random_state, n_init=10
            )
            time_clusters = kmeans_time.fit_predict(X_time_scaled)

            plt.figure(figsize=(15, 8))
            scatter = plt.scatter(
                clustering_data["DEP_TIME_DECIMAL"],
                clustering_data["PRICE"],
                c=time_clusters,
                cmap="viridis",
                alpha=0.6,
                s=100,
            )
            plt.xlabel("Departure Time (24h)", fontsize=12)
            plt.ylabel("Price ($)", fontsize=12)
            plt.title(
                f"Price Clusters by Departure Time (K={optimal_k_time})",
                fontsize=16,
                pad=20,
            )
            cbar = plt.colorbar(scatter)
            cbar.set_label("Cluster")
            time_ticks = np.arange(0, 25, 3)
            plt.xticks(time_ticks, [f"{int(t):02d}:00" for t in time_ticks])
            plt.tight_layout()
            plt.show()

            # 4. Remaining Seats vs Departure Time Clustering
            print("\n--- Clustering: Remaining Seats vs Departure Time ---")
            X_seats_time = np.column_stack(
                [
                    clustering_data["DEP_TIME_DECIMAL"],
                    clustering_data["REMAINING_SEATS"],
                ]
            )
            X_seats_time_scaled = self.scaler.fit_transform(X_seats_time)

            optimal_k_seats_time = self.determine_optimal_k(
                X_seats_time_scaled, max_k=10
            )
            kmeans_seats_time = KMeans(
                n_clusters=optimal_k_seats_time,
                random_state=self.random_state,
                n_init=10,
            )
            seats_time_clusters = kmeans_seats_time.fit_predict(X_seats_time_scaled)

            plt.figure(figsize=(15, 8))
            scatter = plt.scatter(
                clustering_data["DEP_TIME_DECIMAL"],
                clustering_data["REMAINING_SEATS"],
                c=seats_time_clusters,
                cmap="viridis",
                alpha=0.6,
                s=100,
            )
            plt.xlabel("Departure Time (24h)", fontsize=12)
            plt.ylabel("Remaining Seats", fontsize=12)
            plt.title(
                f"Remaining Seats Clusters by Departure Time (K={optimal_k_seats_time})",
                fontsize=16,
                pad=20,
            )
            cbar = plt.colorbar(scatter)
            cbar.set_label("Cluster")
            plt.xticks(time_ticks, [f"{int(t):02d}:00" for t in time_ticks])
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error creating clustering plots: {str(e)}")

    def print_analysis(self):
        """Print statistical analysis."""
        try:
            factors = ["SEASON", "DEPARTURE PART OF THE DAY", "WEEKDAY"]
            for factor in factors:
                print(f"\n{factor} Analysis:")
                stats = (
                    self.df.groupby(factor, observed=False)
                    .agg(
                        {
                            "PRICE": ["mean", "std", "min", "max"],
                            "REMAINING_SEATS": ["mean", "std", "min", "max"],
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
        SAMPLE_SIZE = 20000  # Set to None to use the entire dataset
        RANDOM_STATE = 42  # For reproducibility

        print("Initializing analyzer...")
        analyzer = FlightAnalyzer(
            data_path=DATA_PATH, sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE
        )

        print("\nPreprocessing data...")
        analyzer.preprocess_data()

        print("\nCreating box plots...")
        analyzer.create_box_plots()

        print("\nCreating clustering plots with optimal K determination...")
        analyzer.create_clustering_plots()

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
