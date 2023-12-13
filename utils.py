from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


def redirect_ratio(df):
    """
    This function creates a new column called 'redirect_ratio'
    It uses the ItineraryRedirects & ODRedirects columns
    """

    # This prevents a warning from working on a slice of a dataframe
    data = df.copy()

    # This creates a new column, which is a ratio of ItineraryRedirects & ODRedirects
    data['redirect_ratio'] = data['ItineraryRedirects'] / data['ODRedirects']

    return data

def create_od_column(df, raw_od_columns):
    """
    This creates a column to identify OD's
    raw_od_columns is a list of two columns - Origin first then Destination
    Depending on the columns, it can be used on City or Airport
    """
    df['OD'] = df[raw_od_columns[0]] + df[raw_od_columns[1]]

    return df

def create_od_ctry(df, raw_od_columns):
    """
    This creates a column to identify OD's
    raw_od_columns is a list of two columns - Origin first then Destination
    Depending on the columns, it can be used on City or Airport
    """
    df['OD_ctry'] = df[raw_od_columns[0]] + df[raw_od_columns[1]]

    return df


def calculate_total_segment_times(df):
    """
    This add all the segment times together
    Creates a new column called total_travel_time
    """

    # Create list of values to fill columns with - if this is not done you get a warning
    values = {'Seg_0_DurationMin': 0 ,'Seg_1_DurationMin': 0, 'Seg_2_DurationMin': 0, 'Seg_3_DurationMin': 0}
    # Fills the values
    df.fillna(value=values, inplace=True)

    # Adds them up
    df['total_travel_time'] = df['Seg_0_DurationMin'] + df['Seg_1_DurationMin'] + df['Seg_2_DurationMin'] + df['Seg_3_DurationMin']

    return df

def calculate_total_layover_time(df, as_ratio=False):
    """
    Creates a total layover time by subtracting flight duration with segment times
    You can choose between a ratio or absolute value
    Creates a new column called total_layover_time
    If as_ratio is True, also creates a column callled total_layover_time_ratio
    """

    # Runs function to create segment times
    df_with_segment_time = calculate_total_segment_times(df)

    # Calculates total layover time and creates a new column
    df_with_segment_time['total_layover_time'] = df_with_segment_time['DurationMin'] - df_with_segment_time['total_travel_time']

    # Creates layover ratio as a new column
    if as_ratio == True:
        df_with_segment_time['total_layover_time_ratio'] = df_with_segment_time['total_layover_time'] / df_with_segment_time['DurationMin']
        return df_with_segment_time

    else:
        return df_with_segment_time


def drop_neg_layover_time(df):
    """
    This drops all rows with neg layover time
    As of writing, we believe this is an error in Skyscanner's data
    """

    masked_df = df[df['total_layover_time'] >= 0]

    return masked_df

def calculate_total_distance(df):
    """
    This calculates the total distance traveled from each segment
    And creates a new column called total_distance_traveled
    """

    # This avoids warnings about working on a slice of a dataframe
    copy = df.copy()

    # Filling null values with 0
    values = {'Seg_0_TravelDistanceKm': 0 ,'Seg_1_TravelDistanceKm': 0, 'Seg_2_TravelDistanceKm': 0, 'Seg_3_TravelDistanceKm': 0}

    # Fills the values
    copy.fillna(value=values, inplace=True)

    # Adds them up and creates a new column
    copy['total_distance_traveled'] = copy['Seg_0_TravelDistanceKm'] + copy['Seg_1_TravelDistanceKm'] + copy['Seg_2_TravelDistanceKm'] + copy['Seg_3_TravelDistanceKm']

    return copy


def calculate_distance_difference(df, as_ratio=False):
    """
    Calculates the difference between total distance traveled and 'straight line' distance
    Can be set to be ratio or absolute difference
    Depending on as_ratio, it creates a new column extra_travel_distance OR extra_travel_distance_ratio
    """

    # This runs a function that calcualtes total distance traveled
    copy = calculate_total_distance(df)

    # Creates a ratio or absolute difference of total distance traveled
    if as_ratio == False:
        copy['extra_travel_distance'] = copy['total_distance_traveled'] - copy['TravelDistanceKm']

    else:
        copy['extra_travel_distance_ratio'] =  copy['total_distance_traveled'] / copy['TravelDistanceKm']

    return copy

def scale_itin_redirects(df, column_to_scale, min, max):
    """
    This scales the ItineraryRedirects column
    """

    # This prevents a warning from working on a slice of a dataframe
    data = df.copy()

    # Z-score normalization per OD
    data['Z_Score'] = data.groupby('OD')[column_to_scale].transform(zscore)

    # Min-Max scaling per OD
    scaler = MinMaxScaler()
    data['MinMax_Scaled'] = data.groupby('OD')[column_to_scale].transform(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())

    data['Score_min_max'] = data['MinMax_Scaled']*np.log(data['ODRedirects']+1)
    data['Score_Z_score'] = data['Z_Score']*np.log(data['ODRedirects']+1)

    # Scale the Z-score to be between 0 and 50
    max_scaled_value = max
    min_scaled_value = min

    data['Score_Z_score_0_50'] = ((data['Z_Score'] - data['Z_Score'].min()) / (data['Z_Score'].max() - data['Z_Score'].min())) * (max_scaled_value - min_scaled_value) + min_scaled_value

    return data


def preprocess(df, raw_od_columns, raw_od_ctry, as_ratio=False):
    """
    This runs all the preprocessing functions
    """

    df_with_ratio = redirect_ratio(df)

    # This creates a column to identify OD's
    df_with_od = create_od_column(df_with_ratio, raw_od_columns)

    df_with_od = create_od_ctry(df_with_ratio, raw_od_ctry)

    # This calculates the total segment times
    df_with_segment_time = calculate_total_segment_times(df_with_od)

    # This calculates the total layover time
    df_with_layover_time = calculate_total_layover_time(df_with_segment_time, as_ratio)

    # This calculates the total distance traveled
    df_with_distance = calculate_total_distance(df_with_layover_time)

    # This calculates the difference between total distance traveled and 'straight line' distance
    df_with_distance_diff = calculate_distance_difference(df_with_distance, as_ratio)

    # This drops all rows with neg layover time
    df_final = drop_neg_layover_time(df_with_distance_diff)

    df_scaled = scale_itin_redirects(df_final, 'ItineraryRedirects', 0, 50)

    return df_scaled


def create_train_test_split(df, target_name:str, random_states=42):
    """
    Returns X_train, X_test, y_train, y_test
    Input is the df, and the name of the target as a string
    If you wish, you can adjust the random state
    """

    features = df.drop(target_name, axis=1)
    target = df[target_name]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=random_states)

    return X_train, X_test, y_train, y_test


def preprocess_test_set(df):
    """
    This runs all the preprocessing functions
    """
    df['SelfTransfer'] = 1

    df['total_layover_time_ratio'] = df['connection_time']/(df['total_minutes']+ df['connection_time'])
    df['extra_travel_distance_ratio'] = df['total_distance'] / df['direct_distance']

    # This drops all rows with neg layover time
    df['PricePerPax'] = df['total_price'] / df['pax']

    df_final = df[['direct_distance', 'days_to_travel', 'SelfTransfer', 'total_layover_time_ratio', 'extra_travel_distance_ratio', 'PricePerPax']]

    df_final.loc[:, 'extra_travel_distance_ratio'] = np.log1p(df_final['extra_travel_distance_ratio']+ 1e-9)  # 1e-9 is a small constant to offset zero values
    df_final.loc[:, 'PricePerPax'] = np.log1p(df_final['PricePerPax']+ 1e-9)  # 1e-9 is a small constant to offset zero values

    return df_final

def classification_evaluation(Dohop_test_dataset, model, scaler):
    """ This function evaluates the regression model on the Dohop Test dataset.
    It scales the dataset and applies the model to it.
    It then buckets the dohop dataset into 2 buckets: booked and not booked.
    It then finds the highest cutoff threshold possible, while maintaining the
    condition that at no booked itinerary lies on or below the threshold.

    If the condition cannot be met, the function returns a warning string.
    """
    # Create a copy of the Dohop datast
    data = Dohop_test_dataset.copy()

    # Create a copy of the bookings column, drop it, and scale all remaining columns
    bookings_column = data["bookings"].copy()

    data.drop(columns=["bookings"], inplace=True)

    data = scaler.transform(data)

    # Create a new DataFrame with the scaled data
    # Exclude the 'bookings' column from the columns list
    scaled_columns = [col for col in Dohop_test_dataset.columns if col != 'bookings']
    data = pd.DataFrame(data, columns=scaled_columns)

    # Applying the prediction model to the Dohop dataset and adding as new column
    data["predicted_score"] = model.predict(data).flatten()

    # Adding the bookings column back to the dataset

    data["bookings"] = bookings_column

    # Create a new colummn bucketed into "booked" and "not-booked"
    data["status"] = np.where(data["bookings"] > 0, "booked", "not-booked")

    # Filter down the dataset to those rows which were booked
    booked_data = data[data["status"] == "booked"]

    # Filter down the dataset to those rows which were not booked
    not_booked_data = data[data["status"] == "not-booked"]

    # Compute minimum score threshold for booked and not-booked data
    min_booked_score = data.loc[data["status"] == "booked", "predicted_score"].min()
    min_not_booked_score = data.loc[data["status"] == "not-booked", "predicted_score"].min()

    # Check for the edge case and issue a warning, if it applies
    if min_booked_score < min_not_booked_score:
        return "Edge case encountered: Min score of booked rows is lower than min score of not-booked rows."

    metrics = {"min_threshold": min_booked_score,
               "total_rows": data.shape[0],
               "TP": booked_data[data["predicted_score"] >= min_booked_score].shape[0],
               "FP": not_booked_data[data["predicted_score"] >= min_booked_score].shape[0],
               "TN": not_booked_data[data["predicted_score"] < min_booked_score].shape[0],
               "FN": booked_data[data["predicted_score"] < min_booked_score].shape[0],
               }

    return metrics
