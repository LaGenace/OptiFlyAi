from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import category_encoders as ce
from scipy import stats



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

def convert_bool_to_num(value):
    """
    This converts bulian values to 0 or 1, but flips True and False
    So False will == 1, and True == 0
    This is is used when False values are seen as 'superior' to True values
    """
    return 0 if value else 1


def all_preprocessing(raw_data, columns_to_process, target_creation_function, target,
                        box_cox_columns=False, yeo_johnson_columns=False, min_max_scaling=False, log_transform_columns=False,
                        od_encoding=False, operator_encoding=False,
                        target_func_param1=None, target_func_param2=None, target_func_param3=None):
    """
    This functions completes all feature engineering, target creation and scaling
    RETURNS: updated dataframe and a Class that holds all the scalers

    Notes:
    - It will only return columns in columns_to_process and the target
    """

    #DATA CLEANING

    # All int64 columns need to be float64, or some functions don't work. e.g zscore
    for column in raw_data.select_dtypes(include=['int64']).columns:
        raw_data[column] = raw_data[column].astype('float64')

    #FEATURE ENGINEERING SECTION

    # This creates a column to identify OD's
    raw_data['OD'] = raw_data['OriginCty'] + raw_data['DestinationCty']

    # This calculates the total layover time with ratio
    raw_data['total_layover_time'] = raw_data['DurationMin'] - raw_data['Total_Flight_Duration']
    raw_data['total_layover_time_ratio'] =raw_data['total_layover_time'] /raw_data['DurationMin']

    # This calculates the difference between total distance traveled and 'straight line' distance
    raw_data['extra_travel_distance'] = raw_data['Total_Flight_Distance'] - raw_data['TravelDistanceKm']
    raw_data['extra_travel_distance_ratio'] =  raw_data['Total_Flight_Distance'] / raw_data['TravelDistanceKm']

    # This drops all rows with neg layover time
    data_engineered = drop_neg_layover_time(raw_data)

    # Create the target
    processed_data = target_creation_function(data_engineered, target_func_param1, target_func_param2, target_func_param3)

    # Seperating target so encoders dont store a df shape that is larger than real-world data
    # This is so encoders do not expect the extra column when running on new data, which will not have a target
    y = processed_data[target]

    model_data = processed_data.drop(columns=[target])

    #BINARY ENCODING
    # Binary encoding origin and destination
    if od_encoding:
        o_encoder = ce.BinaryEncoder()
        origin_apt_encoded = o_encoder.fit_transform(model_data['OriginApt'])
        columns_to_process.extend(origin_apt_encoded.columns.to_list())

        d_encoder = ce.BinaryEncoder()
        destination_apt_encoded = d_encoder.fit_transform(model_data['DestinationApt'])
        columns_to_process.extend(destination_apt_encoded.columns.to_list())

        #Concatinating newly encoded columns
        origin_binary = pd.concat([model_data, origin_apt_encoded], axis=1)
        dest_binary = pd.concat([origin_binary, destination_apt_encoded], axis=1)
    else:
        o_encoder = None
        d_encoder = None
        dest_binary = model_data.copy()

    # Binary encoding Operator IATA'
    if operator_encoding:
        seg_0_encoder = ce.BinaryEncoder()
        seg_0_binary = seg_0_encoder.fit_transform(model_data['Seg_0_OperatingCarrierIATA'])
        columns_to_process.extend(seg_0_binary.columns.to_list())

        seg_1_encoder = ce.BinaryEncoder()
        seg_1_binary = seg_1_encoder.fit_transform(model_data['Seg_1_OperatingCarrierIATA'])
        columns_to_process.extend(seg_1_binary.columns.to_list())

        seg_2_encoder = ce.BinaryEncoder()
        seg_2_binary = seg_2_encoder.fit_transform(model_data['Seg_2_OperatingCarrierIATA'])
        columns_to_process.extend(seg_2_binary.columns.to_list())

        seg_3_encoder = ce.BinaryEncoder()
        seg_3_binary = seg_3_encoder.fit_transform(model_data['Seg_3_OperatingCarrierIATA'])
        columns_to_process.extend(seg_3_binary.columns.to_list())

        #Concatinating newly encoded columns
        seg0_bin = pd.concat([dest_binary, seg_0_binary], axis=1)
        seg1_bin = pd.concat([seg0_bin, seg_1_binary], axis=1)
        seg2_bin = pd.concat([seg1_bin, seg_2_binary], axis=1)
        all_binary = pd.concat([seg2_bin, seg_3_binary], axis=1)
    else:
        seg_0_encoder = None
        seg_1_encoder = None
        seg_2_encoder = None
        seg_3_encoder = None
        all_binary = dest_binary.copy()

    all_binary = all_binary[columns_to_process]

    #SCALING
    # Box cox

    # Dictionary to store best_lambda per column for new data processing
    box_lambdas = {}

    if box_cox_columns:
        for col in box_cox_columns:
            all_binary[col], box_lambda = stats.boxcox(all_binary[col])
            box_lambdas[col] = box_lambda

    # Yeo-johnson
    # Dictionary to store best_lambda per column for new data processing
    yeo_lambdas = {}

    if yeo_johnson_columns:
        for col in yeo_johnson_columns:
            all_binary[col], yeo_lambda = stats.yeojohnson(all_binary[col])
            yeo_lambdas[col] = yeo_lambda

    # Log transformations
    if log_transform_columns:
        for column in log_transform_columns:
            all_binary.loc[:, column] = np.log1p(model_data[column])

    #Min max scaling
    # Dictionary to store min max scaler per column for new data processing
    min_max_scalers = {}

    if min_max_scaling:
        for col in min_max_scaling:
            minmax_scaler = MinMaxScaler()
            all_binary[col] = minmax_scaler.fit_transform(all_binary[[col]])
            min_max_scalers[col] = minmax_scaler


    # Cyclical encoding
    all_binary['sin_day'] = np.sin(2 * np.pi * all_binary['dayofweek'] / 7)
    all_binary['cos_day'] = np.cos(2 * np.pi * all_binary['dayofweek'] / 7)

    all_binary.drop(columns='dayofweek', inplace=True)

    #Inversing the importance of SelfTransfer, so Non Self Transfer is seen as better by the model
    all_binary['SelfTransfer'] = all_binary['SelfTransfer'].apply(convert_bool_to_num)

    #STORING SCALERS
    class PreprocessScalers:
        def __init__(self, o_encoder, d_encoder, box_lambdas, yeo_lambdas, min_max_scalers,seg_0_encoder, seg_1_encoder, seg_2_encoder, seg_3_encoder):
                self.o_encoder = o_encoder
                self.d_encoder = d_encoder
                self.box_lambda = box_lambdas
                self.yeo_lambda = yeo_lambdas
                self.minmax_scaler = min_max_scalers
                self.seg_0_encoder = seg_0_encoder
                self.seg_1_encoder = seg_1_encoder
                self.seg_2_encoder = seg_2_encoder
                self.seg_3_encoder = seg_3_encoder

    scalers = PreprocessScalers(o_encoder, d_encoder, box_lambdas, yeo_lambdas, minmax_scaler,seg_0_encoder, seg_1_encoder, seg_2_encoder, seg_3_encoder)

    #Adding y into dataset
    all_binary[target] = y

    # Returning dataframe and scalers
    return all_binary, scalers
