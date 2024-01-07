from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import category_encoders as ce
from scipy import stats
from typing import List, Callable, ClassVar

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


def all_preprocessing(raw_data:pd.DataFrame, columns_to_process:List[str], target_creation_function:Callable, target:str,
                        box_cox_columns:List[str]=False, yeo_johnson_columns:List[str]=False, min_max_scaling:List[str]=False, log_transform_columns:List[str]=False,
                        od_encoding:bool=False, operator_encoding:bool=False, od_combined_encoding=False,
                        target_func_param1=None, target_func_param2=None, target_func_param3=None):
    """
    This functions completes all feature engineering, target creation and scaling
    RETURNS: updated dataframe and a Class that holds all the scalers


    columns_to_process: what you want it to return, if “daysofweek” it will only return the cosine columns

    target_creation_function: function for making the target, allows for future flexibility

    target = name of the target, this is so no encoders/scalers are looking for it in the new data

    _column variables: the rest is just a list of columns that require different scaling/encoding.

    _encoding: Boolean, True only if you want to encode

    _func_params: Room for adding params to the target creation function.
    """

    # Creating a copy of the list so that we do not update the list outside of this function
    list_of_columns = columns_to_process.copy()

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
    if target_func_param1:
        processed_data = target_creation_function(data_engineered, target_func_param1, target_func_param2, target_func_param3)
    else:
        processed_data = target_creation_function(data_engineered)

    # Seperating target so encoders dont store a df shape that is larger than real-world data
    # This is so encoders do not expect the extra column when running on new data, which will not have a target
    y = processed_data[target]

    model_data = processed_data.drop(columns=[target])

    #BINARY ENCODING
    # Binary encoding combined OD
    if od_combined_encoding:
        od_encoder = ce.BinaryEncoder()
        od_encoded_columns = od_encoder.fit_transform(model_data['OD'])
        list_of_columns.extend(od_encoded_columns.columns.to_list())

        #Concatinating newly encoded columns
        combined_od_encoded = pd.concat([model_data, od_encoded_columns], axis=1)

    else:
        od_encoder = None
        combined_od_encoded = model_data.copy()


    # Binary encoding origin and destination
    if od_encoding:
        o_encoder = ce.BinaryEncoder()
        origin_apt_encoded = o_encoder.fit_transform(model_data['OriginApt'])
        list_of_columns.extend(origin_apt_encoded.columns.to_list())

        d_encoder = ce.BinaryEncoder()
        destination_apt_encoded = d_encoder.fit_transform(model_data['DestinationApt'])
        list_of_columns.extend(destination_apt_encoded.columns.to_list())

        #Concatinating newly encoded columns
        origin_binary = pd.concat([combined_od_encoded, origin_apt_encoded], axis=1)
        dest_binary = pd.concat([origin_binary, destination_apt_encoded], axis=1)
    else:
        o_encoder = None
        d_encoder = None
        dest_binary = combined_od_encoded.copy()

    # Binary encoding Operator IATA'
    if operator_encoding:
        seg_0_encoder = ce.BinaryEncoder()
        seg_0_binary = seg_0_encoder.fit_transform(model_data['Seg_0_OperatingCarrierIATA'])
        list_of_columns.extend(seg_0_binary.columns.to_list())

        seg_1_encoder = ce.BinaryEncoder()
        seg_1_binary = seg_1_encoder.fit_transform(model_data['Seg_1_OperatingCarrierIATA'])
        list_of_columns.extend(seg_1_binary.columns.to_list())

        seg_2_encoder = ce.BinaryEncoder()
        seg_2_binary = seg_2_encoder.fit_transform(model_data['Seg_2_OperatingCarrierIATA'])
        list_of_columns.extend(seg_2_binary.columns.to_list())

        seg_3_encoder = ce.BinaryEncoder()
        seg_3_binary = seg_3_encoder.fit_transform(model_data['Seg_3_OperatingCarrierIATA'])
        list_of_columns.extend(seg_3_binary.columns.to_list())

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

    all_binary = all_binary[list_of_columns].copy()

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


    if 'dayofweek' in list_of_columns:
        # Cyclical encoding
        all_binary['sin_day'] = np.sin(2 * np.pi * all_binary['dayofweek'] / 7)
        all_binary['cos_day'] = np.cos(2 * np.pi * all_binary['dayofweek'] / 7)

        all_binary.drop(columns='dayofweek', inplace=True)

    if 'SelfTransfer' in list_of_columns:
        #Inversing the importance of SelfTransfer, so Non Self Transfer is seen as better by the model
        all_binary['SelfTransfer'] = all_binary['SelfTransfer'].apply(convert_bool_to_num)

    #STORING SCALERS
    class PreprocessScalers:
        def __init__(self, od_encoder, o_encoder, d_encoder, box_lambdas, yeo_lambdas, min_max_scalers,seg_0_encoder, seg_1_encoder, seg_2_encoder, seg_3_encoder):
                self.od_encoder = od_encoder
                self.o_encoder = o_encoder
                self.d_encoder = d_encoder
                self.box_lambda = box_lambdas
                self.yeo_lambda = yeo_lambdas
                self.minmax_scaler = min_max_scalers
                self.seg_0_encoder = seg_0_encoder
                self.seg_1_encoder = seg_1_encoder
                self.seg_2_encoder = seg_2_encoder
                self.seg_3_encoder = seg_3_encoder

    scalers = PreprocessScalers(od_encoder, o_encoder, d_encoder, box_lambdas, yeo_lambdas, min_max_scalers, seg_0_encoder, seg_1_encoder, seg_2_encoder, seg_3_encoder)

    #Adding y into dataset
    all_binary[target] = y

    # Returning dataframe and scalers
    return all_binary, scalers

def create_df_of_all_categories(raw_data, data_to_be_processed, column):
    """
    This creates a list of all categories, and appends it to existing data that needs processing
    it appends the data, processes it, and drops the unnecesary columns
    It then appends it back to the dataframe so it can be used in the next step
    """

    # Creating dummy data
    categories  = pd.DataFrame(raw_data[column].unique(), columns=[column])

    # data_to_be_processed = data_to_be_processed.reset_index(drop=True)
    # categories = categories.reset_index(drop=True)

    # Merged data for encoding
    merged_data = pd.concat([data_to_be_processed[[column]],categories], axis=0)

    return merged_data


def encoding_new_data(original_data, data_to_be_processed, column, encoder):
    """
    This function allows for accurate encoding, that matches the training data
    In order to do that, we need to give the encoder all the options it previously had, otherwise the binary encoding won't align
    RETURNS: Only the encoded columns
    """
    # Creating dummy data from the original data
    data_with_dummies = create_df_of_all_categories(original_data, data_to_be_processed, column)

    # Encoding the column
    encoded_columns = encoder.transform(data_with_dummies[column])

    # Removing dummy data
    real_data = encoded_columns[:len(data_to_be_processed)].copy()

    return real_data


def process_new_data(original_data:pd.DataFrame, new_data:pd.DataFrame, scalers, columns_to_keep:List[str],
                     box_cox_columns:List[str]=False, yeo_johnson_columns:List[str]=False, log_transform_columns:List[str]=False, min_max_columns:List[str]=False,
                     od_encoding:bool=False, operator_encoding:bool=False, od_combined_encoding=False):
    """
    This function processes new data, using scalers and encoders from the training set
    It only returns the columns stated in columns_to_keep, and encoded columns if those options flipped to True

    original_data:
                This is used for encoding - can have dummy data if no encoding is taking place.
                IMPORTANT: Use the index of your processed data to slice the original_data.
                In order to encode OD's we need to re-create the OD column in original_data.
                We are slicing by data that we train on so no encoded columns stay consistent

    new_data: The data you are processing

    scalers: Must be a Class variable of scalers from the all_preprocessing function

    columns_to_keep: this is the list of columns you want to keep, if daysofweek, it only returns the cosin columns and drops daysofweek

    _columns variables: List of columns you want in that scaling step

    od_encoding + operator_encoding: Boolean, only flip if you want to encode new data.

    """

    # Creating a copy of the list so that we do not update the list outside of this function
    list_of_columns = columns_to_keep

    # DATA CLEANING

    # Filling the Null itinerary_fare data with booked_fare
    new_data['itinerary_fare'].fillna(new_data['booked_fare'], inplace=True)

    # Dropping data where itinerary_fare remains Null
    clean_data = new_data.dropna(subset=['itinerary_fare']).copy().reset_index()

    # FEATURE ENGINEERING
    clean_data['DurationMin'] = clean_data['flight_time'] + clean_data['connection_time']

    clean_data['total_layover_time_ratio'] = clean_data['connection_time'] / clean_data['DurationMin']

    clean_data['extra_travel_distance'] = clean_data['total_distance'] - clean_data['direct_distance']
    clean_data['extra_travel_distance_ratio'] =  clean_data['total_distance'] / clean_data['direct_distance']

    # Creating OD column in both original and new data, otherwise the binary encoders will not have the reference point for original information
    clean_data['OD'] = clean_data['origin'] + clean_data['destination']

    original_data['OD'] = original_data['OriginApt'] + original_data['DestinationApt']

    if 'seg_0' not in clean_data.columns:
        clean_data['seg_0'] = 0
        clean_data['seg_1'] = 0

        for i in range(len(clean_data)):
            listtt = clean_data['flights'][i].split(',')
            clean_data['seg_0'][i] = listtt[0][:2]
            clean_data['seg_1'][i] = listtt[1].strip()[:2]

    # Renaming the columns
    col_rename_dict = {'origin': 'OriginApt', 'destination':'DestinationApt', 'days_to_travel':'TravelHorizonDays', 'total_distance':'Total_Flight_Distance',
                    'direct_distance':'TravelDistanceKm', 'connection_time':'total_layover_time', 'flight_time':'Total_Flight_Duration','itinerary_fare':'PricePerPax',
                    'seg_0':'Seg_0_OperatingCarrierIATA', 'seg_1':'Seg_1_OperatingCarrierIATA', 'seg_2':'Seg_2_OperatingCarrierIATA', 'seg_3':'Seg_3_OperatingCarrierIATA'}

    clean_data = clean_data.rename(columns=col_rename_dict).copy()

    #TEMP creating Stops and SelfTransfer data
    clean_data['Stops'] = 1
    clean_data['SelfTransfer'] = True

    #DATA CLEANING
    for column in clean_data.select_dtypes(include=['int64']).columns:
        clean_data[column] = clean_data[column].astype('float64')

    # ENCODING
    if od_combined_encoding:
        #Binary encoding origin
        combined_od_encoding = encoding_new_data(original_data=original_data, data_to_be_processed=clean_data, column='OD', encoder=scalers.od_encoder)

        # Updating the dataset with the encoded columns
        od_clean_data = pd.concat([clean_data, combined_od_encoding], axis=1)

        # Ensuring the columns are returned at the end of the function
        list_of_columns.extend(combined_od_encoding.columns.to_list())

    else:
        od_clean_data = clean_data.copy()

    if od_encoding:
        #Binary encoding origin
        origin_encoded = encoding_new_data(original_data=original_data, data_to_be_processed=od_clean_data, column='OriginApt', encoder=scalers.o_encoder)

        # Binary encoding Destination
        destination_encoded = encoding_new_data(original_data, od_clean_data, 'DestinationApt', scalers.d_encoder)

        # Updating the dataset with the encoded columns
        both_ods_clean_data = pd.concat([od_clean_data, origin_encoded, destination_encoded], axis=1)

        # Ensuring the columns are returned at the end of the function
        list_of_columns.extend(origin_encoded.columns.to_list())
        list_of_columns.extend(destination_encoded.columns.to_list())

    else:
        both_ods_clean_data = od_clean_data.copy()

    if operator_encoding:
        seg_0_op_iata = encoding_new_data(original_data, both_ods_clean_data, 'Seg_0_OperatingCarrierIATA', scalers.seg_0_encoder)
        seg_1_op_iata = encoding_new_data(original_data, both_ods_clean_data, 'Seg_1_OperatingCarrierIATA', scalers.seg_1_encoder)

        if 'Seg_2_OperatingCarrierIATA' in both_ods_clean_data.columns:
            seg_2_op_iata = encoding_new_data(original_data, both_ods_clean_data, 'Seg_2_OperatingCarrierIATA', scalers.seg_2_encoder)
        else:
            seg_2_op_iata = False

        if 'Seg_3_OperatingCarrierIATA' in both_ods_clean_data.columns:
            seg_3_op_iata = encoding_new_data(original_data, both_ods_clean_data, 'Seg_3_OperatingCarrierIATA', scalers.seg_3_encoder)
        else:
            seg_3_op_iata = False

        # Updating the dataset with the encoded columns

        dfs_to_concat = [both_ods_clean_data, seg_0_op_iata, seg_1_op_iata]

        if seg_2_op_iata:
            dfs_to_concat.append(seg_2_op_iata)
        if seg_3_op_iata:
            dfs_to_concat.append(seg_3_op_iata)

        seg_clean_data = pd.concat(dfs_to_concat, axis=1)

        # Ensuring the columns are returned at the end of the function
        if operator_encoding:
            list_of_columns.extend(seg_0_op_iata.columns.to_list())
            list_of_columns.extend(seg_1_op_iata.columns.to_list())
            if seg_2_op_iata:
                list_of_columns.extend(seg_2_op_iata.columns.to_list())
            if seg_3_op_iata:
                list_of_columns.extend(seg_3_op_iata.columns.to_list())
    else:
        seg_clean_data = both_ods_clean_data.copy()

    list_of_columns.append('bookings')

    # SCALING
    # Box cox
    if box_cox_columns:
        for col in box_cox_columns:
            seg_clean_data.loc[:,col]  = stats.boxcox(seg_clean_data[col], lmbda=scalers.box_lambda[col])

    # Yeo-johnson
    if yeo_johnson_columns:
        for col in yeo_johnson_columns:
            seg_clean_data.loc[:,col] = stats.yeojohnson(seg_clean_data[col], lmbda=scalers.yeo_lambda[col])

    # Log transformations
    if log_transform_columns:
        for col in log_transform_columns:
            seg_clean_data.loc[:,col] = np.log1p(seg_clean_data[col])

    #Min max scaling
    if min_max_columns:
        for col in min_max_columns:
            seg_clean_data.loc[:,col] = scalers.minmax_scaler[col].transform(seg_clean_data[[col]])

    if 'SelfTransfer' in list_of_columns:
        #Inversing the importance of SelfTransfer, so Non Self Transfer is seen as better by the model
        seg_clean_data['SelfTransfer'] = seg_clean_data['SelfTransfer'].apply(convert_bool_to_num)

    data_to_return = seg_clean_data[list_of_columns].copy()

    if 'dayofweek' in list_of_columns:
        # Cyclical encoding
        data_to_return['sin_day'] = np.sin(2 * np.pi * data_to_return['dayofweek'] / 7)
        data_to_return['cos_day'] = np.cos(2 * np.pi * data_to_return['dayofweek'] / 7)

        # Dropping day of week as it is no longer neccesary
        data_to_return = data_to_return.drop(columns=['dayofweek']).copy()

    return data_to_return

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

def log_od_by_itredirects(data):
    """
    This function creates a target. log(ODRedirects) * ItineraryRedirects
    """
    data['log_ODRedirects'] = np.log1p(data['ODRedirects'])

    data['log_od_by_itredirect'] = data['ItineraryRedirects'] * data['log_ODRedirects']

    return data
