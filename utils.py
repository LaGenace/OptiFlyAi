from sklearn.model_selection import train_test_split


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

    return df_final


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
