
def create_od_column(df, raw_od_columns):
    """
    This creates a column to identify OD's
    Depending on the columns, it can be used on City or Airport
    """
    df['OD'] = df[raw_od_columns[0]] + df[raw_od_columns[1]]

    return df


def calculate_total_segment_times(df):
    """
    This add all the segment times together
    """

    # Create list of values to fill columns with - if this is not done you get a warning
    values = {'Seg_0_DurationMin': 0 ,'Seg_1_DurationMin': 0, 'Seg_2_DurationMin': 0, 'Seg_3_DurationMin': 0}
    # Fills the values
    df.fillna(value=values, inplace=True)

    # Adds them up
    df['total_seg_time'] = df['Seg_0_DurationMin'] + df['Seg_1_DurationMin'] + df['Seg_2_DurationMin'] + df['Seg_3_DurationMin']

    return df

def calculate_total_layover_time(df, as_ratio=False):
    """
    Creates a total layover time by subtracting flight duration with segment times
    You can choose between a ratio or absolute value
    """

    # Runs function to create segment times
    df_with_segment_time = calculate_total_segment_times(df)

    # Calculates total layover time
    df_with_segment_time['total_layover_time'] = df_with_segment_time['DurationMin'] - df_with_segment_time['total_seg_time']

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
    """
    # This avoids warnings about working on a slice of a dataframe
    copy = df.copy()

    # Filling null values with 0
    values = {'Seg_0_TravelDistanceKm': 0 ,'Seg_1_TravelDistanceKm': 0, 'Seg_2_TravelDistanceKm': 0, 'Seg_3_TravelDistanceKm': 0}

    # Fills the values
    copy.fillna(value=values, inplace=True)

    # Adds them up
    copy['total_distance_traveled'] = copy['Seg_0_TravelDistanceKm'] + copy['Seg_1_TravelDistanceKm'] + copy['Seg_2_TravelDistanceKm'] + copy['Seg_3_TravelDistanceKm']

    return copy


def calculate_distance_difference(df, as_ratio=False):
    """
    Calculates the difference between total distance traveled and 'straight line' distance
    Can be set to be ratio or absolute difference
    """

    # This runs a function that calcualtes total distance traveled
    copy = calculate_total_distance(df)

    # Creates a ratio or absolute difference of total distance traveled
    if as_ratio == False:
        copy['extra_travel_distance'] = copy['total_distance_traveled'] - copy['TravelDistanceKm']

    else:
        copy['extra_travel_distance_ratio'] =  copy['total_distance_traveled'] / copy['TravelDistanceKm']

    return copy
