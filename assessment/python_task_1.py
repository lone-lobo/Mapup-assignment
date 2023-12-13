from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def generate_car_matrix(df):
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)
    matrix.values[np.arange(matrix.shape[0]), np.arange(matrix.shape[0])] = 0
    return matrix

def get_type_count(df):
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    df['car_type'] = pd.cut(df['car'], bins=[-np.inf, 15, 25, np.inf], labels=['low', 'medium', 'high'], right=False)
    car_type_counts = df['car_type'].value_counts().to_dict()
    sorted_car_type_counts = dict(sorted(car_type_counts.items()))
    return sorted_car_type_counts

def get_bus_indexes(df):
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    mean_bus = df['bus'].mean()
    bus_indexes = df[df['bus'] > 2 * mean_bus].index.tolist()
    sorted_bus_indexes = sorted(bus_indexes)
    return sorted_bus_indexes

def filter_routes(df):
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    avg_truck_by_route = df.groupby('route')['truck'].mean()
    routes_above_7 = avg_truck_by_route[avg_truck_by_route > 7].index.tolist()
    sorted_routes_above_7 = sorted(routes_above_7)
    return sorted_routes_above_7

def multiply_matrix(matrix):
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    modified_matrix = matrix.copy()
    modified_matrix[modified_matrix > 20] *= 0.75
    modified_matrix[modified_matrix <= 20] *= 1.25
    modified_matrix = modified_matrix.round(1)
    return modified_matrix

def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    days = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }

    today = datetime.today().weekday()

    def weekday_to_date(weekday):
        day_diff = days[weekday] - today
        return datetime.today() + timedelta(days=day_diff)

    df['startDate'] = df['startDay'].apply(weekday_to_date)
    df['startDate'] = pd.to_datetime(df['startDate']).dt.round('1s')
    df['startTime'] = pd.to_datetime(df['startTime'])
    df['startDayTime'] = df.apply(lambda row: row['startDate'].replace(hour=row['startTime'].hour, minute=row['startTime'].minute, second=row['startTime'].second), axis=1)
    df['endDate'] = df['endDay'].apply(weekday_to_date)
    df['endDate'] = pd.to_datetime(df['endDate']).dt.round('1s')
    df['endTime'] = pd.to_datetime(df['endTime'])
    df['endDayTime'] = df.apply(lambda row: row['endDate'].replace(hour=row['endTime'].hour, minute=row['endTime'].minute, second=row['endTime'].second), axis=1)
    df = df.drop(columns = ['startDate' , 'endDate'])

    df = df.groupby(['id', 'id_2']).agg({'startDayTime': 'min', 'endDayTime': 'max'}).reset_index()
    df['TimeDifference'] = df['endDayTime'] - df['startDayTime']
    time_delta = pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)
    df['time_completeness'] = df['TimeDifference'] == time_delta
    # need to recheck on the multiindex part ( workaround)
    df.index = pd.MultiIndex.from_frame(df[['id', 'id_2']])
    df = df.drop(columns=['id' , 'id_2' ])
    return df

