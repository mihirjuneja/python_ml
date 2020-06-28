import pandas as pd
import numpy as np
import datetime as dt

#-------------------------- Functions --------------------------

def choose_columns(df):
    cols=[]
    for col in df.columns:
        if col == 'DateAndTime':
            continue
        else:
            if col == 'BLOCK_10_INV_2':
                cols.append(col)
                break
            else:
                cols.append(col)

    return cols

def normalize_inv_data(df):

    cols = choose_columns(df)

    print('Normalizing values for ', cols[0], ' to ', cols[len(cols) - 1])

    df[cols] = df[cols].apply(lambda x:(x-x.min()) / (x.max()-x.min()))

    return df

def std_time_stamp(dt_column, VAL):
    # Convert strings to date time values in DateAndTime series column
    dt_series = pd.to_datetime(dt_column)

    # subrtract all seconds and nearest VAL minutes
    dt_series = dt_series.apply(lambda x: x - dt.timedelta(minutes=x.minute % VAL,
                                seconds=x.second))

    print('Standardizing time stamps to nearest ', VAL, 'minutes')

    return dt_series
#-------------------------- Main --------------------------

df = pd.read_csv('solar_data.csv')

df = normalize_inv_data(df)

VAL = 5 #Nearest 5 min
df['DateAndTime'] = std_time_stamp(df['DateAndTime'], VAL)

df.to_excel('normalize_ouput.xlsx')