import polars as pl

DTYPES = {
    'Duration': pl.Int8,
    'Distance': pl.Int32,
    'PLong': pl.Float32,
    'PLatd': pl.Float32,
    'DLong': pl.Float32,
    'DLatd': pl.Float32,
    'Haversine': pl.Float32,
    'Pmonth': pl.Int8,
    'Pday': pl.Int8,
    'Phour': pl.Int8,
    'Pmin': pl.Int8,
    'PDweek': pl.Int8,
    'Dmonth': pl.Int8,
    'Dday': pl.Int8,
    'Dhour': pl.Int8,
    'Dmin': pl.Int8,
    'DDweek': pl.Int8,
    'Temp': pl.Float32,
    'Precip': pl.Float32,
    'Wind': pl.Float32,
    'Humid': pl.Float32,
    'Solar': pl.Float32,
    'Snow': pl.Float32,
    'GroundTemp': pl.Float32,
    'Dust': pl.Float32
}

SAMPLE_SIZE = 300000

RANDOM_STATE = 13

TARGET_VARIABLE = 'Duration'

TEST_SIZE = 0.3
