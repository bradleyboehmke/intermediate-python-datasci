import pandas as pd
import numpy as np
import seaborn as sns
import sklearn

planes_df = pd.read_csv('data/planes.csv')
planes_df.head()

planes_df.assign(seats_and_crew = planes_df['seats'] + 5)
planes_df.groupby('type').agg({'speed': 'mean'})


# Conditionals
denominator = 0
numerator = 329
if denominator != 0:
    quotient = numerator / denominator
else:
    print('Cannot divide by zero')

my_list = [23, 40, 25, 9, 33]
if len(my_list) > 0:
    mean_of_my_list = np.mean(my_list)


# vectorized if statement
planes_df['large'] = np.where(planes_df['seats'] > 150, 'Yes', 'No')
planes_df['modern'] = np.where(planes_df['year'] > 2008, 'Yes', 'No')
planes_df['large_and_modern'] = np.where((planes_df['large'] == 'Yes') & (planes_df['modern'] == 'Yes'), 'Yes', 'No')


any([number < 3 for number in range(10)])
all([number < 3 for number in range(10)])


# Iteration
summation = 0
for number in range(100):
    summation = summation + number

neighborhoods = ['oakley', 'hyde park', 'clifton', 'corryville', 'northside']
for hood in neighborhoods:
    print(hood)

for index, value in enumerate(neighborhoods):
    print(index, '-', value)


# iterating over a series
flights_df = pd.read_csv('data/flights.csv')

time_variables = ['dep_delay', 'arr_delay', 'air_time']
for variable in time_variables:
    flights_df[variable] = flights_df[variable] / 60

for element in [0, 2, 4, 6, 8, 10]:
    print(element ** element)

for column in flights_df.columns:
    if flights_df[column].dtype != np.float64 and flights_df[column].dtype != np.int64:
        flights_df[column] = flights_df[column].str.lower()

for column in flights_df.columns:
    missing = flights_df[column].isna().sum()
    if missing > 0:
        print(column + ' - ' + str(missing))

number = 0
while number < 15:
    if number % 2 == 0:
        print(number)
    number = number + 1


# Comprehensions
[number for number in range(10)]
[number < 5 for number in range(10)]
[number for number in range(10) if number not in [3, 5]]



# Functions
def area_of_triangle(h, b):
    return (h * b) / 2

# quick and dirty test
assert area_of_triangle(2, 3) == 3

# more formal
if area_of_triangle(2, 3) != 3:
    raise Exception('You have made an error')

# docstring
def double(x):
    '''Multiply x by 2 and return the result.'''
    doubled = x * 2
    return doubled

# **kwargs passes named arguments into a dictionary
def myfunc(**kwargs):
    print(kwargs)

myfunc(a = 1, b = 2)


def cat(str1, str2, str3):
    print('{} {} {}'.format(str1, str2, str3))

cat('Bradley', 'Carl', 'Boehmke')

def cat(str1, str2, str3 = '.'):
    print('{} {} {}'.format(str1, str2, str3))

cat('Bradley', 'Boehmke')

def cat(str1, str2, str3 = '.'):
    '''
    Concatenate string inputs.

    The user can supply any three strings and the resulting
    output will be those three strings concatenated with a
    single white space in between each input.

    Parameters
    ----------
    str1 : str
        First character string to concatenate
    str2 : str
        Second character string to concatenate
    str3 : str
        Third character string to concatenate

    Returns
    -------
    str
    A concatenated string of str1, str2, and str3

    Examples
    --------
    >>> cat('Bradley', 'Carl', 'Boehmke')
    Bradley Carl Boehmke
    '''
    return '{} {} {}'.format(str1, str2, str3)


# apply function
my_series = pd.Series(['b', 'r', 'a', 'd'])

def my_upper(x):
    return x.upper()

my_series.apply(my_upper)

my_numbers = list(range(100))
my_numbers = pd.Series(my_numbers)
(my_numbers - my_numbers.mean()) / my_numbers

my_series = pd.Series([3, 1, -4, 4, -9])
my_series.abs()

directions = [('North', 'West'),
              ('East', 'North'),
              ('South', 'East'),
              ('West', 'South')]

directions = pd.DataFrame(directions, columns = ['facing', 'leftward'])

def my_func(x, y):
    return 'Looking {} means {} is to your left'.format(x, y)

directions.apply(lambda x: my_func(x['facing'], x['leftward']), axis = 1)


########################
#      CASE STUDY      #
########################

companies = pd.read_csv('data/companies.csv')
prices = pd.read_csv('data/prices.csv')

def is_incorporated(name: str) -> str:
    return 'inc' in name.lower().strip()

assert is_incorporated('my_Inc')
assert is_incorporated('my_org')

for name in companies['Name']:
    print(is_incorporated(name))

companies['Name'].apply(is_incorporated).sum()

def get_length(name):
    length = len(name)
    if length > 12:
        return 'long'
    elif 8 <= length <= 11:
        return 'medium'
    else:
        return 'short'

companies['Name'].apply(get_length)


def make_colname_string(df):
    return ','.join(df.columns)

make_colname_string(prices)

def projected_growth(name_length, price):
    if name_length == 'long':
        return price * 2
    elif name_length == 'medium':
        return price * 1
    else:
        return price * 0.5

df_merge = pd.merge(companies, prices, on = 'Symbol')
df_merge['name_length'] = df_merge['Name'].apply(get_length)
df_merge['Projected'] = df_merge.apply(lambda x: projected_growth(x['name_length'], x['Price']), axis = 1)
df[['Name', 'Symbol', 'Projected']].head()
