import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ssl

# read the dataset
url = "https://data.cityofnewyork.us/api/views/6fi9-q3ta/rows.csv?accessType=DOWNLOAD"
ssl._create_default_https_context = ssl._create_unverified_context

df = pd.read_csv(url)

# 1. Filter the data to include only weekdays (Monday to Friday)
# and plot a line graph showing the pedestrian counts for each day of the week.
df['hour_beginning'] = pd.to_datetime(df['hour_beginning'],
                                      format="%m/%d/%Y %I:%M:%S %p")
df_weekdays = df[df['hour_beginning'].dt.weekday < 5].copy()
df_weekdays['weekday'] = df_weekdays['hour_beginning'].dt.day_name()
weekday_counts = df_weekdays.groupby('weekday')['Pedestrians'].sum()
weekday_counts = weekday_counts.reindex(['Monday', 'Tuesday',
                                         'Wednesday', 'Thursday', 'Friday'])

plt.figure(figsize=(10, 5))
weekday_counts.plot(kind='bar', color='orange')
plt.title('Pedestrian Counts by Weekdays')
plt.xlabel("Day of the Week")
plt.ylabel("Pedestrian Counts")
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# 2. Track pedestrian counts on the Brooklyn Bridge for the year 2019
# and analyze how different weather conditions
# influence pedestrian activity in that year.
# Sort the pedestrian count data by weather summary
# to identify any correlations( with a correlation matrix)
# between weather patterns and pedestrian counts for the selected year.
df['weather_summary'] = df['weather_summary'].ffill().bfill()
df_2019 = df[df['hour_beginning'].dt.year == 2019].copy()
count_2019_weather = df_2019.groupby('weather_summary')['Pedestrians'].sum()

weather_mapping = {
    'clear-day': 1,
    'clear-night': 2,
    'cloudy': 5,
    'fog': 8,
    'partly-cloudy-day': 3,
    'partly-cloudy-night': 4,
    'rain': 6,
    'sleet': 10,
    'snow': 7,
    'wind': 9
}
df_2019['weather_summary'] = df_2019['weather_summary'].map(weather_mapping)

correlation_matrix = df_2019[['Pedestrians', 'weather_summary']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Pedestrians and Weather Summaries')
plt.tight_layout()
plt.show()

# There is a moderate correlation between the weather and pedestrian counts
# More pedestrians in better weathers


# 3. Implement a custom function to categorize time of day
# into morning, afternoon, evening, and night,
# and create a new column in the DataFrame to store these categories.
# Use this new column to analyze pedestrian activity patterns throughout the day.
def categorize(hour):
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 21:
        return "evening"
    else:
        return "night"

df['hour'] = df['hour_beginning'].dt.hour
df['time of day'] = df['hour'].apply(categorize)
time_counts = df.groupby('time of day')['Pedestrians'].sum()
time_counts = time_counts.reindex(['morning', 'afternoon', 'evening', 'night'])
print(df.head().to_string())
print(time_counts)

plt.figure(figsize=(12, 6))
time_counts.plot(kind='bar', color='orange')
plt.title('Total Pedestrian Counts by Time of Day')
plt.xlabel('Time of the Day')
plt.ylabel('Pedestrian Count')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Pedestrians most likely to walk on the bridge in afternoon
# and least likely at night
