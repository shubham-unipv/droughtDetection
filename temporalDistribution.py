import pandas as pd
import matplotlib.pyplot as plt
import os

def load_season_data(district, year):
    """Load data for a specific district and Rabi season"""
    current_file = f"MoreData/timeSeriesData/TimeSeries_{district}_{year}.csv"
    prev_file = f"MoreData/timeSeriesData/TimeSeries_{district}_{year-1}.csv"
    
    season_df = pd.DataFrame()
    
    # Load previous year's October-December data
    if os.path.exists(prev_file):
        prev_df = pd.read_csv(prev_file)
        prev_df['date'] = pd.to_datetime(prev_df['date'])
        prev_df = prev_df[(prev_df['date'] >= f"{year-1}-10-01") & 
                         (prev_df['date'] <= f"{year-1}-12-31")]
        season_df = pd.concat([season_df, prev_df], ignore_index=True)
    
    # Load current year's January-April data
    if os.path.exists(current_file):
        curr_df = pd.read_csv(current_file)
        curr_df['date'] = pd.to_datetime(curr_df['date'])
        curr_df = curr_df[(curr_df['date'] >= f"{year}-01-01") & 
                         (curr_df['date'] <= f"{year}-04-30")]
        season_df = pd.concat([season_df, curr_df], ignore_index=True)
    
    return season_df

def plot_single_season(district, year):
    """Plot temporal distribution for a single season"""
    season_df = load_season_data(district, year)
    
    if not season_df.empty:
        plt.figure(figsize=(12, 6))
        
        # Drop duplicate dates and sort
        season_df = season_df.drop_duplicates(subset=['date'])
        season_df = season_df.sort_values('date')
        
        # Create date range for x-axis (full season)
        date_range = pd.date_range(start=f"{year-1}-10-01", end=f"{year}-04-30")
        
        # Plot full date range as background markers
        plt.plot(date_range, [1] * len(date_range), 'o', color='lightgray', 
                alpha=0.3, label='Possible Dates')
        
        # Plot actual data points
        plt.plot(season_df['date'], [1] * len(season_df), 'o', 
                color='darkblue', label='Available Data')
        
        plt.title(f'Temporal Distribution - {district}\nRabi Season {year}')
        plt.xlabel('Date')
        plt.yticks([])  # Hide y-axis ticks since we only have one level
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add grid and legend
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'MoreData/TemporalDistribution/TemporalDistribution_{district}_season_{year}.png', 
                    bbox_inches='tight', dpi=300)
        plt.close()

# Create output directory if it doesn't exist
os.makedirs('MoreData/TemporalDistribution', exist_ok=True)

# Generate all 27 plots
districts = ['Jodhpur', 'Amravati', 'Thanjavur']
for district in districts:
    for year in range(2016, 2026):
        plot_single_season(district, year)
        print(f"Completed plot for {district} - Season {year-1}-{year}")