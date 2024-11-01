import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from collections import Counter


class Preprocessor:
    def __init__(self, data_path):
        self.__df = pd.read_csv(data_path)
        self.__df_scores = pd.DataFrame()
        
    def apply_strategy(self):
        # Drop unnecessary columns and initialize df_scores with Companies column
        self.__df = self.__df.drop(['URL', 'Text', 'Speech & audio', 'Code', 'Visual media'], axis=1, errors='ignore')
        self.__df_scores['Companies'] = self.__df['Companies']

        # Aplly each score computing strategy
        self.__compute_industry_score() # 1
        self.__comput_date_scores() # 3
        self.__compute_latest_funding_round_score() # 1
        self.__compute_normalized_investor_application_scores()
        self.__compute_acquirers_score() # 1
        self.__compute_exit_round_score() # 1
        self.__recompute_mosaic_score() # 1
        self.__recompute_numerical_scores() # 3


        # Fill the missing scores with zeros to have no effect while computing 
        for column in self.__df_scores.columns:
            if 'Score' in column:
                self.__df_scores[column] = self.__df_scores[column].fillna(value=0)

    def __normalization_op(self, col):
            min_score, max_score = self.__df_scores[col].min(), self.__df_scores[col].max()
            self.__df_scores[col] = (self.__df_scores[col] - min_score) / (max_score - min_score)

    def __compute_industry_score(self):
        # Calculate mean and median valuations by Industry
        industry_mean = self.__df.groupby('Industry')['Latest Valuation'].mean().reset_index()
        industry_median = self.__df.groupby('Industry')['Latest Valuation'].median().reset_index()

        # Rename columns for clarity
        industry_mean.columns = ['Industry', 'Mean_Valuation']
        industry_median.columns = ['Industry', 'Median_Valuation']

        # Normalize mean and median valuations
        scaler = MinMaxScaler()
        industry_mean['Industry_Score_Mean'] = scaler.fit_transform(industry_mean[['Mean_Valuation']])
        industry_median['Industry_Score_Median'] = scaler.fit_transform(industry_median[['Median_Valuation']])

        # Merge mean and median scores
        industry_scores = pd.merge(
            industry_mean[['Industry', 'Industry_Score_Mean']],
            industry_median[['Industry', 'Industry_Score_Median']],
            on='Industry'
        )

        # Normalize the scores to sum to 1
        industry_scores['Normalized_Score_Mean'] = industry_scores['Industry_Score_Mean'] / industry_scores['Industry_Score_Mean'].sum()
        industry_scores['Normalized_Score_Median'] = industry_scores['Industry_Score_Median'] / industry_scores['Industry_Score_Median'].sum()

        # Calculate combined mean of normalized scores
        industry_scores['Mean_of_MeanMedian_Scores'] = (
            industry_scores['Normalized_Score_Mean'] + industry_scores['Normalized_Score_Median']
        ) / 2

        # Map industry scores to the main dataframe
        industry_score_map = dict(zip(industry_scores['Industry'], industry_scores['Mean_of_MeanMedian_Scores']))
        self.__df_scores['Industry'] = self.__df['Industry']
        self.__df_scores['Industry Score'] = self.__df_scores['Industry'].map(industry_score_map)

    def __compute_latest_funding_round_score(self):
        funding_round_scores = {
            'Line of Credit': 1, 'Grant': 1, 'Angel': 2, 'Biz Plan Competition': 2, 'Pre-Seed': 3,
            'Seed': 3, 'Seed VC': 3, 'Bridge': 3, 'Debt': 3, 'Loan': 3, 'Incubator/Accelerator': 4,
            'Unattributed': 4, 'Unattributed VC': 5, 'Convertible Note': 5, 'Corporate Minority': 5,
            'Series A': 5, 'Series B': 6, 'Series C': 7, 'Series D': 8, 'Series E': 9, 'Series F': 10,
            'Series G': 10.5, 'Series H': 11, 'Series I': 11.5, 'Private Equity': 9, 'Growth Equity': 10,
            np.nan: 0  # Handling missing values
        }

        # Process funding round names
        def get_cleaned_funding_round(funding_round):
            if isinstance(funding_round, str):
                cleaned_round = funding_round.split(' - ')[0]
                return funding_round_scores.get(cleaned_round, 0)
            return 0  # Handle missing or non-string values

        self.__df_scores['Latest Funding Round'] = self.__df['Latest Funding Round']
        self.__df_scores['Latest Funding Round Score'] = self.__df_scores['Latest Funding Round'].apply(get_cleaned_funding_round)
        self.__normalization_op('Latest Funding Round Score')

        
    def __process_date_column(self, df, df_scores, date_column):
        # Convert the date column to datetime
        df[date_column] = pd.to_datetime(df[date_column], dayfirst=True, yearfirst=False)
        self.__df_scores[date_column] = self.__df[date_column]
        
        # Calculate days since the given date
        today = pd.to_datetime(datetime.now())
        days_since_col = f'Days Since {date_column}'
        self.__df_scores[days_since_col] = (today - self.__df_scores[date_column]).dt.days
        
        # Calculate and normalize the score
        max_days = self.__df_scores[days_since_col].max()
        score_col = f'{date_column} Score'
        df_scores[score_col] = ((max_days - df_scores[days_since_col]) / max_days) * 100  # Scores between 0 and 100
        
        self.__normalization_op(score_col)

        # Drop the days since column
        self.__df_scores.drop(columns=[days_since_col], inplace=True)

    def __comput_date_scores(self):
        # Apply the function to each date column
        date_columns = ['Latest Funding Date', 'Date Added', 'Exit Date']
        for date_column in date_columns:
            self.__process_date_column(self.__df, self.__df_scores, date_column)

    def __recompute_numerical_scores(self):
        # Columns to transform and normalize
        columns = ['Total Funding', 'Latest Valuation', 'Latest Funding Amount']

        # Apply log transformation and normalization
        for col in columns:
            score_col = f'{col} Score'
            
            # Log transformation
            self.__df_scores[score_col] = np.log1p(self.__df[col])  # log1p handles zero values
            
            # Min-Max Normalization
            self.__normalization_op(score_col)

    def __recompute_mosaic_score(self):
        self.__df_scores['Mosaic (Overall)'] = self.__df['Mosaic (Overall)']
        self.__df_scores['Mosaic (Overall) Score'] = self.__df['Mosaic (Overall)']
        self.__normalization_op('Mosaic (Overall) Score')

    def __compute_normalized_investor_application_scores(self):
        # Copy 'All Investors' and 'Applications' columns to df_scores
        self.__df_scores[['All Investors', 'Applications']] = self.__df[['All Investors', 'Applications']]

        # Step 1: Calculate frequency scores for each unique investor and application
        # Flatten lists and calculate frequencies
        all_investors = [
            investor.strip()
            for sublist in self.__df_scores['All Investors'].dropna()
            for investor in sublist.split(',')
        ]

        all_applications = [
            app.strip()
            for sublist in self.__df_scores['Applications'].dropna()
            for app in sublist.split('.')
        ]

        # Calculate frequencies
        self.investor_freq = Counter(all_investors)
        self.application_freq = Counter(all_applications)


        # Step 2: Assign and sum scores for each company
        # Function to calculate score based on frequency dictionary
        def calculate_score(items, freq_dict):
            if isinstance(items, str):  # If items is a string, split it based on the relevant delimiter
                items = items.split(',') if freq_dict == self.investor_freq else items.split('.')
            elif pd.isna(items):  # Check for NaN values
                return 0  # Return a score of 0 for NaN values
            return sum(freq_dict[item.strip()] for item in items if item.strip() in freq_dict)

        # Apply function to each row
        self.__df_scores['All Investor Score'] = self.__df_scores['All Investors'].apply(lambda x: calculate_score(x, self.investor_freq))
        self.__df_scores['Applications Score'] = self.__df_scores['Applications'].apply(lambda x: calculate_score(x, self.application_freq))
        self.__normalization_op('All Investor Score')
        self.__normalization_op('Applications Score')

    def __compute_exit_round_score(self):
        exit_round_scores = {
            'Acquired': 1,
            'Acq - Fin - II': 2,
            'Merger': 2,
            'Spinoff / Spinout': 3,
            'Shareholder Liquidity': 4,
            'IPO': 5,
        }
        # Function to preprocess funding round names
        def get_cleaned_exit_round(exit_round):
            return exit_round_scores.get(exit_round, 0)  # Return score or 0 if not found

        self.__df_scores['Exit Round'] = self.__df['Exit Round']

        # Apply the function to create a new 'Funding Round Score' column
        self.__df_scores['Exit Round Score'] = self.__df_scores['Exit Round'].apply(get_cleaned_exit_round)

        self.__normalization_op('Exit Round Score')


    def  __compute_acquirers_score(self):
        # Fill acquirers column with values using investor_freq list calculated before
        def acquirers_score(acquirers):
            return self.investor_freq.get(acquirers, 0)  # Return score or 0 if not found

        self.__df_scores['Acquirers'] = self.__df['Acquirers']

        # Apply the function to create a new 'Funding Round Score' column
        self.__df_scores['Acquirers Score'] = self.__df_scores['Acquirers'].apply(acquirers_score)

        self.__normalization_op('Acquirers Score')

    def get_data_with_scores(self):
        return self.__df_scores