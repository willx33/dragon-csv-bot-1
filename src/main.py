import os
import pandas as pd
from typing import List
from simple_term_menu import TerminalMenu

class CSVParser:
    def __init__(self, unparsed_dir: str = "unparsed_csvs", parsed_dir: str = "parsed_csvs"):
        self.unparsed_dir = unparsed_dir
        self.parsed_dir = parsed_dir
        self.ensure_directories()
        
    def ensure_directories(self):
        os.makedirs(self.unparsed_dir, exist_ok=True)
        os.makedirs(self.parsed_dir, exist_ok=True)
    
    def get_available_files(self) -> List[str]:
        return [f for f in os.listdir(self.unparsed_dir) if f.endswith('.csv')]

    def calculate_standard_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        wins_50_plus = ['50% - 199%', '200% - 499%', '500% - 600%', '600% +']
        wins_200_plus = ['200% - 499%', '500% - 600%', '600% +']
        
        df['ratio_50_plus'] = df[wins_50_plus].sum(axis=1) / df['buy_7d']
        df['ratio_200_plus'] = df[wins_200_plus].sum(axis=1) / df['buy_7d']
        
        return df

    def apply_base_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        df['failed_criteria'] = df.apply(lambda x: [], axis=1)
        
        if 'sol_balance' in df.columns:
            df.loc[df['sol_balance'].fillna(0) <= 0, 'failed_criteria'] = \
                df.loc[df['sol_balance'].fillna(0) <= 0, 'failed_criteria'].apply(lambda x: x + ['zero_sol_balance'])
        
        if 'buy_7d' in df.columns:
            df.loc[df['buy_7d'].fillna(0) <= 5, 'failed_criteria'] = \
                df.loc[df['buy_7d'].fillna(0) <= 5, 'failed_criteria'].apply(lambda x: x + ['low_buy_count'])
        
        if 'winrate_7d' in df.columns:
            winrate_series = pd.to_numeric(
                df['winrate_7d'].replace('?', None).str.rstrip('%'), 
                errors='coerce'
            ) / 100
            df.loc[winrate_series.fillna(0) <= 0.30, 'failed_criteria'] = \
                df.loc[winrate_series.fillna(0) <= 0.30, 'failed_criteria'].apply(lambda x: x + ['low_winrate'])

        for col, (min_val, fail_tag) in {
            '7dUSDProfit': (100, 'low_7d_profit'),
            '30dUSDProfit': (1000, 'low_30d_profit')
        }.items():
            if col in df.columns:
                profit_series = df[col].replace('?', None)
                profit_series = profit_series.str.replace('$', '').str.replace(',', '')
                profit_series = profit_series.apply(
                    lambda x: float(x.strip('()')) * -1 if isinstance(x, str) and '(' in x 
                    else float(x) if pd.notnull(x) else None
                )
                df.loc[profit_series.fillna(0) <= min_val, 'failed_criteria'] = \
                    df.loc[profit_series.fillna(0) <= min_val, 'failed_criteria'].apply(lambda x: x + [fail_tag])

        check_columns = ['sol_balance', 'buy_7d', 'winrate_7d', '7dUSDProfit', '30dUSDProfit']
        missing_count = df[check_columns].isna().sum(axis=1)
        df.loc[missing_count > 2, 'failed_criteria'] = \
            df.loc[missing_count > 2, 'failed_criteria'].apply(lambda x: x + ['too_many_missing_datapoints'])

        df['meets_base_criteria'] = df['failed_criteria'].apply(lambda x: len(x) == 0)
        return df

    def parse_basic_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.apply_base_criteria(df)
        df = self.calculate_standard_ratios(df)
        
        profit_series = pd.to_numeric(df['totalProfitPercent'].str.rstrip('%'), errors='coerce')
        
        seven_d_profit = df['7dUSDProfit'].replace('?', None).str.replace('$', '').str.replace(',', '')
        seven_d_profit = seven_d_profit.apply(
            lambda x: float(x.strip('()')) * -1 if isinstance(x, str) and '(' in x 
            else float(x) if pd.notnull(x) else None
        )
        
        thirty_d_profit = df['30dUSDProfit'].replace('?', None).str.replace('$', '').str.replace(',', '')
        thirty_d_profit = thirty_d_profit.apply(
            lambda x: float(x.strip('()')) * -1 if isinstance(x, str) and '(' in x 
            else float(x) if pd.notnull(x) else None
        )
        
        winrate = pd.to_numeric(df['winrate_7d'].replace('?', None).str.rstrip('%'), errors='coerce')
        
        mask = (
            (profit_series > 60) &
            (seven_d_profit > 30000) &
            (thirty_d_profit > 75000) &
            (winrate > 30) &
            (winrate < 97) &
            (df['sol_balance'] > 2) &
            (df['buy_7d'] > 20) &
            (df['buy_7d'] < 2000) &
            df['meets_base_criteria']
        )
        
        result = df[mask].copy()
        result['rank'] = range(1, len(result) + 1)
        
        base_cols = ['rank', 'Identifier', 'totalProfitPercent', '7dUSDProfit', '30dUSDProfit',
                     'winrate_7d', 'winrate_30d', 'tags', 'sol_balance', 'directLink', 'buy_7d',
                     '-50% +', '0% - -50%', '0 - 50%', '50% - 199%', '200% - 499%',
                     '500% - 600%', '600% +', 'ratio_200_plus', 'ratio_50_plus', 'failed_criteria']
        
        final_cols = [col for col in base_cols if col in result.columns]
        return result[final_cols]

    def parse_ratio_buys_to_percent_wins(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.apply_base_criteria(df)
        df = self.calculate_standard_ratios(df)
        
        mask = (
            (df[['50% - 199%', '200% - 499%', '500% - 600%', '600% +']].sum(axis=1) >= 10) &
            (df[['200% - 499%', '500% - 600%', '600% +']].sum(axis=1) >= 5) &
            df['meets_base_criteria']
        )
        
        result = df[mask].sort_values('ratio_200_plus', ascending=False)
        result['rank'] = range(1, len(result) + 1)
        
        base_cols = ['rank', 'Identifier', 'totalProfitPercent', '7dUSDProfit', '30dUSDProfit',
                     'winrate_7d', 'winrate_30d', 'tags', 'sol_balance', 'directLink', 'buy_7d',
                     '-50% +', '0% - -50%', '0 - 50%', '50% - 199%', '200% - 499%',
                     '500% - 600%', '600% +', 'ratio_200_plus', 'ratio_50_plus', 'failed_criteria']
        
        final_cols = [col for col in base_cols if col in result.columns]
        return result[final_cols]

    def parse_by_profit(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.apply_base_criteria(df)
        df = self.calculate_standard_ratios(df)
        
        profit_series = df['7dUSDProfit'].replace('?', None)
        profit_series = profit_series.str.replace('$', '').str.replace(',', '')
        profit_series = profit_series.apply(
            lambda x: float(x.strip('()')) * -1 if isinstance(x, str) and '(' in x 
            else float(x) if pd.notnull(x) else None
        )
        
        mask = df['meets_base_criteria']
        
        result = df[mask].copy()
        result['7dUSDProfit_numeric'] = profit_series
        result = result.sort_values('7dUSDProfit_numeric', ascending=False)
        
        result['rank'] = range(1, len(result) + 1)
        
        base_cols = ['rank', 'Identifier', 'totalProfitPercent', '7dUSDProfit', '30dUSDProfit',
                     'winrate_7d', 'winrate_30d', 'tags', 'sol_balance', 'directLink', 'buy_7d',
                     '-50% +', '0% - -50%', '0 - 50%', '50% - 199%', '200% - 499%',
                     '500% - 600%', '600% +', 'ratio_200_plus', 'ratio_50_plus', 'failed_criteria']
        
        final_cols = [col for col in base_cols if col in result.columns]
        return result[final_cols]

    def process_file(self, filename: str, method: str) -> bool:
        try:
            input_path = os.path.join(self.unparsed_dir, filename)
            df = pd.read_csv(input_path)
            
            if method == "csv-parsing-1":
                processed_df = self.parse_ratio_buys_to_percent_wins(df)
            elif method == "csv-parsing-2":
                processed_df = self.parse_by_profit(df)
            elif method == "csv-parsing-3":
                processed_df = self.parse_basic_filters(df)
            else:
                raise ValueError(f"Unknown parsing method: {method}")
            
            output_filename = f"parsed_{method}_{filename}"
            output_path = os.path.join(self.parsed_dir, output_filename)
            
            processed_df.to_csv(output_path, index=False)
            return True
            
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            return False

def main():
    parser = CSVParser()
    
    files = parser.get_available_files()
    if not files:
        print("No CSV files found in unparsed_csvs directory!")
        return

    print("\nSelect a CSV file to parse:")
    file_menu = TerminalMenu(files)
    file_index = file_menu.show()
    
    if file_index is None:
        print("No file selected!")
        return
        
    selected_file = files[file_index]
    
    methods = [
        "CSV Parsing Method 1: High Win Efficiency",
        "CSV Parsing Method 2: Highest 7-Day Profit",
        "CSV Parsing Method 3: Basic Filters"
    ]
    print("\nSelect parsing method:")
    method_menu = TerminalMenu(methods)
    method_index = method_menu.show()
    
    if method_index is None:
        print("No method selected!")
        return
        
    method_mapping = {
        0: "csv-parsing-1",
        1: "csv-parsing-2",
        2: "csv-parsing-3"
    }
    
    selected_method = method_mapping[method_index]
    
    success = parser.process_file(selected_file, selected_method)
    if success:
        print(f"\nSuccessfully processed {selected_file}")
        print(f"Output saved to parsed_csvs/parsed_{selected_method}_{selected_file}")
    else:
        print("\nProcessing failed!")

if __name__ == "__main__":
    main()