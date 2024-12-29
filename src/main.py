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
        """Ensure required directories exist."""
        os.makedirs(self.unparsed_dir, exist_ok=True)
        os.makedirs(self.parsed_dir, exist_ok=True)
    
    def get_available_files(self) -> List[str]:
        """Get list of CSV files in unparsed directory."""
        return [f for f in os.listdir(self.unparsed_dir) if f.endswith('.csv')]

    def apply_base_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply base filtering criteria that applies to all parsing methods.
        Returns DataFrame with a single column listing any failed criteria.
        """
        # Initialize failed_criteria column with empty lists
        df['failed_criteria'] = df.apply(lambda x: [], axis=1)
        
        # Check SOL balance (nonzero)
        if 'sol_balance' in df.columns:
            df.loc[df['sol_balance'].fillna(0) <= 0, 'failed_criteria'] = \
                df.loc[df['sol_balance'].fillna(0) <= 0, 'failed_criteria'].apply(lambda x: x + ['zero_sol_balance'])
        
        # Check buy_7d (> 5)
        if 'buy_7d' in df.columns:
            df.loc[df['buy_7d'].fillna(0) <= 5, 'failed_criteria'] = \
                df.loc[df['buy_7d'].fillna(0) <= 5, 'failed_criteria'].apply(lambda x: x + ['low_buy_count'])
        
        # Check winrate_7d (> 30%)
        if 'winrate_7d' in df.columns:
            winrate_series = pd.to_numeric(
                df['winrate_7d'].replace('?', None).str.rstrip('%'), 
                errors='coerce'
            ) / 100
            df.loc[winrate_series.fillna(0) <= 0.30, 'failed_criteria'] = \
                df.loc[winrate_series.fillna(0) <= 0.30, 'failed_criteria'].apply(lambda x: x + ['low_winrate'])

        # Check USD profits
        for col, (min_val, fail_tag) in {
            '7dUSDProfit': (100, 'low_7d_profit'),
            '30dUSDProfit': (1000, 'low_30d_profit')
        }.items():
            if col in df.columns:
                # Convert profit values
                profit_series = df[col].replace('?', None)
                profit_series = profit_series.str.replace('$', '').str.replace(',', '')
                profit_series = profit_series.apply(
                    lambda x: float(x.strip('()')) * -1 if isinstance(x, str) and '(' in x 
                    else float(x) if pd.notnull(x) else None
                )
                
                df.loc[profit_series.fillna(0) <= min_val, 'failed_criteria'] = \
                    df.loc[profit_series.fillna(0) <= min_val, 'failed_criteria'].apply(lambda x: x + [fail_tag])

        # Check for too many missing data points
        check_columns = ['sol_balance', 'buy_7d', 'winrate_7d', '7dUSDProfit', '30dUSDProfit']
        missing_count = df[check_columns].isna().sum(axis=1)
        df.loc[missing_count > 2, 'failed_criteria'] = \
            df.loc[missing_count > 2, 'failed_criteria'].apply(lambda x: x + ['too_many_missing_datapoints'])

        # Create base criteria mask
        df['meets_base_criteria'] = df['failed_criteria'].apply(lambda x: len(x) == 0)

        return df

    def parse_ratio_buys_to_percent_wins(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and display two ratios:
        1. ratio_200_plus = (200-600+% wins) / buy_7d 
        2. ratio_50_plus = (50-600+% wins) / buy_7d
        Rank by ratio_200_plus
        """
        # First apply base criteria
        df = self.apply_base_criteria(df)
        
        # Define our column ranges
        wins_50_plus = ['50% - 199%', '200% - 499%', '500% - 600%', '600% +']
        wins_200_plus = ['200% - 499%', '500% - 600%', '600% +']
        
        # Calculate wins in each range
        df['ratio_50_plus'] = df[wins_50_plus].sum(axis=1) / df['buy_7d']
        df['ratio_200_plus'] = df[wins_200_plus].sum(axis=1) / df['buy_7d']
        
        # Keep the raw totals to check minimums only
        df['_total_50_plus'] = df[wins_50_plus].sum(axis=1)  
        df['_total_200_plus'] = df[wins_200_plus].sum(axis=1)
        
        # Apply minimum thresholds and base criteria
        mask = (
            (df['_total_50_plus'] >= 10) &  # Min 10 wins in 50%+ range
            (df['_total_200_plus'] >= 5) &   # Min 5 wins in 200%+ range
            df['meets_base_criteria']
        )
        
        # Sort by ratio_200_plus
        result = df[mask].sort_values('ratio_200_plus', ascending=False)
        
        # Add rank column
        result['rank'] = range(1, len(result) + 1)
        
        # Get base columns in order, excluding our ratio and temp columns
        base_cols = ['rank', 'Identifier', 'totalProfitPercent', '7dUSDProfit', '30dUSDProfit', 
                     'winrate_7d', 'winrate_30d', 'tags', 'sol_balance', 'directLink', 'buy_7d',
                     '-50% +', '0% - -50%', '0 - 50%', '50% - 199%', '200% - 499%', 
                     '500% - 600%', '600% +']
        
        # Add ratios after percentage columns
        final_cols = base_cols + ['ratio_200_plus', 'ratio_50_plus', 'failed_criteria']
        
        # Filter to only columns that exist
        final_cols = [col for col in final_cols if col in result.columns]
        
        result = result[final_cols]
        
        return result

    def process_file(self, filename: str, method: str) -> bool:
        """Process a single file with specified method."""
        try:
            input_path = os.path.join(self.unparsed_dir, filename)
            df = pd.read_csv(input_path)
            
            if method == "csv-parsing-1":  # Updated method name
                processed_df = self.parse_ratio_buys_to_percent_wins(df)
            else:
                raise ValueError(f"Unknown parsing method: {method}")
            
            # Generate output filename
            output_filename = f"parsed_{method}_{filename}"
            output_path = os.path.join(self.parsed_dir, output_filename)
            
            # Save processed file
            processed_df.to_csv(output_path, index=False)
            return True
            
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            return False

def main():
    parser = CSVParser()
    
    # Get available files
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
    
    # Method selection
    methods = ["CSV Parsing Method 1: High Win Efficiency"]  # Updated method name 
    print("\nSelect parsing method:")
    method_menu = TerminalMenu(methods)
    method_index = method_menu.show()
    
    if method_index is None:
        print("No method selected!")
        return
        
    # Map the method selection to our internal method name
    method_mapping = {
        0: "csv-parsing-1"  # Updated method name
    }
    
    selected_method = method_mapping[method_index]
    
    # Process the file
    success = parser.process_file(selected_file, selected_method)
    if success:
        print(f"\nSuccessfully processed {selected_file}")
        print(f"Output saved to parsed_csvs/parsed_{selected_method}_{selected_file}")
    else:
        print("\nProcessing failed!")

if __name__ == "__main__":
    main()