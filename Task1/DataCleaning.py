import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


file_path = '../Data/online_retail_II.xlsx'


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


class DataCleaner:
    output_path = '../Data/ProcessedData/cleaned_online_retail_data.csv'
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.df_cleaned = None



    def load_data(self):

        try:
            if self.file_path.endswith('.xlsx'):
                self.df = pd.read_excel(self.file_path)
            elif self.file_path.endswith('.csv'):
                self.df = pd.read_csv(self.file_path, encoding='latin1')

            print(f"Data loaded successfully with shape: {self.df.shape}")
            print(f"Columns: {self.df.columns.tolist()}")
            return self.df

        except Exception as e:
            print(f"Error loading data: {e}")
            return None


    def explore_data(self):
        if self.df is None:
            print("Data not loaded")
            return

        print("\n" + "=" * 70)
        print ("Initial Data Exploration")
        print("=" * 70)

        #Basic Info
        print("\n1. Dataset Info:")
        print(f"   Rows: {self.df.shape[0]:,}")

        print(f"   Columns: {self.df.shape[1]}")

        #First few rows

        print("\n2. First 5 Rows:")
        print(self.df.head())


        print("\n3. Dataset Types:")
        print(self.df.dtypes)

        #Missing values
        print("\n4. Missing Values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing Values'] > 0])

        #Basic Statistics
        print("\n5. Numerical Columns Statistics:")
        print(self.df.describe())

        #Unique Values
        print("\n6. Unique Value Per Column:")
        for col in self.df.columns:
            print(f"   - {col}: {self.df[col].nunique()} unique values")

    def remove_null_values(self):
        if self.df is None:
            print("Data not loaded")
            return

        initial_rows = len(self.df)
        #remove null CustomerID
        self.df = self.df.dropna(subset=['Customer ID'])

        #remove null descriptions
        self.df = self.df.dropna(subset=['Description'])

        rows_removed = initial_rows - len(self.df)
        print(f"Removed {rows_removed} rows with null Customer ID or Description")
        print(f"Remaining rows: {len(self.df)}")

        return self.df

    def remove_invalid_values(self):
        if self.df is None:
            print("Data not loaded")
            return
        initial_rows =  len(self.df)
        #remove negative or zero quantities
        self.df = self.df[self.df['Quantity'] > 0]
        #remove negative or zero prices
        self.df = self.df[self.df['Price'] > 0]

        #remove cancelled orders (Invoice starts with C)
        self.df = self.df[~self.df['Invoice'].astype(str).str.startswith('C')]

        rows_removed = initial_rows - len(self.df)
        print(f"\n Invalid values removed:")
        print(f"  Rows removed: {rows_removed:,}")
        print(f"  Remaining rows: {len(self.df):,}")

        return self.df

    def handle_duplicates(self):
        if self.df is None:
            print("Data not loaded")
            return

        initial_rows = len(self.df)

        #Check for exact duplicates across all columns
        exact_duplicates = self.df.drop_duplicates()

        rows_removed = initial_rows - len(self.df)
        print(f"\n Exact Duplicates removed:")
        print(f"  Rows removed: {rows_removed:,}")
        print(f"  Remaining rows: {len(self.df):,}")

        # Show example of same product with different prices which are kept
        print("\n  Example : Same products with different proces :")
        sample_product = self.df.groupby('StockCode')['Price'].nunique()
        multi_price_products = sample_product[sample_product > 1].head(5)

        for stock_code in multi_price_products.index:
            product_data =  self.df[self.df['StockCode'] == stock_code][['Description', 'Price']].drop_duplicates()
            print(f"  Product {stock_code}:")
            print(product_data.head())

        return self.df

    def create_revenue_column(self):
        if self.df is None:
            print("Data not loaded")
            return

        self.df['Revenue'] = self.df['Quantity'] * self.df['Price']
        print("\n'Revenue' column created.")
        print(f" Total Revenue: {self.df['Revenue'].sum():,.2f}")
        return self.df

    def clean_dataset(self):
        print("\n" + "=" * 70)
        print("\nStarting data cleaning process...")
        print("=" * 70)
        print("\nStep 1: Removing null values...")
        self.remove_null_values()

        print("\nStep 2: Removing invalid values...")
        self.remove_invalid_values()

        print("\nStep 3: Handling duplicates...")
        self.handle_duplicates()

        print("\nStep 4: Creating 'Revenue' column...")
        self.create_revenue_column()

        self.df_cleaned = self.df.copy()



        print("\n" + "=" * 70)
        print("\nData Cleaning Completed!")
        print("\n" + "=" * 70)
        print(f"Final dataset shape: {self.df_cleaned.shape}")

        return self.df_cleaned

    def perform_eda(self):
        if self.df is None:
            print("Data not loaded")
            return

        # Debug: Check DataFrame structure
        print("\nDataFrame Info:")
        print(self.df.info())
        print("\nDataFrame Columns:")
        print(self.df.columns.tolist())
        print("\nFirst few rows:")
        print(self.df.head())

        print("\n" + "=" * 70)
        print ("Exploratory Data Analysis (EDA)")
        print("=" * 70)

        # 1. Average price and quantity per product
        product_stats = self.df_cleaned.groupby(['StockCode', 'Description']).agg({
            'Price':['mean', 'min' , 'max', 'std', 'count'],
            'Quantity' : ['mean', 'sum'],
            'Revenue': 'sum'
        }).round(2)

        product_stats.columns = ['Avg_Price', 'Min_Price', 'Max_Price', 'Std_PriceDev',
                                 'Num_Transactions', 'Avg_Quantity', 'Total_Quantity',
                                 'Total_Revenue']
        print("\nTop 10 Products by Total Revenue:")
        print(product_stats.head(10))

        #2. Products with multiple prices (potential sellers)
        product_stats['Price_Variation'] = product_stats['Max_Price'] - product_stats['Min_Price']
        multi_price = product_stats[product_stats['Price_Variation'] > 0].sort_values('Num_Transactions', ascending=False)

        print(f"\n Products with Multiple Prices: {len(multi_price)} found")
        print(multi_price.head(10))

        #3. Customer Statistics
        customer_stats = self.df_cleaned.groupby('Customer ID').agg({
            'Invoice': 'nunique',
            'Revenue': 'sum',
            'Quantity': 'sum'
        }).round(2)

        customer_stats.columns = ['Num_Orders', 'Total_Spent', 'Total_Quantity']

        print(f"\n Total unique customers: {len(customer_stats)}")
        print(f"   Average orders per customer: {customer_stats['Num_Orders'].mean():.2f}")
        print(f"   Average spent per customer: {customer_stats['Total_Spent'].mean():.2f}")
        print(f"   Average quantity per customer: {customer_stats['Total_Quantity'].mean():.2f}")


        # Save product stats for next tasks
        self.product_stats = product_stats

        return product_stats

    def visualize_data(self):
        if self.df_cleaned is None or self.product_stats  is None:
            print("Data not cleaned or EDA not performed")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        #1. Price Distribution
        axes[0, 0].hist(self.df_cleaned['Price'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Price Distribution')
        axes[0, 0].set_xlabel('Unit Price')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_xlim(0, 50)

        #2. Top products by revenue
        top_products = self.product_stats.head(15)
        axes[0, 1].barh(range(len(top_products)), top_products['Total_Revenue'])
        axes[0, 1].set_yticks(range(len(top_products)))
        axes[0, 1].set_yticklabels([desc[:30] for desc in top_products.index.get_level_values('Description')])
        axes[0, 1].set_xlabel('Top 15 Products by Revenue', fontsize=14, fontweight='bold')
        axes[0, 1].set_title('Total Revenue')

        #3. Price Variation across products
        price_var = self.product_stats[self.product_stats['Price_Variation'] > 0 ].head(20)
        axes[1, 0].scatter(price_var['Avg_Price'], price_var['Price_Variation'], s=price_var['Num_Transactions']*2, alpha=0.6)
        axes[1, 0].set_title('Price Variation vs Average Price', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Average Price')
        axes[1, 0].set_ylabel('Price Variation')
        axes[1, 0].grid(True, alpha=0.3)


        #4. Monthly Revenue Trend
        self.df_cleaned['InvoiceDate'] = pd.to_datetime(self.df_cleaned['InvoiceDate'])
        monthly_revenue = self.df_cleaned.groupby(self.df_cleaned['InvoiceDate'].dt.to_period('M'))['Revenue'].sum()
        axes[1, 1].plot(monthly_revenue.index.astype(str), monthly_revenue.values, marker ='o', linewidth=2)
        axes[1, 1].set_title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Total Revenue')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)


        plt.tight_layout()
        plt.savefig('Task1_EDA_Results.png', dpi=300, bbox_inches='tight')
        print("\n Visualizations saved as 'Task1_EDA_Results.png'")
        plt.show()

    def save_cleaned_data(self, output_path):
        if self.df_cleaned is None:
            print("Data not cleaned")
            return

        try:
            self.df_cleaned.to_csv(output_path, index=False)
            print(f"Cleaned data saved to {output_path}")
        except Exception as e:
            print(f"Error saving cleaned data: {e}")


    def generate_report(df_original, df_cleaned, output_path='cleaning_report.txt'):
        """
        Generate a comprehensive data cleaning report.

        Args:
            df_original: Original DataFrame before cleaning
            df_cleaned: Cleaned DataFrame after processing
            output_path: Path to save the report
        """
        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DATA CLEANING REPORT\n")
            f.write("=" * 60 + "\n\n")

            # Original dataset statistics
            f.write("1. ORIGINAL DATASET\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total rows: {len(df_original)}\n")
            f.write(f"Total columns: {len(df_original.columns)}\n")
            f.write(f"Memory usage: {df_original.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB\n\n")

            # Missing values
            f.write("2. MISSING VALUES\n")
            f.write("-" * 60 + "\n")
            missing_original = df_original.isnull().sum()
            missing_original = missing_original[missing_original > 0]
            if not missing_original.empty:
                for col, count in missing_original.items():
                    pct = (count / len(df_original)) * 100
                    f.write(f"{col}: {count} ({pct:.2f}%)\n")
            else:
                f.write("No missing values found.\n")
            f.write("\n")

            # Duplicates
            f.write("3. DUPLICATE ROWS\n")
            f.write("-" * 60 + "\n")
            duplicates = df_original.duplicated().sum()
            f.write(f"Duplicate rows found: {duplicates}\n")
            f.write(f"Percentage: {(duplicates / len(df_original)) * 100:.2f}%\n\n")

            # Cleaned dataset statistics
            f.write("4. CLEANED DATASET\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total rows: {len(df_cleaned)}\n")
            f.write(f"Rows removed: {len(df_original) - len(df_cleaned)}\n")
            f.write(f"Removal percentage: {((len(df_original) - len(df_cleaned)) / len(df_original)) * 100:.2f}%\n\n")

            # Data types
            f.write("5. DATA TYPES\n")
            f.write("-" * 60 + "\n")
            for col in df_cleaned.columns:
                f.write(f"{col}: {df_cleaned[col].dtype}\n")
            f.write("\n")

            # Summary statistics
            f.write("6. SUMMARY STATISTICS\n")
            f.write("-" * 60 + "\n")
            f.write(df_cleaned.describe().to_string())
            f.write("\n\n")

            f.write("=" * 60 + "\n")
            f.write("Report generated successfully.\n")
            f.write("=" * 60 + "\n")

        print(f"Report saved to {output_path}")


def handle_outliers(self, df, columns=['Quantity', 'UnitPrice'], method='capping'):
    """
    Detect and handle outliers using IQR method with capping/winsorizing.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Columns to check for outliers
    method : str
        Method to handle outliers ('capping' or 'removal')

    Returns:
    --------
    pd.DataFrame
        DataFrame with outliers handled
    dict
        Dictionary containing outlier statistics
    """
    df_copy = df.copy()
    outlier_stats = {}

    for col in columns:
        if col not in df.columns:
            continue

        # Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier bounds using IQR method
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers
        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        n_outliers = outliers_mask.sum()
        outlier_percentage = (n_outliers / len(df)) * 100

        # Calculate capping values (1st and 99th percentiles)
        lower_cap = df[col].quantile(0.01)
        upper_cap = df[col].quantile(0.99)

        # Store original min/max
        original_min = df[col].min()
        original_max = df[col].max()

        # Apply capping using clip method
        df_copy[col] = df_copy[col].clip(lower=lower_cap, upper=upper_cap)

        # Store new min/max
        new_min = df_copy[col].min()
        new_max = df_copy[col].max()

        # Store statistics
        outlier_stats[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'n_outliers': n_outliers,
            'outlier_percentage': outlier_percentage,
            'lower_cap': lower_cap,
            'upper_cap': upper_cap,
            'original_min': original_min,
            'original_max': original_max,
            'new_min': new_min,
            'new_max': new_max
        }

    return df_copy, outlier_stats


def visualize_outliers(self, df_before, df_after, columns=['Quantity', 'UnitPrice'],
                       save_path='../Data/ProcessedData/outlier_handling_results.png'):
    """
    Create before/after boxplot visualizations for outlier handling.

    Parameters:
    -----------
    df_before : pd.DataFrame
        DataFrame before outlier handling
    df_after : pd.DataFrame
        DataFrame after outlier handling
    columns : list
        Columns to visualize
    save_path : str
        Path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Outlier Handling Results: Before vs After Capping', fontsize=16, fontweight='bold')

    for idx, col in enumerate(columns):
        if col not in df_before.columns:
            continue

        # Before capping (left column)
        axes[idx, 0].boxplot(df_before[col].dropna(), vert=True)
        axes[idx, 0].set_title(f'{col} - Before Capping', fontsize=12, fontweight='bold')
        axes[idx, 0].set_ylabel(col)
        axes[idx, 0].grid(True, alpha=0.3)

        # After capping (right column)
        axes[idx, 1].boxplot(df_after[col].dropna(), vert=True)
        axes[idx, 1].set_title(f'{col} - After Capping', fontsize=12, fontweight='bold')
        axes[idx, 1].set_ylabel(col)
        axes[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Outlier visualization saved to: {save_path}")


def generate_outlier_report(self, outlier_stats, output_path='../Data/ProcessedData/outlier_report.txt'):
    """
    Generate a detailed report on outlier detection and handling.

    Parameters:
    -----------
    outlier_stats : dict
        Dictionary containing outlier statistics
    output_path : str
        Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("OUTLIER DETECTION AND HANDLING REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Method: IQR (Interquartile Range) with Capping (Winsorizing)\n")
        f.write(f"Capping Strategy: 1st percentile (lower) and 99th percentile (upper)\n\n")

        for col, stats in outlier_stats.items():
            f.write("-" * 80 + "\n")
            f.write(f"Column: {col}\n")
            f.write("-" * 80 + "\n\n")

            f.write("IQR Method Statistics:\n")
            f.write(f"  Q1 (25th percentile): {stats['Q1']:.4f}\n")
            f.write(f"  Q3 (75th percentile): {stats['Q3']:.4f}\n")
            f.write(f"  IQR (Q3 - Q1): {stats['IQR']:.4f}\n")
            f.write(f"  Lower Bound (Q1 - 1.5*IQR): {stats['lower_bound']:.4f}\n")
            f.write(f"  Upper Bound (Q3 + 1.5*IQR): {stats['upper_bound']:.4f}\n\n")

            f.write("Outlier Detection Results:\n")
            f.write(f"  Number of Outliers: {stats['n_outliers']}\n")
            f.write(f"  Percentage of Outliers: {stats['outlier_percentage']:.2f}%\n\n")

            f.write("Capping Values Applied:\n")
            f.write(f"  Lower Cap (1st percentile): {stats['lower_cap']:.4f}\n")
            f.write(f"  Upper Cap (99th percentile): {stats['upper_cap']:.4f}\n\n")

            f.write("Before vs After Capping:\n")
            f.write(f"  Original Min: {stats['original_min']:.4f} → New Min: {stats['new_min']:.4f}\n")
            f.write(f"  Original Max: {stats['original_max']:.4f} → New Max: {stats['new_max']:.4f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"Outlier report saved to: {output_path}")



if __name__ == "__main__":
    cleaner = DataCleaner(file_path)
    df = cleaner.load_data()

    cleaner.explore_data()
    df_no_nulls = cleaner.remove_null_values()
    df_cleaned = cleaner.remove_invalid_values()

    df_no_duplicates = cleaner.handle_duplicates()
    df_with_revenue = cleaner.create_revenue_column()
    df_final = cleaner.clean_dataset()
    product_stats = cleaner.perform_eda()
    cleaner.visualize_data()
    cleaner.save_cleaned_data(output_path=DataCleaner.output_path)
    DataCleaner.generate_report(df, df_final, output_path='../Data/ProcessedData/data_cleaning_report.txt')
    # Outlier handling
    df_outliers_handled, outlier_stats = handle_outliers(cleaner, df_final
, columns=['Quantity', 'Price'], method='capping')
    visualize_outliers(cleaner, df_final, df_outliers_handled,
                          columns=['Quantity', 'Price'],
                          save_path='../Data/ProcessedData/outlier_handling_results.png')
    generate_outlier_report(cleaner, outlier_stats,
                            output_path='../Data/ProcessedData/outlier_report.txt')



