#!/usr/bin/env python3

import argparse
import pandas as pd

from pathlib import Path

pd.set_option('mode.chained_assignment', None)

def process_benchmark_column(df: pd.DataFrame):
    '''Remove all benchmarks but the following and rename them'''

    benchmarks_map = {
        'Classification_ECC_OFF': 'Classification',
        'Detection_ECC_OFF': 'Detection',
        'Coral-Conv2d_ECC_OFF': 'Conv',
    }

    initial_nrows = len(df)

    df = df.loc[df['benchmark'].isin(benchmarks_map.keys())]
    df.loc[df['header'].str.contains('depthwise'), 'benchmark'] = 'DepthwiseConv'
    df['benchmark'].replace(benchmarks_map, inplace=True)

    print(f"process_benchmark_column: {initial_nrows - len(df)} rows dropped")

    return df

def process_header_column(df: pd.DataFrame):
    '''Turn the `header` column into `model`'''

    header_prefix = ' model_file: '
    initial_nrows = len(df)

    # Remove runs with invalid headers
    df = df.loc[df['header'].str.startswith(header_prefix)]

    # Extract model name from header
    df['header'] = df['header'].apply(lambda header: Path(header.split(',')[0][len(header_prefix):]).name)

    # Rename column
    df.rename(columns={ 'header': 'model' }, inplace=True)

    print(f"process_header_column: {initial_nrows - len(df)} rows dropped")

    return df

def process_sdc_column(df: pd.DataFrame):
    '''Remove runs with no SDCs'''
    initial_nrows = len(df)

    df = df.loc[df['#SDC'] > 0]

    print(f"process_sdc_column: {initial_nrows - len(df)} rows dropped")

    return df

def calc_aggregate_cross_sections(df: pd.DataFrame):
    benchmarks_models_summary_df = df.groupby(['benchmark', 'model']).agg({
        '#SDC': 'sum',
        '#DUE': 'sum',
        'Fluency(Flux * $AccTime)': 'sum',
    })
    benchmarks_models_summary_df['Cross Section SDC'] = benchmarks_models_summary_df['#SDC'] / benchmarks_models_summary_df['Fluency(Flux * $AccTime)']
    benchmarks_models_summary_df['Cross Section DUE'] = benchmarks_models_summary_df['#DUE'] / benchmarks_models_summary_df['Fluency(Flux * $AccTime)']
    print("BENCHMARKS/MODELS SUMMARY")
    print(benchmarks_models_summary_df)

    print("")

    benchmarks_summary_df = benchmarks_models_summary_df.groupby('benchmark').agg({
        '#SDC': 'sum',
        '#DUE': 'sum',
        'Fluency(Flux * $AccTime)': 'sum'
    })
    benchmarks_summary_df['Cross Section SDC'] = benchmarks_summary_df['#SDC'] / benchmarks_summary_df['Fluency(Flux * $AccTime)']
    benchmarks_summary_df['Cross Section DUE'] = benchmarks_summary_df['#DUE'] / benchmarks_summary_df['Fluency(Flux * $AccTime)']
    print("BENCHMARKS SUMMARY")
    print(benchmarks_summary_df)

    return benchmarks_models_summary_df, benchmarks_summary_df

def process_dataframe(df):
    df = process_benchmark_column(df)
    df = process_header_column(df)
    df = process_sdc_column(df)

    df.sort_values(['benchmark', 'model'], inplace=True)

    benchmarks_models_summary_df, benchmarks_summary_df = calc_aggregate_cross_sections(df)

    return df, benchmarks_models_summary_df, benchmarks_summary_df

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('cross_sections_file', help='Path to cross sections CSV file')
    # parser.add_argument('-t', '--threshold', type=float, default=0.5, help='Threshold for detection/classification score')
    args = parser.parse_args()

    input_file = args.cross_sections_file

    cross_sections_df = pd.read_csv(input_file, index_col=0)

    post_process_cross_sections_df, benchmarks_models_summary_df, benchmarks_summary_df = process_dataframe(cross_sections_df)

    post_process_cross_sections_output_file = f"{Path(input_file).with_suffix('').absolute()}-post_process.csv"
    post_process_cross_sections_df.to_csv(post_process_cross_sections_output_file)
    print(f"Cross Sections for all runs saved to: {post_process_cross_sections_output_file}")

    benchmarks_models_summary_output_file = f"{Path(input_file).with_suffix('').absolute()}-benchs_models_summary.csv"
    benchmarks_models_summary_df.to_csv(benchmarks_models_summary_output_file)
    print(f"Cross Sections for every pair (benchmark, model) saved to: {benchmarks_models_summary_output_file}")

    benchmarks_summary_output_file = f"{Path(input_file).with_suffix('').absolute()}-benchs_summary.csv"
    benchmarks_summary_df.to_csv(benchmarks_summary_output_file)
    print(f"Cross Sections for every benchmark saved to: {benchmarks_summary_output_file}")

if __name__ == "__main__":
    main()