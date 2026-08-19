import glob
import os
import pandas as pd


def calc_simple_jitter(folder_path='data/jitter'):
    # Get all CSV files in the folder
    csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))

    results = []
    all_jitter = []  # collect raw values across every file for a correct pooled summary

    for file in csv_files:
        # Skip result files if they already exist
        if 'wyniki' in file:
            continue

        df = pd.read_csv(file)

        if 'jitter_ms' in df.columns:
            j = df['jitter_ms'].dropna()
            all_jitter.append(j)

            results.append({
                'File': os.path.basename(file),
                'N': len(j),
                'Mean [ms]': j.mean(),
                'SD [ms]': j.std(),
                'Min [ms]': j.min(),
                'Max [ms]': j.max(),
            })

    if not results:
        print('No CSV files with column jitter_ms found.')
        return

    # Create summary table
    df_res = pd.DataFrame(results)

    # Pool the RAW values across all files/blocks/participants and compute the
    # summary statistics directly on the pooled sample. This is the statistically
    # correct way to get an "overall" mean/SD/min/max: averaging the per-file SDs
    # (as the previous version did) ignores between-file variance in the means
    # and systematically underestimates the true pooled SD.
    pooled = pd.concat(all_jitter, ignore_index=True)

    grand_avg = pd.DataFrame([{
        'File': 'Overall',
        'N': len(pooled),
        'Mean [ms]': pooled.mean(),
        'SD [ms]': pooled.std(),
        'Min [ms]': pooled.min(),
        'Max [ms]': pooled.max(),
    }])

    final_df = pd.concat([df_res, grand_avg], ignore_index=True)

    # Display in console formatted to 3 decimal places
    print(final_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    # Save to clean CSV file
    final_df.to_csv('results/tables/jitter_overall_results.csv', index=False)


if __name__ == '__main__':
    calc_simple_jitter()