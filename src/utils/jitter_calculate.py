import glob
import os
import pandas as pd


def calc_simple_jitter(folder_path='data/jitter'):

    csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))

    results = []
    all_jitter = []                                                                     

    for file in csv_files:

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

        return


    df_res = pd.DataFrame(results)


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


    final_df.to_csv('results/tables/jitter_overall_results.csv', index=False)


if __name__ == '__main__':
    calc_simple_jitter()