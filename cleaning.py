import pandas as pd


def remove_outliers(df):
    new_data = pd.DataFrame(df.keys())
    df = df.groupby('event_id')
    for group in df:
        Q1 = group[1]['prediction_min'].quantile(0.25)
        Q3 = group[1]['prediction_min'].quantile(0.75)
        IQR = Q3 - Q1
        filtered = group[1].query('(@Q1 - 1.5 * @IQR) <= prediction_min <= (@Q3 + 1.5 * @IQR)')

        Q1 = filtered['prediction_max'].quantile(0.25)
        Q3 = filtered['prediction_max'].quantile(0.75)
        IQR = Q3 - Q1
        filtered = filtered.query('(@Q1 - 1.5 * @IQR) <= prediction_max <= (@Q3 + 1.5 * @IQR)')
        new_data = new_data.append(filtered, ignore_index=True)

    return new_data
