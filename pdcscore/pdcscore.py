import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

class pdcCalc:
    def __init__(self, dataframe, patient_id_col, drugname_col, filldate_col,
                 supply_days_col, msr_start_dt_col, msr_end_dt_col):
        self.dataframe = dataframe
        self.patient_id_col = patient_id_col
        self.drugname_col = drugname_col
        self.filldate_col = filldate_col
        self.supply_days_col = supply_days_col
        self.msr_start_dt_col = msr_start_dt_col
        self.msr_end_dt_col = msr_end_dt_col

    def calculate_pdc(self, n_workers=cpu_count()):
        pool = Pool(processes=n_workers)
        df = self.dataframe.copy()

        # Calculate the date difference
        df['date_diff'] = df[self.msr_start_dt_col] - df[self.filldate_col] + pd.Timedelta(days=1)

        # Filter rows where date difference is less than or equal to 180 days
        df = df[df['date_diff'] < pd.Timedelta(days=180)]

        # Remove the date_diff column
        df = df.drop('date_diff', axis=1)

        # Get the fill_enddate
        df['fill_enddate'] = df[self.filldate_col] + pd.to_timedelta(df[self.supply_days_col], unit='D')

        # Group the dataframe and select the first value of each column within each group
        new_df = df.groupby([self.patient_id_col, self.drugname_col]).first().reset_index()

        new_df['effective_msr_start_dt'] = new_df.apply(lambda row: row[self.filldate_col]
                                                        if row[self.filldate_col] > row[self.msr_start_dt_col]
                                                        else row[self.msr_start_dt_col], axis=1)

        # Calculate the totaldays: denominator for each patient-drugname pair
        new_df['totaldays'] = (new_df[self.msr_end_dt_col] + pd.Timedelta(days=1) - new_df[
            'effective_msr_start_dt']).dt.days

        df = df.merge(new_df[[self.patient_id_col, self.drugname_col, 'effective_msr_start_dt', 'totaldays']],
                      on=[self.patient_id_col, self.drugname_col], how='left')

        # Create a date array for each row that represents all valid rows based on logic below:
        # if the filldate_col is less than effective_msr_start_dt (effective start date of the measurement period) then use effective_msr_start_dt
        # else use filldate_col as the start of the range, the end date of the range is the fill_enddate less of msr_end_dt_col

        def generate_date_array(row):
            date_range = pd.date_range(start=row['effective_msr_start_dt']
                                       if row[self.filldate_col] < row['effective_msr_start_dt']
                                       else row[self.filldate_col],
                                       end=row['fill_enddate'])
            return [str(date.date()) for date in date_range if date <= row[self.msr_end_dt_col]]

        df['date_array'] = [generate_date_array(row) for _, row in df.iterrows()]
        
        # Group by 'patient_id_col', 'drugname_col', 'totaldays' and aggregate by a concatenated list of unique dates (to avoid overlapping dates)
        # the array length represents the dayscovered
        pdc = df.groupby([self.patient_id_col, self.drugname_col, 'totaldays'])['date_array'].apply(
            lambda x: len(np.unique(np.concatenate(x.tolist())))).reset_index(name='dayscovered')

        # Calculate pdc column as dayscovered divided by totaldays
        pdc['pdc_score'] = pdc['dayscovered'] / pdc['totaldays']

        return pdc

