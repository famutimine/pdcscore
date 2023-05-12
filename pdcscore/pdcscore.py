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
        df=df[df[self.filldate_col]>=df[self.msr_start_dt_col]]

        # Get the fill_enddate
        df['fill_enddate'] = df[self.filldate_col] + pd.to_timedelta(df[self.supply_days_col], unit='D') - pd.Timedelta(days=1)

        # Group the dataframe and select the first value of each column within each group
        df_sorted=df.sort_values(by=[self.patient_id_col, self.drugname_col,self.filldate_col])
        first_fill_dates=df_sorted.groupby([self.patient_id_col, self.drugname_col])[self.filldate_col].first().reset_index().rename(columns={self.filldate_col:'first_filldate'})
        
        # derive the effective measurement start date
        base_data = first_fill_dates.merge(df[[self.patient_id_col, self.drugname_col,self.msr_start_dt_col,self.msr_end_dt_col]],on=[self.patient_id_col, self.drugname_col], how='left').drop_duplicates()
        base_data['effective_msr_start_dt'] = base_data.apply(lambda row: row['first_filldate']
                                                        if row['first_filldate'] > row[self.msr_start_dt_col]
                                                        else row[self.msr_start_dt_col], axis=1)

          
        # Calculate the totaldays: denominator for each patient-drugname pair
        base_data['totaldays']=(base_data[self.msr_end_dt_col] + pd.Timedelta(days=1) - base_data['effective_msr_start_dt']).dt.days
        base_data=base_data.drop_duplicates()

        # Merge dataframes to be able to calculate dayscovered
        dc=df.merge(base_data[[self.patient_id_col, self.drugname_col,self.msr_start_dt_col
                       ,self.msr_end_dt_col,'effective_msr_start_dt','totaldays']],
           on=[self.patient_id_col, self.drugname_col,self.msr_start_dt_col,self.msr_end_dt_col],how='inner').drop_duplicates()


        # Create a date array for each row that represents all valid rows based on logic below:
        # if the filldate_col is less than effective_msr_start_dt (effective start date of the measurement period) then use effective_msr_start_dt
        # else use filldate_col as the start of the range, the end date of the range is the fill_enddate less of msr_end_dt_col

        def generate_date_array(row):
            date_range = pd.date_range(start=row['effective_msr_start_dt'],end=row['fill_enddate'])
            return [str(date.date()) for date in date_range if date <= row[self.msr_end_dt_col]]

        dc['date_array'] = [generate_date_array(row) for _, row in dc.iterrows()]
        
        # Group by 'patient_id_col', 'drugname_col', 'totaldays' and aggregate by a concatenated list of unique dates (to avoid overlapping dates)
        # the array length represents the dayscovered
        pdc = dc.groupby([self.patient_id_col, self.drugname_col, 'totaldays'])['date_array'].apply(lambda x: len(np.unique(np.concatenate(x.tolist())))).reset_index(name='dayscovered')

        # Calculate pdc column as dayscovered divided by totaldays
        pdc['pdc_score'] = pdc['dayscovered'] / pdc['totaldays']

        pool.close()
        pool.join()

        return pdc

