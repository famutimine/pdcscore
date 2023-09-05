import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from datetime import timedelta

class pdcCalc:
    def __init__(self, dataframe, patient_id_col, drugname_col, filldate_col,
                 supply_days_col, msr_start_dt_col, msr_end_dt_col, overstock):
        self.dataframe = dataframe
        self.patient_id_col = patient_id_col
        self.drugname_col = drugname_col
        self.filldate_col = filldate_col
        self.supply_days_col = supply_days_col
        self.msr_start_dt_col = msr_start_dt_col
        self.msr_end_dt_col = msr_end_dt_col
        self.overstock=overstock

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
        
        if self.overstock is False:            

            # Merge dataframes to be able to calculate dayscovered
            dc=df.merge(base_data[[self.patient_id_col, self.drugname_col,self.msr_start_dt_col
                           ,self.msr_end_dt_col,'effective_msr_start_dt','totaldays']],
               on=[self.patient_id_col, self.drugname_col,self.msr_start_dt_col,self.msr_end_dt_col],how='inner').drop_duplicates()


            # Create a date array for each row that represents all valid rows based on logic below:
            # if the filldate_col is less than effective_msr_start_dt (effective start date of the measurement period) then use effective_msr_start_dt
            # else use filldate_col as the start of the range, the end date of the range is the fill_enddate less of msr_end_dt_col

            def generate_date_array(row):
                date_range = pd.date_range(start=row[self.filldate_col], end=row['fill_enddate'])
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
        else:
            # Sort the DataFrame by 'patient_id', 'drugname', and 'filldate'
            df.sort_values(by=[self.patient_id_col, self.drugname_col, self.filldate_col], inplace=True)
                # Create 'prior_fill_enddate' column
            df['prior_fill_enddate'] = df.groupby([self.patient_id_col, self.drugname_col])['fill_enddate'].shift(1)

            # Create 'fill_start_date' column
            df['fill_start_date'] = df.apply(lambda row: row[self.filldate_col] if pd.isnull(row['prior_fill_enddate']) else (row['prior_fill_enddate'] + timedelta(days=1)), axis=1)

            # Create 'fill_stop_date' column
            df['fill_stop_date'] = df.apply(lambda row: row['fill_enddate'] if pd.isnull(row['prior_fill_enddate']) else min(row['fill_start_date'] + timedelta(days=row[self.supply_days_col]) - timedelta(days=1), row[self.msr_end_dt_col]), axis=1)

            # Create a new column 'prior_fill_stop_date' by shifting 'fill_stop_date' within each group
            df['prior_fill_stop_date'] = df.groupby([self.patient_id_col, self.drugname_col])['fill_stop_date'].shift(1)

            # Modify 'fill_start_date' column
            df['fill_start_date'] = df.groupby([self.patient_id_col, self.drugname_col])['fill_stop_date'].shift(1)
            df['fill_start_date'] += pd.Timedelta(days=1)

            # Modify 'fill_stop_date' column
            df['fill_stop_date'] = df.apply(lambda row: min(row['fill_start_date'] + timedelta(days=row[self.supply_days_col] - 1), row[self.msr_end_dt_col]), axis=1)

            df = df[df['fill_stop_date'] != df['prior_fill_stop_date']]

            # Calculate 'dayscovered'
            df['dayscovered'] = df.apply(lambda row: row[self.supply_days_col] if pd.isnull(row['fill_start_date']) else (row['fill_stop_date'] - max(row['fill_start_date'],row[self.filldate_col])).days + 1, axis=1)
            df = df[[self.patient_id_col, self.drugname_col, 'dayscovered']]
            result_df = df.groupby([self.patient_id_col, self.drugname_col]).sum('dayscovered').reset_index()          
            
       
            result_df.columns = [self.patient_id_col, self.drugname_col, 'dayscovered']
            pdc=result_df.merge(base_data[[self.patient_id_col, self.drugname_col,'totaldays']],
               on=[self.patient_id_col, self.drugname_col],how='inner').drop_duplicates()
            
            pdc.columns = [self.patient_id_col, self.drugname_col, 'dayscovered', 'totaldays']
            pdc['pdc_score'] = pdc['dayscovered'] / pdc['totaldays']
            pdc=pdc[[self.patient_id_col, self.drugname_col, 'totaldays','dayscovered','pdc_score']]
            pool.close()
            pool.join()

            return pdc
