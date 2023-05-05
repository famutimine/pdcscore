import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

class pdccalc:
    def __init__(self, dataframe, patient_id_col, drugname_col, filldate_col,
                          supply_days_col, mbr_elig_start_dt_col,
                          mbr_elig_end_dt_col):
        self.dataframe = dataframe
        self.patient_id_col = patient_id_col
        self.drugname_col = drugname_col
        self.filldate_col = filldate_col
        self.supply_days_col = supply_days_col
        self.mbr_elig_start_dt_col = mbr_elig_start_dt_col
        self.mbr_elig_end_dt_col = mbr_elig_end_dt_col
        self.patient_ids = set(dataframe[patient_id_col])
        self.drugs = set(dataframe[drugname_col])
        self.pdc_scores_df = pd.DataFrame(columns=[patient_id_col, drugname_col, 'DAYSCOVERED', 'TOTALDAYS', 'PDC_SCORE'])
        
    def calculate_pdc_for_patient_drug(self, patient_id, drug):
        subset = self.dataframe[(self.dataframe[self.patient_id_col] == patient_id) & (self.dataframe[self.drugname_col] == drug)]
        if subset.empty:
            return None
        
        # determine if the first fill date precedes mbr_elig_start_dt_col by <=6 months
        first_fill_date = subset[self.filldate_col].min()
        mbr_elig_start_dt = subset[self.mbr_elig_start_dt_col].iloc[0]
        if (first_fill_date - mbr_elig_start_dt) <= pd.Timedelta(days=6*30):
            start_date = mbr_elig_start_dt
        else:
            start_date = max(first_fill_date, mbr_elig_start_dt)
        
        dates = []
        for i, row in subset.iterrows():
            end_date = min(row[self.filldate_col] + pd.Timedelta(days=row[self.supply_days_col]), row[self.mbr_elig_end_dt_col])
            if end_date >= start_date:
                dates.append((start_date, end_date))
        dates.sort()
        merged_dates = [dates[0]]
        for date_range in dates[1:]:
            if date_range[0] <= merged_dates[-1][1]:
                merged_dates[-1] = (merged_dates[-1][0], max(date_range[1], merged_dates[-1][1]))
            else:
                merged_dates.append(date_range)
        total_covered = sum([(date_range[1] - date_range[0]).days + 1 for date_range in merged_dates])
        total_days = (subset[self.mbr_elig_end_dt_col].iloc[0] - subset[self.mbr_elig_start_dt_col].iloc[0]).days + 1
        # set PDC_SCORE to zero if TOTALDAYS or DAYSCOVERED is zero
        if total_days == 0 or total_covered == 0:
            proportion = 0
        else:
            proportion = total_covered / total_days
        return {self.patient_id_col: patient_id,
                self.drugname_col: drug,
                'DAYSCOVERED': total_covered,
                'TOTALDAYS': total_days,
                'PDC_SCORE': proportion}
        
    def calculate_pdc(self, n_workers=cpu_count()):
        pool = Pool(processes=n_workers)
        results = []
        for patient_id in self.patient_ids:
            for drug in self.drugs:
                results.append(pool.apply_async(self.calculate_pdc_for_patient_drug, (patient_id, drug)))
        pool.close()
        pool.join()
        
        for result in results:
            data = result.get()
            if data is not None:
                self.pdc_scores_df = pd.concat([self.pdc_scores_df, pd.DataFrame.from_records([data])])
                
        self.pdc_scores_dataframe = self.pdc_scores_df.sort_values(by=[self.patient_id_col, self.drugname_col]).reset_index(drop=True)
        return self.pdc_scores_dataframe
