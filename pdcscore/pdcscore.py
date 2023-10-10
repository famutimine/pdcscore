import pandas as pd
from datetime import timedelta

class pdcCalc:
    def __init__(self, 
                 df, 
                 patient_id_col, 
                 drugname_col, 
                 filldate_col, 
                 supply_days_col, 
                 msr_start_dt_col, 
                 msr_end_dt_col,
                 overlap_adjustment):
        self.df = df
        self.patient_id_col = patient_id_col
        self.drugname_col = drugname_col
        self.filldate_col = filldate_col
        self.supply_days_col = supply_days_col
        self.msr_start_dt_col = msr_start_dt_col
        self.msr_end_dt_col = msr_end_dt_col
        self.overlap_adjustment = overlap_adjustment

    def calculate_pdc_scores(self):
        if self.overlap_adjustment is True:
            results = []

            # Group the DataFrame by patient_id and drugname
            grouped = self.df.groupby([self.patient_id_col, self.drugname_col])

            for (patient_id, drugname), group_df in grouped:
                # Calculate the total days
                max_start_date = max(group_df[self.filldate_col].min(), group_df[self.msr_start_dt_col].min())
                end_date = group_df[self.msr_end_dt_col].max()
                total_days = (end_date - max_start_date).days + 1  # Include the end date

                # Calculate total covered days
                total_covered_days = 0

                for _, row in group_df.iterrows():
                    start_date = row[self.filldate_col]
                    supply_days = row[self.supply_days_col]
                    msr_end_date = row[self.msr_end_dt_col]

                    # Calculate the overlap between the supply and measurement period
                    overlap_start = max(start_date, max_start_date)
                    overlap_end = min(start_date + timedelta(days=supply_days - 1), msr_end_date)

                    # Calculate the days covered by this prescription
                    days_covered = (overlap_end - overlap_start).days + 1
                    total_covered_days += days_covered
                    total_covered_days = min(total_covered_days,total_days)

                # PDC score calculation
                pdc_score = total_covered_days / total_days

                # Append the results to the list
                results.append([patient_id, drugname, total_days, total_covered_days, pdc_score])

            # Create a DataFrame from the results
            result_df = pd.DataFrame(results, columns=[self.patient_id_col, self.drugname_col, 'totaldays', 'dayscovered', 'pdc_score'])
            return result_df
        else:
            results = []

            # Group the DataFrame by patient_id and drugname
            grouped = self.df.groupby([self.patient_id_col, self.drugname_col])

            for (patient_id, drugname), group_df in grouped:
                # Calculate the total days
                max_start_date = max(group_df[self.filldate_col].min(), group_df[self.msr_start_dt_col].min())
                end_date = group_df[self.msr_end_dt_col].max()
                total_days = (end_date - max_start_date).days + 1  # Include the end date

                # Calculate total covered days (accounting for unique days)
                unique_covered_days = set()

                for _, row in group_df.iterrows():
                    start_date = row[self.filldate_col]
                    supply_days = row[self.supply_days_col]
                    msr_end_date = row[self.msr_end_dt_col]

                    # Calculate the overlap between the supply and measurement period
                    overlap_start = max(start_date, max_start_date)
                    overlap_end = min(start_date + timedelta(days=supply_days - 1), msr_end_date)

                    # Add the unique days to the set
                    for day in pd.date_range(overlap_start, overlap_end):
                        unique_covered_days.add(day)

                # Calculate the number of unique days covered
                total_covered_days = len(unique_covered_days)
                pdc_score = total_covered_days / total_days

                # PDC score calculation
                pdc_score = min(total_covered_days / total_days, 1.0)

                # Append the results to the list
                results.append([patient_id, drugname, total_days, total_covered_days, pdc_score])

            # Create a DataFrame from the results
            result_df = pd.DataFrame(results, columns=[self.patient_id_col, self.drugname_col, 'totaldays', 'dayscovered', 'pdc_score'])
            return result_df
