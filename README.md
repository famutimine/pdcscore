The objective of this package is to offer a Python-based solution for computing the Proportion of Days Covered (PDC), a widely used metric in the healthcare industry to evaluate medication adherence. As the healthcare analytics sector shifts away from SAS, there is a growing need to recreate key metrics in alternative platforms. This package aims to simplify the process and reduce the workload for business analysts in the healthcare ecosystem by providing a readily available PDC calculation tool, thereby eliminating the need to build it from scratch.

I followed the original implementation logic of PDC in SAS, this can be found at https://support.sas.com/resources/papers/proceedings13/167-2013.pdf 

This paper offers a gentle, yet detailed introduction to the topic, and will serve as a reference to anyone new to the subject.

Current update is optimized for multiprocessing large datasets.

Please use as described below:

**INPUT PARAMETERS:**

**dataframe** - *A pandas dataframe containing the required columns described below.*

**patient_id_col** - *A unique patient identifier. Format = STRING or INTEGER*

**drugname_col** - *The name of the drug being filled or drug class or Generic name, per usual PDC requirements. Format = STRING*

**filldate_col** - *The date of the fill being dispensed. Format = DATE*

**supply_days_col** - *Days of supply being dispensed at fill. Format = INTEGER*

**mbr_elig_start_dt_col** - *First date of coverage eligiblity for patient or a reference START DATE. Format = DATE*

**mbr_elig_end_dt_col** - *Last date of coverage eligiblity for patient or a reference END DATE. Format = DATE*



**OUTPUT DATAFRAME** - *A Pandas dataframe containing the following columns*

**patient_id_col** - *This will return a column name representing a unique patient identifier as provided in original input dataframe. FORMAT = STRING*

**drugname_col** - *The name of the drug being filled or drug class or Generic name, as provided in original input dataframe.*

**DAYSCOVERED**- *The number of unique days of drug coverage, after shifting coverage to accommodate early refills. FORMAT = INTEGER*

**TOTALDAYS** - *The total number of days in patient analysis window. Set to 0 if days of coverage is 0. FORMAT = INTEGER*

**PDC_SCORE** - *The patient's PDC score, calculated as DAYSCOVERED / TOTALDAYS. Set to 0 if days of coverage is 0. FORMAT = FLOAT*



## USAGE EXAMPLE
```python

#  Import required libraries
import pandas as pd
import numpy as np
from pdcscore import pdccalc

# Create a sample dataframe
df = pd.DataFrame({
    'MCID': ['A001', 'A001', 'A001', 'B001', 'B001', 'B001'],
    'DRUGNAME': ['DRUG_X', 'DRUG_X', 'DRUG_X', 'DRUG_Y', 'DRUG_Y', 'DRUG_Y'],
    'RX_FILLED_DT': pd.to_datetime(['2022-01-01', '2022-01-21', '2022-03-20',
                                '2022-01-01', '2022-02-01', '2022-03-01']),
    'DAYS_SPLY_NBR': [30, 30, 30, 30, 30, 30],
    'START_DT': pd.to_datetime(['2022-01-01', '2022-01-01', '2022-01-01',
                                         '2022-02-01', '2022-02-01', '2022-02-01']),
    'END_DT': pd.to_datetime(['2022-03-31', '2022-03-31', '2022-03-31',
                                       '2022-03-31', '2022-03-31', '2022-03-31'])
})

# Inspect sample data
df.head(n=len(df))

# calculate PDC scores on the input DataFrame
calcfunc = pdccalc(dataframe=df, patient_id_col='MCID', drugname_col='DRUGNAME'
                                         , filldate_col='RX_FILLED_DT', supply_days_col='DAYS_SPLY_NBR'
                                         , mbr_elig_start_dt_col='START_DT', mbr_elig_end_dt_col='END_DT')
pdc_scores_df = calcfunc.calculate_pdc()

# Inspect output
pdc_scores_df.head()