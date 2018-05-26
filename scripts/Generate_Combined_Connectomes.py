import numpy as np
import os
import glob
import pandas as pd
import csv

out_dir ='/Users/briedel/Desktop/EPIC_test_con/'

list_of_files = glob.glob('/ifshome/briedel/EPIC_test_con*/Test*.csv')


with open('/ifshome/briedel/Epic_Tools/scripts/Names_con.csv', 'r') as f:
  reader = csv.reader(f)
  col_headers_list = list(reader)
    
flattened = []

for file_name in list_of_files: 
    df = pd.read_csv(file_name, header=None)
    a = np.array(df)
    tri_upper_diag = np.triu(a, k=1)
    test = np.ndarray.flatten(tri_upper_diag[tri_upper_diag > 0])
    flattened.append(test)

Names_df = pd.DataFrame(col_headers_list).T    
flattened = pd.DataFrame(flattened)

combo = Names_df.append(flattened)

combo_flattened = pd.DataFrame(combo)
combo_flattened.to_csv(os.path.join(out_dir, 'connectomics_test.csv'), header=False, index=False)
