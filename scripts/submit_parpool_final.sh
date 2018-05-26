#!/bin/bash
#$ -S /bin/bash
#$ -o /ifshome/briedel/ -j y 

python /ifshome/briedel/working_dir_epic_tools/scripts/parpool_final.py ${max_combinatorial} ${c_stat} ${min_diff_thresh}
