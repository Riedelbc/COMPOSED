#!/bin/bash


for max_combinatorial in 8 9 10 11 12
do
  for c_stat in 0.1 10.0 25.0 50.0 75.0
  do
    for min_diff_thresh in 0.9 1.1 1.3 1.5
    do
      qsub -v max_combinatorial=$max_combinatorial,c_stat=$c_stat,min_diff_thresh=$min_diff_thresh -N qsub_parpool_${max_combinatorial}_${c_stat}_${min_diff_thresh}_job /ifshome/briedel/working_dir_epic_tools/scripts/submit_parpool_final.sh
    done
  done
done

#EOF
