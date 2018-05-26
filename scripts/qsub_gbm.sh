#!/bin/bash
#$ -t 1-80 -o /ifshome/briedel/log -j y

# Make sure the number of tasks matches the multiplication of combo, learning_rate, min_diff, max_depth
# Max tasks at once is 600 slots/16 cores (37.5)
# Run with qsub -q compute.q -pe smp 1-16 qsub_gbm.sh

# The variable SGE_TASK_ID is used by the scheduler. When you submit a job with -t 0:50
# and the job gets distributed throughout the grid, the first job is run with SGE_TASK_ID = 1
# on the second node that receives the job, SGE_TASK_ID = 2, etc.
# Since BASH arrays start with an index 0, we match SGE_TASK_ID = 1 to array index 0 of the $subjectArray

subject=$[$SGE_TASK_ID-1]

# Now run your executable. You'll need to change around your variables as needed
#echo $srcList | awk -v subject=$subject -F' ' 'NR==subject{system("md5sum " $1 "; cat "$1".md5")}'

max_combinatorial_array=(14 16 18 20)
learning_rate_array=(0.01 0.05 0.1)
min_diff_thresh_array=(1.05 1.08 1.1)
max_depth_array=(5 6 7 8 9 10)

for max_combinatorial in ${max_combinatorial_array[@]}
do
  for learning_rate in ${learning_rate_array[@]}
  do
    for min_diff_thresh in ${min_diff_thresh_array[@]}
    do
      for max_depth in ${max_depth_array[@]}
      do
      taskArray+=($max_combinatorial,$learning_rate,$min_diff_thresh,$max_depth)
      done
    done
  done
done


max_combinatorial=`echo ${taskArray[$subject]} | cut -d, -f1`
learning_rate=`echo ${taskArray[$subject]} | cut -d, -f2`
min_diff_thresh=`echo ${taskArray[$subject]} | cut -d, -f3`
max_depth=`echo ${taskArray[$subject]} | cut -d, -f4`

python /ifshome/briedel/working_dir_epic_tools/scripts/residuals_age_gbm.py $max_combinatorial $learning_rate $min_diff_thresh $max_depth
