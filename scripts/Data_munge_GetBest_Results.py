import os

dirname = '/ifs/loni/faculty/thompson/four_d/briedel/MDD/Results/Fin/Over21_Complete_Fin/'
outdirname = '/ifs/loni/faculty/thompson/four_d/briedel/MDD/Results/Fin/'
logs = os.listdir(dirname)

csv_from_the_dead = []

for log in logs:
    lname, ext = os.path.splitext(log)
    if not ext == '.log':
        print('nope: {}'.format(log))
        continue
    with open(os.path.join(dirname, log), 'r') as f_obj:
        split_idx = log.index('_cstat')
        split_log = '{}, {}'.format(log[:split_idx], log[split_idx:])
        flist = f_obj.readlines()
        perf_l = []
        prev_res = 0
        try:
            while prev_res < len(flist):
                _idx = flist[prev_res:].index('Mean Performance of the runs:\n')
                new_res = prev_res + _idx + 4
                perf_l.append([float(val.strip()) for val in flist[new_res].split(',')])
                prev_res = new_res
        except ValueError:
            if not perf_l:
                continue
            elif len(perf_l) != 10:
                print("WARNING: ignoring results from {}. Expecting to have ten repeats, only received {}".format(os.path.join(dirname, log), len(perf_l)))
                continue
            avgs = ['{}'.format(float(sum([row[idx] for row in perf_l]))/len(perf_l)) for idx, _ in enumerate(perf_l[0])]
            csv_from_the_dead.append('{}, {}'.format(', '.join(avgs), split_log))
           
with open(os.path.join(outdirname, 'All_results_Over21_Complete.csv'), 'w') as f_obj:
    f_obj.write('\n'.join(csv_from_the_dead))

##########################################################################
########################################################################## 
dirname = '/ifs/loni/faculty/thompson/four_d/briedel/MDD/Results/Fin/Recurrent_Complete_Fin/'
outdirname = '/ifs/loni/faculty/thompson/four_d/briedel/MDD/Results/Fin/'
logs = os.listdir(dirname)

csv_from_the_dead = []

for log in logs:
    lname, ext = os.path.splitext(log)
    if not ext == '.log':
        print('nope: {}'.format(log))
        continue
    with open(os.path.join(dirname, log), 'r') as f_obj:
        split_idx = log.index('_cstat')
        split_log = '{}, {}'.format(log[:split_idx], log[split_idx:])
        flist = f_obj.readlines()
        perf_l = []
        prev_res = 0
        try:
            while prev_res < len(flist):
                _idx = flist[prev_res:].index('Mean Performance of the runs:\n')
                new_res = prev_res + _idx + 4
                perf_l.append([float(val.strip()) for val in flist[new_res].split(',')])
                prev_res = new_res
        except ValueError:
            if not perf_l:
                continue
            elif len(perf_l) != 10:
                print("WARNING: ignoring results from {}. Expecting to have ten repeats, only received {}".format(os.path.join(dirname, log), len(perf_l)))
                continue
            avgs = ['{}'.format(float(sum([row[idx] for row in perf_l]))/len(perf_l)) for idx, _ in enumerate(perf_l[0])]
            csv_from_the_dead.append('{}, {}'.format(', '.join(avgs), split_log))

with open(os.path.join(outdirname, 'All_results_Recurrent_Complete.csv'), 'w') as f_obj:
    f_obj.write('\n'.join(csv_from_the_dead))

