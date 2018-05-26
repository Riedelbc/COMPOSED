###############
Getting Started
###############

.. contents::

*************************
Installation Instructions
*************************

Install requires a python3 environment. I recommend using anaconda for linux, once
anaconda is installed, I recommend making an environment to use for composed
analysis, and ipython3 to use debugging capabilities::

   conda update conda
   conda install pip
   conda create --name composed_analysis python=3
   conda install ipython3
   source activate composed_analysis

Once conda is installed, create your local composed library by checking out composed
from brandy's ifshome::

   git clone /ifshome/briedel/Composed_Tools/ my_composed_branch

Make a branch to do your local development. Replace ``my_composed_branch``
with a meaningful name for your analysis project::

   cd my_composed_tools
   git checkout -b my_composed_branch

Install composed tools dependencies::

   pip install -r requirements.txt

**********
Parameters
**********
In your_script_name.py you need to set: the number of folds, classifier type, 
filepath to your csv, first/last column names of your csv

In your data.py in the "composed" dir, you need to set: prefixes of all variable classes on line 132
This is on the 'assert self.prefixes.issubset' line - if your feature type has the prefix 'VL_ROI' and
'Thk_ROI' then you would add 'VL' and 'Thk' - not ID or Group
Also set the SVM baseline cutoff threshold 'cutoff_threshold' - you can also set this to 0

In your composed.py in the "composed" dir, you need to set: the c statistic to optimize your results
This is on the 'clf = svm.LinearSVC' line

In your freesurfer.py script (or whatever it's called) there is a QC parameter
in composed.py line 308 onwards That will prevent you from being able to run
with more folds than you have data/group size for YOU NEED TO HAVE AT LEAST 3
DX/CN PEOPLE IN EACH FOLD! If not, reduce folds!

***********************
How to execute COMPOSED
***********************

Make a script in the scripts folder, you can use one of the existing ones as a
template. If you are using a conda environment, you can't just include
``#!/usr/bin/env python`` as the first line to run your script.  Instead,
activate your python 3 environment, and then run::

   source activate composed_analysis
   ipython3 --pdb
   %run filepath/scripts/your_script_name.py

This will make sure to use the local version of python that's located in your
working python environment.

When you are finished running analyses, to return to the normal environment, hit
control-d enter and then::

   source deactivate composed_analysis

*********
Debugging
*********

If/when it reaches an uncaught exception ipython3 will jump into debug mode.
If/when that happens, here's the basics you need to know in order to figure out
what went wrong.

* You can list out the surrounding lines of code if you type ``l``
* You can travel up the call stack by typing ``u``
* You can travel back down the call stack by typing ``d``
* You can list out values of variables that are located at your current level
  of the call stack by typing the variable name.
* You quit the debugger by typing ``q``

Those are about the only commands that I use. If you want to learn more,
check out the `pdb docs`_.

.. _pdb docs: https://docs.python.org/3.5/library/pdb.html


*********************************************
Saving Your Work and Sharing it with your Lab
*********************************************

Make sure to run::

   git commit -a

Every once in a while in order to version control your work. If you make some
interesting changes to composed tools and you want to share, push your branch
back to Brandy's repository so she can merge your changes with the master
branch. To do this, run the following command after running git commit::

   git push origin my_composed_branch


****
Misc
****

If you want to run multiple verions or csvs you can call them in separate
terminal tabs after activating composed_tools and opening ipython3 with the
following commands (you can do this in 1 tab, but they won't run in parallel)::

   %run Composed_Tools/scripts/freesurfer_analysis2.py /ifs/loni/faculty/thompson/four_d/
      briedel/ABIDE/MDD_ISBI/MDD_full_matched_21.csv
   %run Composed_Tools/scripts/freesurfer_analysis3.py /ifs/loni/faculty/thompson/four_d/
      briedel/ABIDE/MDD_ISBI/MDD_full_matched_21.csv
