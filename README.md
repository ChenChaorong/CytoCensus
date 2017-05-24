# QBrain

Software repository for QBrain, machine learning software for identifying cells in 3-D tissue.


## Corresponding software publication:

MS ID#: BIORXIV/2017/137406

MS TITLE: Brain Development: Machine Learning Analysis Of Individual Stem Cells In Live 3D Tissue

Authors:
Martin Hailstone, Lu Yang, Dominic Waithe, Tamsin J Samuels, Yoav Arava, Tomasz Dobrzycki, Richard M Parton and Ilan Davis

http://biorxiv.org/content/early/2017/05/14/137406

## Installation instructions:

### Run from Source

If you are a capable Python user then  you can clone the above repository and run from source. Clone the repository and run "python v2_release.py" for the training interface to be loaded. Run "python v2_evaluate.py" to load the interface for the bulk running of files.
Be aware the repository has a number of dependencies.

### Compiled software:

The latest compiled releases can be found here:
https://github.com/dwaithe/QBrain/releases/
Simply download the version for your operating system and run the software directly.

## FAQ
* Q: The software is slow to load A: It should only be slow the first time you run it on a particular computer. Subsequent usuage should be much faster.

* Q: Which executable should I run first the train or the bulk software? A: The train software should be run first as this software allows you to train the software and produce models which can than be applied in bulk. The bulk software allows you to apply your previously trained model to one or more datasets.