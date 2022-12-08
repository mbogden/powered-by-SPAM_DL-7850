# powered-by-SPAM_DL-7850
Stellar Particle Animation Module (SPAM)


## Installation

```
git clone https://github.com/mbogden/powered-by-SPAM_DL-7850.git
git clone git@github.com:mbogden/powered-by-SPAM_DL-7850.git
```

Run script to do some directory preperation.  This will create some empty folder and unzip test data to use. 
`bash prep-dir.bash`


## Building and Testing
1. Open `finess_score.ipynb` in Juptyer Notebook too see the code.  
This notebook can be ran within Jupyter Notebook for direct feedback and results.  Notice, there is a variable named `buildEnv` that is only true within Jupyther Notebooks.  This variable allows for visualizing a lot of extra information only within the Jupyter Notebook environment

2. Run build script to convert Jupyter Notebook into a python script.
`bash build-fitness-score.sh`

This will convert the notebook into a python script.  If you append the cmd `test` to the end of the builder, it will also run the script after it's built to test if it can run.

`bash build-fitness-score.sh test`


## Script Execution
The script has a variety of command line arguments to choose from.  However most have default values.  The only required argument is `-runName demo-name`.  This informs the script how to name and save the results and models.  
`python fitness_score -runName demo-name`

