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
1. Open `finess_score.ipynb` in Juptyer Notebook to see the code.  

This notebook is the primary file for creating and altering the scipt.  The notebook can be ran within Jupyter Notebook for direct feedback and results as well providing extra details and information.  Notice, there is a variable named `buildEnv` that is only true within Jupyther Notebooks.  This variable allows for visualizing a lot of extra information only within the Jupyter Notebook environment.

2. Run build script to convert Jupyter Notebook into a python script.

`bash build-fitness-score.sh`

This will convert the notebook into a python script.  If you append the cmd `test` to the end of the builder, it will also run the script after it's built to test if it can run.

`bash build-fitness-score.sh test`


## Script Execution
The script has a variety of command line arguments to choose from.  However most have default values.  The only required argument is `-runName demo-name`.  This informs the script how to name and save the results and models.  

`python fitness_score -runName demo-name`

Below is a snippet of the code that shows a list of available command line arguments.  They can also be accessed by using the help command with the program.

`python fitness_score --help`

```
parser.add_argument( '-runName', )
parser.add_argument( '-modelLoc', )
parser.add_argument( "-tid",      default = '587722984435351614',  type=str )
parser.add_argument( "-start",    default = 0,  type=int, )
parser.add_argument( "-stop",     default = 3,  type=int, )
parser.add_argument( "-verbose",  default = 1,  type=int, )
parser.add_argument( "-num_epochs",    default=2,       type=int )
parser.add_argument( "-learning_rate", default=0.0001,  type=float )
parser.add_argument( "-batch_size",    default=16,      type=int )
parser.add_argument( "-save_model",    default='False', type=str )
parser.add_argument( "-data_gen",      default='True',  type=str )

# Core Model types
parser.add_argument( "-model",   default = 'efficientNetB0', type=str)
parser.add_argument( "-pool",    default = 'None',           type=str )
parser.add_argument( "-weights", default = 'imagenet',       type=str )

# Final layers
parser.add_argument( "-f_depth", default = 8,  type=int )
parser.add_argument( "-f_width", default = 32, type=int )
parser.add_argument( "-f_activation", default = 'relu', type=str )
parser.add_argument( "-output_activation", default = 'sigmoid' )
```