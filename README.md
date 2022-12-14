# powered-by-SPAM_DL-7850
Author: Matthew Ogden

## Installation

```
git clone https://github.com/mbogden/powered-by-SPAM_DL-7850.git
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

`python fitness_score.py --help`

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


## Bulk Scripts: analysis-fitness-score.ipynb

The Notebook `analysis-fitness-score.ipynb` provides an example of how to build runs of the script in bulk.  These full command lines are then saved in individual run files in the `runs` folder.  Once these runs are executed, the results should be saved within the `results` folder.  The analysis notebook also provides an example of how to read the results files and plot some statistical analysis on the learning curves.  Saved within this repository are a handful of results.


## References
* A. Holincheck, J. Wallin and A. Harvey, JSPAM: A restricted three-body code for simulating interacting galaxies, ArXiv e-prints, arXiv:1604.00435. 
* Holincheck, "Galaxy Zoo: Mergers - Dynamical Models of Interacting GalaxiesArXiv e-prints, arXiv:1604.00435," ArXiv e-prints, arXiv:1604.00435, 2016.
* G. West, "On fitting the morphology of simulations of interacting galaxies to synthetic data," Middle Tennessee State University, 2021.
* He, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778.
* M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," arXiv, 2019. 
* Ogden, Matthew. "Optimizing Numerical Simulations of Colliding Galaxies. II. Comparing Simulations to Astronomical Observations.," Research Notes of the AAS, 4, 2020. 
