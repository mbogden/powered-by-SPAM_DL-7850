jupyter nbconvert --to script 'fitness_score.ipynb'
chmod +x fitness_score.py


# Check for valid $1 
case $1 in 

    'test')
        echo 'Testing '
        python fitness_score.py -runName build-test -stop 1 -num_epoch 1
        ;;
        
    *)
        exit
        ;;
esac