# superpixel-annotation
Annotating coastline image segmentaiton training set using superpixels


## Update SWED
python train.py --model_name SWED_TEST --satellite sentinel --early_stopping 10 --train_path /Users/conorosullivan/Documents/git/COASTAL\ MONITORING/data/SWED/train/ --device mps --sample True

## Apply superpixels 
python apply_superpixels.py --input_dir /home/people/22205288/scratch/training --satellite landsat

python apply_superpixels.py --input_dir /home/people/22205288/scratch/SWED/train --satellite sentinel 

## Test datasets

python dataset_test.py /home/people/22205288/scratch/SWED/train sentinel 

python superpixel-annotation/src/dataset_test.py /home/people/22205288/scratch/training landsat 
