# PGC View V2 - Training Framework
This repo exists to hold the model training and test inference code for the PGC View V2 image analysis pipeline

Since our dataset is largely unlabeled and unbalanced, we are using a modified version of the FlexMatch semi-supervised learning algorithm to train the segmentation models. You can read the original publication here: [FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling](https://arxiv.org/abs/2110.08263)

## To Do

- [x] Finish the FlexMatch trainer class
- [x] Format dataset and run Welford calculator to find RGB means and Std.
- [x] Create a Labelbox API call script to pull in the annotations from our labeling project and automatically move images from 'all_images' into the labeled and unlabeled folders.
- [ ] Work on training supervised script
- [ ] Create a new train and inference script.
