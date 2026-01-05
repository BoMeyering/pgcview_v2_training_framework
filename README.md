# PGC View V2 - Training Framework
This repo holds the model training and test inference code for the PGC View V2 image analysis pipeline

Since our dataset is largely unlabeled and unbalanced, we are using a modified version of the FlexMatch semi-supervised learning algorithm to train the segmentation models. You can read the original publication here: [FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling](https://arxiv.org/abs/2110.08263)

## Run the training script
Both the fully supervised and semi-supervised training scripts are set up to run using ```torchrun``` and currently work with a single node with multiple GPUs.

```train_supervised.py``` has two flags. ```--config``` is the path to the training config YAML file and defaults to ```configs/train_config.yaml```. ```--backend``` is the torchrun backend engine you want to use - ```nccl``` for CUDA and ```gloo``` for CPU based computation.
Run the supervised script like
```
$ torchrun --standalon --nproc-per-node=$NUM_PROCESSES train_supervised.py --backend='nccl'
```
## To Do

- [x] Finish the FlexMatch trainer class
- [x] Format dataset and run Welford calculator to find RGB means and Std.
- [x] Create a Labelbox API call script to pull in the annotations from our labeling project and automatically move images from 'all_images' into the labeled and unlabeled folders.
- [x] Work on training supervised script
- [ ] Create a new train and inference script.

## Loss Functions
### Multiclass Cross Entropy
The most common loss function for multiclass semantic segmentation is vanilla cross entropy defined in equation 1 as:
$$
\begin{equation}
    \mathcal{L}_{CE}(\bf{y}, \bf{\hat{y}})=-\sum^{C}_{c=1}y_c \log{\hat{y}_c}
\end{equation}
$$

where $y_{c}$ is the ground truth, one-hot encoded vector of class indices for a pixel, and $\hat{y}_c$ is the probability vector over $C$ classes for the same pixel.

### Weighted Cross Entropy
Since class imbalance is common in semantic segmentation modeling problems, it is common to use a loss function that rebalances, or reweights the loss values for a given class $k$. One of the easiest methods is to assign a class weight vector $\bf{\alpha}$ which can be used directly in cross entropy as shown in equation 2:
$$
\begin{equation}
    \mathcal{L}_{WCE}(\bf{y}, \bf{\hat{y}})=-\sum^{C}_{c=1}\alpha_c y_c \log{\hat{y}_c}
\end{equation}
$$

where $\alpha_c$ is the $c^{th}$ class weight. Commonly the weights are created using an inverse frequency scheme such that $\alpha_c = {\frac{N_c}{N}}^{-1}$ where $N$ is the total number of samples in the training set, and $N_c$ is the total number of samples in class $c$. Notice when $\forall k, \alpha_k = 1$, weighted cross entropy is equal to common cross entropy.

### Focal Loss
Focal loss is another way to rebalance the cross entropy loss by forcing the function to "focus" on difficult to classify samples by introducing a focusing parameter $\gamma$ on the term $(1-\hat{y}_c)^\gamma$ which places more weight on incorrect classifications with a low probability. This can be added in to standard cross entropy loss as shown in equation 3:

$$
\begin{equation}
    \mathcal{L}_{FL}(\bf{y}, \bf{\hat{y}})=-\sum^{C}_{c=1} y_c (1-\hat{y}_c)^\gamma\log{\hat{y}_c}
\end{equation}
$$

### Weighted Focal Loss
Note that we can also express accept a weighted term $\alpha_c$ similar to weighted cross entropy loss such that

$$
\begin{equation}
    \mathcal{L}_{WFL}(\bf{y}, \bf{\hat{y}})=-\sum^{C}_{c=1}\alpha_c y_c (1-\hat{y}_c)^\gamma\log{\hat{y}_c}
\end{equation}
$$

### Class Balanced Loss
Inverse weighting methods tend towards Type I errors since they cause the model to overfit the minority class. Thus class balanced loss introduces a more nuanced weighting term that uses the effective sample number ${E_{n}}_{c}=\frac{1-\beta^{N_c}}{1-\beta}$ where $N_c$ is the number of samples in class $c$ and $\beta$ is a user set hyperparameter. Then define $\alpha_c \propto \frac{1}{{E_n}_c}$ and plug the weight term into either weighted cross entropy or weighted focal loss. Mathematical details can be found [here](https://arxiv.org/abs/1901.05555).

### Adaptive Class Balanced Loss
Adaptive class balanced loss builds off of class balanced loss by allowing $\beta$ to be set dynamically on the data rather than manually tuned. Weights are set by defining $\beta=F(f(u, v, b))=\tanh\bigl(\frac{u}{v^{\sqrt{b}}}\bigr)$. Then ${E_n}_{c}$ is calculated from this $\beta$ as above in class balanced loss and incorporated into the loss function of choices. More details can be found [here](https://ieeexplore.ieee.org/document/10137858).

### Recall Loss
Another strategy is to use the model's recall for class $c$ at step $t$ to adjust the weights dynamically throughout training. This was first proposed [here](https://arxiv.org/abs/2106.14917). The full loss can be expressed as below in equation 5.

$$
\begin{equation}
    \mathcal{L}_{R}(\bf{y}, \bf{\hat{y}})=-\sum^{C}_{c=1}(1-\mathcal{R}_{c,t})y_c\log{\hat{y}_{c,t}}
\end{equation}
$$

### Dice Loss
Many semantic segmentation models are evaluated based on the Dice Score or Jaccard Index. However these are non-differentiable functions. The Dice Score can be written in terms of positives and negatives as 
$$
\begin{equation}
    Dice = \frac{2TP}{2TP + FP + FN}
\end{equation}
$$

This can be rewritten as a differentiable function as shown [here](https://arxiv.org/abs/1707.03237).

### Tversky Loss
Finally Tversky loss, which is an adaptation of the Tversky similarity score allows one to set hyperparameters $\alpha$ and $\beta$ to weight the $FP$ and $FN$ differently and generalize both Dice score and Jaccard index, which after some math can be expressed as
$$
\begin{equation}
    \mathcal{L}_{Tversky}(\bf{y}, \bf{\hat{y}})= 1-\frac{TP}{TP + \alpha FP + \beta FN}
\end{equation}
$$

More details can be found [here](https://arxiv.org/abs/1706.05721).

## Loss Function Implementation in Pytorch
We wrote implementations of all these loss functions in Pytorch using a consistent API call such that they can all be easily called from within the trainer class. Additionally, since the FlexMatch algorithm requires masking non-confident pixels, each ```forward()``` call of the loss module can accept a binary mask of shape ```(H, W)``` to get rid of any pixels in the loss matrix that do not pass the threshold.

 Additionally we implemented label smoothing for all CE/NLL loss functions. Label smoothing is a regularization technique that replaces the one-hot encoded ground truth vector with a smoothed version such that for a given class $c$ and smoothing parameter $\epsilon$, the new ground truth vector is defined as:
 $$
 \begin{equation}
     y_{c}^{LS} = (1-\epsilon)y_c + \frac{\epsilon}{C}
 \end{equation}
 $$ 