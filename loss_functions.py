import torch
import numpy as np


# Helper function to enable loss function to be flexibly used for 
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn https://www.sciencedirect.com/science/article/pii/S0895611121001750 

def identify_axis(shape):
    """Identifies the axis to aggregate over for the loss function. 
    Expects BxCxHxW [2D] or BxCxDxHxW [3D]. Aggregation to happen over (D), H & W.

    Args:
        shape (list): input shape of tensor for GT or prediction

    Returns:
        list: list of dimensions to aggregate over
    """
    # Two dimensional
    if len(shape) == 4 : return [-1,-2]
    # Three dimensional
    elif len(shape) == 5 : return [-1,-2,-3]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')

################################
#           Dice loss          #
################################
def dice_loss(delta = 0.5, smooth = 0.000001):
    """Dice loss originates from Sørensen–Dice coefficient, which is a statistic developed in 1940s to gauge the similarity between two samples.
    
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """ 
    def loss_function(y_pred, y_true):
        axis = identify_axis(y_true.shape)
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1-y_pred), dim=axis)
        fp = torch.sum((1-y_true) * y_pred, dim=axis)
        # Calculate Dice score
        dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Average class scores
        dice_loss = torch.mean(1-dice_class)

        return dice_loss
        
    return loss_function


################################
#         Tversky loss         #
################################
def tversky_loss(delta = 0.7, smooth = 0.000001):
    """Tversky loss function for image segmentation using 3D fully convolutional deep networks
	Link: https://arxiv.org/abs/1706.05721
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """
    def loss_function(y_pred, y_true):
        axis = identify_axis(y_true.shape)
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)   
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1-y_pred), dim=axis)
        fp = torch.sum((1-y_true) * y_pred, dim=axis)
        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Average class scores
        tversky_loss = torch.mean(1-tversky_class)

        return tversky_loss

    return loss_function

################################
#       Dice coefficient       #
################################
def dice_coefficient(delta = 0.5, smooth = 0.000001):
    """The Dice similarity coefficient, also known as the Sørensen–Dice index or simply Dice coefficient, is a statistical tool which measures the similarity between two sets of data.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """
    def loss_function(y_pred, y_true):
        axis = identify_axis(y_true.shape)
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)   
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1-y_pred), dim=axis)
        fp = torch.sum((1-y_true) * y_pred, dim=axis)
        dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Average class scores
        dice = torch.mean(dice_class)

        return dice

    return loss_function

################################
#          Combo loss          #
################################
def combo_loss(alpha=0.5,beta=0.5):
    """Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation
    Link: https://arxiv.org/abs/1805.02798
    Parameters
    ----------
    alpha : float, optional
        controls weighting of dice and cross-entropy loss., by default 0.5
    beta : float, optional
        beta > 0.5 penalises false negatives more than false positives., by default 0.5
    """
    def loss_function(y_pred, y_true):
        dice = dice_coefficient()(y_pred, y_true)
        axis = identify_axis(y_true.shape)
        # Clip values to prevent division by zero error
        epsilon = 10e-8
        y_pred = torch.clamp(input=y_pred, min=epsilon, max=1.0 - epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        if beta is not None:
            beta_weight = np.array([beta, 1-beta])
            cross_entropy = beta_weight * cross_entropy
        # sum over classes
        cross_entropy = torch.mean(torch.sum(cross_entropy, dim=[-1]))
        if alpha is not None:
            combo_loss = (alpha * cross_entropy) - ((1 - alpha) * dice)
        else:
            combo_loss = cross_entropy - dice
        return combo_loss

    return loss_function

################################
#      Focal Tversky loss      #
################################
def focal_tversky_loss(delta=0.7, gamma=0.75, smooth=0.000001):
    """A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
    Link: https://arxiv.org/abs/1810.07842
    Parameters
    ----------
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_pred, y_true):
        # Clip values to prevent division by zero error
        epsilon = 10e-8
        y_pred = torch.clamp(input=y_pred, min=epsilon, max=1.0 - epsilon) 
        axis = identify_axis(y_true.shape)
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1-y_pred), dim=axis)
        fp = torch.sum((1-y_true) * y_pred, dim=axis)
        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Average class scores
        focal_tversky_loss = torch.mean(torch.pow((1-tversky_class), gamma))
	
        return focal_tversky_loss

    return loss_function


################################
#          Focal loss          #
################################
def focal_loss(alpha=None, gamma_f=2.):
    """Focal loss is used to address the issue of the class imbalance problem. A modulation term applied to the Cross-Entropy loss function.
    Parameters
    ----------
    alpha : float, optional
        controls relative weight of false positives and false negatives. alpha > 0.5 penalises false negatives more than false positives, by default None
    gamma_f : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 2.
    """
    def loss_function(y_pred, y_true):
        axis = identify_axis(y_true.shape)
        # Clip values to prevent division by zero error
        epsilon = 10e-8
        y_pred = torch.clamp(input=y_pred, min=epsilon, max=1.0 - epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        if alpha is not None:
            alpha_weight = np.array(alpha, dtype=np.float32)
            focal_loss = alpha_weight * torch.pow(1 - y_pred, gamma_f) * cross_entropy
        else:
            focal_loss = torch.pow(1 - y_pred, gamma_f) * cross_entropy

        focal_loss = torch.mean(torch.sum(focal_loss, dim=[-1]))
        return focal_loss
        
    return loss_function

################################
#       Symmetric Focal loss      #
################################
def symmetric_focal_loss(delta=0.7, gamma=2.):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    """
    def loss_function(y_pred, y_true):

        axis = identify_axis(y_true.shape)  

        epsilon = 10e-8
        y_pred = torch.clamp(input=y_pred, min=epsilon, max=1.0 - epsilon)
        if y_pred.shape[1] == 1:
            y_pred = torch.cat((1-y_pred, y_pred), dim=1)
            y_true = torch.cat((1-y_true, y_true), dim=1)
        cross_entropy = -y_true * torch.log(y_pred)
        #calculate losses separately for each class
        back_ce = torch.pow(1 - y_pred[:,:,:,0], gamma) * cross_entropy[:,:,:,0]
        back_ce =  (1 - delta) * back_ce

        fore_ce = torch.pow(1 - y_pred[:,:,:,1], gamma) * cross_entropy[:,:,:,1]
        fore_ce = delta * fore_ce

        loss = torch.mean(torch.sum(torch.concat([back_ce, fore_ce],dim=-1),dim=-1))

        return loss

    return loss_function

#################################
# Symmetric Focal Tversky loss  #
#################################
def symmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_pred, y_true):
        # Clip values to prevent division by zero error
        epsilon = 10e-8
        y_pred = torch.clamp(input=y_pred, min=epsilon, max=1.0 - epsilon)

        if y_pred.shape[1] == 1:
            y_pred = torch.cat((1-y_pred, y_pred), dim=1)
            y_true = torch.cat((1-y_true, y_true), dim=1)

        axis = identify_axis(y_true.shape)
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1-y_pred), dim=axis)
        fp = torch.sum((1-y_true) * y_pred, dim=axis)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

        #calculate losses separately for each class, enhancing both classes
        back_dice = ((1-dice_class[:,0]) * torch.pow(1-dice_class[:,0], -gamma)).unsqueeze(1)
        fore_dice = ((1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -gamma)).unsqueeze(1)

        # Average class scores
        loss = torch.mean(torch.concat([back_dice,fore_dice],dim=-1))
        return loss

    return loss_function


################################
#     Asymmetric Focal loss    #
################################
def asymmetric_focal_loss(delta=0.7, gamma=2.):
    """For Imbalanced datasets
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    """
    def loss_function(y_pred, y_true):
        axis = identify_axis(y_true.shape)  

        epsilon = 10e-8
        y_pred = torch.clamp(input=y_pred, min=epsilon, max=1.0 - epsilon)
        if y_pred.shape[1] == 1:
            y_pred = torch.cat((1-y_pred, y_pred), dim=1)
            y_true = torch.cat((1-y_true, y_true), dim=1)

        cross_entropy = -y_true * torch.log(y_pred)
        #calculate losses separately for each class, only suppressing background class
        back_ce = torch.pow(1 - y_pred[:,:1,...], gamma) * cross_entropy[:,:1,...]
        back_ce =  (1 - delta) * back_ce

        fore_ce = cross_entropy[:,1:,...]
        fore_ce = delta * fore_ce.sum(1, keepdim=True)

        loss = torch.concat([back_ce, fore_ce], dim=1)
        loss = loss.sum(1)
        loss = loss.mean(axis)

        return loss

    return loss_function

#################################
# Asymmetric Focal Tversky loss #
#################################
def asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_pred, y_true):
        # Clip values to prevent division by zero error
        epsilon = 10e-8
        y_pred = torch.clamp(input=y_pred, min=epsilon, max=1.0 - epsilon)

        if y_pred.shape[1] == 1:
            y_pred = torch.cat((1-y_pred, y_pred), dim=1)
            y_true = torch.cat((1-y_true, y_true), dim=1)

        axis = identify_axis(y_true.shape)
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1-y_pred), dim=axis)
        fp = torch.sum((1-y_true) * y_pred, dim=axis)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

        #calculate losses separately for each class, only enhancing foreground class
        back_dice = (1-dice_class[:,:1])
        fore_dice = (1-dice_class[:,1:]) * torch.pow(1-dice_class[:,1:], -gamma)
        
        # Average class scores
        loss = torch.concat([back_dice, fore_dice], dim=1)
        loss = loss.mean(1) 
        return loss

    return loss_function


###########################################
#      Symmetric Unified Focal loss       #
###########################################
def sym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framewortorch.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def loss_function(y_pred, y_true):
      symmetric_ftl = symmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_pred,y_true)
      symmetric_fl = symmetric_focal_loss(delta=delta, gamma=gamma)(y_pred,y_true)
      if weight is not None:
        return (weight * symmetric_ftl) + ((1-weight) * symmetric_fl)  
      else:
        return symmetric_ftl + symmetric_fl

    return loss_function

###########################################
#      Asymmetric Unified Focal loss      #
###########################################
def asym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framewortorch.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def loss_function(y_pred, y_true):
      asymmetric_ftl = asymmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_pred,y_true)
      asymmetric_fl = asymmetric_focal_loss(delta=delta, gamma=gamma)(y_pred,y_true)
      if weight is not None:
        return (weight * asymmetric_ftl) + ((1-weight) * asymmetric_fl)  
      else:
        return asymmetric_ftl + asymmetric_fl

    return loss_function



def ce_plus_dice_loss(smooth:float=1e-3, reduction:str='mean', dice:bool=True, ce:bool=False):

    def loss_function(y_pred:torch.Tensor, y_true:torch.Tensor)->torch.Tensor:
        if ce and dice:
            return ce_loss(y_pred, y_true, reduction) + dice_loss(y_pred, y_true, smooth, reduction)
        elif dice:
            return dice_loss(y_pred, y_true, smooth, reduction)
        elif ce:
            return ce_loss(y_pred, y_true, reduction)
        else:
            raise ValueError('At least one of dice or ce must be True')

    return loss_function


def ce_loss(pred:torch.Tensor, target:torch.Tensor, reduction:str='mean')->torch.Tensor:
    """Assuming shape BxCxHxW.

    Args:
        pred: prediction array
        true: ground truth array
    
    Returns:
        cross entropy loss

    """
    epsilon = 10e-8
    # scale preds so that the class probs of each sample sum to 1
    pred = pred / torch.sum(pred, 1, keepdim=True)
    # manual computation of crossentropy
    pred = torch.clamp(input=pred, min=epsilon, max=1.0 - epsilon)
    loss = -torch.sum((target * torch.log(pred)), 1, keepdim=True)
    loss = loss.mean((1,2,3)) 
    loss = loss.sum() if reduction == "sum" else loss

    return loss

def dice_loss(pred:torch.Tensor, target:torch.Tensor, smooth:float=1e-3, reduction:str='mean')->torch.Tensor:
    """Assuming shape BxCxHxW."""
    if reduction =='sum':
        intse = torch.sum(pred * target, (0, 2, 3)) # sum of intersection per class
        union = torch.sum(pred**2, (0, 2, 3)) + torch.sum(target**2, (0, 2, 3)) # sum of union per class
        loss = 1.0 - (2.0 * intse + smooth) / (union + smooth) # loss per class
        loss = loss.sum() # loss (aggregation over classes)
        
    else:
        intse = torch.sum(pred * target, (2,3)) # sum of intersection per mask and class
        union = torch.sum(pred**2, (2, 3)) + torch.sum(target**2, (2, 3)) # sum of union per mask and class
        loss = 1.0 - (2.0 * intse + smooth) / (union + smooth) # loss per mask and class
        loss = loss.sum(-1) # loss per mask (aggregation over classes)
    
    return loss