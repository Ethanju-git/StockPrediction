import torch
import torch.nn as nn

def generate_weights(length, weighting_factor):
    """
    Generates exponentially decaying weights for the loss function.
    Ie [h^(n-1) h^(n-2) ... h^0] where h^(n-1) is oldest data point and
    h^0 is the most recent. We divide these weights by the sum to ensure
    that the num timesteps doesnt skew analysis.
    """
    weights = torch.pow(weighting_factor, torch.arange(length - 1, -1, -1).float())
    weights /= torch.sum(weights)  # Normalize weights to sum to 1
    return weights


class MAELoss(nn.Module):
    def __init__(self, weigh_under, weighting_factor):
        """
        MAE with the parameters to punish overprediction and
        underprediction differently as well as a parameter that 
        controls how to weigh older vs recent samples differently.
        These params can be set to 1 which makes them ineffective.
        """
        super(MAELoss, self).__init__()
        self.weigh_under = weigh_under
        self.weigh_recent = weigh_recent
        self.weighting_factor = weighting_factor

    def forward(self, y_true, y_pred):
        errors = torch.abs(y_true - y_pred)

        # Apply asymmetry for under-predictions
        under_mask = (y_pred < torch.abs(y_pred)).bool() 
        errors = torch.where(under_mask, errors * self.weigh_under, errors)

        weights = generate_weights(len(errors), self.weighting_factor).to(errors.device)
        errors = errors * weights

        return torch.mean(errors)



class MSELoss(nn.Module):
    def __init__(self, weigh_under, weighting_factor):
        """
        MSE with the parameters to punish overprediction and
        underprediction differently as well as a parameter that 
        controls how to weigh older vs recent samples differently.
        These params can be set to 1 which makes them ineffective.
        """
        super(MSELoss, self).__init__()
        self.weigh_under = weigh_under
        self.weigh_recent = weigh_recent
        self.weighting_factor = weighting_factor

    def forward(self, y_true, y_pred):
        residuals = y_pred - y_true
        errors = residuals**2
        under_mask = (y_pred < torch.abs(y_pred)).bool() 
        errors = torch.where(under_mask, errors * self.weigh_under, errors)

        weights = generate_weights(len(errors), self.weighting_factor).to(errors.device)
        errors = errors * weights

        return torch.mean(errors)


class QuantileLoss(nn.Module):
    def __init__(self, tau, weighting_factor):
        """
        Loss func with param tao that controls over vs under prediction punishment.
        0 < Tao < 1 and underpredictions are multiplied by Tao and over by (1 - Tao).
        Also has the ability to punish differently based upon recency. 
        """
        super(QuantileLoss, self).__init__()
        self.tau = tau
        self.weigh_recent = weigh_recent
        self.weighting_factor = weighting_factor

    def forward(self, y_true, y_pred):
        diff = y_true - y_pred
        errors = torch.where(diff > 0, self.tau * diff, (1 - self.tau) * -diff)

        weights = generate_weights(len(errors), self.weighting_factor).to(errors.device)
        errors = errors * weights

        return torch.mean(errors)


class LinExLoss(nn.Module):
    def __init__(self, a, weigh_recent, weighting_factor):
        """
        Specialized loss func for over vs underprediction that
        that utilizes an exponential term and a term to smooth
        behavior when error is small. Also has ability to weigh
        data different based upon recency.  
        """
        super(LinExLoss, self).__init__()
        self.a = a
        self.weigh_recent = weigh_recent
        self.weighting_factor = weighting_factor

    def forward(self, y_true, y_pred):
        errors = y_true - y_pred
        losses = torch.exp(self.a * errors) - self.a * errors - 1

        weights = generate_weights(len(losses), self.weighting_factor).to(losses.device)
        losses = losses * weights

        return torch.mean(losses)
