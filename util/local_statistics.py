"""
Functions that help with local statistics
"""

import random
import torch

class LocalStatisticsDestroyer():
    """
    Mess up local statistics
    """

    def __init__(self):

        ###################################
        ### Base Convs
        ###################################

        # Beta 1
        beta_1_conv = torch.nn.Conv2d(3, 3, kernel_size=(1, 2), stride=1, padding=0, groups=3)
        beta_1_conv.weight.data = torch.zeros_like(beta_1_conv.weight.data)
        beta_1_conv.weight.data[0][0][0][0] = 1
        beta_1_conv.weight.data[0][0][0][1] = -1
        beta_1_conv.weight.data[1][0][0][0] = 1
        beta_1_conv.weight.data[1][0][0][1] = -1
        beta_1_conv.weight.data[2][0][0][0] = 1
        beta_1_conv.weight.data[2][0][0][1] = -1
        beta_1_conv.bias.data = torch.zeros_like(beta_1_conv.bias.data)
        self.beta_1_conv = beta_1_conv

        # Beta 2
        beta_2_conv = torch.nn.Conv2d(3, 3, kernel_size=(2, 1), stride=1, padding=0, groups=3)
        beta_2_conv.weight.data = torch.zeros_like(beta_2_conv.weight.data)
        beta_2_conv.weight.data[0][0][0][0] = 1
        beta_2_conv.weight.data[0][0][1][0] = -1
        beta_2_conv.weight.data[1][0][0][0] = 1
        beta_2_conv.weight.data[1][0][1][0] = -1
        beta_2_conv.weight.data[2][0][0][0] = 1
        beta_2_conv.weight.data[2][0][1][0] = -1
        beta_2_conv.bias.data = torch.zeros_like(beta_2_conv.bias.data)
        self.beta_2_conv = beta_2_conv

        # Beta 3
        beta_3_conv = torch.nn.Conv2d(3, 3, kernel_size=(2, 2), stride=1, padding=0, groups=3)
        beta_3_conv.weight.data = torch.zeros_like(beta_3_conv.weight.data)
        beta_3_conv.weight.data[0][0][0][0] = 1
        beta_3_conv.weight.data[0][0][1][1] = -1
        beta_3_conv.weight.data[1][0][0][0] = 1
        beta_3_conv.weight.data[1][0][1][1] = -1
        beta_3_conv.weight.data[2][0][0][0] = 1
        beta_3_conv.weight.data[2][0][1][1] = -1
        beta_3_conv.bias.data = torch.zeros_like(beta_3_conv.bias.data)
        self.beta_3_conv = beta_3_conv

        # Beta 4
        beta_4_conv = torch.nn.Conv2d(3, 3, kernel_size=(2, 2), stride=1, padding=0, groups=3)
        beta_4_conv.weight.data = torch.zeros_like(beta_4_conv.weight.data)
        beta_4_conv.weight.data[0][0][1][0] = 1
        beta_4_conv.weight.data[0][0][0][1] = -1
        beta_4_conv.weight.data[1][0][1][0] = 1
        beta_4_conv.weight.data[1][0][0][1] = -1
        beta_4_conv.weight.data[2][0][1][0] = 1
        beta_4_conv.weight.data[2][0][0][1] = -1
        beta_4_conv.bias.data = torch.zeros_like(beta_4_conv.bias.data)
        self.beta_4_conv = beta_4_conv

        self.loss_fns = [
            self.beta1,
            self.beta2,
            self.beta3,
            self.beta4,
            lambda x: (self.beta1(x) + self.beta3(x)) / 2.0,
            lambda x: (self.beta1(x) + self.beta4(x)) / 2.0,
            lambda x: (self.beta2(x) + self.beta3(x)) / 2.0,
            lambda x: (self.beta2(x) + self.beta4(x)) / 2.0
        ]
    
    
    def __call__(self, image):
        return self.optimize(
            loss_fn      = random.choice(self.loss_fns),
            image        = image,
            lr           = 1e-2,
            print_losses = False,
            iterations   = 10
        )


    def optimize(self, loss_fn, image, lr=1e-3, print_losses=True, iterations=25):
        img_test = image.clone().detach().unsqueeze(0)
        img_test.requires_grad = True

        optimizer = torch.optim.SGD(
            [img_test], 
            lr, # Learning Rate 
            momentum=0.9,
            weight_decay=0, 
            nesterov=False
        )

        for i in range(iterations):
            loss = loss_fn(img_test)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if print_losses:
                print(loss)
        
        return img_test.clone().detach().squeeze(0)

    def beta1(self, image):
        """
        High values: sub-pixels are far apart
        Low values: sub-pixels are close together
        
        Note this is the opposite of the paper
        """
        return torch.sum(self.beta_1_conv(image) ** 2)


    def beta2(self, image):
        """
        High values: sub-pixels are far apart
        Low values: sub-pixels are close together
        
        Note this is the opposite of the paper
        """
        return torch.sum(self.beta_2_conv(image) ** 2)


    def beta3(self, image):
        """
        High values: sub-pixels are far apart
        Low values: sub-pixels are close together
        
        Note this is the opposite of the paper
        """
        return torch.sum(self.beta_3_conv(image) ** 2)


    def beta4(self, image):
        """
        High values: sub-pixels are far apart
        Low values: sub-pixels are close together
        
        Note this is the opposite of the paper
        """
        return torch.sum(self.beta_4_conv(image) ** 2)

