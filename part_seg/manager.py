import os
import sys
sys.path.append('../')
from util import *
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from ShapeNetPart import get_valid_labels
from visualize import visualize

def get_miou(pred: "tensor (point_num, )", target: "tensor (point_num, )", valid_labels: list):
    pred, target = pred.cpu().numpy(), target.cpu().numpy()
    part_ious = []
    for part_id in valid_labels:
        pred_part = (pred == part_id)
        target_part = (target == part_id)
        I = np.sum(np.logical_and(pred_part, target_part))
        U = np.sum(np.logical_or( pred_part, target_part))
        if U == 0:
            part_ious.append(1)
        else:
            part_ious.append(I/U)
    miou = np.mean(part_ious)
    return miou

        
class IouTable():
    def __init__(self):
        self.obj_miou = {}
        
    def add_obj_miou(self, category: str, miou: float):
        if category not in self.obj_miou:
            self.obj_miou[category] = [miou]
        else:
            self.obj_miou[category].append(miou)

    def get_category_miou(self):
        """
        Return: moiu table of each category
        """
        category_miou = {}
        for c, mious in self.obj_miou.items():
            category_miou[c] = np.mean(mious)
        return category_miou

    def get_mean_category_miou(self):
        category_miou = []
        for c, mious in self.obj_miou.items():
            c_miou = np.mean(mious)
            category_miou.append(c_miou)
        return np.mean(category_miou)
    
    def get_mean_instance_miou(self):
        object_miou = []
        for c, mious in self.obj_miou.items():
            object_miou += mious
        return np.mean(object_miou)

    def get_string(self):
        mean_c_miou = self.get_mean_category_miou()
        mean_i_miou = self.get_mean_instance_miou()
        first_row  = "| {:5} | {:5} ||".format("Avg_c", "Avg_i")
        second_row = "| {:.3f} | {:.3f} ||".format(mean_c_miou, mean_i_miou)
        
        categories = list(self.obj_miou.keys())
        categories.sort()
        cate_miou = self.get_category_miou()

        for c in categories:
            miou = cate_miou[c]
            first_row  += " {:5} |".format(c[:3])
            second_row += " {:.3f} |".format(miou)
        
        string = first_row + "\n" + second_row
        return string 

