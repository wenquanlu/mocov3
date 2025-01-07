import csv

import os
import json
import matplotlib.pyplot as plt

import numpy as np
import pickle

def get_eval(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        last_line = lines[-1]
        print(last_line)
        num = float(last_line.split()[2])
        return num

def get_accuracies(output_dir):

    steps = ['0004', '0009', '0014', '0019', '0024', '0029', '0034', '0039', '0044', '0049', '0054', '0059', '0064', '0069', '0074', '0079', '0084', '0089', '0094', '0099', '0104', '0109', '0114', '0119', '0124', '0129', '0134', '0139', '0144', '0149', '0154', '0159', '0164', '0169', '0174', '0179', '0184', '0189', '0194', '0199']

    accuracies = []
    for step in steps:
        max_acc = 0
        file_path = output_dir + f"-{step}/eval_log.txt"
        max_acc = get_eval(file_path)
        accuracies.append(max_acc)
    return accuracies

def get_general_accuracies(output_dir, n):

    steps = ['0004', '0009', '0014', '0019', '0024', '0029', '0034', '0039', '0044', '0049', '0054', '0059', '0064', '0069', '0074', '0079', '0084', '0089', '0094', '0099', '0104', '0109', '0114', '0119', '0124', '0129', '0134', '0139', '0144', '0149', '0154', '0159', '0164', '0169', '0174', '0179', '0184', '0189', '0194', '0199']

    accuracies = []
    if n > 0:
        for step in steps[:n]:
            max_acc = 0
            file_path = output_dir + f"-{step}/eval_log.txt"
            max_acc = get_eval(file_path)
            accuracies.append(max_acc)
    if n < 0:
        for step in steps[n:]:
            max_acc = 0
            file_path = output_dir + f"-{step}/eval_log.txt"
            max_acc = get_eval(file_path)
            accuracies.append(max_acc)
    return accuracies

epochs = [i * 5 for i in range(1, 41)]
gauss100_restart = get_general_accuracies("output_gauss100-resume-0-140-200-0-60-60", 12)
gauss100_noisy = get_general_accuracies("output_gauss100-200", -4)
plt.figure(figsize=(10, 6))
plt.plot(epochs[-12:], gauss100_restart, marker='o', label="Ours")
plt.plot(epochs[-4:], gauss100_noisy, marker='o', label="Noisy")
print(gauss100_restart)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.savefig("first.png")