import torch
import torchvision
import torchvision.transforms as transforms

import os
import shutil
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count

if not os.path.exists("data"):
    os.mkdir("data")

if not os.path.exists(os.path.join("data", "train")):
    os.mkdir(os.path.join("data", "train"))

train_dataset = torchvision.datasets.MNIST(root='data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
batch_size = 1
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

cnt_list = np.zeros((10))


def process(i, data):
    if i % 100 == 0:
        print("Done", i)

    image = data[0]
    label = data[1]

    folder_name = os.path.join(os.path.join(
        "data", "train"), str(int(label[0])))
    file_name = str(int(cnt_list[int(label[0])])) + ".txt"
    file_name = os.path.join(folder_name, file_name)

    cnt_list[int(label[0])] = cnt_list[int(label[0])] + 1

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    img = image.numpy()[0][0].reshape((28*28)).tolist()

    with open(file_name, "w") as f:
        for i in range(len(img)):
            f.write(str(img[i]) + " ")


# executor = ProcessPoolExecutor(max_workers=cpu_count())
# futures = []
data_list = list()

for i, data in enumerate(train_loader):
    # if i == 100:
    #     break
    # print(i)
    data_list.append((i, data))

for (i, data) in data_list:
    process(i, data)
    # futures.append(executor.submit(partial(process, i, data)))

# [future.result() for future in futures]

folder_list = [i for i in range(10)]


def randomarr(start, end, length):
    cnt = 0
    out_list = list()
    while(True):
        ele = random.randint(start, end-1)
        if ele not in out_list:
            out_list.append(ele)
            cnt = cnt + 1
        if cnt == length:
            break

    return out_list


# print(randomarr(0, 10, 3))
if not os.path.exists(os.path.join("data", "test")):
    os.mkdir(os.path.join("data", "test"))

for folder_name in folder_list:
    list_file = randomarr(0, len(os.listdir(os.path.join(
        os.path.join("data", "train"), str(folder_name)))), 500)
    for file_name in list_file:
        filename = os.path.join(os.path.join(
            "data", "train"), str(folder_name))
        filename = os.path.join(filename, str(file_name) + ".txt")
        # print(file_name)

        target = os.path.join("data", "test")
        target = os.path.join(target, str(folder_name))
        if not os.path.exists(target):
            os.mkdir(target)
        target = os.path.join(target, str(file_name) + ".txt")

        shutil.move(filename, target)
