import json
import numpy as np
import matplotlib.pyplot as plt

def get_log(log_path:str)->dict:
    with open(log_path, 'r') as f:
        d = json.load(f)
    return d

def get_log_train_values(log_dict):
    return [(k, v) for k, v in log_dict["train"].items()]

# def get_log_train_values(log_dict):
#     return [(k, v) for k, v in log_dict.items()]

def get_log_val_values(log_dict):
    return [(k, v) for k, v in log_dict["val"].items()]

def get_min_with_loc(values):
    smallest = float('inf')
    smallestIdx = -1
    for k, v in values:
        k = int(k)
        v = float(v)
        if v < smallest:
            smallest = v
            smallestIdx = k
    return smallest, smallestIdx

def get_last_completed_epoch(d):
    epochs = [int(k) for k in d["train"].keys()]
    return max(epochs)

# def get_last_completed_epoch(d):
#     epochs = [int(k) for k in d.keys()]
#     return max(epochs)

def graph_losses(train_values, val_values, save_path="loss.png", title="MSE Loss for ShadowFormer"):
    x = [k for k, _ in train_values]
    y = [v for _, v in train_values]
    xTicks = [t for t in range(0, len(x), 5)]
    plt.plot(x, y, label="train")
    xVal = [k for k, _ in val_values]
    yVal = [v for _, v in val_values]
    plt.scatter(xVal, yVal, color="red", label="val")
    plt.xticks(xTicks)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.savefig(save_path)

if __name__=="__main__":
    title = "Pseudolog L1 Loss for ShadowFormer"
    log_path="/work/SuperResolutionData/ShadowRemovalResults/ShadowFormer/gamma_rgb1/ShadowFormer_istd/gamma_rgb1.json"
    chart_save_path="results/train_curves/gamma_rgb.png"
    #log_path = "/work/SuperResolutionData/ShadowRemovalResults/ShadowFormer/PNG/ShadowFormer_istd/control.json"
    #log_path = "/work/SuperResolutionData/ShadowRemovalResults/ShadowFormer/ShadowFormer_pseudolog/ShadowFormer_istd/pseudolog3.json"
    d = get_log(log_path)
    train_values = get_log_train_values(d)
    val_values = get_log_val_values(d)
    print("Epochs completed: ", get_last_completed_epoch(d))
    print("Minimum loss and minimum epoch: ", get_min_with_loc(train_values))
    graph_losses(train_values, val_values, save_path=chart_save_path, title=title)
