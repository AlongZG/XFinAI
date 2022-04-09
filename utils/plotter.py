import matplotlib.pyplot as plt
import sys

sys.path.append('../')

from utils import path_wrapper
import xfinai_config

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

color = ['#dc2624', '#2b4750', '#45a0a2', '#e87a59',
         '#7dcaa9', '#649E7D', '#dc8018', '#C89F91',
         '#6c6d6c', '#4f6268', '#c7cccf']
rc_params = {'font.sans-serif': 'SimHei',
             'axes.unicode_minus': False}


def plot_loss(losses, epoch_num, loss_name, model_name, future_name):
    plt.figure(figsize=[6, 3], dpi=100)
    plt.plot(list(range(epoch_num)), losses, 'r--', label=loss_name)
    plt.legend()
    plt.title(f"{loss_name} ")
    plt.xlabel('轮数')
    plt.ylabel('损失函数')
    plt.subplots_adjust(bottom=0.15)

    losses_dir = path_wrapper.wrap_path(f"{xfinai_config.losses_path}/{future_name}/{model_name}")
    plt.savefig(f"{losses_dir}/{loss_name}.png")
