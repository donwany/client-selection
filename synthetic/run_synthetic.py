import numpy as np
from optimizer import *
from FedAvg import *

from matplotlib import rcParams
import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.pylab as pylab

train_data_dir = './data/'
test_data_dir = './data/'

c_t = cm.get_cmap('tab10')
lr = 0.05
bs = 50
le = 30
total_rnd = 800

sample_ratio = 1
etamu = 0

seltype_keys = {'rand': ('rand', 2, 'k', '-'),
                'powd2': ('pow-d', sample_ratio * 2, c_t(3), '-.'),
                'powd5': ('pow-d', sample_ratio * 10, c_t(0), '--')
                }

run_keys = ['rand', 'powd2', 'powd5']
ftsize = 45 # 90
params = {'legend.fontsize': ftsize,
          'axes.labelsize': ftsize,
          'axes.titlesize': ftsize,
          'xtick.labelsize': ftsize,
          'ytick.labelsize': ftsize}

pylab.rcParams.update(params)
lw = 8
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.labelweight'] = 'bold'

plt.figure(figsize=(16, 14.5))
plt.subplots_adjust(right=1.1, top=0.9)
rcParams['axes.titlepad'] = 14

all_sel_freq = []
for key in run_keys:
    np.random.seed(12345)
    sel_freq = np.zeros(30)
    seltype, powd, color, lstyle = seltype_keys[key]

    opt = FedAvg(lr, bs, le, seltype, powd, train_data_dir, test_data_dir, sample_ratio)
    errors, local_errors = list(), list()
    for rnd in range(total_rnd):
        if rnd == 300 or rnd == 600:
            opt.lr /= 2
        Delta, workers = opt.local_update(local_errors)
        for i in workers:
            sel_freq[i] += 1

        opt.aggregate(Delta)
        error, local_errors = opt.evaluate()
        errors.append(error)

    all_sel_freq.append(sel_freq)
    if seltype == 'rand':
        p_label = seltype
    else:
        p_label = seltype + ', d={}'.format(powd)

    plt.plot(errors, lw=lw, color=color, ls=lstyle, label=p_label)

plt.ylabel('Global loss')
plt.xlabel('Communication round')
legend_properties = {'weight': 'bold'}
plt.xticks()
plt.yticks()
plt.legend(loc=1)
plt.grid()
plt.title('K=30, m={}'.format(sample_ratio))
plt.show()

if __name__ == '__main__':
    pass
