from utils import read_data
import numpy as np


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex, axis=1).reshape(ex.shape[0], 1)
    return ex / sum_ex


class FederatedOptimizer(object):
    def __init__(self, lr, bs, le, seltype, powd, train_data_dir, test_data_dir, sample_ratio):
        _, _, self.train_data, self.test_data = read_data(train_data_dir, test_data_dir)
        self.size = len(self.train_data.keys()) # 30
        self.dim = np.array(self.train_data['f_00000']['x']).shape[1] # 61
        self.central_parameter = np.zeros((self.dim, 10)) # (61,10) 61 rows, 10 columns
        self.init_central = self.central_parameter + 0 # (61, 10)
        self.local_parameters = np.zeros([self.size, self.dim]) # (30, 61)

        self.powd = powd
        self.sample_ratio = sample_ratio
        self.bs = bs
        self.seltype = seltype
        self.le = le
        self.lr = lr
        self.ratio = self.get_ratio()
        self.local_losses = []
        self.iter = 0
        self.print_flg = True

    def get_ratio(self):
        total_size = 0
        ratios = np.zeros(self.size)
        for i in range(self.size):
            key = 'f_{0:05d}'.format(i)
            local_size = np.array(self.train_data[key]['x']).shape[0]
            ratios[i] = local_size
            total_size += local_size

        return ratios / total_size

    def loss(self, A, y):
        x = self.central_parameter
        y_hat = np.zeros((len(y), 10))
        y_hat[np.arange(len(y)), y.astype(int)] = 1
        loss = - np.sum(y_hat * np.log(softmax(A @ x))) / A.shape[0]

        return loss

    def compute_gradient_template(self, x, i):
        uname = 'f_{0:05d}'.format(i)
        A = np.array(self.train_data[uname]['x'])
        y = np.array(self.train_data[uname]['y'])

        sample_idx = np.random.choice(A.shape[0], self.bs)
        a = A[sample_idx]
        targets = np.zeros((self.bs, 10))
        targets[np.arange(self.bs), y[sample_idx].astype('int')] = 1

        grad = - a.T @ (targets - softmax(a @ x)) / self.bs
        grad[:61] += 10e-4 * self.central_parameter[:61]

        return grad

    def evaluate(self):
        glob_losses, local_losses = [], []
        for i in range(self.size):
            uname = 'f_{0:05d}'.format(i)
            A = np.array(self.train_data[uname]['x'])
            y = np.array(self.train_data[uname]['y'])
            glob_losses.append(self.loss(A, y) * self.ratio[i])
            local_losses.append(self.loss(A, y))

        glob_losses = np.array(glob_losses)

        return np.sum(glob_losses), local_losses

    def select_client(self, loc_loss):
        if not loc_loss:
            idxs_users = np.random.choice(self.size, size=self.sample_ratio, replace=False)

        else:
            if self.seltype == 'rand':
                idxs_users = np.random.choice(self.size, p=self.ratio, size=self.sample_ratio, replace=True)

            elif self.seltype == 'pow-d':
                rnd_idx = np.random.choice(self.size, p=self.ratio, size=self.powd, replace=False)
                repval = list(zip([loc_loss[i] for i in rnd_idx], rnd_idx))
                repval.sort(key=lambda x: x[0], reverse=True)
                rep = list(zip(*repval))
                idxs_users = rep[1][:int(self.sample_ratio)]

        return idxs_users


# if __name__ == '__main__':
#     train_data_dir = './data/'
#     test_data_dir = './data/'
#
#     fd = FederatedOptimizer(lr=0.04,
#                             bs=50,
#                             le=0.01,
#                             seltype='pow-d',
#                             powd=9,
#                             train_data_dir=train_data_dir,
#                             test_data_dir=test_data_dir,
#                             sample_ratio=0.2)

    # print(fd.get_ratio())
    # print(fd.ratio)
    # print(fd.init_central)
    # print(fd.train_data['f_00010'])
    # print(fd.loss(A=np.array(fd.train_data['f_00010']['x']), y=np.array(fd.train_data['f_00010']['y'])))
    # for i in range(1, 30):
    #     uname = 'f_{0:05d}'.format(i)
    #     print(uname)
    #     print(fd.compute_gradient_template(x=np.array(fd.train_data[uname]['x'], i)))

    # _, local_loss = fd.evaluate()
    # print(fd.select_client(loc_loss=local_loss))

    # loc_loss = [i for i in np.random.sample(10)]
    # rnd_idx = np.random.choice(10, p=[0.1] * 10, size=9, replace=False)
    # repval = list(zip([loc_loss[i] for i in rnd_idx], rnd_idx))
    # repval.sort(key=lambda x: x[0], reverse=True)
    # rep = list(zip(*repval))
    # idxs_users = rep[1][:int(9)]
    # print(rnd_idx)
    # print(loc_loss)
    # print(repval)
    # print(rep)
    # print(idxs_users)
