import numpy as np
import numpy.random as rv
import numpy.linalg as lin
import logging
from quadoption import args_parser
import itertools

args = args_parser()


def select_clients(currcli_loss, p_dat, it):
    size = args.num_users * args.frac
    selidxs_freq = np.zeros(args.num_users)
    p_dat = np.ndarray.flatten(p_dat)

    for ii in range(it + 1):
        if args.seltype == 'rand':
            idxs_users = np.random.choice(args.num_users, p=p_dat, size=int(size),
                                          replace=True)

        elif args.seltype == 'powd':
            rnd_idx = np.random.choice(args.num_users, size=args.powd, replace=False, p=p_dat)
            repval = list(zip([currcli_loss[i] for i in rnd_idx], rnd_idx))
            repval.sort(key=lambda x: x[0], reverse=True)
            rep = list(zip(*repval))
            idxs_users = rep[1][:int(size)]

        if ii != it:
            for j in idxs_users:
                selidxs_freq[j] += 1

    return idxs_users, selidxs_freq / it


def get_rhos2(granul):
    """
    Get a close proxy of the theoretical rho values by grid search of all possible $w$s within a certain range
    :param granul: the granularity of the search
    :return: rhobar, rhotilde (proxies of the theoretical values)
    """

    # First get the approximate area we want to search by the maximum l2-distance globmin-locmin-initialpoint

    # max_{k}|w^*-w_k^*|:
    max_globlocdist = np.max([lin.norm(glob_min - loc_fmin[i], ord=2) for i in range(args.num_users)])
    max_locmininit, max_globmininit = -np.inf, -np.inf

    for __ in range(1000):  # trial for different initial points
        init_tmp = rnd.randn(1, args.dim) * 100
        max_locmininittmp = np.max([lin.norm(loc_fmin[i] - init_tmp, ord=2) for i in range(args.num_users)])

        if max_locmininittmp > max_locmininit:
            max_locmininit = max_locmininittmp  # max_{k}|w^0-w_k^*|

        max_globmininittmp = lin.norm(glob_min - init_tmp, ord=2)
        if max_globmininittmp > max_globmininit:
            max_globmininit = max_globmininittmp  # max_{k}|w^0-w^*|

    area = max(max_globmininit, max_locmininit) + max_globlocdist
    grid_dict = {dim_i: np.hstack([glob_min[dim_i] - np.arange(0, round(area) / 2, granul), glob_min[dim_i] +
                                   np.arange(0, round(area) / 2, granul)]) for dim_i in range(args.dim)}

    # Get rho values by grid search
    rb, rt = np.inf, -np.inf
    it = 0
    all_it = len(grid_dict[0]) ** args.dim

    for i0, i1, i2, i3, i4 in itertools.product(grid_dict[0], grid_dict[1], grid_dict[2], grid_dict[3], grid_dict[4]):

        grid_x_temp = [[i0, i1, i2, i3, i4]]

        grid_loss = callocloss(np.ndarray.flatten(np.transpose(grid_x_temp)), all_mult_mat, x_mult_opt)
        __, selidx_freq = select_clients(grid_loss, p_dat, it=10000)
        rb_numi = (sum(np.multiply(selidx_freq, np.array(grid_loss) - np.array(loc_fmin)))) / \
                  (args.frac * args.num_users)
        rb_loss = sum(np.multiply(grid_loss, p_dat)) - sum(np.multiply(p_dat, loc_fmin))  # Global loss F(w^i)

        if rb >= rb_numi / rb_loss:
            rb = rb_numi / rb_loss

        rt_numi = sum(np.multiply(selidx_freq, np.array(local_optloss) - np.array(loc_fmin))) / \
                  (args.frac * args.num_users)
        if rt <= rt_numi / rhotilde_deno:
            rt = rt_numi / rhotilde_deno

        logging.info('it{} out of {}, rhobar:{}_rhot/rhob:{}'.format(it, all_it, rb, rt / rb))
        it += 1

    return rb, rt


def callocloss(x, H, e):
    ans = []

    if len(np.shape(x)) == 1 or np.shape(x)[0] == 1:
        for i in range(args.num_users):
            term1 = np.matmul(np.matmul(x, H[i]), np.transpose(x)) / 2
            term2 = -np.matmul(x, np.transpose(e[i][:]))
            term3 = np.matmul(np.matmul(e[i][:], lin.inv(H[i])), np.transpose(e[i][:])) / 2

            ans.append(term1 + term2 + term3)

    else:
        for i in range(args.num_users):
            term1 = np.matmul(np.matmul(x[i][:], H[i]), np.transpose(x[i][:])) / 2
            term2 = -np.matmul(x[i][:], np.transpose(e[i][:]))
            term3 = np.matmul(np.matmul(e[i][:], lin.inv(H[i])), np.transpose(e[i][:])) / 2

            ans.append(term1 + term2 + term3)

    return ans


# START MAIN
if __name__ == '__main__':

    logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)

    size = args.num_users * args.frac
    rnd = rv.RandomState(args.seed)
    x_mult_opt = rnd.random_sample((args.num_users, args.dim)) * 100
    x_mult_mat = rnd.random_sample(args.num_users) ** args.high

    all_mult_mat = {}
    loc_min, loc_fmin = [], []

    for i in range(args.num_users):
        all_mult_mat[i] = np.diag(np.full(args.dim, x_mult_mat[i]))
        loc_min.append(np.matmul(x_mult_opt[i][:], lin.inv(all_mult_mat[i])))  # x_k^*

    # Get Datasize
    if args.eq == 1:
        p_dat = np.ones(args.num_users) / args.num_users

    else:
        p_dat = rnd.power(args.alpha, args.num_users)
        p_dat = p_dat / sum(p_dat)

    opt1 = np.sum(p_dat * np.array(x_mult_mat))
    opt2 = np.matmul(p_dat.T, x_mult_opt)
    glob_min = np.matmul(opt2, lin.inv(np.diag(np.full(args.dim, opt1))))  # x^*

    local_optloss = callocloss(glob_min, all_mult_mat, x_mult_opt)
    loc_fmin = callocloss(loc_min, all_mult_mat, x_mult_opt)
    glob_loss = np.sum(np.multiply(p_dat, local_optloss))

    disto_glob, x_global = [], []
    cli_loss, currcli_loss, global_loss = [], [], []
    rhobarest_dat, rhotildest_dat = [], []
    rhotilde_deno = sum(np.multiply(local_optloss, p_dat)) - sum(np.multiply(loc_fmin, p_dat))

    rhobar_theo, rhotilde_theo = get_rhos2(100)
