import sys
import logging
import itertools

import numpy as np 
import scipy.optimize as optimize
import matplotlib.pyplot as plt 

def sub_tensors(ts):
    '''
    return iterable for all sub_tensors of ts 
    '''
    # which==0 for left half, !=0 for right half
    def take_which_part(t, where, which):
        half = t.shape[where] // 2
        t2 = t.swapaxes(0, where)
        t2 = t2[:half] if which == 0 else t2[half:]
        return t2.swapaxes(0, where)

    ndims = len(ts.shape)
    for i in range(2**ndims):
        initA = ts
        for where in range(ndims):
            initA = take_which_part(initA, where, i & (1 << where))
        yield initA

def matrix_views(ts):
    '''
    return iterable for all unfoldings of ts 
    '''
    shape = ts.shape
    ndims = len(shape)
    for where in range(ndims):
        yield ( ts
                .swapaxes(0, where)
                .reshape(shape[where], ts.size//shape[where])
              )

#def awe_one_dim():
#    alpha = 2
#    def f(x):
#        return (
#                  1/(1**alpha)*np.sin(1*x)
#                + 1/(2**alpha)*np.sin(2*x)
#                + 1/(3**alpha)*np.sin(3*x)
#                + 1/(4**alpha)*np.sin(4*x)
#               )
#    def g(x):
#        return sum(1/k**alpha * np.sin(k*x) for k in range(1, 1000))
#
#    print(g(23))


def assemble_tss(eight_tss):
    shape = 2 * np.array(eight_tss[0].shape)

    res = np.zeros(np.prod(shape)).reshape(shape)
    for part, one_ts in zip(sub_tensors(res), eight_tss):
        part[...] = one_ts

    return res

def load_sum_of(first, last):
    ts = 0
    for i in range(first, last):
        ts += np.load("sample{}.data.npy".format(i))
    return ts
    

def create_monster():
    eight_tss = list(range(8))

    eight_tss[0b000] = load_sum_of(1, 7)
    eight_tss[0b111] = load_sum_of(7, 12)
    eight_tss[0b010] = load_sum_of(12, 17)
    eight_tss[0b100] = load_sum_of(17, 21)

    eight_tss[0b101] = load_sum_of(22, 22)
    eight_tss[0b110] = load_sum_of(23, 23)
    eight_tss[0b001] = load_sum_of(24, 24)
    eight_tss[0b011] = load_sum_of(25, 25)

    res = assemble_tss(eight_tss)
    return res


def awesome_function():
    k = 2
    #alpha = 1.55
    #alpha = 1.25
    #alpha = 1.60
    alpha = 1.50
    def f(x, y, z):
        pass
        #return 1/k*np.cos(x)*np.sin(y)*np.cos(z) + 1/k*np.sin(x)*

    def g(x, y, z):
        #return sum(1/kx**alpha * 1/ky**alpha * 1/kz**alpha * np.sin(kx*x) * np.cos(ky*y) * np.cos(kz*z)
        #return sum(1/(kx*ky*kz)**alpha * np.sin(kz*z) * np.cos(kx*x) * np.cos(ky*y)
        return sum((y*y*y*y + x*x)/(kz*ky*kx)**alpha * np.sin(kz*z) * np.sin(kz*z) * np.sin(kz*z)
                for kx in range(2, 20)
                for ky in range(2, 20)
                for kz in range(2, 20)
                )

    l = np.linspace(0, np.pi, 40)
    xs, ys, zs = np.meshgrid(l, l, l)
    tensor = g(xs, ys, zs)

    np.save("sample25.data", tensor)

def test_sample_tensor():
    ts = 0
    
    cur_last = 12
    for i in range(1, cur_last+1):
        ts += np.load("sample{}.data.npy".format(i))

    for view in matrix_views(ts):
        _, sigs, _ = np.linalg.svd(view)
        plt.semilogy(sigs)
        plt.show()
        print(sigs)

    np.save("awesome_sample.data", ts)

def remain_estimate2(ts):
    def fit_func_pattern(x, a, b, c):
        return a * (x**b) * np.exp(-c*x)

    remain_square_sum_list = []
    for view in matrix_views(ts):
        _, sigs, _ = np.linalg.svd(view)
        #print(sigs)
        remain_square_sum = (sigs * sigs)[1:][::-1].cumsum()[::-1]

        min_sig_to_use = 1e-10
        try:
            my_end = next(i for i, sig in enumerate(sigs) if sig < min_sig_to_use) - 1
        except StopIteration:
            my_end = len(sigs) - 1

        if my_end < 5:
            logging.error("big sig only {} not enough".format(my_end))
            sys.exit(-1)

        remain_square_sum = remain_square_sum[:my_end]

        def func_to_minimize(abc):
            return sum( (fit_func_pattern(x, abc[0], abc[1], abc[2])-remain_square_sum[x-1])**2
                    for x in np.arange(my_end) + 1 )

        def fun_gen(x):
            return lambda abc: fit_func_pattern(x, abc[0], abc[1], abc[2]) - remain_square_sum[x-1]

        cons = []
        for x in np.arange(my_end) + 1:
            cons.append( {
                'type': 'ineq',
                'fun': fun_gen(x),
                })

        res = optimize.minimize(
                  func_to_minimize
                , x0=(5, 5, 5)
                , method='SLSQP'
                , constraints=cons
                )

        print (res)

        a, b, c = res.x
        remain_square_sum_list.append(lambda x: fit_func_pattern(x, a, b, c))

    return lambda x: (remain_square_sum_list[0](x[0])
                    + remain_square_sum_list[1](x[1])
                    + remain_square_sum_list[2](x[2])
                    )



def remain_estimate(ts):
    def fit_func_pattern(x, a, b, c):
        return a + np.log(x)*b - c*x

    def real_fit_func_pattern(x, a, b, c):
        return a * (x**b) * np.exp(-c*x)

    remain_square_sum_list = []
    for view in matrix_views(ts):
        _, sigs, _ = np.linalg.svd(view)
        #print(sigs)
        remain_square_sum = (sigs * sigs)[1:][::-1].cumsum()[::-1]

        min_sig_to_use = 1e-10
        try:
            my_end = next(i for i, sig in enumerate(sigs) if sig < min_sig_to_use) - 1
        except StopIteration:
            my_end = len(sigs) - 1

        if my_end < 3:
            if True:
                # my_end == 0 ad hoc 
                remain_square_sum_list.append(lambda x: remain_square_sum[0] * np.exp(-16.0*(x-1)))
                continue
            else:
                logging.error("big sig only {} not enough".format(my_end))
                sys.exit(-1)
        
        remain_square_sum = remain_square_sum[:my_end]
        logged_sum = np.log(remain_square_sum)

        paras, _ = optimize.curve_fit(
                  fit_func_pattern
                , np.arange(my_end, dtype=np.float64) + 1
                , logged_sum
                , p0=(5, 5, 5)
                )
        a, b, c = paras

        xdata = np.arange(my_end) + 1
        bigger = max(logged_sum - fit_func_pattern(xdata, *paras))
        #print ("BIGGER:", bigger)
        a += bigger

        #plt.plot(xdata, np.log(remain_square_sum))
        #plt.plot(xdata, fit_func_pattern(xdata, a, b, c))
        #plt.show()

        #plt.plot(xdata, remain_square_sum)
        #plt.plot(xdata, (lambda x: real_fit_func_pattern(x, np.exp(a), b, c))(xdata))
        #plt.show()

        #print ("orig:   ", logged_sum)
        #print ("fitted: ", fit_func_pattern(xdata, a, b, c))

        #print (remain_square_sum)
        #print ((lambda x: a * (x**b) * np.exp(-c*x))(xdata))

        remain_square_sum_list.append(lambda x: real_fit_func_pattern(x, np.exp(a), b, c))

    return lambda x: (remain_square_sum_list[0](x[0])
                    + remain_square_sum_list[1](x[1])
                    + remain_square_sum_list[2](x[2])
                    )


def best_storage_without_divide(ts, eps):
    remain_square_sum_list = []
    for view in matrix_views(ts):
        _, sigs, _ = np.linalg.svd(view)
        remain_square_sum = (sigs * sigs)[1:][::-1].cumsum()[::-1]
        remain_square_sum_list.append(np.append(remain_square_sum, 0))

    im, jm, km = ts.shape 
    res = []
    for i in range(1, 1+im):
        for j in range(1, 1+jm):
            for k in range(1, 1+km):
                x = remain_square_sum_list[0][i-1]
                y = remain_square_sum_list[1][j-1]
                z = remain_square_sum_list[2][k-1]
                if x + y + z <= eps:
                    res.append( (i*j*k
                            + i*ts.shape[0]
                            + j*ts.shape[1]
                            + k*ts.shape[2]
                            , i, j, k) )

    storage, i, j, k = min(res)
    #print ("best_ks_without_divide2: eps:{}, storage:{}, i={},j={},k={}".format(eps, storage, i, j, k))
    ijk = (i, j, k)
    print ("best_ks_without_divide2: eps:{}, storage:{}, ijk:{}, ts:{}".format(eps, storage, ijk, ts.shape))
    return storage


# return storage
def best_storage(ts, total_eps, current_level=0, smallest_cube_to_divide=10):

    #if min(ts.shape) <= smallest_cube_to_divide:
    if current_level == 1 or min(ts.shape) <= smallest_cube_to_divide:
        logging.debug("level:{}, already smallest".format(current_level))
        return best_storage_without_divide(ts, total_eps)

    need_divide, storage, epss = inspect_one_level(ts, total_eps)
    print (sum(epss))
    if not need_divide:
        logging.debug("level:{}, no divide: {}".format(current_level, storage))
        return storage
    else:
        storage_with_divide = 0
        for i, sub_tensor in enumerate(sub_tensors(ts)):
            storage_with_divide += best_storage(sub_tensor, epss[i], current_level + 1, smallest_cube_to_divide)
    
        logging.debug("level:{}, divide: {}".format(current_level, storage_with_divide))
        return storage_with_divide


# return (need_divide, storage, epss=None)
def inspect_one_level(ts, eps):
    remain_estimate_lst = []
    for s in sub_tensors(ts):
        remain_estimate_lst.append( remain_estimate(s) )

    ndims = len(ts.shape)
    def storage_func(ks_lst):
        kss = [ ks_lst[i*ndims:(i+1)*ndims] for i in range(2**ndims) ]
        return sum(np.prod(ks) + sum(ks * np.array(ts.shape) / 2) for ks in kss)

    def total_remain_estimate(ks_lst):
        return eps - sum(func(ks) for ks, func in
                zip([ ks_lst[i*ndims:(i+1)*ndims] for i in range(2**ndims) ],
                    remain_estimate_lst))
            
    cons = ({
        'type': 'ineq',
        'fun': total_remain_estimate,
        })
    
    res = optimize.minimize(
              storage_func
            , x0=[2]*(ndims*2**ndims)
            , method='SLSQP'
            , bounds=[(1.0, 1.0*n//2) for n in ts.shape] * (2**ndims)
            , constraints=cons
            )

    #print (res)

    values = np.ceil(res.x)
    values = values.astype(int)
    kss = [ values[i*ndims:(i+1)*ndims] for i in range(2**ndims) ]
    epss = [ func(ks) for func, ks in zip(remain_estimate_lst, kss) ]

    # epss should be normalized so that sum(epss) == 1
    epss = [ eps/sum(epss) for eps in epss ]


    storage = sum([ i*j*k + i*ts.shape[0]//2 + j*ts.shape[1]//2 + k*ts.shape[2]//2 for i, j, k in kss ])
    print ("kss:{}".format(kss))
    print ("total_remain_estimate:{}".format(total_remain_estimate(values)))
    print ("total storage:{}".format(storage))
    print ("epss: {}".format(epss))

    print(kss[0])
    
    storage_without_divide = best_storage_without_divide(ts, eps)
    logging.debug("    shape:{},with_divide:{},without_divide:{}".format(ts.shape, storage, storage_without_divide))

    return (True, storage, epss) if storage < storage_without_divide else (False, storage_without_divide, None)


def load_ts_and_do_something():
    #ts = np.load("awesome_sample.data.npy")
    ##ts = next(sub_tensors(ts))
    #inspect_one_level(ts, 1000)
    #print ("forb norm squared:{}".format( np.linalg.norm(np.ravel(ts)) ** 2 ))
    ##print("best:{}".format(best_storage(ts, 100)))

    ts = create_monster()
    inspect_one_level(ts, 1000)


if __name__ == '__main__':
    #test_sample_tensor()
    #awesome_function()
    #logging.basicConfig(level=logging.DEBUG)
    load_ts_and_do_something()
