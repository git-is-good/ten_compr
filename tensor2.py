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
    
    cur_last = 25
    for i in range(1, cur_last+1):
        ts += np.load("sample{}.data.npy".format(i))

    for view in matrix_views(ts):
        _, sigs, _ = np.linalg.svd(view)
        plt.semilogy(sigs)
        plt.show()
        print(sigs)

    np.save("awesome_sample.data", ts)

def remain_estimate(ts):
    def fit_func_pattern(x, a, b, c):
        #return a * (x**b) * np.exp(-c*x)
        return a + np.log(x)*b - c*x
    
    remain_square_sum_list = []
    for view in matrix_views(ts):
        _, sigs, _ = np.linalg.svd(view)
        print(sigs)
        remain_square_sum = (sigs * sigs)[1:][::-1].cumsum()[::-1]

        # ad hoc
        my_end = 15
        remain_square_sum = remain_square_sum[:my_end]

        paras, _ = optimize.curve_fit(
                  fit_func_pattern
                , np.arange(my_end, dtype=np.float64) + 1
                , np.log(remain_square_sum)
                , p0=(5, 5, 5)
                )

        #xdata = np.arange(my_end) + 1
        #plt.plot(xdata, np.log(remain_square_sum))
        #plt.plot(xdata, fit_func_pattern(xdata, *paras))
        ##plt.semilogy(xdata, remain_square_sum)
        ##plt.semilogy(xdata, fit_func_pattern(xdata, *paras))
        #plt.show()

        a, b, c = paras
        remain_square_sum_list.append(lambda x: a * (x**b) * np.exp(-c*x))

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
    print ("best_ks_without_divide2: storage:{}, i={},j={},k={}".format(storage, i, j, k))
    return storage


# return is_intact ?
def best_storage(ts, current_level=0, smallest_cube_to_divide=5, total_eps=1.0):
    if min(ts.shape) < smallest_cube_to_divide:
        logging.debug("level:{}, already smallest".format(current_level))
        return np.prod(ts.shape)

    howmany_part = 2**len(ts.shape)
    storage_with_divide = 0
    for sub_tensor in sub_tensors(ts):
        sub_storage = best_storage(sub_tensor, current_level + 1, smallest_cube_to_divide, total_eps // howmany_part)
        storage_with_divide += sub_storage

    storage_without_divide = best_storage_without_divide(ts, total_eps)
    logging.debug("level:{}, with_divide:{}, without_divide:{}, {}".format(
                current_level, storage_with_divide, storage_without_divide, 
                "no more divide" if storage_without_divide <= storage_with_divide else "divide further"))
    return min(storage_with_divide, storage_without_divide)


def inspect_one_level(ts):
    remain_es = remain_estimate(ts)
    eps = 1.0

    remain_estimate_lst = []
    for s in sub_tensors(ts):
        remain_estimate_lst.append( remain_estimate(s) )

    ndims = len(ts.shape)
    def storage_func(ks_lst):
        kss = [ ks_lst[i*ndims:(i+1)*ndims] for i in range(2**ndims) ]
        return sum(sum(np.prod(ks) + ks * np.array(ts.shape)) for ks in kss)

    def total_remain_estimate(ks_lst):
        return eps - sum(func(ks) for ks, func in
                zip([ ks_lst[i*ndims:(i+1)*ndims] for i in range(2**ndims) ],
                    remain_estimate_lst))
            
    cons = ({
        'type': 'ineq',
        'fun': total_remain_estimate,
        })
    
    res = optimize.minimize(storage_func, x0=[2]*(ndims*2**ndims),
            method='SLSQP',
            bounds=[(1.0, 1.0*n//2) for n in ts.shape] * (2**ndims),
            constraints=cons)

    print (res)

    values = np.ceil(res.x)
    kss = [ values[i*ndims:(i+1)*ndims] for i in range(2**ndims) ]
    storages = [ i*j*k + i*ts.shape[0] + j*ts.shape[1] + k*ts.shape[2] for i, j, k in kss ]
    print ("total_remain_estimate:{}".format(total_remain_estimate(values)))
    print ("total storage:{}".format(sum(storages)))
    
    best_storage_without_divide(ts, eps)


def load_ts_and_do_something():
    ts = np.load("awesome_sample.data.npy")
    inspect_one_level(ts)


if __name__ == '__main__':
    #test_sample_tensor()
    #awesome_function()
    load_ts_and_do_something()
