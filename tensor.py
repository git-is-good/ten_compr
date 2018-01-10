import numpy as np 
import scipy.optimize as optimize
import matplotlib.pyplot as plt 

def sub_tensors(tensorA):
    '''
    return iterable for all sub_tensors of tensorA 
    '''
    # which==0 for left half, !=0 for right half
    def take_which_part(t, where, which):
        half = t.shape[where] / 2
        t2 = t.swapaxes(0, where)
        t2 = t2[:half] if which == 0 else t2[half:]
        return t2.swapaxes(0, where)

    ndims = len(tensorA.shape)
    for i in range(2**ndims):
        initA = tensorA
        for where in range(ndims):
            initA = take_which_part(initA, where, i & (1 << where))
        yield initA

def matrix_views(tensorA):
    '''
    return iterable for all unfoldings of tensorA 
    '''
    shape = tensorA.shape
    ndims = len(shape)
    for where in range(ndims):
        yield ( tensorA
                .swapaxes(0, where)
                .reshape(shape[where], tensorA.size/shape[where])
              )


def remain_estimate_for_a_mode(view):
    '''
    Keep k sigs, approximate remain_square_sum as a * (k**b) * exp(-ck)
    return estimated function as lambda
    '''
    def fit_func_pattern(x, a, b, c): return a * (x**b) * np.exp(-c * x)
    _, sigs, _ = np.linalg.svd(view)
    remain_square_sum = (sigs * sigs)[1:][::-1].cumsum()[::-1]
    remain_square_sum = np.append(remain_square_sum, 0)

    paras, _ = optimize.curve_fit(fit_func_pattern,
            np.arange(len(sigs), dtype=np.float64) + 1, remain_square_sum) #, p0=(1, 1, 1))

    print remain_square_sum
    print sigs
    #print paras
    xdata = np.arange(len(sigs)) + 1
    plt.plot(xdata, remain_square_sum)
    plt.plot(xdata, fit_func_pattern(xdata, *paras))
    plt.show()
    return lambda x: fit_func_pattern(x, *paras)

def best_ks_without_divide(tensorA, eps = 10):
    '''
    return [bestks...], regard tensorA as a whole, no partition
    '''
    shape = tensorA.shape
    ndims = len(shape)
    remain_estims = []
    for view in matrix_views(tensorA):
        f = remain_estimate_for_a_mode(view)
        remain_estims.append(f)

    def final_remain_estim_constraint(ks):
        return eps - sum(f(k) for f, k in zip(remain_estims, ks)) 

    def storage_func(ks):
        return sum(np.prod(ks) + ks * np.array(shape))

    cons = ({
        'type': 'ineq',
        'fun': final_remain_estim_constraint,
        })

    res = optimize.minimize(storage_func, x0=shape, method='SLSQP',
           bounds=(zip([1]*ndims, shape)), constraints=cons)
    print res
    print final_remain_estim_constraint(res.x)


def best_storage(tensorA):
    '''
    return storage
    '''
    storage_with_divide = 0
    for sub_tensor in sub_tensors(tensorA):
        sub_storage = best_storage(sub_tensor)
        storage_with_divide += sub_storage

    # compute the optimal storage cost without divide
    ks_without_divide = best_ks_without_divide(tensorA)
    storage_without_divide = ( np.prod(ks_without_divide) 
            + np.sum(ks_without_divide * np.array(tensorA)) )

    return min(storage_with_divide, storage_without_divide)


def main():
    #a = np.arange(4 * 4 * 4).reshape(4, 4, 4)
    #for sa in sub_tensors(a):
    #    print sa
    #    print '****'

    #b = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    #for mb in matrix_views(b):
    #    print b
    #    print '****'

    #c = np.random.uniform(size=(10, 10))
    #remain_estimate_for_a_mode(c)

    #d = np.random.uniform(size=(50,50,50))
    #best_ks_without_divide(d)
   
    def gaussian(x, y, z, alpha):
        return np.exp(
                (x - alpha) ** 2
              + (y - alpha) ** 2
              + (z - alpha) ** 2
              )
    def many_gaussian(x, y, z):
        return ( gaussian(x, y, z, 0.3)
               + gaussian(x, y, z, 0.5)
               + gaussian(x, y, z, 0.8)
               + gaussian(x, y, z, 0.9)
               )

    def fourier(x, y, z):
        return 100 * np.cos(x) * np.sin(y) * np.cos(z)

    x = np.arange(1, 5, 0.1)
    y = np.arange(1, 5, 0.1)
    z = np.arange(1, 5, 0.1)
    xx, yy, zz = np.meshgrid(x, y, z)
    e = fourier(xx, yy, zz)
    best_ks_without_divide(e)


#    e = np.fromfunction(gaussian, (10, 10, 10), dtype=int)
#    best_ks_without_divide(e)


if __name__ == '__main__':
    main()
