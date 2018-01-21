import logging
import itertools

import numpy as np 
import scipy.optimize as optimize
import matplotlib.pyplot as plt 

def sub_tensors(tensorA):
    '''
    return iterable for all sub_tensors of tensorA 
    '''
    # which==0 for left half, !=0 for right half
    def take_which_part(t, where, which):
        half = t.shape[where] // 2
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
                .reshape(shape[where], tensorA.size//shape[where])
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
            np.arange(len(sigs), dtype=np.float64) + 1, remain_square_sum, p0=(100, 10, 10))

    return lambda x: fit_func_pattern(x, *paras)

def remain_estimate(tensorA):
    def fit_func_pattern(x, a, b, c): return a * (x**b) * np.exp(-c * x)

    remain_square_sum_list = []
    for view in matrix_views(tensorA):
        _, sigs, _ = np.linalg.svd(view)
        #print ("Sigs: ", sigs)

        remain_square_sum = (sigs * sigs)[1:][::-1].cumsum()[::-1]
        remain_square_sum = np.append(remain_square_sum, 0)
        paras, _ = optimize.curve_fit(fit_func_pattern,
                np.arange(len(sigs), dtype=np.float64) + 1, remain_square_sum, p0=(5, 5, 5))
        remain_square_sum_list.append(lambda x: fit_func_pattern(x, *paras))

        #xdata = np.arange(len(sigs)) + 1
        #plt.plot(xdata, remain_square_sum)
        #plt.plot(xdata, fit_func_pattern(xdata, *paras))
        #plt.semilogy(xdata, remain_square_sum)
        #plt.semilogy(xdata, fit_func_pattern(xdata, *paras))
        #plt.show()

    return lambda xs: sum(remains(x) for x, remains in zip(xs, remain_square_sum_list))


def best_storage2(tensorA, eps=1.0):
    remain_estimate_lst = []
    for s in sub_tensors(tensorA):
        remain_estimate_lst.append( remain_estimate(s) )

    ndims = len(tensorA.shape)
    def storage_func(ks_lst):
        kss = [ ks_lst[i*ndims:(i+1)*ndims] for i in range(2**ndims) ]
        return sum(sum(np.prod(ks) + ks * np.array(tensorA.shape)) for ks in kss)

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
            bounds=[(1.0, 1.0*n//2) for n in tensorA.shape] * (2**ndims),
            constraints=cons)

    print (res)

    values = np.ceil(res.x)
    kss = [ values[i*ndims:(i+1)*ndims] for i in range(2**ndims) ]
    storages = [ i*j*k + i*tensorA.shape[0] + j*tensorA.shape[1] + k*tensorA.shape[2] for i, j, k in kss ]
    print ("total_remain_estimate:{}".format(total_remain_estimate(values)))
    print ("total storage:{}".format(sum(storages)))

def best_ks_without_divide2(tensorA, eps=1.0):
    remain_square_sum_list = []
    for view in matrix_views(tensorA):
        _, sigs, _ = np.linalg.svd(view)
        remain_square_sum = (sigs * sigs)[1:][::-1].cumsum()[::-1]
        remain_square_sum_list.append(np.append(remain_square_sum, 0))

    im, jm, km = tensorA.shape 
    res = []
    for i in range(1, 1+im):
        for j in range(1, 1+jm):
            for k in range(1, 1+km):
                x = remain_square_sum_list[0][i-1]
                y = remain_square_sum_list[1][j-1]
                z = remain_square_sum_list[2][k-1]
                if x + y + z <= eps:
                    res.append( (i*j*k
                            + i*tensorA.shape[0]
                            + j*tensorA.shape[1]
                            + k*tensorA.shape[2]
                            , i, j, k) )

    storage, i, j, k = min(res)
    print ("best_ks_without_divide2: storage:{}, i={},j={},k={}".format(storage, i, j, k))
    return storage

def best_ks_without_divide2_curve_fit(tensorA, eps=1.0):
    rem = remain_estimate(tensorA)
    
    def total_remain_estimate(ks):
        return eps - rem(ks)

    def storage_func(ks):
        return np.prod(ks) + np.sum(ks*tensorA.shape)
    
    cons = ({
        'type': 'ineq',
        'fun': total_remain_estimate,
        })

    res = optimize.minimize(storage_func, x0=(2.0, 2.0, 2.0),
            method='SLSQP',
            bounds=[(1.0, 1.0*n) for n in tensorA.shape],
            constraints=cons)
    
    print ("best_ks_without_divide2_curve_fit: {}".format(res))
    i, j, k = np.ceil(res.x)
    storage = i*j*k + i*tensorA.shape[0] + j*tensorA.shape[1] + k*tensorA.shape[2]
    print ("best_ks_without_divide2_curve_fit: storage:{}, i={}, j={}, k={}".format(storage, i, j, k))

def best_ks_without_divide(tensorA, eps=1.0):
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

    res = optimize.minimize(storage_func, x0=[2]*ndims,
           method='SLSQP',
           bounds=(zip([1]*ndims, shape)),
           constraints=cons)

    if not res.success:
        print (res)
        print (final_remain_estim_constraint(res.x))

    return res.x


def best_storage(tensorA, current_level=0, smallest_cube_to_divide=5, total_eps=1.0):
    '''
    return storage
    '''
    if min(tensorA.shape) < smallest_cube_to_divide:
        logging.debug("level:{}, already smallest".format(current_level))
        return np.prod(tensorA.shape)

    howmany_part = 2**len(tensorA.shape)
    storage_with_divide = 0
    for sub_tensor in sub_tensors(tensorA):
        sub_storage = best_storage(sub_tensor, current_level + 1, smallest_cube_to_divide, total_eps // howmany_part)
        storage_with_divide += sub_storage

    # compute the optimal storage cost without divide
    ks_without_divide = best_ks_without_divide(tensorA, total_eps)
    storage_without_divide = ( np.prod(ks_without_divide) 
            + np.sum(ks_without_divide * np.array(tensorA.shape)) )

    logging.debug("level:{}, with_divide:{}, without_divide:{}:{}, {}".format(
                current_level, storage_with_divide, storage_without_divide, ks_without_divide,
                "no more divide" if storage_without_divide <= storage_with_divide else "divide further"))

    return min(storage_with_divide, storage_without_divide)

def best_storage3(tensorA, current_level=0, smallest_cube_to_divide=5, total_eps=1.0):
    if min(tensorA.shape) < smallest_cube_to_divide:
        logging.debug("level:{}, already smallest".format(current_level))
        return np.prod(tensorA.shape)

    howmany_part = 2**len(tensorA.shape)
    storage_with_divide = 0
    for sub_tensor in sub_tensors(tensorA):
        sub_storage = best_storage3(sub_tensor, current_level + 1, smallest_cube_to_divide, total_eps // howmany_part)
        storage_with_divide += sub_storage

    storage_without_divide = best_ks_without_divide2(tensorA, total_eps)
    logging.debug("level:{}, with_divide:{}, without_divide:{}, {}".format(
                current_level, storage_with_divide, storage_without_divide, 
                "no more divide" if storage_without_divide <= storage_with_divide else "divide further"))
    return min(storage_with_divide, storage_without_divide)

def test1_useless():
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
        return ( 100 * y * np.cos(x)
               + 88 * x * np.cos(z)
               + 88 * z * np.cos(y)
               )
    x = np.arange(1, 5, 0.1)
    y = np.arange(1, 5, 0.1)
    z = np.arange(1, 5, 0.1)
    xx, yy, zz = np.meshgrid(x, y, z)
    #e = fourier(xx, yy, zz)
    #best_ks_without_divide(e)

def test():
    howbig = 200
    def sing_gen(x):
        return 3.0 * (x**2) * np.exp( -0.019 * x**2 )

    q, _ = np.linalg.qr( np.random.uniform(size=(howbig, howbig)) )

    mymat = q.dot(np.diag( sing_gen(np.arange(howbig, dtype=np.float64)) )).dot(q)
    _, s, _ = np.linalg.svd(mymat)
    #print (s)
    #remain_estimate_for_a_mode(mymat)

    mymat = mymat.reshape(50, 40, 20)

    def gaussian2(x, y, z):
        return (
                0.003 * (2*x+y*z+z*z+x*x+y*y+x+x+y*y+x*y+x*z) * np.sin( -0.019 * (x**2 + y**2 + z**2) )
             +  0.002*x*z*np.cos( -0.023 * (x**2+y**2) )
             +  0.004*y*z*np.cos( -0.013 * (z**2+y**2) )
               )

    def gaussian(x, y, z):
        return np.exp( -0.019 * (x**2 + y**2 + z**2) )
    for i in range(mymat.shape[0]):
        for j in range(mymat.shape[1]):
            for k in range(mymat.shape[2]):
                mymat[i, j, k] = gaussian(i, j, k)

    #print (mymat.shape)
    ##best_ks_without_divide(mymat)
    #res = best_storage(mymat, total_eps=0.8)

    #print ("best_storage:{}".format(res))
    print ("forb norm squared:{}".format( np.linalg.norm(np.ravel(mymat)) ** 2 ))

    #best_storage2(mymat)
    #best_ks_without_divide2(mymat)
    #best_ks_without_divide2_curve_fit(mymat)
    best_storage3(mymat)
    print ("forb norm squared:{}".format( np.linalg.norm(np.ravel(mymat)) ** 2 ))
    

def main():
    #a = np.arange(4 * 4 * 4).reshape(4, 4, 4)
    #for sa in sub_tensors(a):
    #    print (sa)
    #    print ('****')

    #b = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    #for mb in matrix_views(b):
    #    print (b)
    #    print ('****')

    #c = np.random.uniform(size=(10, 10))
    #remain_estimate_for_a_mode(c)

    #d = np.random.uniform(size=(50,50,50))
    #best_ks_without_divide(d)

    #test()

    ts = np.load("sample.data.npy")
    fin = 19
    for i in range(2, fin+1):
        ts += np.load("sample{}.data.npy".format(i))

    best_storage3(ts)
    print ("forb norm squared:{}".format( np.linalg.norm(np.ravel(ts)) ** 2 ))
    
   

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
