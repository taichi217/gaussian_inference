import numpy as np
import math as mt
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, wishart, dirichlet, multinomial
from matplotlib.patches import Ellipse
from matplotlib import animation

def sn_sampler(X, mus, lamdas, pis, N, K) :
    """
    variables
    new_sn : one hot vectol
    eta : given parameters of categorical distribution
    eta_n : given parameter of categorical distribution
    eta_nk : components of eta_n
    """
    new_sn = []
    eta = []
    eta_n = []
    eta_nk = 0
    for n in range(N) :
        eta_n = []
        for k in range(K) :
            zk = X[n] - mus[k]
            eta_nk = np.exp((-0.5 * zk.T@lamdas[k]@zk + 0.5*np.log(np.linalg.det(lamdas[k] + 1e-7)) + np.log(pis[k] + 1e-7)))#1e-7はln0を避けるため
            eta_n.append(eta_nk + 1e-10) #1e-10はeta_nの全ての要素が０になるのを避けるための項
        """
        eta_nの要素の和が1になるように正規化
        """
        eta.append(eta_n / np.sum(eta_n))


    for eta_n in eta:
        new_sn.append(multinomial.rvs(1, eta_n))

    return new_sn

def mu_sampler(new_sn, k, beta, m, lamdas, X, N) :
    beta_hat = 0
    muk = 0
    sn_sum = np.sum(new_sn, axis = 0)
    beta_hat = sn_sum[k] + beta
    mhat_k = 0
    for n in range(N) :
        mhat_k += new_sn[n][k] * X[n]
    mhat_k += beta * m
    mhat_k = mhat_k / beta_hat
    lamda_k_hat = lamdas[k]
    muk = multivariate_normal.rvs(mhat_k, np.linalg.inv(lamda_k_hat), size = 1)
    # print(muk)
    #cheack
    return muk

def lamda_sampler(sn_new, k, beta, m, nu, w, X, N) :
    lamda_k_hat = 0
    w_hat_k_inv = 0
    sn_sum = np.sum(sn_new, axis = 0)
    beta_hat = sn_sum[k] + beta
    nu_hat = sn_sum[k] + nu
    mhat_k = 0
    for n in range(N) :
        mhat_k+= sn_new[n][k] * X[n]
        w_hat_k_inv += sn_new[n][k] * np.outer(X[n], X[n].T)

    mhat_k += beta * m
    mhat_k /= beta_hat
    w_hat_k_inv += beta * np.outer(m, m.T) - beta_hat * np.outer(mhat_k, mhat_k.T) + np.linalg.inv(w)
    w_hat = np.linalg.inv(w_hat_k_inv)
    #cheack
    # print(beta*np.outer(ms[k].T, ms[k]))
    lamda_k_hat = wishart.rvs(nu_hat, w_hat)
    #cheack
    # print(lamda_k_hat)
    return lamda_k_hat

def pi_sampler(sn_new, K, alpha) :
    pis_hat = []
    alpha_hat = []
    sn_sum = np.sum(sn_new, axis = 0)
    for i in range(K) :
        alpha_hat_k = 0
        alpha_hat_k += sn_sum[i] + alpha[i]
        alpha_hat.append(alpha_hat_k)
    pis_hat = dirichlet.rvs(alpha_hat)
    return pis_hat

def gibbs_sampling(iter_num, N, K, X, mus, lamdas, pis, beta, m, w, nu, alpha) :
    """
    iter_num : サンプリング回数
    N : データ数
    K : クラスタ数
    mus_log : 各クラスタのサンプリングされた平均パラメータの記録
    lamdas_log :　各クラスタのサンプリングされた精度パラメータの記録
    pis_log :　各クラスタのサンプリングされた混合比率の記録
    """
    mus_log = []
    lamdas_log = []
    pis_log = []
    mus_log.append(mus)
    lamdas_log.append(lamdas)
    pis_log.append(pis)
    mus_hat = mus
    lamdas_hat = lamdas
    pis_hat = pis
    for i in range(iter_num) :
        sn_new = []
        """
        カテゴリカル分布から，snをサンプリング
        """
        sn_new = sn_sampler(X, mus_hat, lamdas_hat, pis_hat, N, K)
        for k in range(K) :
            """
            ウィシャーと分布から，lamda_kをサンプリング
            ガウス分布から，mu_kをサンプリング
            """
            lamdas_hat[k] = lamda_sampler(sn_new, k, beta, m, nu, w, X, N)
            mus_hat[k] = mu_sampler(sn_new, k, beta, m, lamdas_hat, X, N)

        print(mus_hat)

        lamdas_log.append(lamdas_hat)
        """
        ディリクレ分布から,piをサンプリング
        """
        pis_hat = pi_sampler(sn_new, K, alpha)[0]
        pis_log.append(pis_hat)
    # print(lamdas_hat)
    # print(mus_hat)
    # print(pis_hat)
    return np.array(mus_log), np.array(lamdas_log), np.array(pis_log)




def main() :
    K = 3
    #the number of data
    N = 500
    #Generate observation
    X1 = multivariate_normal.rvs(np.array([20, 20]), np.array([[10, 0], [0, 6]]), N)
    X2 = multivariate_normal.rvs(np.array([20, 27]), np.array([[15, 3], [10, 5]]), N)
    X3 = multivariate_normal.rvs(np.array([35, 25]), np.array([[25, 5], [5, 5]]), N)

    obs = np.vstack([X1, X2, X3])
    N = obs.shape[0]
    #cheack
    # print(obs)

    # fig = plt.figure()
    # plt.scatter(X1[:,0], X1[:,1])
    # plt.scatter(X2[:,0], X2[:,1])
    # plt.scatter(X3[:,0], X3[:,1])
    # plt.show()

    # parameters of prior distribution
    alpha = np.array([1.0, 1.0, 1.0])
    beta = 1.0
    pis_prior = dirichlet.rvs(alpha)
    pis_prior = pis_prior[0]
    m = np.array([20, 20], dtype = "float") #クラスタの種類に限らず，共通の値
    sigmas_m = np.array([np.eye(2), np.eye(2), np.eye(2)])
    lamdas_m = np.array([np.linalg.inv(sigmas_m[0]), np.linalg.inv(sigmas_m[1]), np.linalg.inv(sigmas_m[2])])
    nu = 2.0
    w = np.array(5*np.eye(2))

    #gaussian mixture parameters
    mus = np.array([[50,50],[45, 45], [30, 30]], dtype = 'float')
    lamdas = np.array([wishart.rvs(2, 5*np.eye(2)), wishart.rvs(2, 5*np.eye(2)), wishart.rvs(2, 5*np.eye(2))])
    simas = np.array([np.linalg.inv(lamdas[0]), np.linalg.inv(lamdas[1]), np.linalg.inv(lamdas[2])])
    pis = dirichlet.rvs(alpha)



    mus_log, lamdas_log, pis_log = gibbs_sampling(50, N, K, obs, mus, lamdas, pis[0], beta, m, w, nu, alpha)
    iter = [i for i in range(50)]
    # for i in mus_log :
    #     print(i)




    # mu2
    # mu3
    # fig1 = plt.figure()
    # plt.plot(iter, mus_log[:0])
    # plt.title("mu_x")
    # fig2 = plt.figure()
    # plt.plot(iter, mus_log[:1])
    # plt.plot("mu_y")
    # plt.show()

if __name__ == "__main__" :
    main()
