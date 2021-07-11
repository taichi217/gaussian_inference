import numpy as np
import math as mt
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, wishart, dirichlet, multinomial
from matplotlib.patches import Ellipse
from matplotlib import animation

import glob
from PIL import Image

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
            tmp = -0.5 * np.dot((zk.T@lamdas[k]),zk)
            # print(tmp)
            eta_nk = np.exp((-0.5 * np.dot(zk.T@lamdas[k],zk) + 0.5*np.log(np.linalg.det(lamdas[k] + 1e-7)) + np.log(pis[k] + 1e-7)))#1e-7はln0を避けるため
            eta_n.append(eta_nk + 1e-7)
        """
        eta_nの要素の和が1になるように正規化
        """
        eta.append(eta_n / np.sum(eta_n))


    for eta_n in eta:
        # print(eta_n)
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
    lamda_k_hat = beta_hat * lamdas[k]
    muk = multivariate_normal.rvs(mhat_k, np.linalg.inv(lamda_k_hat), size = 1)

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

    lamda_k_hat = wishart.rvs(nu_hat, w_hat)

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

def gibbs_sampling(iter_num, N, K, X, X1, X2, X3, mus, lamdas, pis, beta, m, w, nu, alpha) :
    """
    iter_num : サンプリング回数
    N : データ数
    K : クラスタ数
    mus_log : 各クラスタのサンプリングされた平均パラメータの記録
    lamdas_log :　各クラスタのサンプリングされた精度パラメータの記録
    pis_log :　各クラスタのサンプリングされた混合比率の記録
    figlist : 描画した画像のファイルパスを記録しておくための配列
    """
    mus_log = np.zeros((iter_num, K, 2), dtype = "float")
    lamdas_log = np.zeros((iter_num, K, 2, 2), dtype = "float")
    pis_log = np.zeros((iter_num, K), dtype = "float")
    mus_hat = mus
    lamdas_hat = lamdas
    pis_hat = pis
    figlist = []
    for i in range(iter_num) :
        sn_new = []
        mus_log[i] = mus_hat
        lamdas_log[i] = lamdas_hat
        pis_log[i] = pis_hat
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
        ell1 = draw_ellipse_CI95(mus_hat[0], np.linalg.inv(lamdas_hat[0]), "blue")
        ell2 = draw_ellipse_CI95(mus_hat[1], np.linalg.inv(lamdas_hat[1]), "orange")
        ell3 = draw_ellipse_CI95(mus_hat[2], np.linalg.inv(lamdas_hat[2]), "green")
        """
        ディリクレ分布から,piをサンプリング
        """
        pis_hat = pi_sampler(sn_new, K, alpha)[0]

        """分布の楕円を描画して指定したディレクトリへ保存していく"""
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(X1[:,0], X1[:,1], c = "blue")
        ax.scatter(X2[:,0], X2[:,1], c = "orange")
        ax.scatter(X3[:,0], X3[:,1], c = "green")
        ax.add_artist(ell1)
        ax.add_artist(ell2)
        ax.add_artist(ell3)
        plt.savefig("./images/distribution_fig/fig{}.png".format(i+1))
        plt.close
        figlist.append("./images/distribution_fig/fig{}.png".format(i+1))

    return np.array(mus_log), np.array(lamdas_log), np.array(pis_log), figlist


"""
アニメーション描画用の関数
"""
def make_animation(figlist) :
    fig = plt.figure()
    plt.axis("off")
    imgs = []
    for i in figlist :
        img = Image.open(i)
        imgs.append([plt.imshow(img)])

    ani = animation.ArtistAnimation(fig, imgs, interval = 500, repeat_delay = 3000)
    ani.save("./gif/Gibbs_Sampling.gif")

"""
確率分布の作図用の関数
"""
def draw_ellipse_CI95(mean, cov, color) :
    w, v = np.linalg.eig(cov)
    p = 0.95

    v1 = v[:, np.argmax(w)]
    angle = 180. / np.pi * np.arctan(v1[1])
    width = 2 * np.sqrt(np.max(w) * 5.991)
    height = 2 * np.sqrt(np.min(w) * 5.991)

    ellipse = Ellipse(mean, width, height, angle, color=color, alpha=1.0, fill = False)
    return ellipse

def main() :
    K = 3
    #the number of data
    N = 150
    #Generate observation
    X1 = multivariate_normal.rvs(np.array([20, 20]), np.array([[10, 0], [0, 6]]), N)
    X2 = multivariate_normal.rvs(np.array([20, 27]), np.array([[15, 3], [10, 5]]), N)
    X3 = multivariate_normal.rvs(np.array([35, 25]), np.array([[10, 5], [5, 5]]), N)

    obs = np.vstack([X1, X2, X3])
    N = obs.shape[0]


    # parameters of prior distribution
    alpha = np.array([1.0, 1.0, 1.0])
    beta = 1.0
    pis_prior = dirichlet.rvs(alpha)
    pis_prior = pis_prior[0]
    m = np.array([20, 20], dtype = "float") #クラスタの種類に限らず，共通の値
    nu = 7.0
    w = np.array(5*np.eye(2))

    #gaussian mixture parameters
    mus = np.array([[50,50],[45, 45], [30, 30]], dtype = 'float')
    sigmas = np.array([np.eye(2), np.eye(2), np.eye(2)])
    lamdas = np.array([np.linalg.inv(sigmas[0]), np.linalg.inv(sigmas[1]), np.linalg.inv(sigmas[2])])
    pis = np.array([0.3,0.4,0.3])#dirichlet.rvs(alpha)[0]



    mus_log, lamdas_log, pis_log, figlist = gibbs_sampling(200, N, K, obs, X1, X2, X3, mus, lamdas, pis, beta, m, w, nu, alpha)
    iter = [i for i in range(200)]

    mu1 = np.array([i[:1][0] for i in mus_log])
    mu2 = np.array([i[:2][1] for i in mus_log])
    mu3 = np.array([i[:3][2] for i in mus_log])
    print(np.linalg.inv(lamdas_log[-1][0]))

    """各クラスタの平均値パラメータのサンプリング結果をプロットする"""

    mu1 = np.array([i[:1][0] for i in mus_log])
    mu2 = np.array([i[:2][1] for i in mus_log])
    mu3 = np.array([i[:3][2] for i in mus_log])

    fig11 = plt.figure(figsize = (8,6))
    plt.plot(iter, [i[0] for i in mu1])
    plt.title("cluster1 mu x")
    plt.savefig("./images/fig_cluster1_mu_x.png")
    fig12 = plt.figure()
    plt.plot(iter, [i[1] for i in mu1])
    plt.title("cluster1 mu y")
    plt.savefig("./images/fig_cluster1_mu_y.png")

    fig21 = plt.figure(figsize = (8,6))
    plt.plot(iter, [i[0] for i in mu2], c = "orange")
    plt.title("cluster2 mu x")
    plt.savefig("./images/fig_cluster2_mu_x.png")
    fig22 = plt.figure(figsize = (8,6))
    plt.plot(iter, [i[1] for i in mu2], c = "orange")
    plt.title("cluster2 mu y")
    plt.savefig("./images/fig_cluster2_mu_y.png")

    fig31 = plt.figure(figsize = (8,6))
    plt.plot(iter, [i[0] for i in mu3], c = "green")
    plt.title("cluster3 mu x")
    plt.savefig("./images/fig_cluster3_mu_x.png")
    fig32 = plt.figure(figsize = (8,6))
    plt.plot(iter, [i[1] for i in mu3], c = "green")
    plt.title("cluster2 mu y")
    plt.savefig("./images/fig_cluster3_mu_y.png")

    make_animation(figlist)



if __name__ == "__main__" :
    main()
