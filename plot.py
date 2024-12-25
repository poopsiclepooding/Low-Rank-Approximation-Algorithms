import numpy as np
import time
import matplotlib.pyplot as plt
from algorithms import randomized_svd, singular_value_decomposition, rank_revealing_qr_decomposition
from utils import generate_fast_decay_matrix

""" Comparing Error Rates for Diff Power Iterations of Randomized SVD and RRQR with Decay Rates """

def compare_error_over_decay(decay_bottom_bound=1, decay_top_bound=0.8, decay_step=-0.01, matrix_size=1000, k=150):
    decay_rates = np.arange(decay_bottom_bound, decay_top_bound, decay_step)
    times_svd = []
    times_randomized_svd_0 = []
    times_randomized_svd_1 = []
    times_randomized_svd_2 = []
    times_rrqr = []
    times_aca = []

    error_svd = []
    error_randomized_svd_0 = []
    error_randomized_svd_1 = []
    error_randomized_svd_2 = []
    error_rrqr = []
    error_aca = []

    size = 1000

    for decay_rate in decay_rates:
        A, _ = generate_fast_decay_matrix(size, size, decay_rate=decay_rate)
        # A = np.random.rand(size, size)
        
        start_time = time.time()
        _, _, _, A_approx = randomized_svd(A, k, n_power_iterations=0)
        error_randomized_svd_0.append(np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro'))
        times_randomized_svd_0.append(time.time() - start_time)

        start_time = time.time()
        _, _, _, A_approx = randomized_svd(A, k, n_power_iterations=1)
        error_randomized_svd_1.append(np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro'))
        times_randomized_svd_1.append(time.time() - start_time)

        start_time = time.time()
        _, _, _, A_approx = randomized_svd(A, k, n_power_iterations=2)
        error_randomized_svd_2.append(np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro'))
        times_randomized_svd_2.append(time.time() - start_time)

        start_time = time.time()
        _, _, _, A_approx = rank_revealing_qr_decomposition(A, 8, k)
        error_rrqr.append(np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro'))
        times_rrqr.append(time.time() - start_time)

        start_time = time.time()
        _, _, _, A_approx = singular_value_decomposition(A, k)
        error_svd.append(np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro'))
        times_svd.append(time.time() - start_time)

    plt.figure(figsize=(10, 6))
    plt.plot(decay_rates, error_svd, label='SVD Error', marker='o')
    plt.plot(decay_rates, error_randomized_svd_0, label='Randomized SVD (PI=0) Error', marker='o')
    plt.plot(decay_rates, error_randomized_svd_1, label='Randomized SVD (PI=1) Error', marker='o')
    plt.plot(decay_rates, error_randomized_svd_2, label='Randomized SVD (PI=2) Error', marker='o')
    plt.plot(decay_rates, error_rrqr, label='RRQR Error', marker='o')
    # plt.plot(size_range, error_aca, label='Adaptive Cross Approximation Time', marker='o')
    plt.xlabel('Decay Rate of Singular Values')
    plt.ylabel('Error (frobenius norm)')
    plt.title('Frobenius Norm Error Comparison for SVD, Various Power Iterations of Randomized SVD, and RRQR')
    plt.legend()
    plt.grid()
    plt.savefig("error.jpg")
    plt.show()

""" Comparing Time Of Computing the Low Rank Approximation Given A Algorithm vs Sizes of the Matrices """

def compare_time_over_sizes(size_bottom_bound=200, size_top_bound=1000, size_step=200, k=150):
    size_range = range(size_bottom_bound, size_top_bound, size_step)
    times_svd = []
    times_randomized_svd_0 = []
    times_randomized_svd_1 = []
    times_randomized_svd_2 = []
    times_rrqr = []
    times_aca = []

    error_svd = []
    error_randomized_svd_0 = []
    error_randomized_svd_1 = []
    error_randomized_svd_2 = []
    error_rrqr = []
    error_aca = []

    print("Running Times for Algos :")

    for size in size_range:
        # A = np.randn((size, size))
        A, _ = generate_fast_decay_matrix(size, size, decay_rate=0.95)

        start_time = time.time()
        _, _, _, A_approx = randomized_svd(A, k, n_power_iterations=0)
        error_randomized_svd_0.append(np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro'))
        times_randomized_svd_0.append(time.time() - start_time)

        start_time = time.time()
        _, _, _, A_approx = randomized_svd(A, k, n_power_iterations=1)
        error_randomized_svd_1.append(np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro'))
        times_randomized_svd_1.append(time.time() - start_time)

        start_time = time.time()
        _, _, _, A_approx = randomized_svd(A, k, n_power_iterations=2)
        error_randomized_svd_2.append(np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro'))
        times_randomized_svd_2.append(time.time() - start_time)

        start_time = time.time()
        _, _, _, A_approx = rank_revealing_qr_decomposition(A, 8, k)
        error_rrqr.append(np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro'))
        times_rrqr.append(time.time() - start_time)

        # start_time = time.time()
        # U, V = adaptive_cross_approximation(A, k)
        # A_approx = reconstruct_matrix(U, V)
        # error_aca.append(np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro'))
        # times_aca.append(time.time() - start_time)

        start_time = time.time()
        _, _, _, A_approx = singular_value_decomposition(A, k)
        error_svd.append(np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro'))
        times_svd.append(time.time() - start_time)

        print("#################")
        print("SVD :", times_svd)
        print("Randomized SVD :", times_randomized_svd_0)
        print("RRQR :", times_rrqr)
        # print("ACA :", times_aca)

    plt.figure(figsize=(10, 6))
    plt.plot(size_range, times_svd, label='SVD Time', marker='o')
    plt.plot(size_range, times_randomized_svd_0, label='Randomized SVD (PI=0) Time', marker='o')
    plt.plot(size_range, times_randomized_svd_1, label='Randomized SVD (PI=1) Time', marker='o')
    plt.plot(size_range, times_randomized_svd_2, label='Randomized SVD (PI=2) Time', marker='o')
    plt.plot(size_range, times_rrqr, label='RRQR Time', marker='o')
    # plt.plot(size_range, times_aca, label='ACA Time', marker='o')
    plt.xlabel('Matrix Dimension (m = n)')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Computation Time Comparison for SVD, Various Power Iterations of Randomized SVD, and RRQR')
    plt.legend()
    plt.grid()
    plt.savefig("time.jpg")
    plt.show()
