#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <queue>
#include <vector>

#define CUDA_CHECK(code)                                            \
  do {                                                              \
    cudaError_t _e = (code);                                        \
    if (_e != cudaSuccess) {                                        \
      fprintf(stderr, "[CUDA]  %s  (error %d)  %s:%d \n",           \
              cudaGetErrorString(_e), (int)_e, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

__device__ __forceinline__ double L(double x0, double x1, double u) {
  double d = x0 - u;
  return std::hypot(d, x1);
}

// <-----  Stream management -----> //

static std::mutex gpu_pool_mutex;
static std::vector<cudaStream_t> gpu_all_streams;
static std::queue<cudaStream_t> gpu_free_streams;

static cudaStream_t acquire_stream() {
  std::lock_guard<std::mutex> lock(gpu_pool_mutex);

  if (!gpu_free_streams.empty()) {
    cudaStream_t stream = gpu_free_streams.front();
    gpu_free_streams.pop();
    return stream;
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  gpu_all_streams.push_back(stream);
  return stream;
}

static void release_stream(cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(gpu_pool_mutex);
  gpu_free_streams.push(stream);
}

// <-----  Helper Functions -----> //

__device__ __forceinline__ void x(const double* __restrict__ Y, long i, long N,
                                  double x0_0, double x0_1, double& o0,
                                  double& o1) {
  if (i == 0) {
    o0 = x0_0;
    o1 = x0_1;
    return;
  }

  long b = 2 * (i - 1);
  o0 = Y[b];
  o1 = Y[b + 1];
}

__device__ __forceinline__ void v(const double* __restrict__ Y, long i, long N,
                                  double v0_0, double v0_1, double& o0,
                                  double& o1) {
  if (i == 0) {
    o0 = v0_0;
    o1 = v0_1;
    return;
  }

  long b = 2 * N + 2 * (i - 1);
  o0 = Y[b];
  o1 = Y[b + 1];
}

__device__ __forceinline__ void lambda(const double* __restrict__ Y, long i,
                                       long N, double lambda_n_0,
                                       double lambda_n_1, double& o0,
                                       double& o1) {
  if (i == N) {
    o0 = lambda_n_0;
    o1 = lambda_n_1;
    return;
  }

  long b = 4 * N + 2 * i;
  o0 = Y[b];
  o1 = Y[b + 1];
}

__device__ __forceinline__ void mu(const double* __restrict__ Y, long i, long N,
                                   double mu_n_0, double mu_n_1, double& o0,
                                   double& o1) {
  if (i == N) {
    o0 = mu_n_0;
    o1 = mu_n_1;
    return;
  }

  long b = 6 * N + 2 * i;
  o0 = Y[b];
  o1 = Y[b + 1];
}

__device__ __forceinline__ double u(const double* __restrict__ Y, long i,
                                    long N) {
  return Y[8 * N + i];
}

// <-----  SE1 -----> //

__global__ void SE1_loop_kernel(const double* __restrict__ Y,
                                double* __restrict__ R, long N, double alpha,
                                double m, double k, double a0, double a1,
                                double t0, double T, double x0_0, double x0_1,
                                double l0, double xd0, double xd1, long M) {
  const long blocks_per_batch = (N + blockDim.x - 1) / blockDim.x;
  const long i = (blockIdx.x % blocks_per_batch) * blockDim.x + threadIdx.x;
  const long b = blockIdx.x / blocks_per_batch;
  if (i >= N || b >= M) return;

  const long batch_length = 9 * N + 1;
  const double* Yb = Y + b * batch_length;
  double* Rb = R + b * batch_length;

  const double h = (T - t0) / (double)N;

  double x_i_0, x_i_1;
  double x_i_plus_1_0, x_i_plus_1_1;
  double v_i_0, v_i_1;
  double v_i_plus_1_0, v_i_plus_1_1;
  double lambda_i_0, lambda_i_1;
  double lambda_i_plus_1_0, lambda_i_plus_1_1;
  double mu_i_0, mu_i_1;
  double mu_i_plus_1_0, mu_i_plus_1_1;

  x(Yb, i, N, x0_0, x0_1, x_i_0, x_i_1);
  x(Yb, i + 1, N, x0_0, x0_1, x_i_plus_1_0, x_i_plus_1_1);
  v(Yb, i, N, 0, 0, v_i_0, v_i_1);
  v(Yb, i + 1, N, 0, 0, v_i_plus_1_0, v_i_plus_1_1);
  lambda(Yb, i, N, 0, 0, lambda_i_0, lambda_i_1);
  lambda(Yb, i + 1, N, 0, 0, lambda_i_plus_1_0, lambda_i_plus_1_1);
  mu(Yb, i, N, 0, 0, mu_i_0, mu_i_1);
  mu(Yb, i + 1, N, 0, 0, mu_i_plus_1_0, mu_i_plus_1_1);

  const double u_i = u(Yb, i, N);

  const double Li = L(x_i_plus_1_0, x_i_plus_1_1, u_i);
  const double Lu = L(x_i_0, x_i_1, u_i);

  const double c1 = k * l0 / (m * Li * Li * Li);
  const double c2 = k / m * (1.0 - l0 / Li);

  const double delta_lambda_0 = (lambda_i_plus_1_0 - lambda_i_0) / h;
  const double delta_lambda_1 = (lambda_i_plus_1_1 - lambda_i_1) / h;
  const double delta_mu_0 = (mu_i_plus_1_0 - mu_i_0) / h;
  const double delta_mu_1 = (mu_i_plus_1_1 - mu_i_1) / h;
  const double delta_v_0 = (v_i_plus_1_0 - v_i_0) / h;
  const double delta_v_1 = (v_i_plus_1_1 - v_i_1) / h;
  const double delta_x_0 = (x_i_plus_1_0 - x_i_0) / h;
  const double delta_x_1 = (x_i_plus_1_1 - x_i_1) / h;

  const double a1_0 = x_i_plus_1_0 - u_i;
  const double a1_1 = x_i_plus_1_1;
  const double a3_0 = x_i_0 - u_i;
  const double a3_1 = x_i_1;

  const double a1_dot_lambda = a1_0 * lambda_i_0 + a1_1 * lambda_i_1;

  const double a1c = k * l0 / (m * Lu * Lu * Lu);
  const double a2c = k / m * (1.0 - l0 / Lu);

  const long base = 9 * i;

  Rb[base] = delta_lambda_0 + mu_i_0;
  Rb[base + 1] = delta_lambda_1 + mu_i_1;

  Rb[base + 2] = delta_x_0 - v_i_plus_1_0;
  Rb[base + 3] = delta_x_1 - v_i_plus_1_1;

  Rb[base + 4] = delta_mu_0 + (x_i_plus_1_0 - xd0) -
                 (c1 * a1_0 * a1_dot_lambda + c2 * lambda_i_0);
  Rb[base + 5] = delta_mu_1 + (x_i_plus_1_1 - xd1) -
                 (c1 * a1_1 * a1_dot_lambda + c2 * lambda_i_1);

  Rb[base + 6] = delta_v_0 + c2 * a1_0 - a0 / m;
  Rb[base + 7] = delta_v_1 + c2 * a1_1 - a1 / m;

  const double fu_0 = -a1c * a3_0 * a3_0 - a2c;
  const double fu_1 = -a1c * a3_0 * a3_1;
  Rb[base + 8] = alpha * u_i - (lambda_i_0 * fu_0 + lambda_i_1 * fu_1);
}

__global__ void SE1_boundary_kernel(const double* __restrict__ Y,
                                    double* __restrict__ R, long N,
                                    double alpha, double m, double k,
                                    double x0_0, double x0_1, double l0,
                                    long M) {
  const long b = (long)(blockIdx.x * blockDim.x + threadIdx.x);
  if (b >= M) return;

  const long batch_length = 9 * N + 1;
  const double* Yb = Y + b * batch_length;
  double* Rb = R + b * batch_length;

  double x_n_0, x_n_1, lambda_n_0, lambda_n_1;
  x(Yb, N, N, x0_0, x0_1, x_n_0, x_n_1);
  lambda(Yb, N, N, 0, 0, lambda_n_0, lambda_n_1);

  const double u_n = u(Yb, N, N);
  const double Lu = L(x_n_0, x_n_1, u_n);
  const double a1c = k * l0 / (m * Lu * Lu * Lu);
  const double a2c = k / m * (1.0 - l0 / Lu);
  const double a3_0 = x_n_0 - u_n;
  const double a3_1 = x_n_1;

  const double fu_0 = -a1c * a3_0 * a3_0 - a2c;
  const double fu_1 = -a1c * a3_0 * a3_1;

  Rb[9 * N] = alpha * u_n - (lambda_n_0 * fu_0 + lambda_n_1 * fu_1);
}

// <-----  SE2 -----> //

__global__ void SE2_loop_kernel(const double* __restrict__ Y,
                                double* __restrict__ R, long N, double alpha,
                                double m, double k, double a0, double a1,
                                double t0, double T, double x0_0, double x0_1,
                                double l0, double xd0, double xd1, long M) {
  const long blocks_per_batch = (N + blockDim.x - 1) / blockDim.x;
  const long i = (blockIdx.x % blocks_per_batch) * blockDim.x + threadIdx.x;
  const long b = blockIdx.x / blocks_per_batch;
  if (i >= N || b >= M) return;

  const long batch_length = 9 * N + 1;
  const double* Yb = Y + b * batch_length;
  double* Rb = R + b * batch_length;

  const double h = (T - t0) / (double)N;

  double x_i_0, x_i_1;
  double x_i_plus_1_0, x_i_plus_1_1;
  double v_i_0, v_i_1;
  double v_i_plus_1_0, v_i_plus_1_1;
  double lambda_i_0, lambda_i_1;
  double lambda_i_plus_1_0, lambda_i_plus_1_1;
  double mu_i_0, mu_i_1;
  double mu_i_plus_1_0, mu_i_plus_1_1;

  x(Yb, i, N, x0_0, x0_1, x_i_0, x_i_1);
  x(Yb, i + 1, N, x0_0, x0_1, x_i_plus_1_0, x_i_plus_1_1);
  v(Yb, i, N, 0, 0, v_i_0, v_i_1);
  v(Yb, i + 1, N, 0, 0, v_i_plus_1_0, v_i_plus_1_1);
  lambda(Yb, i, N, 0, 0, lambda_i_0, lambda_i_1);
  lambda(Yb, i + 1, N, 0, 0, lambda_i_plus_1_0, lambda_i_plus_1_1);
  mu(Yb, i, N, 0, 0, mu_i_0, mu_i_1);
  mu(Yb, i + 1, N, 0, 0, mu_i_plus_1_0, mu_i_plus_1_1);

  const double u_i = u(Yb, i, N);
  const double u_i_plus_1 = u(Yb, i + 1, N);

  const double Li = L(x_i_0, x_i_1, u_i_plus_1);
  const double Lu = L(x_i_0, x_i_1, u_i);

  const double c1 = k * l0 / (m * Li * Li * Li);
  const double c2 = k / m * (1.0 - l0 / Li);

  const double delta_lambda_0 = (lambda_i_plus_1_0 - lambda_i_0) / h;
  const double delta_lambda_1 = (lambda_i_plus_1_1 - lambda_i_1) / h;
  const double delta_mu_0 = (mu_i_plus_1_0 - mu_i_0) / h;
  const double delta_mu_1 = (mu_i_plus_1_1 - mu_i_1) / h;
  const double delta_v_0 = (v_i_plus_1_0 - v_i_0) / h;
  const double delta_v_1 = (v_i_plus_1_1 - v_i_1) / h;
  const double delta_x_0 = (x_i_plus_1_0 - x_i_0) / h;
  const double delta_x_1 = (x_i_plus_1_1 - x_i_1) / h;

  const double a1_0 = x_i_0 - u_i_plus_1;
  const double a1_1 = x_i_1;

  const double a3_0 = x_i_0 - u_i;
  const double a3_1 = x_i_1;

  const double a1_dot_lambda_plus_1 =
      a1_0 * lambda_i_plus_1_0 + a1_1 * lambda_i_plus_1_1;

  const double a1c = k * l0 / (m * Lu * Lu * Lu);
  const double a2c = k / m * (1.0 - l0 / Lu);

  const long base = 9 * i;

  Rb[base] = delta_lambda_0 + mu_i_plus_1_0;
  Rb[base + 1] = delta_lambda_1 + mu_i_plus_1_1;

  Rb[base + 2] = delta_x_0 - v_i_0;
  Rb[base + 3] = delta_x_1 - v_i_1;

  Rb[base + 4] = delta_mu_0 + (x_i_0 - xd0) -
                 (c1 * a1_0 * a1_dot_lambda_plus_1 + c2 * lambda_i_plus_1_0);
  Rb[base + 5] = delta_mu_1 + (x_i_1 - xd1) -
                 (c1 * a1_1 * a1_dot_lambda_plus_1 + c2 * lambda_i_plus_1_1);

  Rb[base + 6] = delta_v_0 + c2 * a1_0 - a0 / m;
  Rb[base + 7] = delta_v_1 + c2 * a1_1 - a1 / m;

  const double fu_0 = -a1c * a3_0 * a3_0 - a2c;
  const double fu_1 = -a1c * a3_0 * a3_1;
  Rb[base + 8] = alpha * u_i - (lambda_i_0 * fu_0 + lambda_i_1 * fu_1);
}

__global__ void SE2_boundary_kernel(const double* __restrict__ Y,
                                    double* __restrict__ R, long N,
                                    double alpha, double m, double k,
                                    double x0_0, double x0_1, double l0,
                                    long M) {
  const long b = (long)(blockIdx.x * blockDim.x + threadIdx.x);
  if (b >= M) return;

  const long batch_length = 9 * N + 1;
  const double* Yb = Y + b * batch_length;
  double* Rb = R + b * batch_length;

  double x_n_0, x_n_1, lambda_n_0, lambda_n_1;
  x(Yb, N, N, x0_0, x0_1, x_n_0, x_n_1);
  lambda(Yb, N, N, 0, 0, lambda_n_0, lambda_n_1);

  const double u_n = u(Yb, N, N);
  const double Lu = L(x_n_0, x_n_1, u_n);
  const double a1c = k * l0 / (m * Lu * Lu * Lu);
  const double a2c = k / m * (1.0 - l0 / Lu);
  const double a3_0 = x_n_0 - u_n;
  const double a3_1 = x_n_1;

  const double fu_0 = -a1c * a3_0 * a3_0 - a2c;
  const double fu_1 = -a1c * a3_0 * a3_1;

  Rb[9 * N] = alpha * u_n - (lambda_n_0 * fu_0 + lambda_n_1 * fu_1);
}

// <-----  Modified SE1 -----> //

__global__ void Modified_SE1_loop_kernel(
    const double* __restrict__ Y, double* __restrict__ R, long N, double alpha,
    double m, double k, double a0, double a1, double t0, double T, double x0_0,
    double x0_1, double l0, double xd0, double xd1, long M) {
  const long blocks_per_batch = (N + blockDim.x - 1) / blockDim.x;
  const long i = (blockIdx.x % blocks_per_batch) * blockDim.x + threadIdx.x;
  const long b = blockIdx.x / blocks_per_batch;
  if (i >= N || b >= M) return;

  const long batch_length = 9 * N;
  const double* Yb = Y + b * batch_length;
  double* Rb = R + b * batch_length;

  const double h = (T - t0) / (double)N;

  double x_i_0, x_i_1;
  double x_i_plus_1_0, x_i_plus_1_1;
  double v_i_0, v_i_1;
  double v_i_plus_1_0, v_i_plus_1_1;
  double lambda_i_0, lambda_i_1;
  double lambda_i_plus_1_0, lambda_i_plus_1_1;
  double mu_i_0, mu_i_1;
  double mu_i_plus_1_0, mu_i_plus_1_1;

  x(Yb, i, N, x0_0, x0_1, x_i_0, x_i_1);
  x(Yb, i + 1, N, x0_0, x0_1, x_i_plus_1_0, x_i_plus_1_1);
  v(Yb, i, N, 0, 0, v_i_0, v_i_1);
  v(Yb, i + 1, N, 0, 0, v_i_plus_1_0, v_i_plus_1_1);
  lambda(Yb, i, N, 0, 0, lambda_i_0, lambda_i_1);
  lambda(Yb, i + 1, N, 0, 0, lambda_i_plus_1_0, lambda_i_plus_1_1);
  mu(Yb, i, N, 0, 0, mu_i_0, mu_i_1);
  mu(Yb, i + 1, N, 0, 0, mu_i_plus_1_0, mu_i_plus_1_1);

  const double u_t = u(Yb, i, N);

  const double Li = L(x_i_plus_1_0, x_i_plus_1_1, u_t);

  const double c1 = k * l0 / (m * Li * Li * Li);
  const double c2 = k / m * (1.0 - l0 / Li);

  const double c3_0 = x_i_plus_1_0 - u_t;
  const double c3_1 = x_i_plus_1_1;

  const double delta_lambda_0 = (lambda_i_plus_1_0 - lambda_i_0) / h;
  const double delta_lambda_1 = (lambda_i_plus_1_1 - lambda_i_1) / h;
  const double delta_mu_0 = (mu_i_plus_1_0 - mu_i_0) / h;
  const double delta_mu_1 = (mu_i_plus_1_1 - mu_i_1) / h;
  const double delta_v_0 = (v_i_plus_1_0 - v_i_0) / h;
  const double delta_v_1 = (v_i_plus_1_1 - v_i_1) / h;
  const double delta_x_0 = (x_i_plus_1_0 - x_i_0) / h;
  const double delta_x_1 = (x_i_plus_1_1 - x_i_1) / h;

  const double c3_dot_lambda = c3_0 * lambda_i_0 + c3_1 * lambda_i_1;

  const long base = 9 * i;

  Rb[base] = delta_lambda_0 + mu_i_0;
  Rb[base + 1] = delta_lambda_1 + mu_i_1;

  Rb[base + 2] = delta_x_0 - v_i_plus_1_0;
  Rb[base + 3] = delta_x_1 - v_i_plus_1_1;

  Rb[base + 4] = delta_mu_0 + (x_i_plus_1_0 - xd0) -
                 (c1 * c3_0 * c3_dot_lambda + c2 * lambda_i_0);
  Rb[base + 5] = delta_mu_1 + (x_i_plus_1_1 - xd1) -
                 (c1 * c3_1 * c3_dot_lambda + c2 * lambda_i_1);

  Rb[base + 6] = delta_v_0 + c2 * c3_0 - a0 / m;
  Rb[base + 7] = delta_v_1 + c2 * c3_1 - a1 / m;

  const double fu_0 = -c1 * c3_0 * c3_0 - c2;
  const double fu_1 = -c1 * c3_0 * c3_1;
  Rb[base + 8] = alpha * u_t - (lambda_i_0 * fu_0 + lambda_i_1 * fu_1);
}

// <-----  Modified SE2 -----> //

__global__ void Modified_SE2_loop_kernel(
    const double* __restrict__ Y, double* __restrict__ R, long N, double alpha,
    double m, double k, double a0, double a1, double t0, double T, double x0_0,
    double x0_1, double l0, double xd0, double xd1, long M) {
  const long blocks_per_batch = (N + blockDim.x - 1) / blockDim.x;
  const long i = (blockIdx.x % blocks_per_batch) * blockDim.x + threadIdx.x;
  const long b = blockIdx.x / blocks_per_batch;
  if (i >= N || b >= M) return;

  const long batch_length = 9 * N;
  const double* Yb = Y + b * batch_length;
  double* Rb = R + b * batch_length;

  const double h = (T - t0) / (double)N;

  double x_i_0, x_i_1;
  double x_i_plus_1_0, x_i_plus_1_1;
  double v_i_0, v_i_1;
  double v_i_plus_1_0, v_i_plus_1_1;
  double lambda_i_0, lambda_i_1;
  double lambda_i_plus_1_0, lambda_i_plus_1_1;
  double mu_i_0, mu_i_1;
  double mu_i_plus_1_0, mu_i_plus_1_1;

  x(Yb, i, N, x0_0, x0_1, x_i_0, x_i_1);
  x(Yb, i + 1, N, x0_0, x0_1, x_i_plus_1_0, x_i_plus_1_1);
  v(Yb, i, N, 0, 0, v_i_0, v_i_1);
  v(Yb, i + 1, N, 0, 0, v_i_plus_1_0, v_i_plus_1_1);
  lambda(Yb, i, N, 0, 0, lambda_i_0, lambda_i_1);
  lambda(Yb, i + 1, N, 0, 0, lambda_i_plus_1_0, lambda_i_plus_1_1);
  mu(Yb, i, N, 0, 0, mu_i_0, mu_i_1);
  mu(Yb, i + 1, N, 0, 0, mu_i_plus_1_0, mu_i_plus_1_1);

  const double u_t = u(Yb, i, N);

  const double Li = L(x_i_0, x_i_1, u_t);
  const double c1 = k * l0 / (m * Li * Li * Li);
  const double c2 = k / m * (1.0 - l0 / Li);
  const double c3_0 = x_i_0 - u_t;
  const double c3_1 = x_i_1;

  const double delta_lambda_0 = (lambda_i_plus_1_0 - lambda_i_0) / h;
  const double delta_lambda_1 = (lambda_i_plus_1_1 - lambda_i_1) / h;
  const double delta_mu_0 = (mu_i_plus_1_0 - mu_i_0) / h;
  const double delta_mu_1 = (mu_i_plus_1_1 - mu_i_1) / h;
  const double delta_v_0 = (v_i_plus_1_0 - v_i_0) / h;
  const double delta_v_1 = (v_i_plus_1_1 - v_i_1) / h;
  const double delta_x_0 = (x_i_plus_1_0 - x_i_0) / h;
  const double delta_x_1 = (x_i_plus_1_1 - x_i_1) / h;

  const double c3_dot_lambda_plus_1 =
      c3_0 * lambda_i_plus_1_0 + c3_1 * lambda_i_plus_1_1;

  const long base = 9 * i;

  Rb[base] = delta_lambda_0 + mu_i_plus_1_0;
  Rb[base + 1] = delta_lambda_1 + mu_i_plus_1_1;

  Rb[base + 2] = delta_x_0 - v_i_0;
  Rb[base + 3] = delta_x_1 - v_i_1;

  Rb[base + 4] = delta_mu_0 + (x_i_0 - xd0) -
                 (c1 * c3_0 * c3_dot_lambda_plus_1 + c2 * lambda_i_plus_1_0);
  Rb[base + 5] = delta_mu_1 + (x_i_1 - xd1) -
                 (c1 * c3_1 * c3_dot_lambda_plus_1 + c2 * lambda_i_plus_1_1);

  Rb[base + 6] = delta_v_0 + c2 * c3_0 - a0 / m;
  Rb[base + 7] = delta_v_1 + c2 * c3_1 - a1 / m;

  const double fu_0 = -c1 * c3_0 * c3_0 - c2;
  const double fu_1 = -c1 * c3_0 * c3_1;
  Rb[base + 8] =
      alpha * u_t - (lambda_i_plus_1_0 * fu_0 + lambda_i_plus_1_1 * fu_1);
}

// <-----  MidPoint -----> //

__global__ void MidPoint_loop_kernel(const double* __restrict__ Y,
                                     double* __restrict__ R, long N,
                                     double alpha, double m, double k,
                                     double a0, double a1, double t0, double T,
                                     double x0_0, double x0_1, double l0,
                                     double xd0, double xd1, long M) {
  const long blocks_per_batch = (N + blockDim.x - 1) / blockDim.x;
  const long i = (blockIdx.x % blocks_per_batch) * blockDim.x + threadIdx.x;
  const long b = blockIdx.x / blocks_per_batch;
  if (i >= N || b >= M) return;

  const long batch_length = 9 * N + 1;
  const double* Yb = Y + b * batch_length;
  double* Rb = R + b * batch_length;

  const double h = (T - t0) / (double)N;

  double x_i_0, x_i_1;
  double x_i_plus_1_0, x_i_plus_1_1;
  double v_i_0, v_i_1;
  double v_i_plus_1_0, v_i_plus_1_1;
  double lambda_i_0, lambda_i_1;
  double lambda_i_plus_1_0, lambda_i_plus_1_1;
  double mu_i_0, mu_i_1;
  double mu_i_plus_1_0, mu_i_plus_1_1;

  x(Yb, i, N, x0_0, x0_1, x_i_0, x_i_1);
  x(Yb, i + 1, N, x0_0, x0_1, x_i_plus_1_0, x_i_plus_1_1);
  v(Yb, i, N, 0, 0, v_i_0, v_i_1);
  v(Yb, i + 1, N, 0, 0, v_i_plus_1_0, v_i_plus_1_1);
  lambda(Yb, i, N, 0, 0, lambda_i_0, lambda_i_1);
  lambda(Yb, i + 1, N, 0, 0, lambda_i_plus_1_0, lambda_i_plus_1_1);
  mu(Yb, i, N, 0, 0, mu_i_0, mu_i_1);
  mu(Yb, i + 1, N, 0, 0, mu_i_plus_1_0, mu_i_plus_1_1);

  const double u_i = u(Yb, i, N);
  const double u_i_plus_1 = u(Yb, i + 1, N);

  const double x_m_0 = 0.5 * (x_i_0 + x_i_plus_1_0);
  const double x_m_1 = 0.5 * (x_i_1 + x_i_plus_1_1);
  const double v_m_0 = 0.5 * (v_i_0 + v_i_plus_1_0);
  const double v_m_1 = 0.5 * (v_i_1 + v_i_plus_1_1);
  const double lambda_m_0 = 0.5 * (lambda_i_0 + lambda_i_plus_1_0);
  const double lambda_m_1 = 0.5 * (lambda_i_1 + lambda_i_plus_1_1);
  const double mu_m_0 = 0.5 * (mu_i_0 + mu_i_plus_1_0);
  const double mu_m_1 = 0.5 * (mu_i_1 + mu_i_plus_1_1);
  const double u_m = 0.5 * (u_i + u_i_plus_1);

  const double Li = L(x_m_0, x_m_1, u_m);
  const double Lu = L(x_i_0, x_i_1, u_i);

  const double c1 = k * l0 / (m * Li * Li * Li);
  const double c2 = k / m * (1.0 - l0 / Li);

  const double delta_lambda_0 = (lambda_i_plus_1_0 - lambda_i_0) / h;
  const double delta_lambda_1 = (lambda_i_plus_1_1 - lambda_i_1) / h;
  const double delta_mu_0 = (mu_i_plus_1_0 - mu_i_0) / h;
  const double delta_mu_1 = (mu_i_plus_1_1 - mu_i_1) / h;
  const double delta_v_0 = (v_i_plus_1_0 - v_i_0) / h;
  const double delta_v_1 = (v_i_plus_1_1 - v_i_1) / h;
  const double delta_x_0 = (x_i_plus_1_0 - x_i_0) / h;
  const double delta_x_1 = (x_i_plus_1_1 - x_i_1) / h;

  const double a1_0 = x_m_0 - u_m;
  const double a1_1 = x_m_1;

  const double a1_dot_lambda_m = a1_0 * lambda_m_0 + a1_1 * lambda_m_1;

  const double a1c = k * l0 / (m * Lu * Lu * Lu);
  const double a2c = k / m * (1.0 - l0 / Lu);
  const double a3_0 = x_i_0 - u_i;
  const double a3_1 = x_i_1;

  const long base = 9 * i;

  Rb[base] = delta_lambda_0 + mu_m_0;
  Rb[base + 1] = delta_lambda_1 + mu_m_1;

  Rb[base + 2] = delta_x_0 - v_m_0;
  Rb[base + 3] = delta_x_1 - v_m_1;

  Rb[base + 4] = delta_mu_0 + (x_m_0 - xd0) -
                 (c1 * a1_0 * a1_dot_lambda_m + c2 * lambda_m_0);
  Rb[base + 5] = delta_mu_1 + (x_m_1 - xd1) -
                 (c1 * a1_1 * a1_dot_lambda_m + c2 * lambda_m_1);

  Rb[base + 6] = delta_v_0 + c2 * a1_0 - a0 / m;
  Rb[base + 7] = delta_v_1 + c2 * a1_1 - a1 / m;

  const double fu_0 = -a1c * a3_0 * a3_0 - a2c;
  const double fu_1 = -a1c * a3_0 * a3_1;
  Rb[base + 8] = alpha * u_i - (lambda_i_0 * fu_0 + lambda_i_1 * fu_1);
}

__global__ void MidPoint_boundary_kernel(const double* __restrict__ Y,
                                         double* __restrict__ R, long N,
                                         double alpha, double m, double k,
                                         double x0_0, double x0_1, double l0,
                                         long M) {
  const long b = (long)(blockIdx.x * blockDim.x + threadIdx.x);
  if (b >= M) return;

  const long batch_length = 9 * N + 1;
  const double* Yb = Y + b * batch_length;
  double* Rb = R + b * batch_length;

  double x_n_0, x_n_1, lambda_n_0, lambda_n_1;
  x(Yb, N, N, x0_0, x0_1, x_n_0, x_n_1);
  lambda(Yb, N, N, 0, 0, lambda_n_0, lambda_n_1);

  const double u_n = u(Yb, N, N);
  const double Lu = L(x_n_0, x_n_1, u_n);
  const double a1c = k * l0 / (m * Lu * Lu * Lu);
  const double a2c = k / m * (1.0 - l0 / Lu);
  const double a3_0 = x_n_0 - u_n;
  const double a3_1 = x_n_1;

  const double fu_0 = -a1c * a3_0 * a3_0 - a2c;
  const double fu_1 = -a1c * a3_0 * a3_1;

  Rb[9 * N] = alpha * u_n - (lambda_n_0 * fu_0 + lambda_n_1 * fu_1);
}

// <-----  Modified_MidPoint -----> //

__global__ void Modified_MidPoint_loop_kernel(
    const double* __restrict__ Y, double* __restrict__ R, long N, double alpha,
    double m, double k, double a0, double a1, double t0, double T, double x0_0,
    double x0_1, double l0, double xd0, double xd1, long M) {
  const long blocks_per_batch = (N + blockDim.x - 1) / blockDim.x;
  const long i = (blockIdx.x % blocks_per_batch) * blockDim.x + threadIdx.x;
  const long b = blockIdx.x / blocks_per_batch;
  if (i >= N || b >= M) return;

  const long batch_length = 9 * N;
  const double* Yb = Y + b * batch_length;
  double* Rb = R + b * batch_length;

  const double h = (T - t0) / (double)N;

  double x_i_0, x_i_1;
  double x_i_plus_1_0, x_i_plus_1_1;
  double v_i_0, v_i_1;
  double v_i_plus_1_0, v_i_plus_1_1;
  double lambda_i_0, lambda_i_1;
  double lambda_i_plus_1_0, lambda_i_plus_1_1;
  double mu_i_0, mu_i_1;
  double mu_i_plus_1_0, mu_i_plus_1_1;

  x(Yb, i, N, x0_0, x0_1, x_i_0, x_i_1);
  x(Yb, i + 1, N, x0_0, x0_1, x_i_plus_1_0, x_i_plus_1_1);
  v(Yb, i, N, 0, 0, v_i_0, v_i_1);
  v(Yb, i + 1, N, 0, 0, v_i_plus_1_0, v_i_plus_1_1);
  lambda(Yb, i, N, 0, 0, lambda_i_0, lambda_i_1);
  lambda(Yb, i + 1, N, 0, 0, lambda_i_plus_1_0, lambda_i_plus_1_1);
  mu(Yb, i, N, 0, 0, mu_i_0, mu_i_1);
  mu(Yb, i + 1, N, 0, 0, mu_i_plus_1_0, mu_i_plus_1_1);

  const double u_t = u(Yb, i, N);

  const double x_m_0 = 0.5 * (x_i_0 + x_i_plus_1_0);
  const double x_m_1 = 0.5 * (x_i_1 + x_i_plus_1_1);
  const double v_m_0 = 0.5 * (v_i_0 + v_i_plus_1_0);
  const double v_m_1 = 0.5 * (v_i_1 + v_i_plus_1_1);
  const double lambda_m_0 = 0.5 * (lambda_i_0 + lambda_i_plus_1_0);
  const double lambda_m_1 = 0.5 * (lambda_i_1 + lambda_i_plus_1_1);
  const double mu_m_0 = 0.5 * (mu_i_0 + mu_i_plus_1_0);
  const double mu_m_1 = 0.5 * (mu_i_1 + mu_i_plus_1_1);

  const double Li = L(x_m_0, x_m_1, u_t);
  const double c1 = k * l0 / (m * Li * Li * Li);
  const double c2 = k / m * (1.0 - l0 / Li);

  const double c3_0 = x_m_0 - u_t;
  const double c3_1 = x_m_1;

  const double delta_lambda_0 = (lambda_i_plus_1_0 - lambda_i_0) / h;
  const double delta_lambda_1 = (lambda_i_plus_1_1 - lambda_i_1) / h;
  const double delta_mu_0 = (mu_i_plus_1_0 - mu_i_0) / h;
  const double delta_mu_1 = (mu_i_plus_1_1 - mu_i_1) / h;
  const double delta_v_0 = (v_i_plus_1_0 - v_i_0) / h;
  const double delta_v_1 = (v_i_plus_1_1 - v_i_1) / h;
  const double delta_x_0 = (x_i_plus_1_0 - x_i_0) / h;
  const double delta_x_1 = (x_i_plus_1_1 - x_i_1) / h;

  const double c3_dot_lambda_m = c3_0 * lambda_m_0 + c3_1 * lambda_m_1;

  const long base = 9 * i;

  Rb[base] = delta_lambda_0 + mu_m_0;
  Rb[base + 1] = delta_lambda_1 + mu_m_1;

  Rb[base + 2] = delta_x_0 - v_m_0;
  Rb[base + 3] = delta_x_1 - v_m_1;

  Rb[base + 4] = delta_mu_0 + (x_m_0 - xd0) -
                 (c1 * c3_0 * c3_dot_lambda_m + c2 * lambda_m_0);
  Rb[base + 5] = delta_mu_1 + (x_m_1 - xd1) -
                 (c1 * c3_1 * c3_dot_lambda_m + c2 * lambda_m_1);

  Rb[base + 6] = delta_v_0 + c2 * c3_0 - a0 / m;
  Rb[base + 7] = delta_v_1 + c2 * c3_1 - a1 / m;

  const double fu_0 = -c1 * c3_0 * c3_0 - c2;
  const double fu_1 = -c1 * c3_0 * c3_1;
  Rb[base + 8] = alpha * u_t - (lambda_m_0 * fu_0 + lambda_m_1 * fu_1);
}

// <-----  Kernel Launcher -----> //

enum class Method {
  SE1 = 0,
  SE2 = 1,
  Modified_SE1 = 2,
  Modified_SE2 = 3,
  MidPoint = 4,
  Modified_MidPoint = 5
};

void SE_launch(Method method, const double* Y_h, double* R_h, long N,
               double alpha, double m, double k, double* a, double t0, double T,
               double* x0, double l0, double* x_d, long M,
               cudaStream_t stream = 0) {
  constexpr int BLOCK = 256;
  long length = (9 * N + 1) * M;

  stream = acquire_stream();

  if (method == Method::Modified_SE1 || method == Method::Modified_SE2 ||
      method == Method::Modified_MidPoint)
    length = 9 * N * M;

  double* Y;
  double* R;

  CUDA_CHECK(cudaMalloc((void**)&Y, sizeof(double) * length));
  CUDA_CHECK(cudaMalloc((void**)&R, sizeof(double) * length));

  CUDA_CHECK(cudaMemcpyAsync(Y, Y_h, sizeof(double) * length,
                             cudaMemcpyHostToDevice, stream));

  if (method == Method::SE1) {
    {
      dim3 grid(((N + BLOCK - 1) / BLOCK) * M);
      SE1_loop_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y, R, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0, x_d[0],
          x_d[1], M);
    }
    {
      dim3 grid((M + BLOCK - 1) / BLOCK);
      SE1_boundary_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y, R, N, alpha, m, k, x0[0], x0[1], l0, M);
    }
  } else if (method == Method::SE2) {
    {
      dim3 grid(((N + BLOCK - 1) / BLOCK) * M);
      SE2_loop_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y, R, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0, x_d[0],
          x_d[1], M);
    }
    {
      dim3 grid((M + BLOCK - 1) / BLOCK);
      SE2_boundary_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y, R, N, alpha, m, k, x0[0], x0[1], l0, M);
    }
  } else if (method == Method::Modified_SE1) {
    {
      dim3 grid(((N + BLOCK - 1) / BLOCK) * M);
      Modified_SE1_loop_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y, R, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0, x_d[0],
          x_d[1], M);
    }
  } else if (method == Method::Modified_SE2) {
    {
      dim3 grid(((N + BLOCK - 1) / BLOCK) * M);
      Modified_SE2_loop_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y, R, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0, x_d[0],
          x_d[1], M);
    }
  } else if (method == Method::MidPoint) {
    {
      dim3 grid(((N + BLOCK - 1) / BLOCK) * M);
      MidPoint_loop_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y, R, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0, x_d[0],
          x_d[1], M);
    }
    {
      dim3 grid((M + BLOCK - 1) / BLOCK);
      MidPoint_boundary_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y, R, N, alpha, m, k, x0[0], x0[1], l0, M);
    }
  } else if (method == Method::Modified_MidPoint) {
    {
      dim3 grid(((N + BLOCK - 1) / BLOCK) * M);
      Modified_MidPoint_loop_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y, R, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0, x_d[0],
          x_d[1], M);
    }
  }

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(R_h, R, sizeof(double) * length,
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(Y));
  CUDA_CHECK(cudaFree(R));

  release_stream(stream);
}

// <-----  Approximate Jacobian -----> //

__global__ void build_jacobian_batch_kernel(const double* __restrict__ x0,
                                            double* __restrict__ Y_batch,
                                            long x_size, long M, double t) {
  const long idx = (long)(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= M * x_size) return;

  const long batch = idx / x_size;
  const long col = idx % x_size;

  double val = x0[col];

  if (batch > 0 && col == (batch - 1)) val += t;
  Y_batch[idx] = val;
}

__global__ void compute_jacobian_kernel(const double* __restrict__ R_batch,
                                        double* __restrict__ J, long x_size,
                                        double t) {
  const long idx = (long)(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= x_size * x_size) return;

  const long col = idx / x_size;
  const long row = idx % x_size;

  const double R_base = R_batch[row];
  const double R_pert = R_batch[(col + 1) * x_size + row];

  J[col * x_size + row] = (R_pert - R_base) / t;
}

void ApproximateJacobian_launch(Method method, const double* Y_h, double* J_h,
                                long N, double alpha, double m, double k,
                                double* a, double t0, double T, double* x0,
                                double l0, double* x_d, double t,
                                cudaStream_t stream = 0) {
  constexpr int BLOCK = 256;
  long x_size = 9 * N;

  stream = acquire_stream();

  if (method == Method::SE1 || method == Method::SE2 ||
      method == Method::MidPoint)
    x_size++;

  const long M = x_size + 1;
  const long batch_total = M * x_size;
  const long J_size = x_size * x_size;

  double *Y0_d, *Y_batch, *R_batch, *J_d;
  CUDA_CHECK(cudaMalloc((void**)&Y0_d, sizeof(double) * x_size));
  CUDA_CHECK(cudaMalloc((void**)&Y_batch, sizeof(double) * batch_total));
  CUDA_CHECK(cudaMalloc((void**)&R_batch, sizeof(double) * batch_total));
  CUDA_CHECK(cudaMalloc((void**)&J_d, sizeof(double) * J_size));

  CUDA_CHECK(cudaMemcpyAsync(Y0_d, Y_h, sizeof(double) * x_size,
                             cudaMemcpyHostToDevice, stream));

  {
    dim3 grid((batch_total + BLOCK - 1) / BLOCK);
    build_jacobian_batch_kernel<<<grid, dim3(BLOCK), 0, stream>>>(Y0_d, Y_batch,
                                                                  x_size, M, t);
  }

  if (method == Method::SE1) {
    {
      dim3 grid(((N + BLOCK - 1) / BLOCK) * M);
      SE1_loop_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y_batch, R_batch, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0,
          x_d[0], x_d[1], M);
    }
    {
      dim3 grid((M + BLOCK - 1) / BLOCK);
      SE1_boundary_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y_batch, R_batch, N, alpha, m, k, x0[0], x0[1], l0, M);
    }
  } else if (method == Method::SE2) {
    {
      dim3 grid(((N + BLOCK - 1) / BLOCK) * M);
      SE2_loop_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y_batch, R_batch, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0,
          x_d[0], x_d[1], M);
    }
    {
      dim3 grid((M + BLOCK - 1) / BLOCK);
      SE2_boundary_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y_batch, R_batch, N, alpha, m, k, x0[0], x0[1], l0, M);
    }
  } else if (method == Method::Modified_SE1) {
    dim3 grid(((N + BLOCK - 1) / BLOCK) * M);
    Modified_SE1_loop_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
        Y_batch, R_batch, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0,
        x_d[0], x_d[1], M);
  } else if (method == Method::Modified_SE2) {
    dim3 grid(((N + BLOCK - 1) / BLOCK) * M);
    Modified_SE2_loop_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
        Y_batch, R_batch, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0,
        x_d[0], x_d[1], M);
  } else if (method == Method::MidPoint) {
    {
      dim3 grid(((N + BLOCK - 1) / BLOCK) * M);
      MidPoint_loop_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y_batch, R_batch, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0,
          x_d[0], x_d[1], M);
    }
    {
      dim3 grid((M + BLOCK - 1) / BLOCK);
      MidPoint_boundary_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y_batch, R_batch, N, alpha, m, k, x0[0], x0[1], l0, M);
    }
  } else if (method == Method::Modified_MidPoint) {
    dim3 grid(((N + BLOCK - 1) / BLOCK) * M);
    Modified_MidPoint_loop_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
        Y_batch, R_batch, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0,
        x_d[0], x_d[1], M);
  }

  {
    dim3 grid((J_size + BLOCK - 1) / BLOCK);
    compute_jacobian_kernel<<<grid, dim3(BLOCK), 0, stream>>>(R_batch, J_d,
                                                              x_size, t);
  }

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaMemcpyAsync(J_h, J_d, sizeof(double) * J_size,
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(Y0_d));
  CUDA_CHECK(cudaFree(Y_batch));
  CUDA_CHECK(cudaFree(R_batch));
  CUDA_CHECK(cudaFree(J_d));

  release_stream(stream);
}

// <-----  Approximate Jacobian Central -----> //

__global__ void build_jacobian_central_batch_kernel(
    const double* __restrict__ x0, double* __restrict__ Y_batch, long x_size,
    long M, double t) {
  const long idx = (long)(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= M * x_size) return;

  const long batch = idx / x_size;
  const long col = idx % x_size;

  const long col_idx = batch / 2;
  const long sign = batch % 2;  // 0 = plus, 1 = minus

  double val = x0[col];
  if (col == col_idx) val += (sign == 0) ? t : -t;

  Y_batch[idx] = val;
}

__global__ void compute_jacobian_central_kernel(
    const double* __restrict__ R_batch, double* __restrict__ J, long x_size,
    double t) {
  const long idx = (long)(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= x_size * x_size) return;

  const long col = idx / x_size;
  const long row = idx % x_size;

  const double R_plus = R_batch[(2 * col) * x_size + row];
  const double R_minus = R_batch[(2 * col + 1) * x_size + row];

  J[col * x_size + row] = (R_plus - R_minus) / (2.0 * t);
}

void ApproximateJacobianCentral_launch(Method method, const double* Y_h,
                                       double* J_h, long N, double alpha,
                                       double m, double k, double* a, double t0,
                                       double T, double* x0, double l0,
                                       double* x_d, double t,
                                       cudaStream_t stream = 0) {
  constexpr int BLOCK = 256;
  long x_size = 9 * N;

  stream = acquire_stream();

  if (method == Method::SE1 || method == Method::SE2 ||
      method == Method::MidPoint)
    x_size++;

  const long M = 2 * x_size;
  const long batch_total = M * x_size;
  const long J_size = x_size * x_size;

  double *Y0_d, *Y_batch, *R_batch, *J_d;
  CUDA_CHECK(cudaMalloc((void**)&Y0_d, sizeof(double) * x_size));
  CUDA_CHECK(cudaMalloc((void**)&Y_batch, sizeof(double) * batch_total));
  CUDA_CHECK(cudaMalloc((void**)&R_batch, sizeof(double) * batch_total));
  CUDA_CHECK(cudaMalloc((void**)&J_d, sizeof(double) * J_size));

  CUDA_CHECK(cudaMemcpyAsync(Y0_d, Y_h, sizeof(double) * x_size,
                             cudaMemcpyHostToDevice, stream));

  {
    dim3 grid((batch_total + BLOCK - 1) / BLOCK);
    build_jacobian_central_batch_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
        Y0_d, Y_batch, x_size, M, t);
  }

  if (method == Method::SE1) {
    {
      dim3 grid(((N + BLOCK - 1) / BLOCK) * M);
      SE1_loop_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y_batch, R_batch, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0,
          x_d[0], x_d[1], M);
    }
    {
      dim3 grid((M + BLOCK - 1) / BLOCK);
      SE1_boundary_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y_batch, R_batch, N, alpha, m, k, x0[0], x0[1], l0, M);
    }
  } else if (method == Method::SE2) {
    {
      dim3 grid(((N + BLOCK - 1) / BLOCK) * M);
      SE2_loop_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y_batch, R_batch, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0,
          x_d[0], x_d[1], M);
    }
    {
      dim3 grid((M + BLOCK - 1) / BLOCK);
      SE2_boundary_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y_batch, R_batch, N, alpha, m, k, x0[0], x0[1], l0, M);
    }
  } else if (method == Method::Modified_SE1) {
    {
      dim3 grid(((N + BLOCK - 1) / BLOCK) * M);
      Modified_SE1_loop_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y_batch, R_batch, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0,
          x_d[0], x_d[1], M);
    }
  } else if (method == Method::Modified_SE2) {
    {
      dim3 grid(((N + BLOCK - 1) / BLOCK) * M);
      Modified_SE2_loop_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y_batch, R_batch, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0,
          x_d[0], x_d[1], M);
    }
  } else if (method == Method::MidPoint) {
    {
      dim3 grid(((N + BLOCK - 1) / BLOCK) * M);
      MidPoint_loop_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y_batch, R_batch, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0,
          x_d[0], x_d[1], M);
    }
    {
      dim3 grid((M + BLOCK - 1) / BLOCK);
      MidPoint_boundary_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y_batch, R_batch, N, alpha, m, k, x0[0], x0[1], l0, M);
    }
  } else if (method == Method::Modified_MidPoint) {
    {
      dim3 grid(((N + BLOCK - 1) / BLOCK) * M);
      Modified_MidPoint_loop_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
          Y_batch, R_batch, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0,
          x_d[0], x_d[1], M);
    }
  }

  {
    dim3 grid((J_size + BLOCK - 1) / BLOCK);
    compute_jacobian_central_kernel<<<grid, dim3(BLOCK), 0, stream>>>(
        R_batch, J_d, x_size, t);
  }

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaMemcpyAsync(J_h, J_d, sizeof(double) * J_size,
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(Y0_d));
  CUDA_CHECK(cudaFree(Y_batch));
  CUDA_CHECK(cudaFree(R_batch));
  CUDA_CHECK(cudaFree(J_d));

  release_stream(stream);
}

// <-----  Library Interface -----> //

extern "C" {
// Direct Methods
void SE1(const double* Y_h, double* R_h, long N, double alpha, double m,
         double k, double* a, double t0, double T, double* x0, double l0,
         double* x_d, long M) {
  SE_launch(Method::SE1, Y_h, R_h, N, alpha, m, k, a, t0, T, x0, l0, x_d, M);
}
void SE2(const double* Y_h, double* R_h, long N, double alpha, double m,
         double k, double* a, double t0, double T, double* x0, double l0,
         double* x_d, long M) {
  SE_launch(Method::SE2, Y_h, R_h, N, alpha, m, k, a, t0, T, x0, l0, x_d, M);
}
void Modified_SE1(const double* Y_h, double* R_h, long N, double alpha,
                  double m, double k, double* a, double t0, double T,
                  double* x0, double l0, double* x_d, long M) {
  SE_launch(Method::Modified_SE1, Y_h, R_h, N, alpha, m, k, a, t0, T, x0, l0,
            x_d, M);
}
void Modified_SE2(const double* Y_h, double* R_h, long N, double alpha,
                  double m, double k, double* a, double t0, double T,
                  double* x0, double l0, double* x_d, long M) {
  SE_launch(Method::Modified_SE2, Y_h, R_h, N, alpha, m, k, a, t0, T, x0, l0,
            x_d, M);
}
void MidPoint(const double* Y_h, double* R_h, long N, double alpha, double m,
              double k, double* a, double t0, double T, double* x0, double l0,
              double* x_d, long M) {
  SE_launch(Method::MidPoint, Y_h, R_h, N, alpha, m, k, a, t0, T, x0, l0, x_d,
            M);
}
void Modified_MidPoint(const double* Y_h, double* R_h, long N, double alpha,
                       double m, double k, double* a, double t0, double T,
                       double* x0, double l0, double* x_d, long M) {
  SE_launch(Method::Modified_MidPoint, Y_h, R_h, N, alpha, m, k, a, t0, T, x0,
            l0, x_d, M);
}
// Approximate Jacobian
void ApproximateJacobian(const Method method, const double* Y_h, double* J_h,
                         long N, double alpha, double m, double k, double* a,
                         double t0, double T, double* x0, double l0,
                         double* x_d, double t) {
  ApproximateJacobian_launch(method, Y_h, J_h, N, alpha, m, k, a, t0, T, x0, l0,
                             x_d, t);
}
void ApproximateJacobianCentral(const Method method, const double* Y_h,
                                double* J_h, long N, double alpha, double m,
                                double k, double* a, double t0, double T,
                                double* x0, double l0, double* x_d, double t) {
  ApproximateJacobianCentral_launch(method, Y_h, J_h, N, alpha, m, k, a, t0, T,
                                    x0, l0, x_d, t);
}
}