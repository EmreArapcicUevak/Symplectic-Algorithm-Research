// Y = 9n + 1
// a 2d vector
// x0 2d vector
// x_d 2d vector

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(code)                                            \
  do {                                                             \
    hipError_t _e = (code);                                        \
    if (_e != hipSuccess) {                                        \
      fprintf(stderr, "[HIP]  %s  (error %d)  %s:%d \n",           \
              hipGetErrorString(_e), (int)_e, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                          \
    }                                                              \
  } while (0)

__device__ __forceinline__ double L(double x0, double x1, double u) {
  double d = x0 - u;
  return std::hypot(d, x1);
}

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
                                double l0, double xd0, double xd1) {
  const long i = (long)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N) return;

  const double h = (T - t0) / (double)N;

  double x_i_0, x_i_1;
  double x_i_plus_1_0, x_i_plus_1_1;
  double v_i_0, v_i_1;
  double v_i_plus_1_0, v_i_plus_1_1;
  double lambda_i_0, lambda_i_1;
  double lambda_i_plus_1_0, lambda_i_plus_1_1;
  double mu_i_0, mu_i_1;
  double mu_i_plus_1_0, mu_i_plus_1_1;

  x(Y, i, N, x0_0, x0_1, x_i_0, x_i_1);
  x(Y, i + 1, N, x0_0, x0_1, x_i_plus_1_0, x_i_plus_1_1);
  v(Y, i, N, 0, 0, v_i_0, v_i_1);
  v(Y, i + 1, N, 0, 0, v_i_plus_1_0, v_i_plus_1_1);
  lambda(Y, i, N, 0, 0, lambda_i_0, lambda_i_1);
  lambda(Y, i + 1, N, 0, 0, lambda_i_plus_1_0, lambda_i_plus_1_1);
  mu(Y, i, N, 0, 0, mu_i_0, mu_i_1);
  mu(Y, i + 1, N, 0, 0, mu_i_plus_1_0, mu_i_plus_1_1);

  const double u_i = u(Y, i, N);

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

  R[base] = delta_lambda_0 + mu_i_0;
  R[base + 1] = delta_lambda_1 + mu_i_1;

  R[base + 2] = delta_x_0 - v_i_plus_1_0;
  R[base + 3] = delta_x_1 - v_i_plus_1_1;

  R[base + 4] = delta_mu_0 + (x_i_plus_1_0 - xd0) -
                (c1 * a1_0 * a1_dot_lambda + c2 * lambda_i_0);
  R[base + 5] = delta_mu_1 + (x_i_plus_1_1 - xd1) -
                (c1 * a1_1 * a1_dot_lambda + c2 * lambda_i_1);

  R[base + 6] = delta_v_0 + c2 * a1_0 - a0 / m;
  R[base + 7] = delta_v_1 + c2 * a1_1 - a1 / m;

  const double fu_0 = -a1c * a3_0 * a3_0 - a2c;
  const double fu_1 = -a1c * a3_0 * a3_1;
  R[base + 8] = alpha * u_i - (lambda_i_0 * fu_0 + lambda_i_1 * fu_1);
}

__global__ void SE1_boundary_kernel(const double* __restrict__ Y,
                                    double* __restrict__ R, long N,
                                    double alpha, double m, double k,
                                    double x0_0, double x0_1, double l0) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  double x_n_0, x_n_1, lambda_n_0, lambda_n_1;
  x(Y, N, N, x0_0, x0_1, x_n_0, x_n_1);
  lambda(Y, N, N, 0, 0, lambda_n_0, lambda_n_1);

  const double u_n = u(Y, N, N);
  const double Lu = L(x_n_0, x_n_1, u_n);
  const double a1c = k * l0 / (m * Lu * Lu * Lu);
  const double a2c = k / m * (1.0 - l0 / Lu);
  const double a3_0 = x_n_0 - u_n;
  const double a3_1 = x_n_1;

  const double fu_0 = -a1c * a3_0 * a3_0 - a2c;
  const double fu_1 = -a1c * a3_0 * a3_1;

  R[9 * N] = alpha * u_n - (lambda_n_0 * fu_0 + lambda_n_1 * fu_1);
}

// <-----  SE2 -----> //

__global__ void SE2_loop_kernel(const double* __restrict__ Y,
                                double* __restrict__ R, long N, double alpha,
                                double m, double k, double a0, double a1,
                                double t0, double T, double x0_0, double x0_1,
                                double l0, double xd0, double xd1) {
  const long i = (long)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N) return;

  const double h = (T - t0) / (double)N;

  double x_i_0, x_i_1;
  double x_i_plus_1_0, x_i_plus_1_1;
  double v_i_0, v_i_1;
  double v_i_plus_1_0, v_i_plus_1_1;
  double lambda_i_0, lambda_i_1;
  double lambda_i_plus_1_0, lambda_i_plus_1_1;
  double mu_i_0, mu_i_1;
  double mu_i_plus_1_0, mu_i_plus_1_1;

  x(Y, i, N, x0_0, x0_1, x_i_0, x_i_1);
  x(Y, i + 1, N, x0_0, x0_1, x_i_plus_1_0, x_i_plus_1_1);
  v(Y, i, N, 0, 0, v_i_0, v_i_1);
  v(Y, i + 1, N, 0, 0, v_i_plus_1_0, v_i_plus_1_1);
  lambda(Y, i, N, 0, 0, lambda_i_0, lambda_i_1);
  lambda(Y, i + 1, N, 0, 0, lambda_i_plus_1_0, lambda_i_plus_1_1);
  mu(Y, i, N, 0, 0, mu_i_0, mu_i_1);
  mu(Y, i + 1, N, 0, 0, mu_i_plus_1_0, mu_i_plus_1_1);

  const double u_i = u(Y, i, N);
  const double u_i_plus_1 = u(Y, i + 1, N);

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

  R[base] = delta_lambda_0 + mu_i_plus_1_0;
  R[base + 1] = delta_lambda_1 + mu_i_plus_1_1;

  R[base + 2] = delta_x_0 - v_i_0;
  R[base + 3] = delta_x_1 - v_i_1;

  R[base + 4] = delta_mu_0 + (x_i_0 - xd0) -
                (c1 * a1_0 * a1_dot_lambda_plus_1 + c2 * lambda_i_plus_1_0);
  R[base + 5] = delta_mu_1 + (x_i_1 - xd1) -
                (c1 * a1_1 * a1_dot_lambda_plus_1 + c2 * lambda_i_plus_1_1);

  R[base + 6] = delta_v_0 + c2 * a1_0 - a0 / m;
  R[base + 7] = delta_v_1 + c2 * a1_1 - a1 / m;

  const double fu_0 = -a1c * a3_0 * a3_0 - a2c;
  const double fu_1 = -a1c * a3_0 * a3_1;
  R[base + 8] = alpha * u_i - (lambda_i_0 * fu_0 + lambda_i_1 * fu_1);
}

__global__ void SE2_boundary_kernel(const double* __restrict__ Y,
                                    double* __restrict__ R, long N,
                                    double alpha, double m, double k,
                                    double x0_0, double x0_1, double l0) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  double x_n_0, x_n_1, lambda_n_0, lambda_n_1;
  x(Y, N, N, x0_0, x0_1, x_n_0, x_n_1);
  lambda(Y, N, N, 0, 0, lambda_n_0, lambda_n_1);

  const double u_n = u(Y, N, N);
  const double Lu = L(x_n_0, x_n_1, u_n);
  const double a1c = k * l0 / (m * Lu * Lu * Lu);
  const double a2c = k / m * (1.0 - l0 / Lu);
  const double a3_0 = x_n_0 - u_n;
  const double a3_1 = x_n_1;

  const double fu_0 = -a1c * a3_0 * a3_0 - a2c;
  const double fu_1 = -a1c * a3_0 * a3_1;

  R[9 * N] = alpha * u_n - (lambda_n_0 * fu_0 + lambda_n_1 * fu_1);
}

// <-----  Modified SE1 -----> //

__global__ void Modified_SE1_loop_kernel(const double* __restrict__ Y,
                                         double* __restrict__ R, long N,
                                         double alpha, double m, double k,
                                         double a0, double a1, double t0,
                                         double T, double x0_0, double x0_1,
                                         double l0, double xd0, double xd1) {
  const long i = (long)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N) return;

  const double h = (T - t0) / (double)N;

  double x_i_0, x_i_1;
  double x_i_plus_1_0, x_i_plus_1_1;
  double v_i_0, v_i_1;
  double v_i_plus_1_0, v_i_plus_1_1;
  double lambda_i_0, lambda_i_1;
  double lambda_i_plus_1_0, lambda_i_plus_1_1;
  double mu_i_0, mu_i_1;
  double mu_i_plus_1_0, mu_i_plus_1_1;

  x(Y, i, N, x0_0, x0_1, x_i_0, x_i_1);
  x(Y, i + 1, N, x0_0, x0_1, x_i_plus_1_0, x_i_plus_1_1);
  v(Y, i, N, 0, 0, v_i_0, v_i_1);
  v(Y, i + 1, N, 0, 0, v_i_plus_1_0, v_i_plus_1_1);
  lambda(Y, i, N, 0, 0, lambda_i_0, lambda_i_1);
  lambda(Y, i + 1, N, 0, 0, lambda_i_plus_1_0, lambda_i_plus_1_1);
  mu(Y, i, N, 0, 0, mu_i_0, mu_i_1);
  mu(Y, i + 1, N, 0, 0, mu_i_plus_1_0, mu_i_plus_1_1);

  const double u_t = u(Y, i, N);

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

  R[base] = delta_lambda_0 + mu_i_0;
  R[base + 1] = delta_lambda_1 + mu_i_1;

  R[base + 2] = delta_x_0 - v_i_plus_1_0;
  R[base + 3] = delta_x_1 - v_i_plus_1_1;

  R[base + 4] = delta_mu_0 + (x_i_plus_1_0 - xd0) -
                (c1 * c3_0 * c3_dot_lambda + c2 * lambda_i_0);
  R[base + 5] = delta_mu_1 + (x_i_plus_1_1 - xd1) -
                (c1 * c3_1 * c3_dot_lambda + c2 * lambda_i_1);

  R[base + 6] = delta_v_0 + c2 * c3_0 - a0 / m;
  R[base + 7] = delta_v_1 + c2 * c3_1 - a1 / m;

  const double fu_0 = -c1 * c3_0 * c3_0 - c2;
  const double fu_1 = -c1 * c3_0 * c3_1;
  R[base + 8] = alpha * u_t - (lambda_i_0 * fu_0 + lambda_i_1 * fu_1);
}

// <-----  Modified SE2 -----> //

__global__ void Modified_SE2_loop_kernel(const double* __restrict__ Y,
                                         double* __restrict__ R, long N,
                                         double alpha, double m, double k,
                                         double a0, double a1, double t0,
                                         double T, double x0_0, double x0_1,
                                         double l0, double xd0, double xd1) {
  const long i = (long)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N) return;

  const double h = (T - t0) / (double)N;

  double x_i_0, x_i_1;
  double x_i_plus_1_0, x_i_plus_1_1;
  double v_i_0, v_i_1;
  double v_i_plus_1_0, v_i_plus_1_1;
  double lambda_i_0, lambda_i_1;
  double lambda_i_plus_1_0, lambda_i_plus_1_1;
  double mu_i_0, mu_i_1;
  double mu_i_plus_1_0, mu_i_plus_1_1;

  x(Y, i, N, x0_0, x0_1, x_i_0, x_i_1);
  x(Y, i + 1, N, x0_0, x0_1, x_i_plus_1_0, x_i_plus_1_1);
  v(Y, i, N, 0, 0, v_i_0, v_i_1);
  v(Y, i + 1, N, 0, 0, v_i_plus_1_0, v_i_plus_1_1);
  lambda(Y, i, N, 0, 0, lambda_i_0, lambda_i_1);
  lambda(Y, i + 1, N, 0, 0, lambda_i_plus_1_0, lambda_i_plus_1_1);
  mu(Y, i, N, 0, 0, mu_i_0, mu_i_1);
  mu(Y, i + 1, N, 0, 0, mu_i_plus_1_0, mu_i_plus_1_1);

  const double u_t = u(Y, i, N);

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

  R[base] = delta_lambda_0 + mu_i_plus_1_0;
  R[base + 1] = delta_lambda_1 + mu_i_plus_1_1;

  R[base + 2] = delta_x_0 - v_i_0;
  R[base + 3] = delta_x_1 - v_i_1;

  R[base + 4] = delta_mu_0 + (x_i_0 - xd0) -
                (c1 * c3_0 * c3_dot_lambda_plus_1 + c2 * lambda_i_plus_1_0);
  R[base + 5] = delta_mu_1 + (x_i_1 - xd1) -
                (c1 * c3_1 * c3_dot_lambda_plus_1 + c2 * lambda_i_plus_1_1);

  R[base + 6] = delta_v_0 + c2 * c3_0 - a0 / m;
  R[base + 7] = delta_v_1 + c2 * c3_1 - a1 / m;

  const double fu_0 = -c1 * c3_0 * c3_0 - c2;
  const double fu_1 = -c1 * c3_0 * c3_1;
  R[base + 8] =
      alpha * u_t - (lambda_i_plus_1_0 * fu_0 + lambda_i_plus_1_1 * fu_1);
}

// <-----  MidPoint -----> //

__global__ void MidPoint_loop_kernel(const double* __restrict__ Y,
                                     double* __restrict__ R, long N,
                                     double alpha, double m, double k,
                                     double a0, double a1, double t0, double T,
                                     double x0_0, double x0_1, double l0,
                                     double xd0, double xd1) {
  const long i = (long)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N) return;

  const double h = (T - t0) / (double)N;

  double x_i_0, x_i_1;
  double x_i_plus_1_0, x_i_plus_1_1;
  double v_i_0, v_i_1;
  double v_i_plus_1_0, v_i_plus_1_1;
  double lambda_i_0, lambda_i_1;
  double lambda_i_plus_1_0, lambda_i_plus_1_1;
  double mu_i_0, mu_i_1;
  double mu_i_plus_1_0, mu_i_plus_1_1;

  x(Y, i, N, x0_0, x0_1, x_i_0, x_i_1);
  x(Y, i + 1, N, x0_0, x0_1, x_i_plus_1_0, x_i_plus_1_1);
  v(Y, i, N, 0, 0, v_i_0, v_i_1);
  v(Y, i + 1, N, 0, 0, v_i_plus_1_0, v_i_plus_1_1);
  lambda(Y, i, N, 0, 0, lambda_i_0, lambda_i_1);
  lambda(Y, i + 1, N, 0, 0, lambda_i_plus_1_0, lambda_i_plus_1_1);
  mu(Y, i, N, 0, 0, mu_i_0, mu_i_1);
  mu(Y, i + 1, N, 0, 0, mu_i_plus_1_0, mu_i_plus_1_1);

  const double u_i = u(Y, i, N);
  const double u_i_plus_1 = u(Y, i + 1, N);

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

  R[base] = delta_lambda_0 + mu_m_0;
  R[base + 1] = delta_lambda_1 + mu_m_1;

  R[base + 2] = delta_x_0 - v_m_0;
  R[base + 3] = delta_x_1 - v_m_1;

  R[base + 4] = delta_mu_0 + (x_m_0 - xd0) -
                (c1 * a1_0 * a1_dot_lambda_m + c2 * lambda_m_0);
  R[base + 5] = delta_mu_1 + (x_m_1 - xd1) -
                (c1 * a1_1 * a1_dot_lambda_m + c2 * lambda_m_1);

  R[base + 6] = delta_v_0 + c2 * a1_0 - a0 / m;
  R[base + 7] = delta_v_1 + c2 * a1_1 - a1 / m;

  const double fu_0 = -a1c * a3_0 * a3_0 - a2c;
  const double fu_1 = -a1c * a3_0 * a3_1;
  R[base + 8] = alpha * u_i - (lambda_i_0 * fu_0 + lambda_i_1 * fu_1);
}

__global__ void MidPoint_boundary_kernel(const double* __restrict__ Y,
                                         double* __restrict__ R, long N,
                                         double alpha, double m, double k,
                                         double x0_0, double x0_1, double l0) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  double x_n_0, x_n_1, lambda_n_0, lambda_n_1;
  x(Y, N, N, x0_0, x0_1, x_n_0, x_n_1);
  lambda(Y, N, N, 0, 0, lambda_n_0, lambda_n_1);

  const double u_n = u(Y, N, N);
  const double Lu = L(x_n_0, x_n_1, u_n);
  const double a1c = k * l0 / (m * Lu * Lu * Lu);
  const double a2c = k / m * (1.0 - l0 / Lu);
  const double a3_0 = x_n_0 - u_n;
  const double a3_1 = x_n_1;

  const double fu_0 = -a1c * a3_0 * a3_0 - a2c;
  const double fu_1 = -a1c * a3_0 * a3_1;

  R[9 * N] = alpha * u_n - (lambda_n_0 * fu_0 + lambda_n_1 * fu_1);
}

// <-----  Modified_MidPoint -----> //

__global__ void Modified_MidPoint_loop_kernel(
    const double* __restrict__ Y, double* __restrict__ R, long N, double alpha,
    double m, double k, double a0, double a1, double t0, double T, double x0_0,
    double x0_1, double l0, double xd0, double xd1) {
  const long i = (long)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N) return;

  const double h = (T - t0) / (double)N;

  double x_i_0, x_i_1;
  double x_i_plus_1_0, x_i_plus_1_1;
  double v_i_0, v_i_1;
  double v_i_plus_1_0, v_i_plus_1_1;
  double lambda_i_0, lambda_i_1;
  double lambda_i_plus_1_0, lambda_i_plus_1_1;
  double mu_i_0, mu_i_1;
  double mu_i_plus_1_0, mu_i_plus_1_1;

  x(Y, i, N, x0_0, x0_1, x_i_0, x_i_1);
  x(Y, i + 1, N, x0_0, x0_1, x_i_plus_1_0, x_i_plus_1_1);
  v(Y, i, N, 0, 0, v_i_0, v_i_1);
  v(Y, i + 1, N, 0, 0, v_i_plus_1_0, v_i_plus_1_1);
  lambda(Y, i, N, 0, 0, lambda_i_0, lambda_i_1);
  lambda(Y, i + 1, N, 0, 0, lambda_i_plus_1_0, lambda_i_plus_1_1);
  mu(Y, i, N, 0, 0, mu_i_0, mu_i_1);
  mu(Y, i + 1, N, 0, 0, mu_i_plus_1_0, mu_i_plus_1_1);

  const double u_t = u(Y, i, N);

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

  R[base] = delta_lambda_0 + mu_m_0;
  R[base + 1] = delta_lambda_1 + mu_m_1;

  R[base + 2] = delta_x_0 - v_m_0;
  R[base + 3] = delta_x_1 - v_m_1;

  R[base + 4] = delta_mu_0 + (x_m_0 - xd0) -
                (c1 * c3_0 * c3_dot_lambda_m + c2 * lambda_m_0);
  R[base + 5] = delta_mu_1 + (x_m_1 - xd1) -
                (c1 * c3_1 * c3_dot_lambda_m + c2 * lambda_m_1);

  R[base + 6] = delta_v_0 + c2 * c3_0 - a0 / m;
  R[base + 7] = delta_v_1 + c2 * c3_1 - a1 / m;

  const double fu_0 = -c1 * c3_0 * c3_0 - c2;
  const double fu_1 = -c1 * c3_0 * c3_1;
  R[base + 8] = alpha * u_t - (lambda_m_0 * fu_0 + lambda_m_1 * fu_1);
}

// <-----  Kernel Launcher -----> //

enum class Method {
  SE1,
  SE2,
  Modified_SE1,
  Modified_SE2,
  MidPoint,
  Modified_MidPoint
};

void SE_launch(Method method, const double* Y_h, double* R_h, long N,
               double alpha, double m, double k, double* a, double t0, double T,
               double* x0, double l0, double* x_d, hipStream_t stream = 0) {
  constexpr int BLOCK = 256;
  long length = 9 * N + 1;

  if (method == Method::Modified_SE1 || method == Method::Modified_SE2 ||
      method == Method::Modified_MidPoint)
    length = 9 * N;

  double* Y;
  double* R;

  HIP_CHECK(hipMalloc((void**)&Y, sizeof(double) * length));
  HIP_CHECK(hipMalloc((void**)&R, sizeof(double) * length));

  HIP_CHECK(hipMemcpyAsync(Y, Y_h, sizeof(double) * length,
                           hipMemcpyHostToDevice, stream));

  dim3 grid((N + BLOCK - 1) / BLOCK);
  if (method == Method::SE1) {
    hipLaunchKernelGGL(SE1_loop_kernel, grid, dim3(BLOCK), 0, stream, Y, R, N,
                       alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0, x_d[0],
                       x_d[1]);

    hipLaunchKernelGGL(SE1_boundary_kernel, dim3(1), dim3(1), 0, stream, Y, R,
                       N, alpha, m, k, x0[0], x0[1], l0);
  } else if (method == Method::SE2) {
    hipLaunchKernelGGL(SE2_loop_kernel, grid, dim3(BLOCK), 0, stream, Y, R, N,
                       alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0, x_d[0],
                       x_d[1]);

    hipLaunchKernelGGL(SE2_boundary_kernel, dim3(1), dim3(1), 0, stream, Y, R,
                       N, alpha, m, k, x0[0], x0[1], l0);
  } else if (method == Method::Modified_SE1) {
    hipLaunchKernelGGL(Modified_SE1_loop_kernel, grid, dim3(BLOCK), 0, stream,
                       Y, R, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1],
                       l0, x_d[0], x_d[1]);
  } else if (method == Method::Modified_SE2) {
    hipLaunchKernelGGL(Modified_SE2_loop_kernel, grid, dim3(BLOCK), 0, stream,
                       Y, R, N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1],
                       l0, x_d[0], x_d[1]);
  } else if (method == Method::MidPoint) {
    hipLaunchKernelGGL(MidPoint_loop_kernel, grid, dim3(BLOCK), 0, stream, Y, R,
                       N, alpha, m, k, a[0], a[1], t0, T, x0[0], x0[1], l0,
                       x_d[0], x_d[1]);

    hipLaunchKernelGGL(MidPoint_boundary_kernel, dim3(1), dim3(1), 0, stream, Y,
                       R, N, alpha, m, k, x0[0], x0[1], l0);
  } else if (method == Method::Modified_MidPoint) {
    hipLaunchKernelGGL(Modified_MidPoint_loop_kernel, grid, dim3(BLOCK), 0,
                       stream, Y, R, N, alpha, m, k, a[0], a[1], t0, T, x0[0],
                       x0[1], l0, x_d[0], x_d[1]);
  }

  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpyAsync(R_h, R, sizeof(double) * length,
                           hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipDeviceSynchronize());

  if (Y) {
    hipFree(Y);
    Y = nullptr;
  }
  if (R) {
    hipFree(R);
    R = nullptr;
  }
}

// <-----  Library Interface -----> //

extern "C" {
void SE1(const double* Y_h, double* R_h, long N, double alpha, double m,
         double k, double* a, double t0, double T, double* x0, double l0,
         double* x_d) {
  SE_launch(Method::SE1, Y_h, R_h, N, alpha, m, k, a, t0, T, x0, l0, x_d);
}
void SE2(const double* Y_h, double* R_h, long N, double alpha, double m,
         double k, double* a, double t0, double T, double* x0, double l0,
         double* x_d) {
  SE_launch(Method::SE2, Y_h, R_h, N, alpha, m, k, a, t0, T, x0, l0, x_d);
}
void Modified_SE1(const double* Y_h, double* R_h, long N, double alpha,
                  double m, double k, double* a, double t0, double T,
                  double* x0, double l0, double* x_d) {
  SE_launch(Method::Modified_SE1, Y_h, R_h, N, alpha, m, k, a, t0, T, x0, l0,
            x_d);
}
void Modified_SE2(const double* Y_h, double* R_h, long N, double alpha,
                  double m, double k, double* a, double t0, double T,
                  double* x0, double l0, double* x_d) {
  SE_launch(Method::Modified_SE2, Y_h, R_h, N, alpha, m, k, a, t0, T, x0, l0,
            x_d);
}
void MidPoint(const double* Y_h, double* R_h, long N, double alpha, double m,
              double k, double* a, double t0, double T, double* x0, double l0,
              double* x_d) {
  SE_launch(Method::MidPoint, Y_h, R_h, N, alpha, m, k, a, t0, T, x0, l0, x_d);
}
void Modified_MidPoint(const double* Y_h, double* R_h, long N, double alpha,
                       double m, double k, double* a, double t0, double T,
                       double* x0, double l0, double* x_d) {
  SE_launch(Method::Modified_MidPoint, Y_h, R_h, N, alpha, m, k, a, t0, T, x0,
            l0, x_d);
}
}