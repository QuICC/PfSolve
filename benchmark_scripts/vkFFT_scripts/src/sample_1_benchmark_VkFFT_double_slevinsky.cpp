//general parts
#include <stdio.h>
#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>

#if(VKFFT_BACKEND==0)
#include "vulkan/vulkan.h"
#include "glslang_c_interface.h"
#elif(VKFFT_BACKEND==1)
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#elif(VKFFT_BACKEND==2)
#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>
#elif(VKFFT_BACKEND==3)
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif 
#endif
#include "vkSolve.hpp"
#include "utils_VkFFT.h"
#include <mpir.h>
#include <mpfr.h>

typedef struct {
    mpfr_t* data;
    int n;
    int b;
} ft_mpfr_triangular_banded;
#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)
//#include <mpir.h>
double mu(double n, double alpha, double beta) {
	return sqrt(2 * (n + beta) * (n + alpha + beta) / (2 * n + alpha + beta) / (2 * n + alpha + beta + 1));
}
double nu(double n, double alpha, double beta) {
	return sqrt(2 * (n + 1) * (n + alpha + 1) / (2 * n + alpha + beta+1) / (2 * n + alpha + beta + 2));
}
mpfr_t* ft_mpfr_init_Id(int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    mpfr_t* A = (mpfr_t*) malloc(n * n * sizeof(mpfr_t));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            mpfr_init2(A[i + j * n], prec);
            mpfr_set_zero(A[i + j * n], 1);
        }
        mpfr_set_d(A[j + j * n], 1.0, rnd);
    }
    return A;
}

void ft_mpfr_destroy_triangular_banded(ft_mpfr_triangular_banded* A) {
    for (int j = 0; j < A->n; j++)
        for (int i = 0; i < A->b + 1; i++)
            mpfr_clear(A->data[i + j * (A->b + 1)]);
    free(A->data);
    free(A);
}

ft_mpfr_triangular_banded* ft_mpfr_calloc_triangular_banded(const int n, const int b, mpfr_prec_t prec) {
    mpfr_t* data = (mpfr_t*)malloc(n * (b + 1) * sizeof(mpfr_t));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < b + 1; i++) {
            mpfr_init2(data[i + j * (b + 1)], prec);
            mpfr_set_zero(data[i + j * (b + 1)], 1);
        }
    ft_mpfr_triangular_banded* A = (ft_mpfr_triangular_banded*)malloc(sizeof(ft_mpfr_triangular_banded));
    A->data = data;
    A->n = n;
    A->b = b;
    return A;
}
void ft_mpfr_set_triangular_banded_index(const ft_mpfr_triangular_banded* A, const mpfr_t v, const int i, const int j, mpfr_rnd_t rnd) {
    int n = A->n, b = A->b;
    if (0 <= i && 0 <= j && 0 <= j - i && j - i <= b && i < n && j < n)
        mpfr_set(A->data[i + (j + 1) * b], v, rnd);
}
static inline ft_mpfr_triangular_banded* ft_mpfr_create_A_legendre_to_chebyshev(const int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded* A = ft_mpfr_calloc_triangular_banded(n, 2, prec);
    mpfr_t v;
    mpfr_init2(v, prec);
    if (n > 1) {
        mpfr_set_d(v, 2.0, rnd);
        ft_mpfr_set_triangular_banded_index(A, v, 1, 1, rnd);
    }
    for (int i = 2; i < n; i++) {
        mpfr_set_d(v, -i * (i - 1.0), rnd);
        ft_mpfr_set_triangular_banded_index(A, v, i - 2, i, rnd);
        mpfr_set_d(v, i * (i + 1.0), rnd);
        ft_mpfr_set_triangular_banded_index(A, v, i, i, rnd);
    }
    mpfr_clear(v);
    return A;
}
static inline ft_mpfr_triangular_banded* ft_mpfr_create_B_legendre_to_chebyshev(const int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded* B = ft_mpfr_calloc_triangular_banded(n, 2, prec);
    mpfr_t v;
    mpfr_init2(v, prec);
    if (n > 0) {
        mpfr_set_d(v, 2.0, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, 0, 0, rnd);
    }
    if (n > 1) {
        mpfr_set_d(v, 1.0, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, 1, 1, rnd);
    }
    for (int i = 2; i < n; i++) {
        mpfr_set_d(v, -1.0, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, i - 2, i, rnd);
        mpfr_set_d(v, 1.0, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, i, i, rnd);
    }
    mpfr_clear(v);
    return B;
}

void ft_mpfr_get_triangular_banded_index(const ft_mpfr_triangular_banded* A, mpfr_t* v, const int i, const int j, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    int n = A->n, b = A->b;
    if (0 <= i && 0 <= j && 0 <= j - i && j - i <= b && i < n && j < n)
        mpfr_set(*v, A->data[i + (j + 1) * b], rnd);
    else
        mpfr_set_zero(*v, 1);
    return;
}
// Assumes eigenvectors are initialized by V[i,j] = 0 for i > j and V[j,j] ≠ 0.
void ft_mpfr_triangular_banded_eigenvectors(ft_mpfr_triangular_banded* A, ft_mpfr_triangular_banded* B, mpfr_t* V, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    int n = A->n, b1 = A->b, b2 = B->b;
    int b = MAX(b1, b2);
    mpfr_t t, t1, t2, t3, t4, lam;
    mpfr_init2(t, prec);
    mpfr_init2(t1, prec);
    mpfr_init2(t2, prec);
    mpfr_init2(t3, prec);
    mpfr_init2(t4, prec);
    mpfr_init2(lam, prec);
    for (int j = 1; j < n; j++) {
        //lam = X(get_triangular_banded_index)(A, j, j)/X(get_triangular_banded_index)(B, j, j);
        ft_mpfr_get_triangular_banded_index(A, &t1, j, j, prec, rnd);
        ft_mpfr_get_triangular_banded_index(B, &t2, j, j, prec, rnd);
        mpfr_div(lam, t1, t2, rnd);
        for (int i = j - 1; i >= 0; i--) {
            //t = 0;
            mpfr_set_zero(t, 1);
            for (int k = i + 1; k < MIN(i + b + 1, n); k++) {
                //t += (lam*X(get_triangular_banded_index)(B, i, k) - X(get_triangular_banded_index)(A, i, k))*V[k+j*n];
                mpfr_set(t3, V[k + j * n], rnd);
                ft_mpfr_get_triangular_banded_index(A, &t1, i, k, prec, rnd);
                ft_mpfr_get_triangular_banded_index(B, &t2, i, k, prec, rnd);
                mpfr_fms(t4, lam, t2, t1, rnd);
                mpfr_fma(t, t4, t3, t, rnd);
            }
            //V[i+j*n] = -t/(lam*X(get_triangular_banded_index)(B, i, i) - X(get_triangular_banded_index)(A, i, i));
            ft_mpfr_get_triangular_banded_index(A, &t1, i, i, prec, rnd);
            ft_mpfr_get_triangular_banded_index(B, &t2, i, i, prec, rnd);
            mpfr_fms(t3, lam, t2, t1, rnd);
            mpfr_div(t4, t, t3, rnd);
            mpfr_neg(V[i + j * n], t4, rnd);
        }
    }
    mpfr_clear(t1);
    mpfr_clear(t2);
    mpfr_clear(t3);
    mpfr_clear(t4);
}

mpfr_t* ft_mpfr_plan_legendre_to_chebyshev(const int normleg, const int normcheb, const int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded* A = ft_mpfr_create_A_legendre_to_chebyshev(n, prec, rnd);
    ft_mpfr_triangular_banded* B = ft_mpfr_create_B_legendre_to_chebyshev(n, prec, rnd);
    mpfr_t* V = (mpfr_t *)malloc(n * n * sizeof(mpfr_t));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            mpfr_init2(V[i + j * n], prec);
            mpfr_set_zero(V[i + j * n], 1);
        }
        mpfr_set_d(V[j + j * n], 1.0, rnd);
    }
    ft_mpfr_triangular_banded_eigenvectors(A, B, V, prec, rnd);

    mpfr_t* sclrow = (mpfr_t *) malloc(n * sizeof(mpfr_t));
    mpfr_t* sclcol = (mpfr_t *) malloc(n * sizeof(mpfr_t));
    mpfr_t t, t1, sqrtpi, sqrtpi2;
    mpfr_init2(t, prec);
    mpfr_init2(t1, prec);
    mpfr_set_d(t, 1.0, rnd);
    mpfr_t half;
    mpfr_init2(half, prec);
    mpfr_set_d(half, 0.5, rnd);
    mpfr_init2(sqrtpi, prec);
    mpfr_gamma(sqrtpi, half, rnd);
    mpfr_t sqrthalf;
    mpfr_init2(sqrthalf, prec);
    mpfr_sqrt(sqrthalf, half, rnd);
    mpfr_init2(sqrtpi2, prec);
    mpfr_mul(sqrtpi2, sqrtpi, sqrthalf, rnd);

    if (n > 0) {
        //sclrow[0] = normcheb ? sqrtpi : 1;
        mpfr_init2(sclrow[0], prec);
        normcheb ? mpfr_set(sclrow[0], sqrtpi, rnd) : mpfr_set_d(sclrow[0], 1.0, rnd);
        //sclcol[0] = normleg ? Y2(sqrt)(0.5) : 1;
        mpfr_init2(sclcol[0], prec);
        normleg ? mpfr_set(sclcol[0], sqrthalf, rnd) : mpfr_set_d(sclcol[0], 1.0, rnd);
    }
    if (n > 1) {
        //sclrow[1] = normcheb ? sqrtpi2 : 1;
        mpfr_init2(sclrow[1], prec);
        normcheb ? mpfr_set(sclrow[1], sqrtpi2, rnd) : mpfr_set_d(sclrow[1], 1.0, rnd);
        //sclcol[1] = normleg ? Y2(sqrt)(1.5) : 1;
        mpfr_init2(sclcol[1], prec);
        mpfr_set_d(t1, 1.5, rnd);
        normleg ? mpfr_sqrt(sclcol[1], t1, rnd) : mpfr_set_d(sclcol[1], 1.0, rnd);
    }
    mpfr_t num, den, rat;
    mpfr_init2(num, prec);
    mpfr_init2(den, prec);
    mpfr_init2(rat, prec);
    for (int i = 2; i < n; i++) {
        //t *= (2*i-ONE(FLT2))/(2*i);
        mpfr_set_d(num, 2 * i - 1, rnd);
        mpfr_set_d(den, 2 * i, rnd);
        mpfr_div(rat, num, den, rnd);
        mpfr_mul(t, rat, t, rnd);
        //sclrow[i] = normcheb ? sqrtpi2 : 1;
        mpfr_init2(sclrow[i], prec);
        normcheb ? mpfr_set(sclrow[i], sqrtpi2, rnd) : mpfr_set_d(sclrow[i], 1.0, rnd);
        //sclcol[i] = (normleg ? Y2(sqrt)(i+0.5) : 1)*t;
        mpfr_init2(sclcol[i], prec);
        mpfr_set_d(t1, i + 0.5, rnd);
        normleg ? mpfr_sqrt(sclcol[i], t1, rnd) : mpfr_set_d(sclcol[i], 1.0, rnd);
        mpfr_mul(sclcol[i], t, sclcol[i], rnd);
    }

    for (int j = 0; j < n; j++)
        for (int i = j; i >= 0; i -= 2) {
            //V[i+j*n] = sclrow[i]*Vl[i+j*n]*sclcol[j];
            mpfr_mul(V[i + j * n], sclrow[i], V[i + j * n], rnd);
            mpfr_mul(V[i + j * n], V[i + j * n], sclcol[j], rnd);
        }
    ft_mpfr_destroy_triangular_banded(A);
    ft_mpfr_destroy_triangular_banded(B);
    for (int i = 0; i < n; i++) {
        mpfr_clear(sclrow[i]);
        mpfr_clear(sclcol[i]);
    }
    free(sclrow);
    free(sclcol);
    mpfr_clear(t);
    mpfr_clear(t1);
    mpfr_clear(sqrtpi);
    mpfr_clear(sqrtpi2);
    mpfr_clear(half);
    mpfr_clear(sqrthalf);
    mpfr_clear(num);
    mpfr_clear(den);
    mpfr_clear(rat);
    return V;
}
void ft_mpfr_trmv(char TRANS, int n, mpfr_t* A, int LDA, mpfr_t* x, mpfr_rnd_t rnd) {
    if (TRANS == 'N') {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < j; i++)
                mpfr_fma(x[i], A[i + j * LDA], x[j], x[i], rnd);
            mpfr_mul(x[j], A[j + j * LDA], x[j], rnd);
        }
    }
    else if (TRANS == 'T') {
        for (int i = n - 1; i >= 0; i--) {
            mpfr_mul(x[i], A[i + i * LDA], x[i], rnd);
            for (int j = i - 1; j >= 0; j--)
                mpfr_fma(x[i], A[j + i * LDA], x[j], x[i], rnd);
        }
    }
}
void ft_mpfr_trmm(char TRANS, int n, mpfr_t* A, int LDA, mpfr_t* B, int LDB, int N, mpfr_rnd_t rnd) {
#pragma omp parallel for
    for (int j = 0; j < N; j++)
        ft_mpfr_trmv(TRANS, n, A, LDA, B + j * LDB, rnd);
}
void ft_mpfr_destroy_plan(mpfr_t* A, int n) {
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            mpfr_clear(A[i + j * n]);
    free(A);
}
static inline ft_mpfr_triangular_banded* ft_mpfr_create_A_chebyshev_to_legendre(const int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded* A = ft_mpfr_calloc_triangular_banded(n, 2, prec);
    mpfr_t v, w, x;
    mpfr_init2(v, prec);
    mpfr_init2(w, prec);
    mpfr_init2(x, prec);
    if (n > 1) {
        mpfr_set_d(w, 1, rnd);
        mpfr_set_d(x, 3, rnd);
        mpfr_div(v, w, x, rnd);
        ft_mpfr_set_triangular_banded_index(A, v, 1, 1, rnd);
    }
    for (int i = 2; i < n; i++) {
        mpfr_set_d(w, -(i + 1.0) * (i + 1.0), rnd);
        mpfr_set_d(x, 2 * i + 1, rnd);
        mpfr_div(v, w, x, rnd);
        ft_mpfr_set_triangular_banded_index(A, v, i - 2, i, rnd);
        mpfr_set_d(w, 1.0 * i * i, rnd);
        mpfr_div(v, w, x, rnd);
        ft_mpfr_set_triangular_banded_index(A, v, i, i, rnd);
    }
    mpfr_clear(v);
    mpfr_clear(w);
    mpfr_clear(x);
    return A;
}

static inline ft_mpfr_triangular_banded* ft_mpfr_create_B_chebyshev_to_legendre(const int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded* B = ft_mpfr_calloc_triangular_banded(n, 2, prec);
    mpfr_t v, w, x;
    mpfr_init2(v, prec);
    mpfr_init2(w, prec);
    mpfr_init2(x, prec);
    if (n > 0) {
        mpfr_set_d(v, 1, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, 0, 0, rnd);
    }
    if (n > 1) {
        mpfr_set_d(w, 1, rnd);
        mpfr_set_d(x, 3, rnd);
        mpfr_div(v, w, x, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, 1, 1, rnd);
    }
    for (int i = 2; i < n; i++) {
        mpfr_set_d(w, -1, rnd);
        mpfr_set_d(x, 2 * i + 1, rnd);
        mpfr_div(v, w, x, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, i - 2, i, rnd);
        mpfr_set_d(w, 1, rnd);
        mpfr_div(v, w, x, rnd);
        ft_mpfr_set_triangular_banded_index(B, v, i, i, rnd);
    }
    mpfr_clear(v);
    mpfr_clear(w);
    mpfr_clear(x);
    return B;
}
mpfr_t* ft_mpfr_plan_chebyshev_to_legendre(const int normcheb, const int normleg, const int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    ft_mpfr_triangular_banded* A = ft_mpfr_create_A_chebyshev_to_legendre(n, prec, rnd);
    ft_mpfr_triangular_banded* B = ft_mpfr_create_B_chebyshev_to_legendre(n, prec, rnd);
    mpfr_t* V = (mpfr_t *) malloc(n * n * sizeof(mpfr_t));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            mpfr_init2(V[i + j * n], prec);
            mpfr_set_zero(V[i + j * n], 1);
        }
        mpfr_set_d(V[j + j * n], 1.0, rnd);
    }
    ft_mpfr_triangular_banded_eigenvectors(A, B, V, prec, rnd);

    mpfr_t* sclrow = (mpfr_t *) malloc(n * sizeof(mpfr_t));
    mpfr_t* sclcol = (mpfr_t *) malloc(n * sizeof(mpfr_t));
    mpfr_t t, t1, sqrtpi, sqrt_1_pi, sqrt_2_pi;
    mpfr_init2(t, prec);
    mpfr_init2(t1, prec);
    mpfr_set_d(t, 1.0, rnd);
    mpfr_t half;
    mpfr_init2(half, prec);
    mpfr_set_d(half, 0.5, rnd);
    mpfr_init2(sqrtpi, prec);
    mpfr_gamma(sqrtpi, half, rnd);
    mpfr_init2(sqrt_1_pi, prec);
    mpfr_div(sqrt_1_pi, t, sqrtpi, rnd);
    mpfr_t sqrt2;
    mpfr_init2(sqrt2, prec);
    mpfr_sqrt_ui(sqrt2, 2, rnd);
    mpfr_init2(sqrt_2_pi, prec);
    mpfr_mul(sqrt_2_pi, sqrt_1_pi, sqrt2, rnd);

    if (n > 0) {
        //sclrow[0] = normleg ? 1/Y2(sqrt)(0.5) : 1;
        mpfr_init2(sclrow[0], prec);
        normleg ? mpfr_set(sclrow[0], sqrt2, rnd) : mpfr_set_d(sclrow[0], 1.0, rnd);
        //sclcol[0] = normcheb ? sqrt_1_pi : 1;
        mpfr_init2(sclcol[0], prec);
        normcheb ? mpfr_set(sclcol[0], sqrt_1_pi, rnd) : mpfr_set_d(sclcol[0], 1.0, rnd);
    }
    if (n > 1) {
        //sclrow[1] = normleg ? 1/Y2(sqrt)(1.5) : 1;
        mpfr_init2(sclrow[1], prec);
        mpfr_set_d(t1, 1.5, rnd);
        normleg ? mpfr_rec_sqrt(sclrow[1], t1, rnd) : mpfr_set_d(sclrow[1], 1.0, rnd);
        //sclcol[1] = normcheb ? sqrt_2_pi : 1;
        mpfr_init2(sclcol[1], prec);
        normcheb ? mpfr_set(sclcol[1], sqrt_2_pi, rnd) : mpfr_set_d(sclcol[1], 1.0, rnd);
    }
    mpfr_t num, den, rat;
    mpfr_init2(num, prec);
    mpfr_init2(den, prec);
    mpfr_init2(rat, prec);
    for (int i = 2; i < n; i++) {
        //t *= (2*i)/(2*i-ONE(FLT2));
        mpfr_set_d(num, 2 * i, rnd);
        mpfr_set_d(den, 2 * i - 1, rnd);
        mpfr_div(rat, num, den, rnd);
        mpfr_mul(t, rat, t, rnd);
        //sclrow[i] = normleg ? 1/Y2(sqrt)(i+0.5) : 1;
        mpfr_init2(sclrow[i], prec);
        mpfr_set_d(t1, i + 0.5, rnd);
        normleg ? mpfr_rec_sqrt(sclrow[i], t1, rnd) : mpfr_set_d(sclrow[i], 1.0, rnd);
        //sclcol[i] = (normcheb ? sqrt_2_pi : 1)*t;
        mpfr_init2(sclcol[i], prec);
        normcheb ? mpfr_set(sclcol[i], sqrt_2_pi, rnd) : mpfr_set_d(sclcol[i], 1.0, rnd);
        mpfr_mul(sclcol[i], t, sclcol[i], rnd);
    }

    for (int j = 0; j < n; j++)
        for (int i = j; i >= 0; i -= 2) {
            //V[i+j*n] = sclrow[i]*Vl[i+j*n]*sclcol[j];
            mpfr_mul(V[i + j * n], sclrow[i], V[i + j * n], rnd);
            mpfr_mul(V[i + j * n], V[i + j * n], sclcol[j], rnd);
        }
    ft_mpfr_destroy_triangular_banded(A);
    ft_mpfr_destroy_triangular_banded(B);
    for (int i = 0; i < n; i++) {
        mpfr_clear(sclrow[i]);
        mpfr_clear(sclcol[i]);
    }
    free(sclrow);
    free(sclcol);
    mpfr_clear(t);
    mpfr_clear(t1);
    mpfr_clear(sqrtpi);
    mpfr_clear(sqrt_1_pi);
    mpfr_clear(sqrt_2_pi);
    mpfr_clear(half);
    mpfr_clear(sqrt2);
    mpfr_clear(num);
    mpfr_clear(den);
    mpfr_clear(rat);
    return V;
}
void ft_mpfr_norm_1arg(mpfr_t* ret, mpfr_t* A, int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    mpfr_set_d(*ret, 0.0, rnd);
    for (int i = 0; i < n; i++)
        mpfr_fma(*ret, A[i], A[i], *ret, rnd);
    mpfr_sqrt(*ret, *ret, rnd);
}

void ft_mpfr_norm_2arg(mpfr_t* ret, mpfr_t* A, mpfr_t* B, int n, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    mpfr_t t;
    mpfr_set_d(*ret, 0.0, rnd);
    mpfr_init2(t, prec);
    for (int i = 0; i < n; i++) {
        mpfr_sub(t, A[i], B[i], rnd);
        mpfr_fma(*ret, t, t, *ret, rnd);
    }
    mpfr_sqrt(*ret, *ret, rnd);
    mpfr_clear(t);
}
void ft_mpfr_checktest(mpfr_t* err, double cst, int* checksum, mpfr_prec_t prec, mpfr_rnd_t rnd) {
    mpfr_t t1, t2, one, oneplus, eps;
    mpfr_init2(t1, prec);
    mpfr_init2(t2, prec);
    mpfr_init2(one, prec);
    mpfr_init2(oneplus, prec);
    mpfr_init2(eps, prec);
    mpfr_set_d(one, 1.0, rnd);
    mpfr_set_d(oneplus, 1.0, rnd);
    mpfr_nextabove(oneplus);
    mpfr_sub(eps, oneplus, one, rnd);
    mpfr_abs(t1, *err, rnd);
    mpfr_mul_d(t2, eps, cst, rnd);
    if (mpfr_cmp(t1, t2) < 0) printf("✓\n");
    else { printf("×\n"); (*checksum)++; }
    mpfr_clear(t1);
    mpfr_clear(t2);
    mpfr_clear(one);
    mpfr_clear(oneplus);
    mpfr_clear(eps);
}
#define FLT double
#define FLT2 long double
typedef struct tbstruct_FMM tb_eigen_FMM;
typedef struct hmat hierarchicalmatrix;
typedef struct {
    FLT* A;
    int m;
    int n;
} densematrix;

typedef struct {
    FLT* U;
    FLT* S;
    FLT* V;
    FLT* t1;
    FLT* t2;
    int m;
    int n;
    int r;
    int p;
    char N;
} lowrankmatrix;
struct hmat {
    hierarchicalmatrix** hierarchicalmatrices;
    densematrix** densematrices;
    lowrankmatrix** lowrankmatrices;
    int* hash;
    int M;
    int N;
    int m;
    int n;
};
typedef struct {
    tb_eigen_FMM* F;
    FLT* s;
    FLT* c;
    FLT* t;
    int n;
} btb_eigen_FMM;
typedef struct {
    FLT* data;
    int n;
    int b;
} triangular_banded;

typedef struct {
    int* p;
    int* q;
    FLT* v;
    int m;
    int n;
    int nnz;
} sparse;

typedef struct {
    FLT* data;
    int m;
    int n;
    int l;
    int u;
} banded;

typedef struct {
    triangular_banded* data[2][2];
    int n;
    int b;
} block_2x2_triangular_banded;
typedef struct {
    int start;
    int stop;
} unitrange;
struct tbstruct_FMM {
    hierarchicalmatrix* F0;
    tb_eigen_FMM* F1;
    tb_eigen_FMM* F2;
    sparse* S;
    FLT* V;
    FLT* X;
    FLT* Y;
    FLT* t1;
    FLT* t2;
    FLT* lambda;
    int* p1;
    int* p2;
    int n;
    int b;
};
#define ZERO(FLT) ((FLT) 0)
#define ONE(FLT) ((FLT) 1)
#define TWO(FLT) ((FLT) 2)
#define M_EPS          0x1p-52                /* pow(2.0, -52)      */
static inline double eps(void) { return M_EPS; }
#define TB_EIGEN_BLOCKSIZE 32
#define BLOCKRANK 2*((int) floor(-log(eps())/2.271667761226165))
#define BLOCKSIZE 4*BLOCKRANK
#define FT_GET_THREAD_NUM() 0
#define FT_GET_NUM_THREADS() 1
#define FT_GET_MAX_THREADS() 1
#define FT_SET_NUM_THREADS(x)
FLT thresholded_cauchykernel(FLT x, FLT y) {
    if (fabs(x - y) < 16 * sqrt(eps()) * MAX(fabs(x), fabs(y)))
        return 0;
    else
        return 1 / (x - y);
}
static inline FLT thresholded_cauchykernel2(FLT x, FLT y) { return thresholded_cauchykernel(x, y); }

triangular_banded* calloc_triangular_banded(const int n, const int b) {
    FLT* data = (FLT*)calloc(n * (b + 1), sizeof(FLT));
    triangular_banded* A = (triangular_banded*)malloc(sizeof(triangular_banded));
    A->data = data;
    A->n = n;
    A->b = b;
    return A;
}
void set_triangular_banded_index(const triangular_banded* A, const FLT v, const int i, const int j) {
    FLT* data = A->data;
    int n = A->n, b = A->b;
    if (0 <= i && 0 <= j && 0 <= j - i && j - i <= b && i < n && j < n)
        data[i + (j + 1) * b] = v;
}
triangular_banded* create_A_legendre_to_chebyshev(const int norm, const int n) {
    triangular_banded* A = calloc_triangular_banded(n, 2);

    /*ft_mpfr_triangular_banded* A2 = ft_mpfr_calloc_triangular_banded(n, 2, 128);
    mpfr_t v;
    mpfr_init2(v, 128);
    if (n > 1) {
        mpfr_set_d(v, 2.0, MPFR_RNDN);
        ft_mpfr_set_triangular_banded_index(A2, v, 1, 1, MPFR_RNDN);
    }
    for (int i = 2; i < n; i++) {
        mpfr_set_d(v, -i * (i - 1.0), MPFR_RNDN);
        ft_mpfr_set_triangular_banded_index(A2, v, i - 2, i, MPFR_RNDN);
        mpfr_set_d(v, i * (i + 1.0), MPFR_RNDN);
        ft_mpfr_set_triangular_banded_index(A2, v, i, i, MPFR_RNDN);
    }
    mpfr_clear(v);
    for (int i = 0; i < A->n * (A->b + 1); i++) {
        A->data[i] = mpfr_get_d(A2->data[i], MPFR_RNDN);
    }
    ft_mpfr_destroy_triangular_banded(A2);*/
    if (n > 1)
        set_triangular_banded_index(A, 2, 1, 1);
    for (int i = 2; i < n; i++) {
        set_triangular_banded_index(A, -i * (i - ONE(FLT)), i - 2, i);
        set_triangular_banded_index(A, i * (i + ONE(FLT)), i, i);
    }
    return A;
}

triangular_banded* create_B_legendre_to_chebyshev(const int norm, const int n) {
    triangular_banded* B = calloc_triangular_banded(n, 2);
    /*ft_mpfr_triangular_banded* B2 = ft_mpfr_calloc_triangular_banded(n, 2, 128);
    mpfr_t v;
    mpfr_init2(v, 128);
    if (n > 0) {
        mpfr_set_d(v, 2.0, MPFR_RNDN);
        ft_mpfr_set_triangular_banded_index(B2, v, 0, 0, MPFR_RNDN);
    }
    if (n > 1) {
        mpfr_set_d(v, 1.0, MPFR_RNDN);
        ft_mpfr_set_triangular_banded_index(B2, v, 1, 1, MPFR_RNDN);
    }
    for (int i = 2; i < n; i++) {
        mpfr_set_d(v, -1.0, MPFR_RNDN);
        ft_mpfr_set_triangular_banded_index(B2, v, i - 2, i, MPFR_RNDN);
        mpfr_set_d(v, 1.0, MPFR_RNDN);
        ft_mpfr_set_triangular_banded_index(B2, v, i, i, MPFR_RNDN);
    }
    mpfr_clear(v);
    for (int i = 0; i < B->n * (B->b + 1); i++) {
        B->data[i] = mpfr_get_d(B2->data[i], MPFR_RNDN);
    }
    ft_mpfr_destroy_triangular_banded(B2);*/
    if (n > 0)
        set_triangular_banded_index(B, norm ? sqrt(2) : 2, 0, 0);
    if (n > 1)
        set_triangular_banded_index(B, 1, 1, 1);
    for (int i = 2; i < n; i++) {
        set_triangular_banded_index(B, -1, i - 2, i);
        set_triangular_banded_index(B, 1, i, i);
    }
    return B;
}
void create_legendre_to_chebyshev_diagonal_connection_coefficient(const int normleg, const int normcheb, const int n, FLT* D, const int INCD) {
    if (normleg) {
        if (normcheb) {
            if (n > 0)
                D[0] = sqrt(0.5) * tgamma(0.5);
            if (n > 1)
                D[INCD] = sqrt(1.5) * D[0];
            for (int i = 2; i < n; i++)
                D[i * INCD] = sqrt((2 * i + 1) * (2 * i - ONE(FLT))) * D[(i - 1) * INCD] / (2 * i);
        }
        else {
            if (n > 0)
                D[0] = sqrt(0.5);
            if (n > 1)
                D[INCD] = sqrt(1.5);
            for (int i = 2; i < n; i++)
                D[i * INCD] = sqrt((2 * i + 1) * (2 * i - ONE(FLT))) * D[(i - 1) * INCD] / (2 * i);
        }
    }
    else {
        if (normcheb) {
            if (n > 0)
                D[0] = tgamma(0.5);
            if (n > 1)
                D[INCD] = D[0] / sqrt(2);
            for (int i = 2; i < n; i++)
                D[i * INCD] = (2 * i - 1) * D[(i - 1) * INCD] / (2 * i);
        }
        else {
            if (n > 0)
                D[0] = 1;
            if (n > 1)
                D[INCD] = 1;
            for (int i = 2; i < n; i++)
                D[i * INCD] = (2 * i - 1) * D[(i - 1) * INCD] / (2 * i);
        }
    }
}
FLT get_triangular_banded_index(const triangular_banded* A, const int i, const int j) {
    FLT* data = A->data;
    int n = A->n, b = A->b;
    if (0 <= i && 0 <= j && 0 <= j - i && j - i <= b && i < n && j < n)
        return data[i + (j + 1) * b];
    else
        return 0;
}
void triangular_banded_eigenvalues(triangular_banded* A, triangular_banded* B, FLT* lambda) {
    for (int j = 0; j < A->n; j++)
        lambda[j] = get_triangular_banded_index(A, j, j) / get_triangular_banded_index(B, j, j);
}

// Assumes eigenvectors are initialized by V[i,j] = 0 for i > j and V[j,j] ≠ 0.
void triangular_banded_eigenvectors(triangular_banded* A, triangular_banded* B, FLT* V) {
    int n = A->n, b = MAX(A->b, B->b);
    FLT t, kt, d, kd, lam;
    for (int j = 1; j < n; j++) {
        lam = get_triangular_banded_index(A, j, j) / get_triangular_banded_index(B, j, j);
        for (int i = j - 1; i >= 0; i--) {
            t = kt = 0;
            for (int k = i + 1; k < MIN(i + b + 1, n); k++) {
                t += (get_triangular_banded_index(A, i, k) - lam * get_triangular_banded_index(B, i, k)) * V[k + j * n];
                kt += (fabs(get_triangular_banded_index(A, i, k)) + fabs(lam * get_triangular_banded_index(B, i, k))) * fabs(V[k + j * n]);
            }
            d = lam * get_triangular_banded_index(B, i, i) - get_triangular_banded_index(A, i, i);
            kd = fabs(lam * get_triangular_banded_index(B, i, i)) + fabs(get_triangular_banded_index(A, i, i));
            if (fabs(d) < 4 * kd * eps() || fabs(t) < 4 * kt * eps())
                V[i + j * n] = 0;
            else
                V[i + j * n] = t / d;
        }
    }
}
triangular_banded* view_triangular_banded(const triangular_banded* A, const unitrange i) {
    triangular_banded* V = (triangular_banded*)malloc(sizeof(triangular_banded));
    V->data = A->data + i.start * (A->b + 1);
    V->n = i.stop - i.start;
    V->b = A->b;
    return V;
}
// x ← A⁻¹*x, x ← A⁻ᵀ*x
void tbsv(char TRANS, triangular_banded* A, FLT* x) {
    int n = A->n, b = A->b;
    FLT* data = A->data, t;
    if (TRANS == 'N') {
        for (int i = n - 1; i >= 0; i--) {
            t = 0;
            for (int k = i + 1; k < MIN(i + b + 1, n); k++)
                t += data[i + (k + 1) * b] * x[k];
            x[i] = (x[i] - t) / data[i + (i + 1) * b];
        }
    }
    else if (TRANS == 'T') {
        for (int i = 0; i < n; i++) {
            t = 0;
            for (int k = MAX(i - b, 0); k < i; k++)
                t += data[k + (i + 1) * b] * x[k];
            x[i] = (x[i] - t) / data[i + (i + 1) * b];
        }
    }
}
// x ← A⁻¹*x, x ← A⁻ᵀ*x
void trsv(char TRANS, int n, FLT* A, int LDA, FLT* x) {
    if (TRANS == 'N') {
        for (int j = n - 1; j >= 0; j--) {
            x[j] /= A[j + j * LDA];
            for (int i = 0; i < j; i++)
                x[i] -= A[i + j * LDA] * x[j];
        }
    }
    else if (TRANS == 'T') {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++)
                x[i] -= A[j + i * LDA] * x[j];
            x[i] /= A[i + i * LDA];
        }
    }
}
static inline int size_hierarchicalmatrix(hierarchicalmatrix* H, int k) {
    if (k == 1) return H->m;
    else if (k == 2) return H->n;
    else return 1;
}
// y ← α*A*x + β*y, y ← α*Aᵀ*x + β*y
void gemv(char TRANS, int m, int n, FLT alpha, FLT* A, int LDA, FLT* x, FLT beta, FLT* y) {
    FLT t;
    if (TRANS == 'N') {
        if (beta != 1) {
            if (beta == 0)
                for (int i = 0; i < m; i++)
                    y[i] = 0;
            else
                for (int i = 0; i < m; i++)
                    y[i] = beta * y[i];
        }
        for (int j = 0; j < n; j++) {
            t = alpha * x[j];
            for (int i = 0; i < m; i++)
                y[i] += A[i + j * LDA] * t;
        }
    }
    else if (TRANS == 'T') {
        if (beta != 1) {
            if (beta == 0)
                for (int i = 0; i < n; i++)
                    y[i] = 0;
            else
                for (int i = 0; i < n; i++)
                    y[i] = beta * y[i];
        }
        for (int i = 0; i < n; i++) {
            t = 0;
            for (int j = 0; j < m; j++)
                t += A[j + i * LDA] * x[j];
            //printf("%.17e %.17e\n", y[i], t);
            y[i] += alpha * t;
        }
    }
}
void demv(char TRANS, FLT alpha, densematrix* A, FLT* x, FLT beta, FLT* y) {
    gemv(TRANS, A->m, A->n, alpha, A->A, A->m, x, beta, y);
}
// y ← α*(USVᵀ)*x + β*y, y ← α*(VSᵀUᵀ)*x + β*y
void lrmv(char TRANS, FLT alpha, lowrankmatrix* L, FLT* x, FLT beta, FLT* y) {
    int m = L->m, n = L->n, r = L->r;
    FLT* t1 = L->t1 + r * FT_GET_THREAD_NUM();
    FLT* t2 = L->t2 + r * FT_GET_THREAD_NUM();
    if (TRANS == 'N') {
        if (L->N == '2') {
            gemv('T', n, r, 1, L->V, n, x, 0, t1);
            gemv('N', m, r, alpha, L->U, m, t1, beta, y);
        }
        else if (L->N == '3') {
            gemv('T', n, r, 1, L->V, n, x, 0, t1);
            gemv('N', r, r, 1, L->S, r, t1, 0, t2);
            gemv('N', m, r, alpha, L->U, m, t2, beta, y);
        }
    }
    else if (TRANS == 'T') {
        if (L->N == '2') {
            gemv('T', m, r, 1, L->U, m, x, 0, t1);
            gemv('N', n, r, alpha, L->V, n, t1, beta, y);
        }
        else if (L->N == '3') {
            gemv('T', m, r, 1, L->U, m, x, 0, t1);
            gemv('T', r, r, 1, L->S, r, t1, 0, t2);
            gemv('N', n, r, alpha, L->V, n, t2, beta, y);
        }
    }
}
static inline int size_densematrix(densematrix* A, int k) {
    if (k == 1) return A->m;
    else if (k == 2) return A->n;
    else return 1;
}

static inline int size_lowrankmatrix(lowrankmatrix* L, int k) {
    if (k == 1) return L->m;
    else if (k == 2) return L->n;
    else return 1;
}
static inline int blocksize_hierarchicalmatrix(hierarchicalmatrix* H, int m, int n, int k) {
    int M = H->M, N = H->N;
    switch (H->hash[n * M + m]) {
    case 1: return size_hierarchicalmatrix(H->hierarchicalmatrices[n * M + m], k);
    case 2: return size_densematrix(H->densematrices[n * M + m], k);
    case 3: return size_lowrankmatrix(H->lowrankmatrices[n * M + m], k);
    default: return 1;
    }
}
// y ← α*H*x + β*y, y ← α*Hᵀ*x + β*y
void ghmv(char TRANS, FLT alpha, hierarchicalmatrix* H, FLT* x, FLT beta, FLT* y) {
    int M = H->M, N = H->N;
    int p, q = 0;
    if (TRANS == 'N') {
        if (beta != 1) {
            if (beta == 0)
                for (int i = 0; i < size_hierarchicalmatrix(H, 1); i++)
                    y[i] = 0;
            else
                for (int i = 0; i < size_hierarchicalmatrix(H, 1); i++)
                    y[i] = beta * y[i];
        }
        for (int n = 0; n < N; n++) {
            p = 0;
            for (int m = 0; m < M; m++) {
                switch (H->hash[n * M + m]) {
                case 1: ghmv(TRANS, alpha, H->hierarchicalmatrices[n * M + m], x + q, 1, y + p); break;
                case 2: demv(TRANS, alpha, H->densematrices[n * M + m], x + q, 1, y + p); break;
                case 3: lrmv(TRANS, alpha, H->lowrankmatrices[n * M + m], x + q, 1, y + p); break;
                }
                p += blocksize_hierarchicalmatrix(H, m, N - 1, 1);
            }
            q += blocksize_hierarchicalmatrix(H, 0, n, 2);
        }
    }
    else if (TRANS == 'T') {
        if (beta != 1) {
            if (beta == 0)
                for (int i = 0; i < size_hierarchicalmatrix(H, 2); i++)
                    y[i] = 0;
            else
                for (int i = 0; i < size_hierarchicalmatrix(H, 2); i++)
                    y[i] = beta * y[i];
        }
        for (int m = 0; m < M; m++) {
            p = 0;
            for (int n = 0; n < N; n++) {
                switch (H->hash[n * M + m]) {
                case 1: ghmv(TRANS, alpha, H->hierarchicalmatrices[n * M + m], x + q, 1, y + p); break;
                case 2: demv(TRANS, alpha, H->densematrices[n * M + m], x + q, 1, y + p); break;
                case 3: lrmv(TRANS, alpha, H->lowrankmatrices[n * M + m], x + q, 1, y + p); break;
                }
                p += blocksize_hierarchicalmatrix(H, 0, n, 2);
            }
            q += blocksize_hierarchicalmatrix(H, m, N - 1, 1);
        }
    }
}
// x ← A⁻¹*x, x ← A⁻ᵀ*x
void bfsv(char TRANS, tb_eigen_FMM* F, FLT* x) {
    int n = F->n;
    if (n < TB_EIGEN_BLOCKSIZE)
        trsv(TRANS, n, F->V, n, x);
    else {
        int s = n >> 1, b = F->b;
        FLT* t1 = F->t1 + s * FT_GET_THREAD_NUM();
        FLT* t2 = F->t2 + (n - s) * FT_GET_THREAD_NUM();
        int* p1 = F->p1, * p2 = F->p2;
        sparse* S = F->S;
        if (TRANS == 'N') {
            bfsv(TRANS, F->F1, x);
            bfsv(TRANS, F->F2, x + s);
            // C(Λ₁, Λ₂) ∘ (-XYᵀ)
            for (int k = 0; k < b; k++) {
                for (int i = 0; i < n - s; i++)
                    t2[i] = F->Y[p2[i] + k * (n - s)] * x[p2[i] + s];
                ghmv(TRANS, 1, F->F0, t2, 0, t1);
                for (int i = 0; i < s; i++)
                    x[p1[i]] += t1[i] * F->X[p1[i] + k * s];
            }
            for (int l = 0; l < S->nnz; l++)
                x[S->p[l]] -= S->v[l] * x[S->q[l] + s];
        }
        else if (TRANS == 'T') {
            // C(Λ₁, Λ₂) ∘ (-XYᵀ)
            for (int k = 0; k < b; k++) {
                for (int i = 0; i < s; i++)
                    t1[i] = F->X[p1[i] + k * s] * x[p1[i]];
                ghmv(TRANS, 1, F->F0, t1, 0, t2);
                for (int i = 0; i < n - s; i++)
                    x[p2[i] + s] += t2[i] * F->Y[p2[i] + k * (n - s)];
            }
            for (int l = 0; l < S->nnz; l++)
                x[S->q[l] + s] -= S->v[l] * x[S->p[l]];
            bfsv(TRANS, F->F1, x);
            bfsv(TRANS, F->F2, x + s);
        }
    }
}
// x ← A*x, x ← Aᵀ*x
void trmv(char TRANS, int n, FLT* A, int LDA, FLT* x) {
    if (TRANS == 'N') {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < j; i++)
                x[i] += A[i + j * LDA] * x[j];
            x[j] *= A[j + j * LDA];
        }
    }
    else if (TRANS == 'T') {
        for (int i = n - 1; i >= 0; i--) {
            x[i] *= A[i + i * LDA];
            for (int j = i - 1; j >= 0; j--)
                x[i] += A[j + i * LDA] * x[j];
        }
    }
}
// x ← A*x, x ← Aᵀ*x
void bfmv(char TRANS, tb_eigen_FMM* F, FLT* x) {
    int n = F->n;
    if (n < TB_EIGEN_BLOCKSIZE)
        trmv(TRANS, n, F->V, n, x);
    else {
        int s = n >> 1, b = F->b;
        FLT* t1 = F->t1 + s * FT_GET_THREAD_NUM();
        FLT* t2 = F->t2 + (n - s) * FT_GET_THREAD_NUM();
        int* p1 = F->p1;
        int* p2 = F->p2;
        sparse* S = F->S;
        if (TRANS == 'N') {
            // C(Λ₁, Λ₂) ∘ (-XYᵀ)
            for (int k = 0; k < b; k++) {
                for (int i = 0; i < n - s; i++)
                    t2[i] = F->Y[p2[i] + k * (n - s)] * x[p2[i] + s];
                ghmv(TRANS, -1, F->F0, t2, 0, t1);
                for (int i = 0; i < s; i++)
                    x[p1[i]] += t1[i] * F->X[p1[i] + k * s];
            }
            for (int l = 0; l < S->nnz; l++)
                x[S->p[l]] += S->v[l] * x[S->q[l] + s];
            bfmv(TRANS, F->F1, x);
            bfmv(TRANS, F->F2, x + s);
        }
        else if (TRANS == 'T') {
            bfmv(TRANS, F->F1, x);
            bfmv(TRANS, F->F2, x + s);
            // C(Λ₁, Λ₂) ∘ (-XYᵀ)
            for (int k = 0; k < b; k++) {
                for (int i = 0; i < s; i++)
                    t1[i] = F->X[p1[i] + k * s] * x[p1[i]];
                ghmv(TRANS, -1, F->F0, t1, 0, t2);
                for (int i = 0; i < n - s; i++)
                    x[p2[i] + s] += t2[i] * F->Y[p2[i] + k * (n - s)];
            }
            for (int l = 0; l < S->nnz; l++)
                x[S->q[l] + s] += S->v[l] * x[S->p[l]];
        }
    }
}
/*
These versions of quicksort sort `a` in-place according to the `by` ordering on
`FLT` types. They also return the permutation `p`.
*/
void swap(FLT* a, int i, int j) {
    FLT temp = a[i];
    a[i] = a[j];
    a[j] = temp;
}

void swapi(int* p, int i, int j) {
    int temp = p[i];
    p[i] = p[j];
    p[j] = temp;
}
static FLT selectpivot_1arg(FLT* a, int* p, int lo, int hi, int (*by)(FLT x, FLT y)) {
    int mid = (lo + hi) / 2;
    if (by(a[mid], a[lo])) {
        swap(a, lo, mid);
        swapi(p, lo, mid);
    }
    if (by(a[hi], a[lo])) {
        swap(a, lo, hi);
        swapi(p, lo, hi);
    }
    if (by(a[mid], a[hi])) {
        swap(a, mid, hi);
        swapi(p, mid, hi);
    }
    return a[hi];
}
static int partition_1arg(FLT* a, int* p, int lo, int hi, int (*by)(FLT x, FLT y)) {
    int i = lo - 1, j = hi + 1;
    FLT pivot = selectpivot_1arg(a, p, lo, hi, by);
    while (1) {
        do i += 1; while (by(a[i], pivot));
        do j -= 1; while (by(pivot, a[j]));
        if (i >= j) break;
        swap(a, i, j);
        swapi(p, i, j);
    }
    return j;
}
void quicksort_1arg(FLT* a, int* p, int lo, int hi, int (*by)(FLT x, FLT y)) {
    if (lo < hi) {
        int mid = partition_1arg(a, p, lo, hi, by);
        quicksort_1arg(a, p, lo, mid, by);
        quicksort_1arg(a, p, mid + 1, hi, by);
    }
}
int lt(FLT x, FLT y) { return x < y; }
// Assumptions: x, y are non-decreasing.
static inline int count_intersections(const int m, const FLT* x, const int n, const FLT* y, const FLT epsilon) {
    int istart = 0, idx = 0;
    for (int j = 0; j < n; j++) {
        int i = istart;
        int thefirst = 1;
        while (i < m) {
            if (fabs(x[i] - y[j]) < epsilon * MAX(fabs(x[i]), fabs(y[j]))) {
                idx++;
                if (thefirst) {
                    istart = i;
                    thefirst--;
                }
            }
            else if (x[i] > y[j])
                break;
            i++;
        }
    }
    return idx;
}
// Assumptions: p and q have been malloc'ed with `idx` integers.
static inline void produce_intersection_indices(const int m, const FLT* x, const int n, const FLT* y, const FLT epsilon, int* p, int* q) {
    int istart = 0, idx = 0;
    for (int j = 0; j < n; j++) {
        int i = istart;
        int thefirst = 1;
        while (i < m) {
            if (fabs(x[i] - y[j]) < epsilon * MAX(fabs(x[i]), fabs(y[j]))) {
                p[idx] = i;
                q[idx] = j;
                idx++;
                if (thefirst) {
                    istart = i;
                    thefirst--;
                }
            }
            else if (x[i] > y[j])
                break;
            i++;
        }
    }
}
sparse* malloc_sparse(const int m, const int n, const int nnz) {
    sparse* A = (sparse *)malloc(sizeof(sparse));
    A->p = (int*)malloc(nnz * sizeof(int));
    A->q = (int*)malloc(nnz * sizeof(int));
    A->v = (FLT*)malloc(nnz * sizeof(FLT));
    A->m = m;
    A->n = n;
    A->nnz = nnz;
    return A;
}
static inline sparse* get_sparse_from_eigenvectors(tb_eigen_FMM* F1, triangular_banded* A, triangular_banded* B, FLT* D, int* p1, int* p2, int* p3, int* p4, int n, int s, int b, int idx) {
    sparse* S = malloc_sparse(s, n - s, idx);
    FLT* V = (FLT*)calloc(n, sizeof(FLT));
    for (int l = 0; l < idx; l++) {
        int j = p2[p4[l]] + s;
        for (int i = 0; i < n; i++)
            V[i] = 0;
        V[j] = D[j];
        FLT t, kt, d, kd, lam;
        lam = get_triangular_banded_index(A, j, j) / get_triangular_banded_index(B, j, j);
        for (int i = j - 1; i >= 0; i--) {
            t = kt = 0;
            for (int k = i + 1; k < MIN(i + b + 1, n); k++) {
                t += (get_triangular_banded_index(A, i, k) - lam * get_triangular_banded_index(B, i, k)) * V[k];
                kt += (fabs(get_triangular_banded_index(A, i, k)) + fabs(lam * get_triangular_banded_index(B, i, k))) * fabs(V[k]);
            }
            d = lam * get_triangular_banded_index(B, i, i) - get_triangular_banded_index(A, i, i);
            kd = fabs(lam * get_triangular_banded_index(B, i, i)) + fabs(get_triangular_banded_index(A, i, i));
            if (fabs(d) < 4 * kd * eps() || fabs(t) < 4 * kt * eps())
                V[i] = 0;
            else
                V[i] = t / d;
        }
        bfsv('N', F1, V);
        S->p[l] = p1[p3[l]];
        S->q[l] = p2[p4[l]];
        S->v[l] = V[p1[p3[l]]];
    }
    free(V);
    return S;
}
hierarchicalmatrix* malloc_hierarchicalmatrix(const int M, const int N) {
    hierarchicalmatrix* H = (hierarchicalmatrix*)malloc(sizeof(hierarchicalmatrix));
    H->hierarchicalmatrices = (hierarchicalmatrix**)malloc(M * N * sizeof(hierarchicalmatrix*));
    H->densematrices = (densematrix**)malloc(M * N * sizeof(densematrix*));
    H->lowrankmatrices = (lowrankmatrix**)malloc(M * N * sizeof(lowrankmatrix*));
    H->hash = (int*) calloc(M * N, sizeof(int));
    H->M = M;
    H->N = N;
    return H;
}
int binarysearch(FLT* x, int start, int stop, FLT y) {
    int j;
    while (stop >= start) {
        j = (start + stop) / 2;
        if (x[j] < y) start = j + 1;
        else if (x[j] > y) stop = j - 1;
        else break;
    }
    if (x[j] < y) j += 1;
    return j;
}
/*
indsplit takes a unitrange `start ≤ ir < stop`, and splits it into
two unitranges `i1` and `i2` such that
    `a ≤ x[i] < (a+b)/2` for `i ∈ i1`, and
    `(a+b)/2 ≤ x[i] ≤ b` for `i ∈ i2`.
*/
void indsplit(FLT* x, unitrange ir, unitrange* i1, unitrange* i2, FLT a, FLT b) {
    int start = ir.start, stop = ir.stop;
    i1->start = start;
    i1->stop = i2->start = binarysearch(x, start, stop, (a + b) / 2);
    i2->stop = stop;
}
densematrix* malloc_densematrix(int m, int n) {
    densematrix* A = (densematrix*)malloc(sizeof(densematrix));
    A->A = (FLT*)malloc(m * n * sizeof(FLT));
    A->m = m;
    A->n = n;
    return A;
}
densematrix* sample_densematrix(FLT(*f)(FLT x, FLT y), FLT* x, FLT* y, unitrange i, unitrange j) {
    int M = i.stop - i.start;
    densematrix* AD = malloc_densematrix(M, j.stop - j.start);
    FLT* A = AD->A;
    for (int n = j.start; n < j.stop; n++)
        for (int m = i.start; m < i.stop; m++)
            A[m - i.start + M * (n - j.start)] = f(x[m], y[n]);
    return AD;
}

// Assumes x and y are increasing sequences
static FLT dist(FLT* x, FLT* y, unitrange i, unitrange j) {
    if (y[j.start] > x[i.stop - 1])
        return y[j.start] - x[i.stop - 1];
    else if (y[j.start] >= x[i.start])
        return ZERO(FLT);
    else if (y[j.stop - 1] >= x[i.start])
        return ZERO(FLT);
    else
        return x[i.start] - y[j.stop - 1];
}

// Assumes x is an increasing sequence
static FLT diam(FLT* x, unitrange i) { return x[i.stop - 1] - x[i.start]; }
lowrankmatrix* malloc_lowrankmatrix(char N, int m, int n, int r) {
    int sz = 0;
    if (N == '2') sz = r;
    else if (N == '3') sz = r * r;
    lowrankmatrix* L = (lowrankmatrix*)malloc(sizeof(lowrankmatrix));
    L->U = (FLT*)malloc(m * r * sizeof(FLT));
    L->S = (FLT*)malloc(sz * sizeof(FLT));
    L->V = (FLT*)malloc(n * r * sizeof(FLT));
    L->t1 = (FLT*)calloc(r * FT_GET_MAX_THREADS(), sizeof(FLT));
    L->t2 = (FLT*)calloc(r * FT_GET_MAX_THREADS(), sizeof(FLT));
    L->m = m;
    L->n = n;
    L->r = r;
    L->p = FT_GET_MAX_THREADS();
    L->N = N;
    return L;
}
#define M_PI 3.1415926535897932384626433832795
FLT* chebyshev_points(char KIND, int n) {
    int nd2 = n >> 1;
    FLT* x = (FLT*)malloc(n * sizeof(FLT));
    if (KIND == '1') {
        for (int k = 0; k <= nd2; k++)
            x[k] = sin(M_PI*(n - 2 * k - ONE(FLT)) / (2 * n));
        for (int k = 0; k < nd2; k++)
            x[n - 1 - k] = -x[k];
    }
    else if (KIND == '2') {
        for (int k = 0; k <= nd2; k++)
            x[k] = sin(M_PI * (n - 2 * k - ONE(FLT)) / (2 * n - 2));
        for (int k = 0; k < nd2; k++)
            x[n - 1 - k] = -x[k];
    }
    return x;
}

FLT* chebyshev_barycentric_weights(char KIND, int n) {
    int nd2 = n >> 1;
    FLT* l = (FLT*)malloc(n * sizeof(FLT));
    if (KIND == '1') {
        for (int k = 0; k <= nd2; k++)
            l[k] = sin(M_PI * (2 * k + ONE(FLT)) / (2 * n));
        for (int k = 0; k < nd2; k++)
            l[n - 1 - k] = l[k];
        for (int k = 1; k < n; k += 2)
            l[k] *= -1;
    }
    else if (KIND == '2') {
        l[0] = ONE(FLT) / TWO(FLT);
        for (int k = 1; k <= nd2; k++)
            l[k] = 1;
        for (int k = 0; k < nd2; k++)
            l[n - 1 - k] = l[k];
        for (int k = 1; k < n; k += 2)
            l[k] *= -1;
    }
    return l;
}
void barycentricmatrix(FLT* A, FLT* x, int m, FLT* y, FLT* l, int n) {
    int k;
    FLT yj, lj, temp;
    for (int j = 0; j < n; j++) {
        yj = y[j];
        lj = l[j];
        for (int i = 0; i < m; i++)
            A[i + m * j] = lj / (x[i] - yj);
    }
    for (int i = 0; i < m; i++) {
        k = -1;
        temp = 0;
        for (int j = 0; j < n; j++) {
            if (isfinite(A[i + m * j])) temp += A[i + m * j];
            else { k = j; break; }
        }
        if (k != -1) {
            for (int j = 0; j < n; j++)
                A[i + m * j] = 0;
            A[i + m * k] = 1;
        }
        else {
            temp = 1 / temp;
            for (int j = 0; j < n; j++)
                A[i + m * j] *= temp;
        }
    }
}
lowrankmatrix* sample_lowrankmatrix(FLT(*f)(FLT x, FLT y), FLT* x, FLT* y, unitrange i, unitrange j) {
    int M = i.stop - i.start, N = j.stop - j.start, r = BLOCKRANK;
    lowrankmatrix* L = malloc_lowrankmatrix('3', M, N, r);

    FLT* xc1 = chebyshev_points('1', r);
    FLT* xc2 = chebyshev_points('1', r);
    FLT* lc = chebyshev_barycentric_weights('1', r);

    FLT a = x[i.start], b = x[i.stop - 1];
    FLT c = y[j.start], d = y[j.stop - 1];
    FLT ab2 = (a + b) / 2, ba2 = (b - a) / 2;
    FLT cd2 = (c + d) / 2, dc2 = (d - c) / 2;

    for (int p = 0; p < r; p++)
        xc1[p] = ab2 + ba2 * xc1[p];
    for (int q = 0; q < r; q++)
        xc2[q] = cd2 + dc2 * xc2[q];

    for (int q = 0; q < r; q++)
        for (int p = 0; p < r; p++)
            L->S[p + r * q] = f(xc1[p], xc2[q]);

    barycentricmatrix(L->U, x + i.start, M, xc1, lc, r);
    barycentricmatrix(L->V, y + j.start, N, xc2, lc, r);

    free(xc1);
    free(xc2);
    free(lc);

    return L;
}

hierarchicalmatrix* sample_hierarchicalmatrix(FLT(*f)(FLT x, FLT y), FLT* x, FLT* y, unitrange i, unitrange j, char SPLITTING) {
    int M = 2, N = 2;
    hierarchicalmatrix* H = malloc_hierarchicalmatrix(M, N);
    hierarchicalmatrix** HH = H->hierarchicalmatrices;
    densematrix** HD = H->densematrices;
    lowrankmatrix** HL = H->lowrankmatrices;

    unitrange i1, i2, j1, j2;
    if (SPLITTING == 'I') {
        i1.start = i.start;
        i1.stop = i2.start = i.start + ((i.stop - i.start) >> 1);
        i2.stop = i.stop;
        j1.start = j.start;
        j1.stop = j2.start = j.start + ((j.stop - j.start) >> 1);
        j2.stop = j.stop;
    }
    else if (SPLITTING == 'G') {
        indsplit(x, i, &i1, &i2, x[i.start], x[i.stop - 1]);
        indsplit(y, j, &j1, &j2, y[j.start], y[j.stop - 1]);
    }

    if (i1.stop - i1.start < BLOCKSIZE || j1.stop - j1.start < BLOCKSIZE) {
        HD[0] = sample_densematrix(f, x, y, i1, j1);
        H->hash[0* M+0] = 2;
    }
    else if (dist(x, y, i1, j1) >= MIN(diam(x, i1), diam(y, j1))) {
        HL[0] = sample_lowrankmatrix(f, x, y, i1, j1);
        H->hash[0 * M + 0] = 3;
    }
    else {
        HH[0] = sample_hierarchicalmatrix(f, x, y, i1, j1, SPLITTING);
        H->hash[0 * M + 0] = 1;
    }

    if (i2.stop - i2.start < BLOCKSIZE || j1.stop - j1.start < BLOCKSIZE) {
        HD[1] = sample_densematrix(f, x, y, i2, j1);
        H->hash[0 * M + 1] = 2;
    }
    else if (dist(x, y, i2, j1) >= MIN(diam(x, i2), diam(y, j1))) {
        HL[1] = sample_lowrankmatrix(f, x, y, i2, j1);
        H->hash[0 * M + 1] = 3;
    }
    else {
        HH[1] = sample_hierarchicalmatrix(f, x, y, i2, j1, SPLITTING);
        H->hash[0 * M + 1] = 1;
    }

    if (i1.stop - i1.start < BLOCKSIZE || j2.stop - j2.start < BLOCKSIZE) {
        HD[2] = sample_densematrix(f, x, y, i1, j2);
        H->hash[1 * M + 0] = 2;
    }
    else if (dist(x, y, i1, j2) >= MIN(diam(x, i1), diam(y, j2))) {
        HL[2] = sample_lowrankmatrix(f, x, y, i1, j2);
        H->hash[1 * M + 0] = 3;
    }
    else {
        HH[2] = sample_hierarchicalmatrix(f, x, y, i1, j2, SPLITTING);
        H->hash[1 * M + 0] = 1;
    }

    if (i2.stop - i2.start < BLOCKSIZE || j2.stop - j2.start < BLOCKSIZE) {
        HD[3] = sample_densematrix(f, x, y, i2, j2);
        H->hash[1 * M + 1] = 2;
    }
    else if (dist(x, y, i2, j2) >= MIN(diam(x, i2), diam(y, j2))) {
        HL[3] = sample_lowrankmatrix(f, x, y, i2, j2);
        H->hash[1 * M + 1] = 3;
    }
    else {
        HH[3] = sample_hierarchicalmatrix(f, x, y, i2, j2, SPLITTING);
        H->hash[1 * M + 1] = 1;
    }

    H->m = i.stop - i.start;
    H->n = j.stop - j.start;

    return H;
}

// No transpose: x .= x[p], or x .= P*x where P = Id[p, :].
// Transpose:    x[p] .= x, or x .= P'x where P = Id[p, :].
void perm(char TRANS, FLT* x, int* p, int n) {
    for (int i = 0; i < n; i++)
        p[i] = p[i] - n;
    if (TRANS == 'N') {
        int j, k;
        for (int i = 0; i < n; i++) {
            if (p[i] >= 0) continue;
            j = i;
            k = p[j] = p[j] + n;
            while (p[k] < 0) {
                swap(x, j, k);
                j = k;
                k = p[j] = p[j] + n;
            }
        }
    }
    else if (TRANS == 'T') {
        int j;
        for (int i = 0; i < n; i++) {
            if (p[i] >= 0) continue;
            j = p[i] = p[i] + n;
            while (p[j] < 0) {
                swap(x, i, j);
                j = p[j] = p[j] + n;
            }
        }
    }
}
tb_eigen_FMM* tb_eig_FMM(triangular_banded* A, triangular_banded* B, FLT* D) {
    int n = A->n, b1 = A->b, b2 = B->b;
    int b = MAX(b1, b2);
    tb_eigen_FMM* F = (tb_eigen_FMM*)malloc(sizeof(tb_eigen_FMM));
    if (n < TB_EIGEN_BLOCKSIZE) {
        FLT* V = (FLT*)calloc(n * n, sizeof(FLT));
        for (int i = 0; i < n; i++)
            V[i + i * n] = D[i];
        F->lambda = (FLT*)malloc(n * sizeof(FLT));
        triangular_banded_eigenvalues(A, B, F->lambda);
        triangular_banded_eigenvectors(A, B, V);
        F->V = V;
        F->n = n;
        F->b = b;
    }
    else {
        int s = n >> 1;
        unitrange i = { 0, s }, j = { s, n };
        triangular_banded* A1 = view_triangular_banded(A, i);
        triangular_banded* B1 = view_triangular_banded(B, i);
        triangular_banded* A2 = view_triangular_banded(A, j);
        triangular_banded* B2 = view_triangular_banded(B, j);

        F->F1 = tb_eig_FMM(A1, B1, D);
        F->F2 = tb_eig_FMM(A2, B2, D + s);

        FLT* lambda = (FLT *)malloc(n * sizeof(FLT));
        for (int i = 0; i < s; i++)
            lambda[i] = F->F1->lambda[i];
        for (int i = 0; i < n - s; i++)
            lambda[i + s] = F->F2->lambda[i];

        FLT* X = (FLT*)calloc(s * b, sizeof(FLT));
        for (int j = 0; j < b; j++) {
            X[s - b + j + j * s] = 1;
            tbsv('N', B1, X + j * s);
            bfsv('N', F->F1, X + j * s);
        }

        FLT* Y = (FLT*)calloc((n - s) * b, sizeof(FLT));
        for (int j = 0; j < b1; j++)
            for (int k = j; k < b1; k++)
                Y[j + k * (n - s)] = get_triangular_banded_index(A, k + s - b1, j + s);
        FLT* Y2 = (FLT*)calloc((n - s) * b2, sizeof(FLT));
        for (int j = 0; j < b2; j++)
            for (int k = j; k < b2; k++)
                Y2[j + k * (n - s)] = get_triangular_banded_index(B, k + s - b2, j + s);

        for (int j = 0; j < b1; j++)
            bfmv('T', F->F2, Y + j * (n - s));
        for (int j = 0; j < b2; j++)
            bfmv('T', F->F2, Y2 + j * (n - s));

        for (int j = 0; j < b2; j++)
            for (int i = 0; i < n - s; i++)
                Y[i + (j + b - b2) * (n - s)] = Y[i + (j + b - b2) * (n - s)] - lambda[i + s] * Y2[i + j * (n - s)];

        int* p1 = (int*)malloc(s * sizeof(int));
        for (int i = 0; i < s; i++)
            p1[i] = i;
        quicksort_1arg(lambda, p1, 0, s - 1, lt);
        int* p2 = (int*)malloc((n - s) * sizeof(int));
        for (int i = 0; i < n - s; i++)
            p2[i] = i;
        quicksort_1arg(lambda + s, p2, 0, n - s - 1, lt);

        int idx = count_intersections(s, lambda, n - s, lambda + s, 16 * sqrt(eps()));
        int* p3 = (int*)malloc(idx * sizeof(int));
        int* p4 = (int*)malloc(idx * sizeof(int));
        produce_intersection_indices(s, lambda, n - s, lambda + s, 16 * sqrt(eps()), p3, p4);
        sparse* S = get_sparse_from_eigenvectors(F->F1, A, B, D, p1, p2, p3, p4, n, s, b, idx);
        free(p3);
        free(p4);

        F->F0 = sample_hierarchicalmatrix(thresholded_cauchykernel, lambda, lambda, i, j, 'G');
        F->X = X;
        F->Y = Y;
        F->S = S;
        F->t1 = (FLT*)calloc(s * FT_GET_MAX_THREADS(), sizeof(FLT));
        F->t2 = (FLT*)calloc((n - s) * FT_GET_MAX_THREADS(), sizeof(FLT));
        perm('T', lambda, p1, s);
        perm('T', lambda + s, p2, n - s);
        F->lambda = lambda;
        F->p1 = p1;
        F->p2 = p2;
        F->n = n;
        F->b = b;
        free(A1);
        A1 = 0;
        free(B1);
        B1 = 0;
        free(A2);
        A2 = 0;
        free(B2);
        B2 = 0;
        free(Y2);
        Y2 = 0;
    }
    return F;
}
sparse* drop_precision_sparse(sparse* S2) {
    sparse* S = malloc_sparse(S2->m, S2->n, S2->nnz);
    for (int l = 0; l < S->nnz; l++) {
        S->p[l] = S2->p[l];
        S->q[l] = S2->q[l];
        S->v[l] = S2->v[l];
    }
    return S;
}
tb_eigen_FMM* drop_precision_tb_eigen_FMM(tb_eigen_FMM* F2) {
    int n = F2->n;
    tb_eigen_FMM* F = (tb_eigen_FMM*)malloc(sizeof(tb_eigen_FMM));
    if (n < TB_EIGEN_BLOCKSIZE) {
        FLT* V = (FLT*)malloc(n * n * sizeof(FLT));
        for (int i = 0; i < n * n; i++)
            V[i] = F2->V[i];
        FLT* lambda = (FLT*)malloc(n * sizeof(FLT));
        for (int i = 0; i < n; i++)
            lambda[i] = F2->lambda[i];
        F->V = V;
        F->lambda = lambda;
        F->n = n;
    }
    else {
        int s = n >> 1, b = F2->b;
        int* p1 = (int*)malloc(s * sizeof(int)), * p2 = (int*)malloc((n - s) * sizeof(int));
        FLT* lambda = (FLT*)malloc(n * sizeof(FLT));
        for (int i = 0; i < s; i++)
            p1[i] = F2->p1[i];
        for (int i = 0; i < n - s; i++)
            p2[i] = F2->p2[i];
        for (int i = 0; i < n; i++)
            lambda[i] = F2->lambda[i];
        perm('N', lambda, p1, s);
        perm('N', lambda + s, p2, n - s);
        unitrange i0 = { 0, s };
        unitrange j0 = { s, n };
        F->F0 = sample_hierarchicalmatrix(thresholded_cauchykernel2, lambda, lambda, i0, j0, 'G');
        perm('T', lambda, p1, s);
        perm('T', lambda + s, p2, n - s);
        F->F1 = drop_precision_tb_eigen_FMM(F2->F1);
        F->F2 = drop_precision_tb_eigen_FMM(F2->F2);
        F->S = drop_precision_sparse(F2->S);
        F->X = (FLT*)malloc(s * b * sizeof(FLT));
        for (int i = 0; i < s * b; i++)
            F->X[i] = F2->X[i];
        F->Y = (FLT*)malloc((n - s) * b * sizeof(FLT));
        for (int i = 0; i < (n - s) * b; i++)
            F->Y[i] = F2->Y[i];
        F->t1 = (FLT*)calloc(s * FT_GET_MAX_THREADS(), sizeof(FLT));
        F->t2 = (FLT*)calloc((n - s) * FT_GET_MAX_THREADS(), sizeof(FLT));
        F->lambda = lambda;
        F->p1 = p1;
        F->p2 = p2;
        F->n = n;
        F->b = b;
    }
    return F;
}
void destroy_triangular_banded(triangular_banded* A) {
    free(A->data);
    A->data = 0;
    free(A);
    A = 0;
}
void destroy_densematrix(densematrix* A) {
    free(A->A);
    A->A = 0;
    free(A);
    A = 0;
}

void destroy_lowrankmatrix(lowrankmatrix* L) {
    free(L->U);
    L->U = 0;
    free(L->S);
    L->S = 0;
    free(L->V);
    L->V = 0;
    free(L->t1);
    L->t1 = 0;
    free(L->t2);
    L->t2 = 0;
    free(L);
    L = 0;
}
void destroy_hierarchicalmatrix(hierarchicalmatrix* H) {
    int M = H->M, N = H->N;
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            switch (H->hash[n * M + m]) {
            case 1: destroy_hierarchicalmatrix(H->hierarchicalmatrices[n * M + m]); break;
            case 2: destroy_densematrix(H->densematrices[n * M + m]); break;
            case 3: destroy_lowrankmatrix(H->lowrankmatrices[n * M + m]); break;
            }
        }
    }
    free(H->hierarchicalmatrices);
    H->hierarchicalmatrices = 0;
    free(H->densematrices);
    H->densematrices = 0;
    free(H->lowrankmatrices);
    H->lowrankmatrices = 0;
    free(H->hash);
    H->hash = 0;
    free(H);
    H = 0;
}
void destroy_sparse(sparse* A) {
    free(A->p);
    A->p = 0;
    free(A->q);
    A->q = 0;
    free(A->v);
    A->v = 0;
    free(A);
    A = 0;
}
void destroy_tb_eigen_FMM(tb_eigen_FMM* F) {
    if (F->n < TB_EIGEN_BLOCKSIZE) {
        free(F->V);
        F->V = 0;
        free(F->lambda);
        F->lambda = 0;
    }
    else {
        destroy_hierarchicalmatrix(F->F0);
        destroy_tb_eigen_FMM(F->F1);
        destroy_tb_eigen_FMM(F->F2);
        destroy_sparse(F->S);
        free(F->X);
        F->X = 0;
        free(F->Y);
        F->Y = 0;
        free(F->t1);
        F->t1 = 0;
        free(F->t2);
        F->t2 = 0;
        free(F->lambda);
        F->lambda = 0;
        free(F->p1);
        F->p1 = 0;
        free(F->p2);
        F->p2 = 0;
    }
    free(F);
}
tb_eigen_FMM* plan_legendre_to_chebyshev(const int normleg, const int normcheb, const int n) {
    triangular_banded* A = create_A_legendre_to_chebyshev(normcheb, n);
    triangular_banded* B = create_B_legendre_to_chebyshev(normcheb, n);
    FLT* D = (FLT*)malloc(n * sizeof(FLT));
    create_legendre_to_chebyshev_diagonal_connection_coefficient(normleg, normcheb, n, D, 1);
    tb_eigen_FMM* F2 = tb_eig_FMM(A, B, D);
    tb_eigen_FMM* F = drop_precision_tb_eigen_FMM(F2);
    destroy_triangular_banded(A);
    destroy_triangular_banded(B);
    destroy_tb_eigen_FMM(F2);
    free(D);
    D = 0;
    return F;
}
triangular_banded* create_A_chebyshev_to_legendre(const int norm, const int n) {
    triangular_banded* A = calloc_triangular_banded(n, 2);
    if (norm) {
        if (n > 1)
            set_triangular_banded_index(A, sqrt(TWO(FLT) / 5), 1, 1);
        for (int i = 2; i < n; i++) {
            set_triangular_banded_index(A, -(i + 1) * sqrt(((i - ONE(FLT)) * i) / ((2 * i - ONE(FLT)) * (2 * i + 1))) * (i + 1), i - 2, i);
            set_triangular_banded_index(A, i * sqrt(((i + ONE(FLT)) * (i + 2)) / ((2 * i + ONE(FLT)) * (2 * i + 3))) * i, i, i);
        }
    }
    else {
        if (n > 1)
            set_triangular_banded_index(A, ONE(FLT) / 3, 1, 1);
        for (int i = 2; i < n; i++) {
            set_triangular_banded_index(A, -(i + 1) / (2 * i + ONE(FLT)) * (i + 1), i - 2, i);
            set_triangular_banded_index(A, i / (2 * i + ONE(FLT)) * i, i, i);
        }
    }
    return A;
}

triangular_banded* create_B_chebyshev_to_legendre(const int norm, const int n) {
    triangular_banded* B = calloc_triangular_banded(n, 2);
    if (norm) {
        if (n > 0)
            set_triangular_banded_index(B, sqrt(TWO(FLT) / 3), 0, 0);
        if (n > 1)
            set_triangular_banded_index(B, sqrt(TWO(FLT) / 5), 1, 1);
        for (int i = 2; i < n; i++) {
            set_triangular_banded_index(B, -sqrt(((i - ONE(FLT)) * i) / ((2 * i - ONE(FLT)) * (2 * i + 1))), i - 2, i);
            set_triangular_banded_index(B, sqrt(((i + ONE(FLT)) * (i + 2)) / ((2 * i + ONE(FLT)) * (2 * i + 3))), i, i);
        }
    }
    else {
        if (n > 0)
            set_triangular_banded_index(B, 1, 0, 0);
        if (n > 1)
            set_triangular_banded_index(B, ONE(FLT) / 3, 1, 1);
        for (int i = 2; i < n; i++) {
            set_triangular_banded_index(B, -1 / (2 * i + ONE(FLT)), i - 2, i);
            set_triangular_banded_index(B, 1 / (2 * i + ONE(FLT)), i, i);
        }
    }
    return B;
}

void create_chebyshev_to_legendre_diagonal_connection_coefficient(const int normcheb, const int normleg, const int n, FLT* D, const int INCD) {
    if (normcheb) {
        if (normleg) {
            if (n > 0)
                D[0] = sqrt(2) / tgamma(0.5);
            if (n > 1)
                D[INCD] = D[0] / sqrt(1.5);
            for (int i = 2; i < n; i++)
                D[i * INCD] = (2 * i) / sqrt((2 * i + 1) * (2 * i - ONE(FLT))) * D[(i - 1) * INCD];
        }
        else {
            if (n > 0)
                D[0] = 1 / tgamma(0.5);
            if (n > 1)
                D[INCD] = sqrt(2) * D[0];
            for (int i = 2; i < n; i++)
                D[i * INCD] = (2 * i) * D[(i - 1) * INCD] / (2 * i - 1);
        }
    }
    else {
        if (normleg) {
            if (n > 0)
                D[0] = sqrt(2);
            if (n > 1)
                D[INCD] = 1 / sqrt(1.5);
            for (int i = 2; i < n; i++)
                D[i * INCD] = (2 * i) / sqrt((2 * i + 1) * (2 * i - ONE(FLT))) * D[(i - 1) * INCD];
        }
        else {
            if (n > 0)
                D[0] = 1;
            if (n > 1)
                D[INCD] = 1;
            for (int i = 2; i < n; i++)
                D[i * INCD] = (2 * i) * D[(i - 1) * INCD] / (2 * i - 1);
        }
    }
}
void bfmm(char TRANS, tb_eigen_FMM* F, FLT* B, int LDB, int N) {
#pragma omp parallel for
    for (int j = 0; j < N; j++)
        bfmv(TRANS, F, B + j * LDB);
}
tb_eigen_FMM* plan_chebyshev_to_legendre(const int normcheb, const int normleg, const int n) {
    triangular_banded* A = create_A_chebyshev_to_legendre(normleg, n);
    triangular_banded* B = create_B_chebyshev_to_legendre(normleg, n);
    FLT* D = (FLT*) malloc(n * sizeof(FLT));
    create_chebyshev_to_legendre_diagonal_connection_coefficient(normcheb, normleg, n, D, 1);
    tb_eigen_FMM* F2 = tb_eig_FMM(A, B, D);
    tb_eigen_FMM* F = drop_precision_tb_eigen_FMM(F2);
    destroy_triangular_banded(A);
    destroy_triangular_banded(B);
    destroy_tb_eigen_FMM(F2);
    free(D);
    D = 0;
    return F;
}
FLT norm_2arg(FLT* A, FLT* B, int n) {
    FLT ret = 0;
    for (int i = 0; i < n; i++)
        ret += (A[i] - B[i]) * (A[i] - B[i]);
    return sqrt(ret);
}
FLT norm_1arg(FLT* A, int n) {
    FLT ret = 0;
    for (int i = 0; i < n; i++)
        ret += A[i] * A[i];
    return sqrt(ret);
}
void checktest(FLT err, FLT cst, int* checksum) {
    if (fabs(err) < cst * eps()) printf("✓\n");
    else { printf("✗\n"); (*checksum)++; }
}
VkSolveResult sample_1_benchmark_VkFFT_double(VkGPU* vkGPU, uint64_t file_output, FILE* output, uint64_t isCompilerInitialized)
{
    int checksum = 0;
   /* {
        double err;
        double* Id, * B, * x;
        tb_eigen_FMM* A;
        btb_eigen_FMM* C;
        banded* M;

        for (int n = 64; n < 1024; n *= 2) {
            err = 0;
            Id = (FLT*)calloc(n * n, sizeof(FLT));
            B = (FLT*)calloc(n * n, sizeof(FLT));
            for (int i = 0; i < n; i++)
                B[i + i * n] = Id[i + i * n] = 1;
            for (int normleg = 0; normleg <= 1; normleg++) {
                for (int normcheb = 0; normcheb <= 1; normcheb++) {
                    A = plan_legendre_to_chebyshev(normleg, normcheb, n);
                    bfmm('N', A, B, n, n);
                    destroy_tb_eigen_FMM(A);
                    A = plan_chebyshev_to_legendre(normcheb, normleg, n);
                    bfmm('N', A, B, n, n);
                    destroy_tb_eigen_FMM(A);
                    err += norm_2arg(B, Id, n * n) / norm_1arg(Id, n * n);
                }
            }
            printf("(n×n) = (%4ix%4i): \t\t\t\t\t |%20.2e ", n, n, (double)err);
            checktest(err, 2 * sqrt(n), &checksum);
            free(Id);
            free(B);
        }
    }*/
    {
        mpfr_prec_t prec = 53;
        mpfr_prec_t prec2 = 100;
        mpfr_t err, t1, t2, t3;
        //mpfr_set_emax(1024);
       // mpfr_set_emin(-1073);
        mpfr_init2(err, prec);
        mpfr_init2(t1, prec);
        mpfr_init2(t2, prec);
        mpfr_t* Id, * A, * B;
        mpfr_t* Id2, * A2, * B2;
        mpfr_rnd_t rnd = MPFR_RNDN;
        for (int n = 64; n < 1024; n *= 2) {
            mpfr_set_d(err, 0.0, rnd);
            Id = ft_mpfr_init_Id(n, prec, rnd);
            B = ft_mpfr_init_Id(n, prec, rnd);
            for (int normleg = 0; normleg <= 1; normleg++) {
                for (int normcheb = 0; normcheb <= 1; normcheb++) {
                    A = ft_mpfr_plan_legendre_to_chebyshev(normleg, normcheb, n, prec2, rnd);
                    for (int i = 0; i < n*n; i++) {
                        mpf_t temp;
                        mpf_init2(temp, prec);
                        mpfr_get_f(temp, A[i], rnd);
                        mpfr_set_prec(A[i], prec);
                        mpfr_set_f(A[i], temp, rnd);
                    }
                    ft_mpfr_trmm('N', n, A, n, B, n, n, rnd);
                    ft_mpfr_destroy_plan(A, n);
                    A = ft_mpfr_plan_chebyshev_to_legendre(normcheb, normleg, n, prec2, rnd);
                    for (int i = 0; i < n * n; i++) {
                        mpf_t temp;
                        mpf_init2(temp, prec);
                        mpfr_get_f(temp, A[i], rnd);
                        mpfr_set_prec(A[i], prec);
                        mpfr_set_f(A[i], temp, rnd);
                    }
                    ft_mpfr_trmm('N', n, A, n, B, n, n, rnd);
                    ft_mpfr_destroy_plan(A, n);
                    ft_mpfr_norm_2arg(&t1, B, Id, n * n, prec, rnd);
                    ft_mpfr_norm_1arg(&t2, Id, n * n, prec, rnd);
                    mpfr_div(t1, t1, t2, rnd);
                    mpfr_add(err, err, t1, rnd);
                }
            }
            printf("(n×n) = (%4ix%4i): \t\t\t\t\t |%20.2e ", n, n, mpfr_get_d(err, rnd));
            ft_mpfr_checktest(&err, n, &checksum, prec, rnd);
            ft_mpfr_destroy_plan(Id, n);
            ft_mpfr_destroy_plan(B, n);
        }
    }
	return VKSOLVE_SUCCESS;
}
