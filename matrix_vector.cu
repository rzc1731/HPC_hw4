#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <iostream>

#define BLOCK_SIZE 512

void inner_product_cpu(double* c, const double* a, const double* b, long N) {
  double sum = 0;
  #pragma omp parallel for reduction(+:sum)
  for (long i = 0; i < N; i++) {
    sum += a[i] * b[i];
  }
  *c = sum;
}

void matrix_vector_cpu(double* c, double *a, double *b, long M, long N) {
  #pragma omp parallel for
  for(long i = 0; i < M; i++) {
    double sum = 0;
    for(long j = 0; j < N; j++) {
        sum += a[i*M + j] * b[j];
    }
    c[i] = sum;
  }
}

// Reduction kernel
__global__ void reduction_kernel(double *a, double *b, long N) {
	__shared__ double smem[BLOCK_SIZE];
	// each thread loads one element from global to shared mem
	int tid = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < N) {
    smem[tid] = a[idx];
  } else {
    smem[tid] = 0;
  }
	__syncthreads();

	//reversed looping
	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			smem[tid] += smem[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		b[blockIdx.x] = smem[0];
	}
}

__global__ void inner_product_kernel(double* c, const double* a, const double* b, long N){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) c[idx] = a[idx] * b[idx];
}

__global__ void matrix_vector_kernel(double* c, const double* a, const double* b, long M, long N){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  double sum = 0;
    if (idx < M){
      for (long i=0; i<N; i++) {
          sum += a[idx*M + i] * b[i];
      }
      c[idx] = sum;
    }
}

int main() {
  long N = 5000;
  long M = N;

  double *v, *u, *v_d, *u_d, *t_d, *r_d, *r, *A, *x, *A_d, *x_d, *Ax_d, *Ax_ref, *Ax;

  // compute for inner product
  cudaMallocHost((void**)&v, N * sizeof(double));
  cudaMallocHost((void**)&u, N * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    v[i] = drand48();
    u[i] = drand48();
  }

  double r_ref;
  inner_product_cpu(&r_ref, v, u, N);

  cudaMalloc(&v_d, N*sizeof(double));
  cudaMalloc(&u_d, N*sizeof(double));

  cudaMemcpyAsync(v_d, v, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(u_d, u, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  cudaMalloc(&t_d, N*sizeof(double));
  cudaMalloc(&r_d, (N/BLOCK_SIZE+1)*sizeof(double));
  cudaMallocHost(&r, (N/BLOCK_SIZE+1)*sizeof(double));

  double tt = omp_get_wtime();
  inner_product_kernel<<<N/BLOCK_SIZE+1,BLOCK_SIZE>>>(t_d, v_d, u_d, N);
  reduction_kernel<<<N/BLOCK_SIZE+1,BLOCK_SIZE>>>(t_d, r_d, N);
  cudaMemcpy(r, r_d, (N/BLOCK_SIZE+1)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  double sum = 0;
  for (int j = 0; j < N/BLOCK_SIZE+1; j++) {
    sum += r[j];
  }
  printf("Inner product\n");
  printf("GPU Bandwidth = %lf GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  printf("Error = %lf\n", fabs(sum - r_ref));

  cudaFree(u_d);
  cudaFree(v_d);
  cudaFree(t_d);
  cudaFree(r_d);
  cudaFreeHost(r);
  cudaFreeHost(u);
  cudaFreeHost(v);

  // compute for matrix vector
  cudaMallocHost((void**)&A, M * N * sizeof(double));
  cudaMallocHost((void**)&x, N * sizeof(double));

  # pragma omp parallel for
  for (long i = 0; i < N; i++) {
    x[i] = drand48();
  }
  # pragma omp parallel for
  for (long i = 0; i < M * N; i++) {
    A[i] = drand48();
  }

  cudaMallocHost((void**)&Ax_ref, N * sizeof(double));
  matrix_vector_cpu(Ax_ref, A, x, M, N);

  cudaMalloc(&A_d, M*N*sizeof(double));
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&Ax_d, N*sizeof(double));
  cudaMallocHost(&Ax, N*sizeof(double));

  cudaMemcpyAsync(A_d, A, M*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  tt = omp_get_wtime();
  matrix_vector_kernel<<<N/BLOCK_SIZE+1,BLOCK_SIZE>>>(Ax_d, A_d, x_d, M, N);
  cudaMemcpyAsync(Ax, Ax_d, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  printf("Matrix vector multiplication\n");
  printf("GPU Bandwidth = %lf GB/s\n", 2*M*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double err = 0;
  for (long i = 0; i < N; i++) {
    err += fabs(Ax_ref[i] - Ax[i]);
  }
  printf("Error = %lf\n", err);

  cudaFree(A_d);
  cudaFree(x_d);
  cudaFree(Ax_d);
  cudaFreeHost(A);
  cudaFreeHost(x);
  cudaFreeHost(Ax);
  cudaFreeHost(Ax_ref);

  return 0;
}
