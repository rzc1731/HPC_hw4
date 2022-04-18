#include <stdio.h>
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#ifdef _OPENMP
	#include <omp.h>
#endif

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

int num_t = 4;

void jacobi_cpu(long N, int max_iter, double residual_factor, double* u, double* u_next) {
	long Nt = N + 2; // add extra two rows and two columns to avoid edge cases
	double h = 1.0 / (N + 1);
	double hsq = h * h;
	double hsq_inv = 1.0 / hsq;

	double res = 0;

	double res_start;

	bool stop_flag = 0;

	#pragma omp parallel num_threads(num_t)
	{
		for (int k = 1; k <= max_iter; k++) {
			if (stop_flag) {
				continue;
			}
			#pragma omp for
			for (long i = Nt+1; i <= Nt*Nt-Nt-1; i++) {
				if (i%Nt != 0 && i%Nt != Nt-1) {
					u_next[i] = 0.25 * (hsq + u[i-Nt] + u[i-1] + u[i+1] + u[i+Nt]);
				}
			}

			#pragma omp for reduction(+:res)
			for (long i = Nt+1; i <= Nt*Nt-Nt-1; i++) {
				if (i%Nt != 0 && i%Nt != Nt-1) {
					// f(x,y) is always 1
					double temp = 1.0 + (u[i-Nt] + u[i-1] - 4.0 * u[i] + u[i+1] + u[i+Nt]) * hsq_inv;
					res += temp * temp;
				}
			}

			#pragma omp single
			{
				double res_norm = sqrt(res);
				if (k == 1) {
					res_start = res_norm * residual_factor;
					std::cout << "res_start = " << res_start << std::endl;
				}
				std::cout << "iter #: " << k << ", residual_norm = " << res_norm << std::endl;
				if (res_norm <= res_start) {
					std::cout << "Goal reached after " << k << " iterations. Final Residual: " << res_norm << std::endl;
					stop_flag = true;
				}
				double *swap;
				swap = u;
				u = u_next;
				u_next = swap;
				res = 0;
			}
		}
	}
}

#define BLOCK_SIZE 32
#define REDUCTION_BLOCK_SIZE 512

// Kernel for one step of Jacobi iteration
__global__ void jacobi_kernel(long Nt, const double* u_d, double* u_next_d, double hsq) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= 1 && i < Nt-1 && j >= 1 && j < Nt-1) {
    u_next_d[index(i,j,Nt)] = 0.25 * (hsq + u_d[index(i-1,j,Nt)] + u_d[index(i+1,j,Nt)] + u_d[index(i,j-1,Nt)] + u_d[index(i,j+1,Nt)]);
  }
}

// Kernel for computing residual
__global__ void res_kernel(long Nt, const double* u_d, double* res_d, double hsq_inv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= 1 && i < Nt-1 && j >= 1 && j < Nt-1) {
		double diff = 1.0 + (u_d[index(i-1,j,Nt)] + u_d[index(i+1,j,Nt)] - 4.0 * u_d[index(i,j,Nt)] + u_d[index(i,j-1,Nt)] + u_d[index(i,j+1,Nt)]) * hsq_inv;
    res_d[index(i,j,Nt)] = diff * diff;
  }
}

// Reduction kernel
__global__ void reduction_kernel(double *res_d, double *res_out_d, long Nt) {
	__shared__ double sdata[REDUCTION_BLOCK_SIZE];
	// each thread loads one element from global to shared mem
	int tid = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = res_d[i];
	__syncthreads();

	//reversed looping
	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		res_out_d[blockIdx.x] = sdata[0];
	}
}

void jacobi_gpu(long N, int max_iter, double residual_factor, double* u, double* res_out) {
  double *u_d, *u_next_d, *res_d, *res_out_d;

	long Nt = N + 2; // add extra two rows and two columns to avoid edge cases
	double h = 1.0 / (N + 1);
	double hsq = h * h;
	double hsq_inv = 1.0 / hsq;

  cudaMalloc(&u_d, Nt*Nt*sizeof(double));
  cudaMalloc(&u_next_d, Nt*Nt*sizeof(double));
	cudaMalloc(&res_d, Nt*Nt*sizeof(double));

	cudaMalloc(&res_out_d, (Nt*Nt/REDUCTION_BLOCK_SIZE+1)*sizeof(double));

  // Initialize both current and previous step to input
  cudaMemcpyAsync(u_d, u, Nt*Nt*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(u_next_d, u, Nt*Nt*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(res_d, u, Nt*Nt*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(res_out_d, res_out, (Nt*Nt/REDUCTION_BLOCK_SIZE+1)*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(Nt / BLOCK_SIZE + 1, Nt / BLOCK_SIZE + 1);

	double res_start;

  // Iterate kernel
  for (int k = 1; k <= max_iter; k++) {
    jacobi_kernel<<<dimGrid,dimBlock>>>(Nt, u_d, u_next_d, hsq);
		res_kernel<<<dimGrid,dimBlock>>>(Nt, u_d, res_d, hsq_inv);
		reduction_kernel<<<Nt*Nt/REDUCTION_BLOCK_SIZE+1,REDUCTION_BLOCK_SIZE>>>(res_d, res_out_d, Nt);
		cudaDeviceSynchronize();
		cudaMemcpy(res_out, res_out_d, (Nt*Nt/REDUCTION_BLOCK_SIZE+1)*sizeof(double), cudaMemcpyDeviceToHost);
		double sum = 0.0;
		for (int j = 0; j < Nt*Nt/REDUCTION_BLOCK_SIZE+1; j++) {
			sum += res_out[j];
		}
		double res_norm = sqrt(sum);
		if (k == 1) {
			double res_start =  res_norm * residual_factor;
			std::cout << "res_start = " << res_start << std::endl;
		}
		std::cout << "iter #: " << k << ", residual_norm = " << res_norm << std::endl;
		if (res_norm <= res_start) {
			std::cout << "Goal reached after " << k << " iterations. Final Residual: " << res_norm << std::endl;
			break;
		}
    cudaMemcpyAsync(u_d, u_next_d, Nt*Nt*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
  }

  cudaFree(u_d);
  cudaFree(u_next_d);
	cudaFree(res_d);
	cudaFree(res_out_d);
}

int main(int argc, char** argv) {
	const int max_iter = 100;
	const double residual_factor = 0.000001;
	int N = 100;

	if (argc == 1) {
		#ifdef _OPENMP
		std::cout << "Run with default N=100, Number_of_threads=4 if openmp." << std::endl;
		#else
		std::cout << "Run with default N=100, Serial version." << std::endl;
		#endif
		std::cout << "Usage: " << argv[0] << " N Number_of_threads" << std::endl;
	} else {
		N = atoi(argv[1]);

		#ifdef _OPENMP
		num_t = atoi(argv[2]);
		std::cout << "There are " << num_t << " threads used." << std::endl;
		#else
		num_t = 1;
		std::cout << "Serial version." << std::endl;
		#endif
	}

	Timer time;
	time.tic();

	double *uu = (double*)calloc(sizeof(double), (N+2)*(N+2));
	double *u_next = (double*)calloc(sizeof(double), (N+2)*(N+2));

	jacobi_cpu(N, max_iter, residual_factor, uu, u_next);

	std::cout << "CPU computation time: " << time.toc() << std::endl;

	free(uu);
	free(u_next);

  // GPU calculation
	double *u, *res_out;
  cudaMallocHost(&u, (N+2)*(N+2)*sizeof(double));
	cudaMallocHost(&res_out, ((N+2)*(N+2)/REDUCTION_BLOCK_SIZE+1)*sizeof(double));
  for (int i = 0; i < (N+2)*(N+2); i++) {
    u[i] = 0.0;
  }
	for (int i = 0; i < (N+2)*(N+2)/REDUCTION_BLOCK_SIZE+1; i++) {
    res_out[i] = 0.0;
  }

  std::cout << ("GPU:") << std::endl;

  time.tic();
  jacobi_gpu(N, max_iter, residual_factor, u, res_out);

  std::cout << "GPU computation time: " << time.toc() << std::endl;

  cudaFree(u);
	cudaFree(res_out);

  return 0;
}
