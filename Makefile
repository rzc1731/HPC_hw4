NVCC = nvcc
NVCCFLAGS = -std=c++11
NVCCFLAGS += -Xcompiler "-fopenmp"
EXECS = matrix_vector jacobi
all: ${EXECS}

matrix_vector: matrix_vector.cu
	${NVCC} ${NVCCFLAGS} $^ -o matrix_vector

jacobi: jacobi.cu
	${NVCC} ${NVCCFLAGS} $^ -o jacobi

clean:
	rm -f ${EXECS}
