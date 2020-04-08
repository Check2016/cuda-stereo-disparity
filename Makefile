
NVCC = nvcc

CUDAPATH = /usr/local/cuda

SOURCE = ./src
INCLUDE = ./include

NVCCFLAGS = -c -w -I$(INCLUDE)
#LFLAGS = -L$(CUDAPATH)/lib64 -lcuda -lcudart

build: disparity_cuda

disparity_cuda: disparity_cuda.o
	$(NVCC) disparity_cuda.o -o disparity_cuda -lm -lpthread -lX11

disparity_cuda.o: $(SOURCE)/disparity_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(SOURCE)/disparity_cuda.cu

clean:
	rm *.o
