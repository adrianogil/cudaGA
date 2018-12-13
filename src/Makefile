NVCC = nvcc

CUDAFLAGS = -arch=sm_30

all: gaapproximate

gaapproximate.o: gaapproximate.cu
	$(NVCC) $(CUDAFLAGS) -std=c++11 -c gaapproximate.cu

gaapproximate: gaapproximate.o
	${NVCC} ${CUDAFLAGS} -o gaapproximate gaapproximate.o

clean:
	rm -f *.o gaapproximate
