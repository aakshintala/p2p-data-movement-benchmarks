# LiveStreamDarknet Makefile.

CXX = clang++
NVCC = nvcc
CPPFLAGS += -g -O3 -std=c++11 -I/usr/local/cuda/include/
LDFLAGS = -ldl -lpthread -L/usr/local/cuda/lib64 -lcuda -lcudart -lnvToolsExt -lnvidia-ml

all: p2pbenchmark

p2pbenchmark: p2pbenchmark.o
	nvcc -ccbin g++ $(CPPFLAGS) -m64 $(LDFLAGS) -arch=sm_60 -o $@ $+

p2pbenchmark.o: benchmark.cu
	nvcc -ccbin g++ -m64 $(CPPFLAGS) $(LDFLAGS) -arch=sm_60 -o $@ -c $^

clean:
	rm -f *.o p2pbenchmark
