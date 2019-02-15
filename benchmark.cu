#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <deque>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <thread>
#include <cmath>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <functional>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>

#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include "nvToolsExt.h"

#include <assert.h>

// Gotta createt the logger before including FFmpegStreamer/Demuxer
#include "Logger.h"
simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

#include <cudaUtils.h>
#include <Timer.h>

using namespace std;

enum copyMode
{
	memcpyThroughHostPinned,
	memcpyThroughHostUnpinned,
	memcpyP2P,
	copyKernelNVLINK,
	copyKernelUVM
};

__global__ void delay(volatile int *flag, unsigned long long timeout_clocks = 10000000)
{
	// Wait until the application notifies us that it has completed queuing up the
	// experiment, or timeout and exit, allowing the application to make progress
	long long int start_clock, sample_clock;
	start_clock = clock64();

	while (!*flag) {
		sample_clock = clock64();

		if (sample_clock - start_clock > timeout_clocks) {
			break;
		}
	}
}

// This kernel is for demonstration purposes only, not a performant kernel for p2p transfers.
__global__ void copyp2p(int4* __restrict__  dest, int4 const* __restrict__ src, size_t num_elems)
{
	size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
	size_t gridSize = blockDim.x * gridDim.x;

	#pragma unroll(5)
	for (size_t i=globalId; i < num_elems; i+= gridSize)
	{
		dest[i] = src[i];
	}
}


void copyKernel(int *dest, int destDevice, int *src, int srcDevice, int bufferSize, int repeat,
				cudaStream_t streamToRun)
{
	int blockSize = 0;
	int numBlocks = 0;

	CUDA_ASSERT(cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, copyp2p));

	for (int r = 0; r < repeat; r++)
		copyp2p<<<numBlocks, blockSize, 0, streamToRun>>>((int4*)dest, (int4*)src,
															bufferSize/(4*sizeof(int)));
}

void measureBandwidthAndUtilization(int numGPUs, int numElems, int objectSize, copyMode mode)
{
	int repeat = 5;
	bool p2p = false;
	size_t bufferSize = numElems * objectSize;
	volatile int *flag = NULL;
	vector<int *> buffers(numGPUs);
	vector<int *> buffersHost(numGPUs);
	vector<int *> buffersD2D(numGPUs); // buffer for D2D, that is, intra-GPU copy
	vector<cudaEvent_t> start(numGPUs);
	vector<cudaEvent_t> stop(numGPUs);
	vector<cudaStream_t> stream(numGPUs);

	switch(mode) {
		case memcpyThroughHostPinned:
			for (int i = 0; i < numGPUs; i++)
				CUDA_ASSERT(cudaMallocHost(&buffersHost[i], bufferSize));
			break;
		case memcpyThroughHostUnpinned:
			for (int i = 0; i < numGPUs; i++)
				buffersHost[i] = new int[bufferSize];
			break
		case memcpyP2P:
		case copyKernelNVLINK:
			p2p = true;
			break
		case copyKernelUVM:
			p2p = false;
			break
	}

	CUDA_ASSERT(cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable));

	for (int d = 0; d < numGPUs; d++) {
		cudaSetDevice(d);
		cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking);
		CUDA_ASSERT(cudaMalloc(&buffers[d], bufferSize));
		CUDA_ASSERT(cudaMalloc(&buffersD2D[d], bufferSize));
		CUDA_ASSERT(cudaEventCreate(&start[d]));
		CUDA_ASSERT(cudaEventCreate(&stop[d]));
	}

	vector<double> bandwidthMatrix(numGPUs * numGPUs);

	for (int i = 0; i < numGPUs; i++) {
		cudaSetDevice(i);

		for (int j = 0; j < numGPUs; j++) {
			if (i == j) {
				bandwidthMatrix[i * numGPUs + j] = -1.0;
				continue;
			}
			int access = 0;
			if (p2p) {
				cudaDeviceCanAccessPeer(&access, i, j);
				if (access) {
					CUDA_ASSERT(cudaDeviceEnablePeerAccess(j, 0));
					CUDA_ASSERT(cudaSetDevice(j));
					CUDA_ASSERT(cudaDeviceEnablePeerAccess(i, 0));
					CUDA_ASSERT(cudaSetDevice(i));
				}
			}

			cudaStreamSynchronize(stream[i]);
			cudaCheckError();

			// Block the stream until all the work is queued up
			// DANGER! - cudaMemcpy*Async may infinitely block waiting for
			// room to push the operation, so keep the number of repetitions
			// relatively low.  Higher repetitions will cause the delay kernel
			// to timeout and lead to unstable results.
			*flag = 0;
			CUDA_ASSERT(delay<<< 1, 1, 0, stream[i]>>>(flag));


			CUDA_ASSERT(cudaEventRecord(start[i], stream[i]));
			for (int r = 0; r < repeat; r++) {
				switch(mode) {
					case memcpyThroughHostPinned:
					case memcpyThroughHostUnpinned:
						CUDA_ASSERT(cudaMemcpyAsync((void *)buffersHost[i], (const void*)buffers[i], bufferSize, cudaMemcpyDeviceToHost, stream[i]));
						CUDA_ASSERT(cudaMemcpyAsync((void *)buffers[j], (const void*)buffersHost[i], bufferSize, cudaMemcpyHostToDevice, stream[i]));
						break
					case memcpyP2P:
						CUDA_ASSERT(cudaMemcpyPeerAsync((void *)buffers[j], j, (const void*) buffers[i], i, bufferSize, stream[i]))
						break
					case copyKernelNVLINK:
					case copyKernelUVM:
						copyKernel(buffers[i], i, buffers[j], j, bufferSize, repeat, stream[i]);
						break
				}
			}
			CUDA_ASSERT(cudaEventRecord(stop[i], stream[i]));

			// Release the queued events
			*flag = 1;
			CUDA_ASSERT(cudaStreamSynchronize(stream[i]));

			float time_ms;
			cudaEventElapsedTime(&time_ms, start[i], stop[i]);
			double time_s = time_ms / 1e3;

			double gb = numElems * objectSize * repeat / (double)1e9;
			bandwidthMatrix[i * numGPUs + j] = gb / time_s;

			if (p2p && access) {
				CUDA_ASSERT(cudaDeviceDisablePeerAccess(j));
				CUDA_ASSERT(cudaSetDevice(j));
				CUDA_ASSERT(cudaDeviceDisablePeerAccess(i));
				CUDA_ASSERT(cudaSetDevice(i));
			}
		}
	}

	sstream header;
	for (int j = 0; j < numGPUs; j++) {
		header <<std::setw(6) <<j;
	}
	LOG(INFO) << header;

	sstream row;
	for (int i = 0; i < numGPUs; i++) {
		row <<std::setw(6) <<i;

		for (int j = 0; j < numGPUs; j++) {
			if (i == j)
				row <<std::setw(6) <<"-";
			row <<std::setw(6) <<std::setprecision(2) << bandwidthMatrix[i * numGPUs + j];
		}
		LOG(INFO) << row;
	}

	for (int d = 0; d < numGPUs; d++) {
		cudaSetDevice(d);
		CUDA_ASSERT(cudaFree(buffers[d]));
		CUDA_ASSERT(cudaFree(buffersD2D[d]));
		CUDA_ASSERT(cudaEventDestroy(start[d]));
		CUDA_ASSERT(cudaEventDestroy(stop[d]));
		CUDA_ASSERT(cudaStreamDestroy(stream[d]));
	}

	switch(mode) {
		case memcpyThroughHostPinned:
			for (int i = 0; i < numGPUs; i++)
				CUDA_ASSERT(cudaFree(buffersHost[i]));
			break;
		case memcpyThroughHostUnpinned:
			for (int i = 0; i < numGPUs; i++)
				delete [] buffersHost[i];
			break
	}

	CUDA_ASSERT(cudaFreeHost((void *)flag));
}

void checkP2Paccess(int numGPUs)
{
	for (int i = 0; i < numGPUs; i++) {
		CUDA_ASSERT(cudaSetDevice(i));

		for (int j = 0; j < numGPUs; j++) {
			int access;
			if (i != j) {
				CUDA_ASSERT(cudaDeviceCanAccessPeer(&access, i, j));
				LOG(INFO)<< "Device "<<i  <<access? "CAN" : "CANNOT" <<" access peer device " <<j;
			}
		}
	}
	LOG(INFO) << "***NOTE: When a device doesn't have P2P access, it falls back to cudaMemCpyAsync through the host, in which case, you'll observe a loss in bandwidth (GB/s) and higher latency (us).***";
}

int main(int argc, char **argv)
{
	int numGPUs;
	int queueDepth = 1;
	int objectSize = sizeof(int);

	CUDA_ASSERT(cudaGetDeviceCount(&numGPUs));


	//process command line args
	for (int i = 1; i < argc; i++) {
		if(0==strcmp(argv[i], "-h")){
			LOG(INFO)<< "Usage:" << argv[0] " [OPTION]..." << std::endl
					<< "Options:" <<std::endl
					<< "-h\tDisplay this Help menu" <<std::endl
					<< "-q\tQueue depth" <<std::endl
					<< "-s\tobject Size";
			return 0;
		} else if (0==strcmp(argv[i], "-q")) {
			queueDepth = atoi(argv[i+1]);
		} else if (0==strcmp(argv[i], "-q")) {
			objectSize = atoi(argv[i+1]);
		}
	}

	LOG(INFO) << "GPU to GPU Bandwidth & Latency Test\n";

	//output devices
	for (int i = 0; i < numGPUs; i++) {
		cudaDeviceProp prop;
		CUDA_ASSERT(cudaGetDeviceProperties(&prop, i));
		LOG(INFO) << "Device:" << i <<" " <<prop.name <<" pciBusID:" <<std::hex <<prop.pciBusID <<"pciDeviceID:" <<prop.pciDeviceID <<"pciDomainID:" << prop.pciDomainID;
	}

	checkP2Paccess(numGPUs);
	measureBandwidthAndUtilization(numGPUs, queueDepth, objectSize, memcpyThroughHostPinned);
	measureBandwidthAndUtilization(numGPUs, queueDepth, objectSize, memcpyThroughHostUnpinned);
	measureBandwidthAndUtilization(numGPUs, queueDepth, objectSize, memcpyP2P);
	measureBandwidthAndUtilization(numGPUs, queueDepth, objectSize, copyKernelNVLINK);
	measureBandwidthAndUtilization(numGPUs, queueDepth, objectSize, copyKernelUVM);
	exit(EXIT_SUCCESS);
}
