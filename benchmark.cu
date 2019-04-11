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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include "nvToolsExt.h"

#include <assert.h>

// Gotta createt the logger before including FFmpegStreamer/Demuxer
#include "Logger.h"
simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();
simplelogger::Logger *fileLogger = simplelogger::LoggerFactory::CreateFileLogger("./GPU-utilization");

#include "cudaUtils.h"
#include "Timer.h"
#include <nvml.h>

using namespace std;

enum copyMode
{
	memcpyThroughHostPinned,
	memcpyThroughHostUnpinned,
	memcpyP2P,
	copyKernelNVLINK,
	copyKernelUVM
};

string getCopyModeString(copyMode mode)
{
	switch(mode){
		case memcpyThroughHostPinned:
			return "memcpyThroughHostPinned";
		case memcpyThroughHostUnpinned:
			return "memcpyThroughHostUnpinned";
		case memcpyP2P:
			return "memcpyP2P";
		case copyKernelNVLINK:
			return "copyKernelNVLINK";
		case copyKernelUVM:
			return "copyKernelUVM";
		default:
			return "";
	}
}

__global__ void delay(volatile int *flag, uint64_t timeout_clocks = 10000000)
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

__global__ void spin_for_cycles(uint64_t timeout_clocks = 10000000)
{
	// Wait until the application notifies us that it has completed queuing up the
	// experiment, or timeout and exit, allowing the application to make progress
	long long int start_clock, sample_clock;
	start_clock = clock64();

	while (true) {
		sample_clock = clock64();

		if (sample_clock - start_clock > timeout_clocks) {
			break;
		}
	}
}

// This kernel is for demonstration purposes only, not a performant kernel for p2p transfers.
__global__ void incKernel(int*  buffer, uint64_t num_elems)
{
	size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
	size_t gridSize = blockDim.x * gridDim.x;

	#pragma unroll(5)
	for (uint64_t i=globalId; i < num_elems; i+= gridSize)
	{
		buffer[i] += 1;
	}
}

void incBuffer(int *buffer, uint64_t bufferSize, cudaStream_t streamToRun)
{
	int blockSize = 0;
	int numBlocks = 0;

	CUDA_ASSERT(cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, incKernel));

	incKernel<<<numBlocks, blockSize, 0, streamToRun>>>((int*)buffer,bufferSize/(sizeof(int)));
}

// This kernel is for demonstration purposes only, not a performant kernel for p2p transfers.
__global__ void copyp2p(int4* __restrict__  dest, int4 const* __restrict__ src, uint64_t num_elems)
{
	size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
	size_t gridSize = blockDim.x * gridDim.x;

	#pragma unroll(5)
	for (uint64_t i=globalId; i < num_elems; i+= gridSize)
	{
		dest[i] = src[i];
	}
}


void copyKernel(int *dest, int *src, uint64_t bufferSize,cudaStream_t streamToRun)
{
	int blockSize = 128;
	int numBlocks = 1024;

	CUDA_ASSERT(cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, copyp2p));

	copyp2p<<<numBlocks, blockSize, 0, streamToRun>>>((int4*)dest, (int4*)src,
															bufferSize/(4*sizeof(int)));
}

void nvmlMeasure(volatile int *start, volatile bool *stop, int deviceIndex) {
	nvmlDevice_t nvmlDeviceHandle;
	char * deviceName;
	nvmlUtilization_t utilization;
	unsigned int power_mW;

	NVML_ASSERT(nvmlDeviceGetHandleByIndex(deviceIndex, &nvmlDeviceHandle));
	deviceName = new char[256];
	NVML_ASSERT(nvmlDeviceGetName(nvmlDeviceHandle, deviceName, 255));

	FILELOG(INFO) <<"Measuring " <<deviceName <<" " <<deviceIndex <<std::endl;
	bool started = false;
	while(*stop != true) {
		if (1==(*start) && !started) {
			FILELOG(INFO) <<"START HERE" <<std::endl;
			started = true;
		}
		NVML_ASSERT(nvmlDeviceGetUtilizationRates(nvmlDeviceHandle, &utilization));
		NVML_ASSERT(nvmlDeviceGetPowerUsage(nvmlDeviceHandle, &power_mW));
		FILELOG(INFO) <<" gpu utilization(%) = " <<utilization.gpu << " power = "<< 1e-3 * power_mW << std::endl;
		usleep(1);
	}

	delete [] deviceName;
}

#define SRCGPU 0
#define DESTGPU 1
#define NUMSTREAMS 1

void measureBandwidthAndUtilization(int srcGPU, int destGPU, uint64_t numElems, size_t objectSize, uint64_t computeTimeCycles, copyMode mode)
{
	bool p2p = false;
	uint64_t bufferSize = numElems * objectSize;
	volatile int *flag = NULL;
	volatile bool nvmlDone = false;
	int numGPUs = 2;
	vector<int *> buffers(numGPUs);
	int * bufferHost;
	vector<cudaEvent_t> start(numGPUs);
	vector<cudaEvent_t> stop(numGPUs);
	cudaStream_t srcGPUStream;
	vector<cudaStream_t> stream(NUMSTREAMS);

	switch(mode) {
		case memcpyThroughHostPinned:
			CUDA_ASSERT(cudaMallocHost(&bufferHost, bufferSize));
			break;
		case memcpyThroughHostUnpinned:
			bufferHost = new int[bufferSize];
			break;
		case memcpyP2P:
		case copyKernelNVLINK:
			p2p = true;
			break;
		case copyKernelUVM:
			p2p = false;
			CUDA_ASSERT(cudaMallocManaged(&bufferHost, bufferSize));
			break;
	}

	CUDA_ASSERT(cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable));

	cudaSetDevice(srcGPU);
	cudaStreamCreateWithFlags(&srcGPUStream, cudaStreamNonBlocking);
	CUDA_ASSERT(cudaMalloc(&buffers[SRCGPU], bufferSize));
	CUDA_ASSERT(cudaEventCreate(&start[SRCGPU]));
	CUDA_ASSERT(cudaEventCreate(&stop[SRCGPU]));

	cudaSetDevice(destGPU);
	for( int i = 0; i < NUMSTREAMS; i++)
		cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
	CUDA_ASSERT(cudaMalloc(&buffers[DESTGPU], bufferSize));
	CUDA_ASSERT(cudaEventCreate(&start[DESTGPU]));
	CUDA_ASSERT(cudaEventCreate(&stop[DESTGPU]));

	if (mode == copyKernelUVM) {
		cudaSetDevice(srcGPU);
		incBuffer(bufferHost, bufferSize, srcGPUStream);
		CUDA_ASSERT(cudaStreamSynchronize(srcGPUStream));
		cudaSetDevice(destGPU);
	}
	int access = 0;
	if (p2p) {
		cudaDeviceCanAccessPeer(&access, destGPU, srcGPU);
		if (access) {
			CUDA_ASSERT(cudaDeviceEnablePeerAccess(srcGPU, 0));
			CUDA_ASSERT(cudaSetDevice(srcGPU));
			CUDA_ASSERT(cudaDeviceEnablePeerAccess(destGPU, 0));
			CUDA_ASSERT(cudaSetDevice(destGPU));
		}
	}

	double bandwidth;
	pid_t pid = fork();

	if ( pid == 0 ) {
		string cuFile =  "./cpu-utilization" + getCopyModeString(mode) + std::to_string(srcGPU)+ std::to_string(destGPU);
		int childOutFD = open(cuFile.c_str(), O_WRONLY|O_CREAT, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);
		dup2(childOutFD, STDOUT_FILENO);
		dup2(childOutFD, STDERR_FILENO);
		close(childOutFD);
		execl( "./cpu-stat", "cpu-stat");
		exit(EXIT_SUCCESS);
	} else {
		cudaSetDevice(destGPU);
		for( int i = 0; i < NUMSTREAMS; i++)
			CUDA_ASSERT(cudaStreamSynchronize(stream[i]));

		// Block the stream until all the work is queued up
		// DANGER! - cudaMemcpy*Async may infinitely block waiting for
		// room to push the operation, so keep the number of repetitions
		// relatively low.  Higher repetitions will cause the delay kernel
		// to timeout and lead to unstable results.
		*flag = 0;
		for( int i = 0; i < NUMSTREAMS; i++)
			delay<<< 1, 1, 0, stream[i]>>>(flag);

		nvmlDone = false;
		std::thread nvmlThread = std::thread(&nvmlMeasure, flag, &nvmlDone, destGPU);

		int blockSize = 128;
		int numBlocks = 1024;
		CUDA_ASSERT(cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, spin_for_cycles));

		float time_ms;
		CUDA_ASSERT(cudaEventRecord(start[DESTGPU], stream[0]));
		uint64_t offset = 0;
		int streamIdx = 0;
		for (int e = 0; e < numElems; e++) {
			streamIdx = e%NUMSTREAMS;
			//usleep(computeTimeMicrosecond);
			switch(mode) {
				case memcpyThroughHostPinned:
				case memcpyThroughHostUnpinned:
					offset = e*objectSize/sizeof(int);
					CUDA_ASSERT(cudaMemcpyAsync((void *)(bufferHost+offset),
								(const void*)(buffers[SRCGPU]+offset),
								objectSize, cudaMemcpyDeviceToHost,	stream[streamIdx]));
								//objectSize, cudaMemcpyDeviceToHost,	stream[DESTGPU]));
					CUDA_ASSERT(cudaMemcpyAsync((void *)(buffers[DESTGPU]+offset),
											(const void*)(bufferHost+offset),
											objectSize, cudaMemcpyHostToDevice, stream[streamIdx]));
											//objectSize, cudaMemcpyHostToDevice, stream[DESTGPU]));
					break;
				case memcpyP2P:
					offset = e*objectSize/sizeof(int);
					CUDA_ASSERT(cudaMemcpyPeerAsync((void *)(buffers[DESTGPU]+offset), destGPU,
													(const void*)(buffers[SRCGPU]+offset),
													srcGPU, objectSize, stream[streamIdx]));
													//srcGPU, objectSize, stream[DESTGPU]));
					break;
				case copyKernelNVLINK:
					offset = e*objectSize/sizeof(int);
					copyKernel(buffers[DESTGPU]+offset, buffers[SRCGPU]+offset, objectSize,
									stream[streamIdx]);
									//stream[DESTGPU]);
					break;
				case copyKernelUVM:
					// Copy from and to UVM managed buffers
					offset = e*objectSize/sizeof(int);
					copyKernel(buffers[DESTGPU]+offset, bufferHost+offset, objectSize,
									stream[streamIdx]);
									//stream[DESTGPU]);
					break;
			}
			spin_for_cycles<<< numBlocks, blockSize, 0, stream[streamIdx]>>>(computeTimeCycles);
		}
		//CUDA_ASSERT(cudaEventRecord(stop[DESTGPU], stream[DESTGPU]));
		CUDA_ASSERT(cudaEventRecord(stop[DESTGPU], stream[streamIdx]));

		// Release the queued events
		*flag = 1;
		for( int i = 0; i < NUMSTREAMS; i++)
			CUDA_ASSERT(cudaStreamSynchronize(stream[i]));
		cudaEventElapsedTime(&time_ms, start[DESTGPU], stop[DESTGPU]);

		sleep(1);
		nvmlDone = true;
		nvmlThread.join();

		kill(pid, SIGTERM);
		int status;
		waitpid(pid, &status,0);

		double time_s = time_ms / 1e3;

		double gb = 0.0;
		gb = bufferSize / (double)1e9;
		LOG(INFO) << "gb = " <<gb <<" time_s = " <<time_s <<std::endl;
		bandwidth = gb / time_s;

		if (p2p && access) {
			CUDA_ASSERT(cudaDeviceDisablePeerAccess(srcGPU));
			CUDA_ASSERT(cudaSetDevice(srcGPU));
			CUDA_ASSERT(cudaDeviceDisablePeerAccess(destGPU));
			CUDA_ASSERT(cudaSetDevice(destGPU));
		}

		LOG(INFO) << srcGPU <<" ->" <<destGPU <<" = " <<std::setprecision(2) << bandwidth <<" GBps" << std::endl;

		cudaSetDevice(srcGPU);
		CUDA_ASSERT(cudaFree(buffers[SRCGPU]));
		CUDA_ASSERT(cudaEventDestroy(start[SRCGPU]));
		CUDA_ASSERT(cudaEventDestroy(stop[SRCGPU]));
		CUDA_ASSERT(cudaStreamDestroy(srcGPUStream));

		cudaSetDevice(destGPU);
		CUDA_ASSERT(cudaFree(buffers[DESTGPU]));
		CUDA_ASSERT(cudaEventDestroy(start[DESTGPU]));
		CUDA_ASSERT(cudaEventDestroy(stop[DESTGPU]));
		for(int i = 0; i < NUMSTREAMS; i++)
			CUDA_ASSERT(cudaStreamDestroy(stream[i]));

		switch(mode) {
			case memcpyThroughHostPinned:
				CUDA_ASSERT(cudaFreeHost(bufferHost));
				break;
			case memcpyThroughHostUnpinned:
				delete [] bufferHost;
				break;
			case copyKernelUVM:
				CUDA_ASSERT(cudaFree(bufferHost));
				break;
			default:
				break;
		}

		CUDA_ASSERT(cudaFreeHost((void *)flag));
	}
}

void measureBandwidthAndUtilizationA2A(int numGPUs, size_t numElems, size_t objectSize, uint64_t computeTimeCycles, copyMode mode)
{
	for (int i = 0; i< numGPUs; i++) {
		for (int j = 0; j < numGPUs; j++) {
			if (i == j) continue;
			measureBandwidthAndUtilization(i, j, numElems, objectSize, computeTimeCycles,  mode);
		}
	}
}

void checkP2Paccess(int numGPUs)
{
	for (int i = 0; i < numGPUs; i++) {
		CUDA_ASSERT(cudaSetDevice(i));

		for (int j = 0; j < numGPUs; j++) {
			int access = 0;
			if (i != j) {
				CUDA_ASSERT(cudaDeviceCanAccessPeer(&access, i, j));
				LOG(INFO)<< "Device "<<i  <<(access? " CAN" : " CANNOT") <<" access peer device " <<j <<std::endl;
			}
		}
	}
	LOG(INFO) << "***NOTE: When a device doesn't have P2P access, it falls back to cudaMemCpyAsync through the host, in which case, you'll observe a loss in bandwidth (GB/s) and higher latency (us).***" << std::endl;
}

int main(int argc, char **argv)
{
	int numGPUs = 0;
	uint64_t queueDepth = 1000000; // 1 million
	size_t objectSize = 1024*sizeof(int);
	int computeTimeMicroseconds = 10;

	CUDA_ASSERT(cudaGetDeviceCount(&numGPUs));
	assert(numGPUs != 0);
	//process command line args
	for (int i = 1; i < argc; i++) {
		if (0==strcmp(argv[i], "-h")) {
			LOG(ERROR) << "Usage:" << argv[0] <<" [OPTION]..." <<std::endl << "Options:" <<std::endl << "-h\tDisplay this Help menu" <<std::endl <<"-q\tQueue depth" <<std::endl << "-s\tobject Size (in increments of sizeof(int))" << std::endl;
			return 0;
		} else if (0==strcmp(argv[i], "-q")) {
			queueDepth = atoi(argv[i+1]);
		} else if (0==strcmp(argv[i], "-s")) {
			objectSize = atoi(argv[i+1])*sizeof(int);
		} else if (0==strcmp(argv[i], "-t")) {
			computeTimeMicroseconds = atoi(argv[i+1]);
		}
	}

	LOG(INFO) << "GPU to GPU Bandwidth & Latency Test\n";
	LOG(INFO) << "Moving " << queueDepth*objectSize/1000/1000 << " MB of data" << std::endl;

	//output devices
	int clockRates[numGPUs];
	for (int i = 0; i < numGPUs; i++) {
		cudaDeviceProp prop;
		CUDA_ASSERT(cudaGetDeviceProperties(&prop, i));
		LOG(INFO) << "Device:" << i <<" " <<prop.name <<" pciBusID:" <<std::hex <<prop.pciBusID <<" pciDeviceID:" <<std::hex <<prop.pciDeviceID <<" pciDomainID:" <<std::hex << prop.pciDomainID <<std::endl <<std::dec;
		clockRates[i] = prop.clockRate;
	}

	// Initialize NVML library
	NVML_ASSERT(nvmlInit());

	checkP2Paccess(numGPUs);
	//uint64_t computeTimeCycles = computeTimeMicroseconds;
	uint64_t computeTimeCycles = (computeTimeMicroseconds/1e6)*clockRates[0]*1e3;
	LOG(INFO) << "computeTimeCycles = " << computeTimeCycles <<std::endl;

	LOG(INFO) <<"\nmemcpyThroughHostPinned\n";
	FILELOG(INFO) <<"\nmemcpyThroughHostPinned\n";
	measureBandwidthAndUtilization(0, 1, queueDepth, objectSize, computeTimeCycles,
		memcpyThroughHostPinned);
	sleep(2);

	LOG(INFO) <<"\nmemcpyThroughHostUnpinned\n";
	FILELOG(INFO) <<"\nmemcpyThroughHostUnpinned\n";
	measureBandwidthAndUtilization(0, 1, queueDepth, objectSize, computeTimeCycles,
		memcpyThroughHostUnpinned);
	sleep(2);

	LOG(INFO) <<"\nmemcpyP2P\n";
	FILELOG(INFO) <<"\nmemcpyP2P\n";
	measureBandwidthAndUtilization(0, 3, queueDepth, objectSize, computeTimeCycles,
		memcpyP2P);
	sleep(2);

	LOG(INFO) <<"\ncopyKernelNVLINK\n";
	FILELOG(INFO) <<"\ncopyKernelNVLINK\n";
	measureBandwidthAndUtilization(0, 3, queueDepth, objectSize, computeTimeCycles,
		copyKernelNVLINK);
	sleep(2);

	LOG(INFO) <<"\ncopyKernelUVM\n";
	FILELOG(INFO) <<"\ncopyKernelUVM\n";
	measureBandwidthAndUtilization(0, 3, queueDepth, objectSize, computeTimeCycles,
		copyKernelUVM);

	//Shutdown NVML
	NVML_ASSERT(nvmlShutdown());
	exit(EXIT_SUCCESS);
}
