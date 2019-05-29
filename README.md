# p2p-data-movement-benchmarks
The benchmark measures the CPU-utilization and the GPU-utilization 
while transferring data between 2 GPUs in the following settings:
1. cudaMemcpy through pinned host memory
2. cudaMemcpy through pinned host memory
3. cudaMemcpyP2P between 2 GPUs
4. One GPU pulls data from another GPU's memory over NVLINK
5. One GPU pulls data from another GPU's memory over PCIe with UVM enabled (NVLINK was disabled)

Currently, the data being moved is all 0s (I presume cudaMalloc zeroes out GPU memory), 
except for the UVM case (where it's all 1s).

The run-benchmark script currently executes the benchmark in the following configurations:
```shell
let totalDataMoved=12*1024*1024*1024/4; # 12GiB; the benchmark uses ints, so divide by sizeof(int) 
let elemSize=1024; # Because each element is a 32-bit int, that's 4kb (1 page)
# at each elem size we loop over 10us to 100ms (multiplicative step of 10) of 'work' (just a busyloop kernel)
while [[ $elemSize -lt 4294967296 ]]; do # upto 4GiB elements (multiplicative factor of 4)
	let numElems=$totalDataMoved/$elemSize;
	let computeTime=10; # Time in microseconds that simulates 'work' Range 10u to 100m
	while [[ $computeTime -lt 100000 ]]; do # stop at 100 milliseconds
		subDir=$numElems-$elemSize-$computeTime
		mkdir $subDir
		cd $subDir
    # The following command times out after 100 mins (yes sometimes it does take very long...)
		sudo timeout 6000 $topdir/p2pbenchmark -q $numElems -s $elemSize -t $computeTime | tee timing
		cd ..
		let computeTime=$computeTime*10;
	done
	let elemSize=$elemSize*4;
done
```
