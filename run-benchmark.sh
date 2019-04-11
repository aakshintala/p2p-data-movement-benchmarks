#! /bin/bash
set -x #echo on

# Make a new directory for each run.
topdir=$(pwd)
benchmarkDir=$topdir"/benchmark-$(date +%Y-%m-%d-%H-%M-%S)"
mkdir $benchmarkDir
cd $benchmarkDir

let totalDataMoved=12*1024*1024*1024/4; # the benchmark uses ints, so divide by sizeof(int)
let elemSize=1024; # Because each element is a 32-bit int, that's 40kb (1 page)
while [[ $elemSize -lt 4294967296 ]]; do
	let numElems=$totalDataMoved/$elemSize;
	let computeTime=10; # Time in microseconds that simulates 'work' Range 10u to 100m
	while [[ $computeTime -lt 100000 ]]; do # stop at 100 milliseconds
		subDir=$numElems-$elemSize-$computeTime
		mkdir $subDir
		cd $subDir
		sudo timeout 6000 $topdir/p2pbenchmark -q $numElems -s $elemSize -t $computeTime | tee timing
		cd ..
		let computeTime=$computeTime*10;
	done
	let elemSize=$elemSize*4;
done
