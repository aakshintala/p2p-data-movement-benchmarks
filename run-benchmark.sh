#! /bin/bash
set -x #echo on

# Make a new directory for each run.
topdir=$(pwd)
benchmarkDir=$topdir"/benchmark-$(date +%Y-%m-%d-%H-%M-%S)"
mkdir $benchmarkDir
cd $benchmarkDir

let totalDataMoved=12*1000*1000*1000/4; # the benchmark uses ints, so divide by sizeof(int)
let elemSize=10000; # Because each element is a 32-bit int, that's 40kb (1 page)
let computeTime=10; # Time in microseconds that simulates 'work' Range 10u to 100m
while [[ $elemSize -lt 4000000000 ]]; do
	let numElems=$totalDataMoved/$elemSize;
	while [[ $computeTime -lt 100000 ]]; do
		subDir=$numElems-$elemSize-$computeTime
		mkdir $subDir
		cd $subDir
		sudo $topdir/p2pbenchmark -q $numElems -s $elemSize -t $computeTime > timing
		cd ..
		let computeTime=$computeTime*10;
	done
	let elemSize=$elemSize*4;
done