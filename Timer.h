#pragma once
#include <chrono>

class Timer {
  public:
	Timer() {
		this->reset();
	}

	void reset() {
		this->time  = std::chrono::high_resolution_clock::now();
	}

	double getElapsedMicroseconds() const {
		return std::chrono::duration_cast<std::chrono::microseconds>(
			std::chrono::high_resolution_clock::now() - this->time).count();
	}

	double getElapsedNanoseconds() const {
		return std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::high_resolution_clock::now() - this->time).count();
	}

  private:
	std::chrono::high_resolution_clock::time_point time;
};