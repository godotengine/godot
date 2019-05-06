/*
 * Copyright 2015-2017 ARM Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SPIRV_CROSS_BARRIER_HPP
#define SPIRV_CROSS_BARRIER_HPP

#include <atomic>
#include <thread>

namespace spirv_cross
{
class Barrier
{
public:
	Barrier()
	{
		count.store(0);
		iteration.store(0);
	}

	void set_release_divisor(unsigned divisor)
	{
		this->divisor = divisor;
	}

	static inline void memoryBarrier()
	{
		std::atomic_thread_fence(std::memory_order_seq_cst);
	}

	void reset_counter()
	{
		count.store(0);
		iteration.store(0);
	}

	void wait()
	{
		unsigned target_iteration = iteration.load(std::memory_order_relaxed) + 1;
		// Overflows cleanly.
		unsigned target_count = divisor * target_iteration;

		// Barriers don't enforce memory ordering.
		// Be as relaxed about the barrier as we possibly can!
		unsigned c = count.fetch_add(1u, std::memory_order_relaxed);

		if (c + 1 == target_count)
		{
			iteration.store(target_iteration, std::memory_order_relaxed);
		}
		else
		{
			// If we have more threads than the CPU, don't hog the CPU for very long periods of time.
			while (iteration.load(std::memory_order_relaxed) != target_iteration)
				std::this_thread::yield();
		}
	}

private:
	unsigned divisor = 1;
	std::atomic<unsigned> count;
	std::atomic<unsigned> iteration;
};
}

#endif
