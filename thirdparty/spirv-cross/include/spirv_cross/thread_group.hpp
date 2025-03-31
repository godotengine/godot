/*
 * Copyright 2015-2017 ARM Limited
 * SPDX-License-Identifier: Apache-2.0
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

#ifndef SPIRV_CROSS_THREAD_GROUP_HPP
#define SPIRV_CROSS_THREAD_GROUP_HPP

#include <condition_variable>
#include <mutex>
#include <thread>

namespace spirv_cross
{
template <typename T, unsigned Size>
class ThreadGroup
{
public:
	ThreadGroup(T *impl)
	{
		for (unsigned i = 0; i < Size; i++)
			workers[i].start(&impl[i]);
	}

	void run()
	{
		for (auto &worker : workers)
			worker.run();
	}

	void wait()
	{
		for (auto &worker : workers)
			worker.wait();
	}

private:
	struct Thread
	{
		enum State
		{
			Idle,
			Running,
			Dying
		};
		State state = Idle;

		void start(T *impl)
		{
			worker = std::thread([impl, this] {
				for (;;)
				{
					{
						std::unique_lock<std::mutex> l{ lock };
						cond.wait(l, [this] { return state != Idle; });
						if (state == Dying)
							break;
					}

					impl->main();

					std::lock_guard<std::mutex> l{ lock };
					state = Idle;
					cond.notify_one();
				}
			});
		}

		void wait()
		{
			std::unique_lock<std::mutex> l{ lock };
			cond.wait(l, [this] { return state == Idle; });
		}

		void run()
		{
			std::lock_guard<std::mutex> l{ lock };
			state = Running;
			cond.notify_one();
		}

		~Thread()
		{
			if (worker.joinable())
			{
				{
					std::lock_guard<std::mutex> l{ lock };
					state = Dying;
					cond.notify_one();
				}
				worker.join();
			}
		}
		std::thread worker;
		std::condition_variable cond;
		std::mutex lock;
	};
	Thread workers[Size];
};
}

#endif
