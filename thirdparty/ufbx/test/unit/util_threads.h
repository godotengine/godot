#pragma once

#include <thread>
#include <vector>

#include "../../extra/ufbxw_cpp_threads.h"

#ifdef UFBXWI_FEATURE_THREAD_POOL

struct ufbxwt_thread_pool
{
	ufbxw_thread_sync sync;
	ufbxwi_thread_pool tp;

	ufbxwt_thread_pool()
	{
		ufbxw_cpp_threads_setup_sync(&sync);
		ufbxwi_thread_pool_init(&tp, &sync);
	}

	~ufbxwt_thread_pool()
	{
		ufbxwi_thread_pool_free(&tp);
	}
};

#endif

template <typename F>
static void thread_runner(size_t thread_id, size_t num_iters, F f) {
	for (size_t i = 0; i < num_iters; i++) {
		f(thread_id, i);
	}
}

template <typename F>
static void fork_threads(size_t num_threads, size_t num_iters, F f) {
	std::vector<std::thread> threads;
	threads.reserve(num_threads);

	for (size_t id = 0; id < num_threads; id++) {
		threads.emplace_back(thread_runner<F>, id, num_iters, f);
	}

	for (std::thread &thread : threads) {
		thread.join();
	}
}
