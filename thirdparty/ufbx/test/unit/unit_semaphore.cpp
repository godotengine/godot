#include "unit_test.h"

#define UFBXW_UNIT_TEST 1
#define UFBXWI_FEATURE_ATOMICS 1
#define UFBXWI_FEATURE_THREAD_POOL 1
#include "../../ufbx_write.c"

#include "util_threads.h"

#include <atomic>

#define UFBXWT_UNIT_CATEGORY "semaphore"

UFBXWT_UNIT_TEST(semaphore_simple)
{
	ufbxwt_thread_pool thread_pool;
	ufbxwi_thread_pool &tp = thread_pool.tp;

	std::atomic_uint value = 0;
	ufbxwi_semaphore sema = { };

	const size_t num_threads = 16;
	const size_t num_iters = 100000;

	auto sema_post = [&]() {
		for (size_t i = 0; i < num_iters; i++) {
			if (i % 10000 == 0) {
				std::this_thread::sleep_for(std::chrono::milliseconds{1});
			}

			if (i % 3 == 0) {
				for (size_t i = 0; i < num_threads; i++) {
					ufbxwi_semaphore_notify(&tp, &sema, 1u);
				}
			} else {
				ufbxwi_semaphore_notify(&tp, &sema, (uint32_t)num_threads);
			}
		}
	};

	std::thread post_thread { sema_post };

	fork_threads(num_threads, num_iters, [&](size_t id, size_t index) {
		ufbxwi_semaphore_wait(&tp, &sema);
		value.fetch_add(1u, std::memory_order_relaxed);
	});

	post_thread.join();

	ufbxwt_assert(value == num_threads * num_iters);
}

UFBXWT_UNIT_TEST(semaphore_try_wait)
{
	ufbxwt_thread_pool thread_pool;
	ufbxwi_thread_pool &tp = thread_pool.tp;

	std::atomic_uint value = 0;
	ufbxwi_semaphore sema = { };

	const size_t num_threads = 16;
	const size_t num_iters = 1000;

	auto sema_post = [&]() {
		for (size_t i = 0; i < num_iters; i++) {
			if (i % 200 == 0) {
				std::this_thread::sleep_for(std::chrono::milliseconds{1});
			}
			if (i % 3 == 0) {
				for (size_t i = 0; i < num_threads; i++) {
					ufbxwi_semaphore_notify(&tp, &sema, 1u);
				}
			} else {
				ufbxwi_semaphore_notify(&tp, &sema, (uint32_t)num_threads);
			}
		}
	};

	std::thread post_thread { sema_post };

	fork_threads(num_threads, num_iters, [&](size_t id, size_t index) {
		while (!ufbxwi_semaphore_try_wait(&tp, &sema)) {
			ufbxwi_atomic_pause();
		}
		value.fetch_add(1u, std::memory_order_relaxed);
	});

	post_thread.join();

	ufbxwt_assert(value == num_threads * num_iters);
}

