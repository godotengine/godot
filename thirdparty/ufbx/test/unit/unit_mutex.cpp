#include "unit_test.h"

#define UFBXW_UNIT_TEST 1
#define UFBXWI_FEATURE_ATOMICS 1
#define UFBXWI_FEATURE_THREAD_POOL 1
#include "../../ufbx_write.c"

#include "util_threads.h"

#define UFBXWT_UNIT_CATEGORY "mutex"

UFBXWT_UNIT_TEST(mutex_lock)
{
	ufbxwt_thread_pool thread_pool;
	ufbxwi_thread_pool &tp = thread_pool.tp;

	uint32_t value = 0;
	ufbxwi_mutex mutex = { };

	const size_t num_threads = 16;
	const size_t num_iters = 100000;
	fork_threads(num_threads, num_iters, [&](size_t id, size_t index) {
		ufbxwi_mutex_lock(&tp, &mutex);
		value++;
		ufbxwi_mutex_unlock(&tp, &mutex);
	});

	ufbxwt_assert(value == num_threads * num_iters);
}

UFBXWT_UNIT_TEST(mutex_try_lock)
{
	ufbxwt_thread_pool thread_pool;
	ufbxwi_thread_pool &tp = thread_pool.tp;

	uint32_t value = 0;
	ufbxwi_mutex mutex = { };

	const size_t num_threads = 16;
	const size_t num_iters = 100000;
	fork_threads(num_threads, num_iters, [&](size_t id, size_t index) {
		for (;;) {
			if (!ufbxwi_mutex_try_lock(&tp, &mutex)) {
				continue;
			}

			value++;
			ufbxwi_mutex_unlock(&tp, &mutex);
			break;
		}
	});

	ufbxwt_assert(value == num_threads * num_iters);
}

UFBXWT_UNIT_TEST(mutex_multiple)
{
	ufbxwt_thread_pool thread_pool;
	ufbxwi_thread_pool &tp = thread_pool.tp;

	constexpr size_t num_slots = 8;
	uint32_t values[num_slots] = { };
	ufbxwi_mutex mutex[num_slots] = { };

	const size_t num_threads = 16;
	const size_t num_iters = 64 * 1024;
	fork_threads(num_threads, num_iters, [&](size_t id, size_t index) {
		size_t slot = index % num_slots;

		ufbxwi_mutex_lock(&tp, &mutex[slot]);
		values[slot]++;
		ufbxwi_mutex_unlock(&tp, &mutex[slot]);
	});

	for (size_t i = 0; i < num_slots; i++) {
		ufbxwt_assert(values[i] == num_threads * num_iters / num_slots);
	}
}

UFBXWT_UNIT_TEST(mutex_sleep)
{
	ufbxwt_thread_pool thread_pool;
	ufbxwi_thread_pool &tp = thread_pool.tp;

	uint32_t value = 0;
	ufbxwi_mutex mutex = { };

	const size_t num_threads = 8;
	const size_t num_iters = 100;
	fork_threads(num_threads, num_iters, [&](size_t id, size_t index) {
		ufbxwi_mutex_lock(&tp, &mutex);

		if ((id * 3 + index + 7) % 23 == 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds{1});
		}

		value++;
		ufbxwi_mutex_unlock(&tp, &mutex);
	});

	ufbxwt_assert(value == num_threads * num_iters);
}
