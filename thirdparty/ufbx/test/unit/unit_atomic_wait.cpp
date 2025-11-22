#include "unit_test.h"

#define UFBXW_UNIT_TEST 1
#define UFBXWI_FEATURE_ATOMICS 1
#define UFBXWI_FEATURE_THREAD_POOL 1
#include "../../ufbx_write.c"

#include "util_threads.h"
#include <atomic>

#define UFBXWT_UNIT_CATEGORY "atomic_wait"

UFBXWT_UNIT_TEST(atomic_wait)
{
	ufbxwt_thread_pool thread_pool;
	ufbxwi_thread_pool &tp = thread_pool.tp;

	uint32_t value = 0;
	ufbxwi_atomic_u32 lock = { };

	const size_t num_threads = 16;
	const size_t num_iters = 1000;
	fork_threads(num_threads, num_iters, [&](size_t id, size_t index) {
		for (;;) {
			if (ufbxwi_atomic_cas(&lock, 0, 1)) {
				break;
			} else {
				ufbxwi_atomic_wait(&tp, &lock, 1);
			}
		}

		value++;

		ufbxwi_atomic_store(&lock, 0);
		ufbxwi_atomic_notify(&tp, &lock, 1);
	});

	ufbxwt_assert(value == num_threads * num_iters);
}

UFBXWT_UNIT_TEST(atomic_wait_multiple)
{
	ufbxwt_thread_pool thread_pool;
	ufbxwi_thread_pool &tp = thread_pool.tp;

	constexpr size_t num_slots = 64;
	uint32_t values[num_slots] = { };
	ufbxwi_atomic_u32 locks[num_slots] = { };

	const size_t num_threads = 32;
	const size_t num_iters = 64 * 1024;
	fork_threads(num_threads, num_iters, [&](size_t id, size_t index) {
		size_t slot = (id * 3 + index) % num_slots;

		for (;;) {
			if (ufbxwi_atomic_cas(&locks[slot], 0, 1)) {
				break;
			} else {
				ufbxwi_atomic_wait(&tp, &locks[slot], 1);
			}
		}

		if (index % 50000 == 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds{1});
		}

		values[slot]++;

		ufbxwi_atomic_store(&locks[slot], 0);
		ufbxwi_atomic_notify(&tp, &locks[slot], 1);
	});

	for (size_t i = 0; i < num_slots; i++) {
		ufbxwt_assert(values[i] == num_threads * num_iters / num_slots);
	}
}

template <size_t NumThreads, size_t NumLocks>
static void test_atomic_wait_stuck_imp()
{
	constexpr size_t num_threads = NumThreads;
	constexpr size_t num_locks = NumLocks;
	constexpr size_t num_iters = 10;

	ufbxwt_thread_pool thread_pool;
	ufbxwi_thread_pool &tp = thread_pool.tp;

	ufbxwi_atomic_u32 locks[num_locks] = { };

	ufbxwi_atomic_u32 barrier = { };

	auto increment_barriers = [&](){
		for (size_t iter = 0; iter < num_iters; iter++) {
			for (;;) {
				uint32_t value = ufbxwi_atomic_load_acquire(&barrier);
				if (value == num_threads) {
					break;
				} else {
					ufbxwt_assert(value < num_threads);
					ufbxwi_atomic_wait(&tp, &barrier, value);
				}
			}

			ufbxwi_atomic_store(&barrier, 0);
			std::this_thread::sleep_for(std::chrono::milliseconds(10));

			for (size_t i = 0; i < num_locks; i++) {
				ufbxwi_atomic_add(&locks[i], 1u);
				ufbxwi_atomic_notify(&tp, &locks[i], NumThreads / NumLocks);
			}
		}
	};

	std::thread barrier_thread{increment_barriers};

	fork_threads(num_threads, num_iters, [&](size_t id, size_t index) {
		const uint32_t ref = (uint32_t)index;
		const uint32_t lock = id % num_locks;
		for (;;) {
			uint32_t value = ufbxwi_atomic_load_relaxed(&locks[lock]);
			if (value == ref) {
				break;
			} else {
				ufbxwt_assert(value < ref);
				ufbxwi_atomic_wait(&tp, &locks[lock], value);
			}
		}

		ufbxwi_atomic_add(&barrier, 1);
		ufbxwi_atomic_notify(&tp, &barrier, 1);
	});

	barrier_thread.join();
}

UFBXWT_UNIT_TEST(atomic_wait_many_single)
{
	test_atomic_wait_stuck_imp<512, 512>();
}

UFBXWT_UNIT_TEST(atomic_wait_many_overlap)
{
	test_atomic_wait_stuck_imp<512, 128>();
}
