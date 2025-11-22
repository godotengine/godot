#include "unit_test.h"

#define UFBXW_UNIT_TEST 1
#define UFBXWI_FEATURE_ATOMICS 1
#include "../../ufbx_write.c"

#include "util_threads.h"

#define UFBXWT_UNIT_CATEGORY "atomic"

UFBXWT_UNIT_TEST(atomic_add)
{
	ufbxwi_atomic_u32 value = { };

	const size_t num_threads = 16;
	const size_t num_iters = 100000;
	fork_threads(num_threads, num_iters, [&](size_t id, size_t index) {
		ufbxwi_atomic_add(&value, 1);
	});

	ufbxwt_assert(ufbxwi_atomic_load_relaxed(&value) == num_threads * num_iters);
}

UFBXWT_UNIT_TEST(atomic_cas)
{
	ufbxwi_atomic_u32 value = { };

	const size_t num_threads = 16;
	const size_t num_iters = 100000;
	fork_threads(num_threads, num_iters, [&](size_t id, size_t index) {
		for (;;) {
			uint32_t v = ufbxwi_atomic_load_relaxed(&value);
			if (ufbxwi_atomic_cas(&value, v, v + 1)) {
				break;
			}
		}
	});

	ufbxwt_assert(ufbxwi_atomic_load_relaxed(&value) == num_threads * num_iters);
}

