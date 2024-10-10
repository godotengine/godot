/**************************************************************************/
/*  test_rid.h                                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef TEST_RID_H
#define TEST_RID_H

#include "core/os/thread.h"
#include "core/templates/local_vector.h"
#include "core/templates/rid.h"
#include "core/templates/rid_owner.h"

#include "tests/test_macros.h"

#include <mach/mach.h>
#include <pthread.h>
#include <unistd.h>

namespace TestRID {
TEST_CASE("[RID] Default Constructor") {
	RID rid;

	CHECK(rid.get_id() == 0);
}

TEST_CASE("[RID] Factory method") {
	RID rid = RID::from_uint64(1);

	CHECK(rid.get_id() == 1);
}

TEST_CASE("[RID] Operators") {
	RID rid = RID::from_uint64(1);

	RID rid_zero = RID::from_uint64(0);
	RID rid_one = RID::from_uint64(1);
	RID rid_two = RID::from_uint64(2);

	CHECK_FALSE(rid == rid_zero);
	CHECK(rid == rid_one);
	CHECK_FALSE(rid == rid_two);

	CHECK_FALSE(rid < rid_zero);
	CHECK_FALSE(rid < rid_one);
	CHECK(rid < rid_two);

	CHECK_FALSE(rid <= rid_zero);
	CHECK(rid <= rid_one);
	CHECK(rid <= rid_two);

	CHECK(rid > rid_zero);
	CHECK_FALSE(rid > rid_one);
	CHECK_FALSE(rid > rid_two);

	CHECK(rid >= rid_zero);
	CHECK(rid >= rid_one);
	CHECK_FALSE(rid >= rid_two);

	CHECK(rid != rid_zero);
	CHECK_FALSE(rid != rid_one);
	CHECK(rid != rid_two);
}

TEST_CASE("[RID] 'is_valid' & 'is_null'") {
	RID rid_zero = RID::from_uint64(0);
	RID rid_one = RID::from_uint64(1);

	CHECK_FALSE(rid_zero.is_valid());
	CHECK(rid_zero.is_null());

	CHECK(rid_one.is_valid());
	CHECK_FALSE(rid_one.is_null());
}

TEST_CASE("[RID] 'get_local_index'") {
	CHECK(RID::from_uint64(1).get_local_index() == 1);
	CHECK(RID::from_uint64(4'294'967'295).get_local_index() == 4'294'967'295);
	CHECK(RID::from_uint64(4'294'967'297).get_local_index() == 1);
}

TEST_CASE("[RID_Owner] Thread safety") {
#ifdef __cpp_lib_hardware_interference_size
	static const size_t CACHE_LINE_BYTES = std::hardware_constructive_interference_size;
#else
	static const size_t CACHE_LINE_BYTES = 64;
#endif

	struct DataHolder {
		char data[CACHE_LINE_BYTES];
	};

	struct RID_OwnerTester {
		RID_Owner<DataHolder, true> rid_owner;
		TightLocalVector<Thread> threads;
		long cpu_cores = 0;
		SafeNumeric<uint32_t> next_thread_idx;
		// Using std::atomic directly since SafeNumeric doesn't support relaxed ordering.
		TightLocalVector<std::atomic<uint64_t>> rids; // Atomic here to prevent false data race warnings.
		std::atomic<uint32_t> sync[2] = {};
		std::atomic<uint32_t> correct = 0;

		// A latch that doesn't introduce memory ordering constraints, only compiler ones.
		// The idea is not to cause any sync traffic that would make the code we want to test
		// seem correct as a side effect.
		void lockstep(uint32_t p_step) {
			uint32_t buf_idx = p_step % 2;
			uint32_t target = (p_step / 2 + 1) * threads.size();
			std::atomic_signal_fence(std::memory_order_seq_cst);
			sync[buf_idx].fetch_add(1, std::memory_order_relaxed);
			do {
				std::atomic_signal_fence(std::memory_order_seq_cst);
			} while (sync[buf_idx].load(std::memory_order_relaxed) != target);
			std::atomic_signal_fence(std::memory_order_seq_cst);
		}

		explicit RID_OwnerTester(bool p_chunk_for_all, bool p_chunks_preallocated) :
				rid_owner(sizeof(DataHolder) * (p_chunk_for_all ? OS::get_singleton()->get_processor_count() : 1)) {
			cpu_cores = sysconf(_SC_NPROCESSORS_ONLN);
			printf("############################# %ld cores\n", cpu_cores);

			threads.resize(OS::get_singleton()->get_processor_count());
			printf("############################# %u threads\n", threads.size());
			rids.resize(threads.size());
			if (p_chunks_preallocated) {
				LocalVector<RID> prealloc_rids;
				for (uint32_t i = 0; i < (p_chunk_for_all ? 1 : threads.size()); i++) {
					prealloc_rids.push_back(rid_owner.make_rid());
				}
				for (uint32_t i = 0; i < prealloc_rids.size(); i++) {
					rid_owner.free(prealloc_rids[i]);
				}
			}
		}

		~RID_OwnerTester() {
			for (uint32_t i = 0; i < threads.size(); i++) {
				rid_owner.free(RID::from_uint64(rids[i].load(std::memory_order_relaxed)));
			}
		}

		void test() {
			for (uint32_t i = 0; i < threads.size(); i++) {
				threads[i].start(
						[](void *p_data) {
							RID_OwnerTester *rot = (RID_OwnerTester *)p_data;

							// 1. Each thread gets a zero-based index.
							uint32_t self_th_idx = rot->next_thread_idx.postincrement();

							// Use pthread to set affinity to the core self_th_idx modulo cpu cores.
							// This is to avoid threads running on the same core.
							thread_port_t mach_thread;
							int core = self_th_idx % rot->cpu_cores;
							thread_affinity_policy_data_t policy = { core };
							mach_thread = pthread_mach_thread_np(pthread_self());
							thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy, 1);
							printf("############################# thread %d - %d\n", self_th_idx, core);

							rot->lockstep(0);

							// 2. Each thread makes a RID holding unique data.
							char unique_byte = ((self_th_idx & 0xff) ^ 0xAA);
							DataHolder initial_data;
							memset(&initial_data, unique_byte, CACHE_LINE_BYTES);
							rot->rids[self_th_idx].store(rot->rid_owner.make_rid(initial_data).get_id(), std::memory_order_relaxed);

							rot->lockstep(1);

							// 3. Each thread verifies all the others.
							uint32_t local_correct = 0;
							for (uint32_t th_idx = 0; th_idx < rot->threads.size(); th_idx++) {
								if (th_idx == self_th_idx) {
									continue;
								}
								char expected_unique_byte = ((th_idx & 0xff) ^ 0xAA);
								RID rid = RID::from_uint64(rot->rids[th_idx].load(std::memory_order_relaxed));
								DataHolder *data = rot->rid_owner.get_or_null(rid);
								bool ok = true;
								for (uint32_t j = 0; j < CACHE_LINE_BYTES; j++) {
									if (data->data[j] != expected_unique_byte) {
										ok = false;
										break;
									}
								}
								if (ok) {
									local_correct++;
								}
							}

							rot->lockstep(2);

							rot->correct.fetch_add(local_correct, std::memory_order_acq_rel);
						},
						this);
			}

			for (uint32_t i = 0; i < threads.size(); i++) {
				threads[i].wait_to_finish();
			}

			CHECK_EQ(correct.load(), threads.size() * (threads.size() - 1));
		}
	};

	SUBCASE("All items in one chunk, pre-allocated") {
		for (int i = 0; i < 1000; i++) {
			RID_OwnerTester tester(true, true);
			tester.test();
		}
	}
	SUBCASE("All items in one chunk, NOT pre-allocated") {
		RID_OwnerTester tester(true, false);
		tester.test();
	}
	SUBCASE("One item per chunk, pre-allocated") {
		RID_OwnerTester tester(false, true);
		tester.test();
	}
	SUBCASE("One item per chunk, NOT pre-allocated") {
		RID_OwnerTester tester(false, false);
		tester.test();
	}
}
} // namespace TestRID

#endif // TEST_RID_H
