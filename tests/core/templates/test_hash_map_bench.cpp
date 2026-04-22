/**************************************************************************/
/*  test_hash_map_bench.cpp                                               */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_hash_map_bench)

#include "core/os/os.h"
#include "core/string/string_name.h"
#include "core/string/ustring.h"
#include "core/templates/a_hash_map.h"
#include "core/templates/hash_map.h"
#include "core/variant/variant.h"

// Benchmarks for the SwissTable HashMap / AHashMap rewrites.
//
// These cases run only when the user explicitly opts in with
// `--test-case=*HashMapBench*`, since they take seconds and would slow down
// the regular CI sweep. The numbers printed here are absolute throughput; to
// compare against the legacy Robin-Hood implementation, check out the parent
// commit of this rewrite, build with the same flags, and re-run.

namespace TestHashMapBench {

static constexpr int kElems = 200'000;
static constexpr int kIters = 5;

// Print a one-line result row in microseconds-per-op, plus megaops/sec.
static void _report(const char *p_workload, uint64_t p_total_usec, int p_ops) {
	const double per_op_ns = static_cast<double>(p_total_usec) * 1000.0 / static_cast<double>(p_ops);
	const double mops = static_cast<double>(p_ops) / static_cast<double>(p_total_usec);
	print_line(vformat("  %s: %.1f ns/op  (%.2f Mops/s, %d ops in %d us)",
			p_workload, per_op_ns, mops, p_ops, (int)p_total_usec));
}

template <typename TMap, typename Make>
static void _bench_int_workload(const char *p_label, Make make_map) {
	print_line(vformat("[%s] int->int (%d entries):", p_label, kElems));

	uint64_t insert_us = 0, hit_us = 0, miss_us = 0, iter_us = 0, erase_us = 0;
	int64_t hit_sink = 0, miss_sink = 0, iter_sink = 0;

	for (int rep = 0; rep < kIters; rep++) {
		TMap m = make_map();
		const uint64_t t0 = OS::get_singleton()->get_ticks_usec();
		for (int i = 0; i < kElems; i++) {
			m.insert(i, i + 1);
		}
		const uint64_t t1 = OS::get_singleton()->get_ticks_usec();
		insert_us += t1 - t0;

		const uint64_t t2 = OS::get_singleton()->get_ticks_usec();
		for (int i = 0; i < kElems; i++) {
			const int *p = m.getptr(i);
			if (p) {
				hit_sink += *p;
			}
		}
		const uint64_t t3 = OS::get_singleton()->get_ticks_usec();
		hit_us += t3 - t2;

		const uint64_t t4 = OS::get_singleton()->get_ticks_usec();
		for (int i = kElems; i < 2 * kElems; i++) {
			const int *p = m.getptr(i);
			miss_sink += p ? 1 : 0;
		}
		const uint64_t t5 = OS::get_singleton()->get_ticks_usec();
		miss_us += t5 - t4;

		const uint64_t t6 = OS::get_singleton()->get_ticks_usec();
		for (const auto &kv : m) {
			iter_sink += kv.value;
		}
		const uint64_t t7 = OS::get_singleton()->get_ticks_usec();
		iter_us += t7 - t6;

		const uint64_t t8 = OS::get_singleton()->get_ticks_usec();
		for (int i = 0; i < kElems; i++) {
			m.erase(i);
		}
		const uint64_t t9 = OS::get_singleton()->get_ticks_usec();
		erase_us += t9 - t8;
	}

	_report("insert", insert_us, kElems * kIters);
	_report("lookup-hit ", hit_us, kElems * kIters);
	_report("lookup-miss", miss_us, kElems * kIters);
	_report("iterate    ", iter_us, kElems * kIters);
	_report("erase      ", erase_us, kElems * kIters);

	// Keep results live so the optimizer can't elide.
	CHECK((hit_sink + miss_sink + iter_sink) != INT64_MIN);
}

TEST_CASE("[HashMapBench] HashMap<int,int>") {
	_bench_int_workload<HashMap<int, int>>("HashMap", []() { return HashMap<int, int>(); });
}

TEST_CASE("[HashMapBench] AHashMap<int,int>") {
	_bench_int_workload<AHashMap<int, int>>("AHashMap", []() { return AHashMap<int, int>(); });
}

TEST_CASE("[HashMapBench] HashMap<String,int>") {
	print_line(vformat("[HashMap] String->int (%d entries):", kElems));

	Vector<String> keys;
	keys.resize(kElems);
	for (int i = 0; i < kElems; i++) {
		keys.write[i] = "key_" + itos(i);
	}

	uint64_t insert_us = 0, hit_us = 0, miss_us = 0, erase_us = 0;
	int64_t hit_sink = 0;

	for (int rep = 0; rep < kIters; rep++) {
		HashMap<String, int> m;
		const uint64_t t0 = OS::get_singleton()->get_ticks_usec();
		for (int i = 0; i < kElems; i++) {
			m.insert(keys[i], i);
		}
		insert_us += OS::get_singleton()->get_ticks_usec() - t0;

		const uint64_t t2 = OS::get_singleton()->get_ticks_usec();
		for (int i = 0; i < kElems; i++) {
			const int *p = m.getptr(keys[i]);
			if (p) {
				hit_sink += *p;
			}
		}
		hit_us += OS::get_singleton()->get_ticks_usec() - t2;

		const String missing = "missing_key_xxx";
		const uint64_t t4 = OS::get_singleton()->get_ticks_usec();
		for (int i = 0; i < kElems; i++) {
			hit_sink += m.has(missing) ? 1 : 0;
		}
		miss_us += OS::get_singleton()->get_ticks_usec() - t4;

		const uint64_t t6 = OS::get_singleton()->get_ticks_usec();
		for (int i = 0; i < kElems; i++) {
			m.erase(keys[i]);
		}
		erase_us += OS::get_singleton()->get_ticks_usec() - t6;
	}

	_report("insert     ", insert_us, kElems * kIters);
	_report("lookup-hit ", hit_us, kElems * kIters);
	_report("lookup-miss", miss_us, kElems * kIters);
	_report("erase      ", erase_us, kElems * kIters);
	CHECK(hit_sink != INT64_MIN);
}

TEST_CASE("[HashMapBench] HashMap<StringName,Variant>") {
	print_line(vformat("[HashMap] StringName->Variant (%d entries):", kElems));

	Vector<StringName> keys;
	keys.resize(kElems);
	for (int i = 0; i < kElems; i++) {
		keys.write[i] = StringName("k_" + itos(i));
	}

	uint64_t insert_us = 0, hit_us = 0, erase_us = 0;
	int64_t hit_sink = 0;

	for (int rep = 0; rep < kIters; rep++) {
		HashMap<StringName, Variant> m;
		const uint64_t t0 = OS::get_singleton()->get_ticks_usec();
		for (int i = 0; i < kElems; i++) {
			m.insert(keys[i], Variant(i));
		}
		insert_us += OS::get_singleton()->get_ticks_usec() - t0;

		const uint64_t t2 = OS::get_singleton()->get_ticks_usec();
		for (int i = 0; i < kElems; i++) {
			const Variant *p = m.getptr(keys[i]);
			if (p) {
				hit_sink += (int64_t)(*p);
			}
		}
		hit_us += OS::get_singleton()->get_ticks_usec() - t2;

		const uint64_t t6 = OS::get_singleton()->get_ticks_usec();
		for (int i = 0; i < kElems; i++) {
			m.erase(keys[i]);
		}
		erase_us += OS::get_singleton()->get_ticks_usec() - t6;
	}

	_report("insert    ", insert_us, kElems * kIters);
	_report("lookup-hit", hit_us, kElems * kIters);
	_report("erase     ", erase_us, kElems * kIters);
	CHECK(hit_sink != INT64_MIN);
}

TEST_CASE("[HashMapBench] AHashMap<int,int> churn") {
	print_line(vformat("[AHashMap] int->int churn (%d cycles of insert+erase, table size 1024):", kElems));

	uint64_t churn_us = 0;
	int64_t sink = 0;
	for (int rep = 0; rep < kIters; rep++) {
		AHashMap<int, int> m;
		for (int i = 0; i < 1024; i++) {
			m.insert(i, i);
		}
		const uint64_t t0 = OS::get_singleton()->get_ticks_usec();
		for (int i = 0; i < kElems; i++) {
			const int k = i & 1023;
			m.erase(k);
			m.insert(k, i);
			sink += m[k];
		}
		churn_us += OS::get_singleton()->get_ticks_usec() - t0;
	}
	_report("churn (erase+insert)", churn_us, kElems * kIters);
	CHECK(sink != INT64_MIN);
}

} // namespace TestHashMapBench
