// Copyright 2009 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <thread>

#include "util/util.h"
#include "util/flags.h"
#include "util/benchmark.h"
#include "re2/re2.h"

DEFINE_string(test_tmpdir, "/var/tmp", "temp directory");

#ifdef _WIN32
#define snprintf _snprintf
#endif

using testing::Benchmark;

static Benchmark* benchmarks[10000];
static int nbenchmarks;

void Benchmark::Register() {
	benchmarks[nbenchmarks] = this;
	if(lo < 1)
		lo = 1;
	if(hi < lo)
		hi = lo;
	nbenchmarks++;
}

static int64_t nsec() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
		std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int64_t bytes;
static int64_t ns;
static int64_t t0;
static int64_t items;

void SetBenchmarkBytesProcessed(int64_t x) {
	bytes = x;
}

void StopBenchmarkTiming() {
	if(t0 != 0)
		ns += nsec() - t0;
	t0 = 0;
}

void StartBenchmarkTiming() {
	if(t0 == 0)
		t0 = nsec();
}

void SetBenchmarkItemsProcessed(int n) {
	items = n;
}

void BenchmarkMemoryUsage() {
	// TODO(rsc): Implement.
}

int NumCPUs() {
	return static_cast<int>(std::thread::hardware_concurrency());
}

static void runN(Benchmark *b, int n, int siz) {
	bytes = 0;
	items = 0;
	ns = 0;
	t0 = nsec();
	if(b->fn)
		b->fn(n);
	else if(b->fnr)
		b->fnr(n, siz);
	else {
		fprintf(stderr, "%s: missing function\n", b->name);
		abort();
	}
	if(t0 != 0)
		ns += nsec() - t0;
}

static int round(int n) {
	int base = 1;
	
	while(base*10 < n)
		base *= 10;
	if(n < 2*base)
		return 2*base;
	if(n < 5*base)
		return 5*base;
	return 10*base;
}

void RunBench(Benchmark* b, int nthread, int siz) {
	int n, last;

	// TODO(rsc): Threaded benchmarks.
	if(nthread != 1)
		return;
	
	// run once in case it's expensive
	n = 1;
	runN(b, n, siz);
	while(ns < (int)1e9 && n < (int)1e9) {
		last = n;
		if(ns/n == 0)
			n = (int)1e9;
		else
			n = (int)1e9 / static_cast<int>(ns/n);
		
		n = std::max(last+1, std::min(n+n/2, 100*last));
		n = round(n);
		runN(b, n, siz);
	}
	
	char mb[100];
	char suf[100];
	mb[0] = '\0';
	suf[0] = '\0';
	if(ns > 0 && bytes > 0)
		snprintf(mb, sizeof mb, "\t%7.2f MB/s", ((double)bytes/1e6)/((double)ns/1e9));
	if(b->fnr || b->lo != b->hi) {
		if(siz >= (1<<20))
			snprintf(suf, sizeof suf, "/%dM", siz/(1<<20));
		else if(siz >= (1<<10))
			snprintf(suf, sizeof suf, "/%dK", siz/(1<<10));
		else
			snprintf(suf, sizeof suf, "/%d", siz);
	}
	printf("%s%s\t%8lld\t%10lld ns/op%s\n", b->name, suf, (long long)n, (long long)ns/n, mb);
	fflush(stdout);
}

static int match(const char* name, int argc, const char** argv) {
	if(argc == 1)
		return 1;
	for(int i = 1; i < argc; i++)
		if(RE2::PartialMatch(name, argv[i]))
			return 1;
	return 0;
}

int main(int argc, const char** argv) {
	for(int i = 0; i < nbenchmarks; i++) {
		Benchmark* b = benchmarks[i];
		if(match(b->name, argc, argv))
			for(int j = b->threadlo; j <= b->threadhi; j++)
				for(int k = std::max(b->lo, 1); k <= std::max(b->hi, 1); k<<=1)
					RunBench(b, j, k);
	}
}

