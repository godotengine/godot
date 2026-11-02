/**************************************************************************/
/*  standalone_driver.cpp                                                 */
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

// Standalone fuzz driver — provides main() when libFuzzer is not available.
//
// Reads seed corpus files, then mutates them randomly for a configurable
// number of iterations. Catches crashes (SIGSEGV, SIGABRT) and reports them.
//
// Usage:
//   ./fuzz_target corpus_file1.spv corpus_file2.spv ...
//   ./fuzz_target --iterations 500000 corpus/*.spv

#include <unistd.h>

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

static const char *g_current_seed = nullptr;
static int g_current_iter = -1;

static void crash_handler(int sig) {
	const char *signame = sig == SIGSEGV ? "SIGSEGV" : sig == SIGABRT ? "SIGABRT"
			: sig == SIGFPE											  ? "SIGFPE"
																	  : "UNKNOWN";
	fprintf(stderr, "\n[FUZZ] CRASH: %s at iteration %d (seed: %s)\n",
			signame, g_current_iter, g_current_seed ? g_current_seed : "mutation");
	_exit(1);
}

static std::vector<uint8_t> read_file(const char *path) {
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f.is_open()) {
		return {};
	}
	auto sz = f.tellg();
	if (sz <= 0) {
		return {};
	}
	std::vector<uint8_t> data((size_t)sz);
	f.seekg(0);
	f.read(reinterpret_cast<char *>(data.data()), sz);
	return data;
}

static void mutate(std::vector<uint8_t> &data) {
	if (data.empty()) {
		data.push_back(rand() & 0xFF);
		return;
	}
	int mutations = 1 + rand() % 4;
	for (int m = 0; m < mutations; m++) {
		int op = rand() % 5;
		switch (op) {
			case 0: // Flip a random bit.
				data[rand() % data.size()] ^= (1 << (rand() % 8));
				break;
			case 1: // Replace a random byte.
				data[rand() % data.size()] = rand() & 0xFF;
				break;
			case 2: // Insert a random byte (cap size at 64KB).
				if (data.size() < 65536) {
					size_t pos = rand() % (data.size() + 1);
					data.insert(data.begin() + pos, rand() & 0xFF);
				}
				break;
			case 3: // Delete a random byte.
				if (data.size() > 1) {
					data.erase(data.begin() + (rand() % data.size()));
				}
				break;
			case 4: // Overwrite a random 4-byte word (SPIR-V is word-oriented).
				if (data.size() >= 4) {
					size_t pos = (rand() % (data.size() / 4)) * 4;
					uint32_t val = (uint32_t)rand();
					memcpy(data.data() + pos, &val, 4);
				}
				break;
		}
	}
}

int main(int argc, char *argv[]) {
	signal(SIGSEGV, crash_handler);
	signal(SIGABRT, crash_handler);
	signal(SIGFPE, crash_handler);

	int iterations = 100000;
	std::vector<std::vector<uint8_t>> seeds;
	std::vector<const char *> seed_names;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
			iterations = atoi(argv[++i]);
			continue;
		}
		auto data = read_file(argv[i]);
		if (!data.empty()) {
			seeds.push_back(std::move(data));
			seed_names.push_back(argv[i]);
		}
	}

	if (seeds.empty()) {
		fprintf(stderr, "Usage: %s [--iterations N] <corpus_files...>\n", argv[0]);
		fprintf(stderr, "No valid seed files found.\n");
		return 1;
	}

	srand((unsigned)time(nullptr));

	fprintf(stderr, "[FUZZ] %zu seed files, %d iterations\n", seeds.size(), iterations);

	// Phase 1: Run all seeds unmodified.
	for (size_t i = 0; i < seeds.size(); i++) {
		g_current_seed = seed_names[i];
		g_current_iter = -(int)(seeds.size() - i);
		LLVMFuzzerTestOneInput(seeds[i].data(), seeds[i].size());
	}
	fprintf(stderr, "[FUZZ] Seed phase complete (%zu files)\n", seeds.size());

	// Phase 2: Random mutations.
	for (int iter = 0; iter < iterations; iter++) {
		g_current_seed = nullptr;
		g_current_iter = iter;

		auto data = seeds[rand() % seeds.size()]; // copy
		mutate(data);
		LLVMFuzzerTestOneInput(data.data(), data.size());

		if ((iter + 1) % 10000 == 0) {
			fprintf(stderr, "[FUZZ] %d / %d iterations\n", iter + 1, iterations);
		}
	}

	fprintf(stderr, "[FUZZ] Complete: %d iterations, %zu seeds. No crashes.\n",
			iterations, seeds.size());
	return 0;
}
