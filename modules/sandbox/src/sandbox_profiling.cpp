/**************************************************************************/
/*  sandbox_profiling.cpp                                                 */
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

#include "sandbox.h"

#include <algorithm>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
static constexpr bool USE_ADDR2LINE = false;

struct ProfilingMachine {
	std::unique_ptr<riscv::Machine<RISCV_ARCH>> machine = nullptr;
	std::vector<uint8_t> binary;
};
static std::unordered_map<std::string, ProfilingMachine> lookup_machines;

void Sandbox::set_profiling(bool enable) {
	enable_profiling(enable);
}

void Sandbox::enable_profiling(bool enable, uint32_t interval) {
	if (enable) {
		if (!m_local_profiling_data) {
			m_local_profiling_data = std::make_unique<LocalProfilingData>();
		}
		m_local_profiling_data->profiling_interval = interval;

		std::scoped_lock lock(profiling_mutex);
		if (!m_profiling_data) {
			m_profiling_data = std::make_unique<ProfilingData>();
		}
	} else {
		if (this->is_in_vmcall()) {
			ERR_PRINT("Cannot disable profiling while a VM call is in progress.");
			return;
		}
		m_local_profiling_data.reset();
	}
}

struct Result {
	std::string elf;
	gaddr_t pc = 0;
	int64_t offset = 0;
	int count = 0;
	int line = 0;
	String function;
	String file;
};

static ProfilingMachine *requisition(const std::string &elf) {
	auto it = lookup_machines.find(elf);
	if (it == lookup_machines.end()) {
		ProfilingMachine pm;
		PackedByteArray array = FileAccess::get_file_as_bytes("res://" + String::utf8(elf.c_str(), elf.size()));
		pm.binary = std::vector<uint8_t>(array.ptr(), array.ptr() + array.size());
		if (pm.binary.empty()) {
			ERR_PRINT("Failed to load ELF file for profiling: " + String::utf8(elf.c_str(), elf.size()));
			return nullptr;
		}
		pm.machine = std::make_unique<riscv::Machine<RISCV_ARCH>>(pm.binary, riscv::MachineOptions<RISCV_ARCH>{
																					 .load_program = false,
																					 .use_memory_arena = false,
																			 });
		lookup_machines[elf] = std::move(pm);
		return &lookup_machines[elf];
	}
	return &it->second;
}

static void resolve(Result &res, const Callable &callback,
		const Sandbox::ProfilingState &gprofstate) {
	// Try to resolve the address using addr2line
#ifdef __linux__
	if (USE_ADDR2LINE && !res.elf.empty()) {
		// execute riscv64-linux-gnu-addr2line -e <binary> -f -C 0x<address>
		// using popen() and fgets() to read the output
		char buffer[4096];
		snprintf(buffer, sizeof(buffer),
				"riscv64-linux-gnu-addr2line -e %s -f -C 0x%lX", res.elf.c_str(), long(res.pc));

		FILE *f = popen(buffer, "r");
		if (f) {
			String output;
			while (fgets(buffer, sizeof(buffer), f) != nullptr) {
				output += String::utf8(buffer, strlen(buffer));
			}
			pclose(f);

			// Parse the output. addr2line returns two lines:
			// 1. The function name
			// _physics_process
			// 2. The path to the source file and line number
			// /home/gonzo/github/thp/scenes/objects/trees/trees.cpp:29
			PackedStringArray lines = output.split("\n");
			if (lines.size() >= 2) {
				res.function = lines[0];
				const String &line2 = lines[1];

				// Parse the function name and file/line number
				int idx = line2.rfind(":");
				if (idx != -1) {
					res.file = line2.substr(0, idx);
					res.line = line2.substr(idx + 1).to_int();
					if (res.file == "??") {
						res.file = String::utf8(res.elf.c_str(), res.elf.size());
					}
				}
			} else {
				res.function = output;
			}
			if (res.file.is_empty()) {
				res.file = String::utf8(res.elf.c_str(), res.elf.size());
			}
			//printf("Hotspot %zu: %.*s\n", results.size(), int(output.utf8().size()), output.utf8().ptr());
			return;
		}
	}
#endif
	// Fallback to the callback
	res.file = "(unknown)";
	res.function = "??";
	if (!res.elf.empty()) {
		// Prefer gprofstate lookup
		gaddr_t best = ~0ULL;
		res.offset = 0x0;
		for (const auto &entry : gprofstate.lookup) {
			if (res.pc < entry.address || entry.address >= best) {
				continue;
			}
			best = entry.address;
			res.offset = res.pc - entry.address;
			res.function = entry.name;
		}

		// Try to resolve the address using the ELF file, if the offset from lookup was too large
		if (best == ~0ULL || res.offset > 0x4000) {
			ProfilingMachine *pm = requisition(res.elf);
			if (pm) {
				riscv::Memory<RISCV_ARCH>::Callsite callsite = pm->machine->memory.lookup(res.pc);
				res.function = String::utf8(callsite.name.c_str(), callsite.name.size());
				res.offset = callsite.offset;
			}
		}
		res.file = String::utf8(res.elf.c_str(), res.elf.size());
	}
	// If a callback is set, use it to resolve the address
	else if (callback.is_null()) {
		res.function = callback.call(res.file, res.pc);
	}
}

Array Sandbox::get_hotspots(unsigned total, const Callable &callable) {
	std::unordered_map<std::string_view, ProfilingState> gprofstate;
	{
		std::scoped_lock lock(profiling_mutex);
		if (!m_profiling_data) {
			ERR_PRINT("Profiling is not currently enabled.");
			return Array();
		}
		ProfilingData &profdata = *m_profiling_data;
		// Copy the profiling data
		gprofstate = profdata.state;
	}
	// Prevent re-entrancy into the profiling data
	std::scoped_lock lock(generate_hotspots_mutex);

	// Gather information about the hotspots
	std::vector<Result> results;
	unsigned total_measurements = 0;

	for (const auto &path : gprofstate) {
		std::string_view elf_path = path.first;
		const auto &state = path.second;

		for (const auto &entry : state.hotspots) {
			Result res;
			res.elf = elf_path;
			res.pc = entry.first;
			res.count = entry.second;
			total_measurements += res.count;

			resolve(res, callable, state);

			results.push_back(std::move(res));
		}
	}

	// Deduplicate the results, now that we have function names
	HashMap<String, unsigned> dedup;
	for (auto &res : results) {
		const String key = res.function + res.file;
		if (dedup.has(key)) {
			const size_t index = dedup[key];
			results[index].count += res.count;
			res.count = 0;
			continue;
		} else {
			dedup.insert(key, &res - results.data());
		}
	}
	total = std::min(total, static_cast<unsigned>(results.size()));

	// Partial sort to find the top N hotspots
	std::partial_sort(results.begin(), results.begin() + total, results.end(), [](const Result &a, const Result &b) {
		return a.count > b.count;
	});

	// Cut off the results to the top N
	results.resize(total);

	// Trim off any zero-count results
	while (!results.empty() && results.back().count == 0) {
		results.pop_back();
	}

	Array result;
	unsigned measured = 0;
	for (const Result &res : results) {
		Dictionary hotspot;
		hotspot["function"] = res.function;
		hotspot["address"] = String::num_int64(res.pc, 16);
		hotspot["offset"] = String::num_int64(res.offset, 16);
		hotspot["file"] = res.file;
		hotspot["line"] = res.line;
		hotspot["samples"] = res.count;
		result.push_back(hotspot);
		measured += res.count;
	}
	Dictionary stats;
	stats["results"] = unsigned(results.size());
	stats["samples_shown"] = measured;
	stats["samples_total"] = total_measurements;
	stats["unique_functions"] = dedup.size();
	result.push_back(stats);
	return result;
}

void Sandbox::clear_hotspots() {
	std::scoped_lock lock(profiling_mutex);
	if (!m_profiling_data) {
		ERR_PRINT("Profiling is not currently enabled.");
		return;
	}
	m_profiling_data->state.clear();
	lookup_machines.clear();
}
