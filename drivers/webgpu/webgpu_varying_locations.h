/**************************************************************************/
/*  webgpu_varying_locations.h                                            */
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

#pragma once

// WebGPU guarantees only 16 inter-stage locations (maxInterStageShaderVariables), and a vertex
// output or fragment input at location 16 or above makes the whole pipeline invalid. Godot numbers
// its own scene varyings 0-14 for every variant and puts shader-language varyings from 15 up
// (`base_varying_index`), so a user shader with more than one varying goes over the limit, even
// though one variant uses only a few of the built-in slots. This renumbers the locations one shader
// actually uses to 0..N-1, keeping their order, in every stage's SPIR-V before it becomes WGSL.
// Vertex inputs and fragment outputs keep their numbers: vertex buffers and color attachments
// refer to them.
//
// Plain C++ so the converter's tests can use it without the engine.

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace webgpu_varyings {

enum : uint32_t {
	SPV_OP_ENTRY_POINT = 15,
	SPV_OP_TYPE_INT = 21,
	SPV_OP_TYPE_FLOAT = 22,
	SPV_OP_TYPE_VECTOR = 23,
	SPV_OP_TYPE_MATRIX = 24,
	SPV_OP_TYPE_ARRAY = 28,
	SPV_OP_TYPE_POINTER = 32,
	SPV_OP_CONSTANT = 43,
	SPV_OP_VARIABLE = 59,
	SPV_OP_DECORATE = 71,
	SPV_OP_MEMBER_DECORATE = 72,
	SPV_DECORATION_LOCATION = 30,
	SPV_STORAGE_INPUT = 1,
	SPV_STORAGE_OUTPUT = 3,
	SPV_MODEL_VERTEX = 0,
	SPV_MODEL_FRAGMENT = 4,
};

// Highest location (exclusive) accepted. Vulkan devices offer at most 32 vec4 locations; anything
// past this is malformed or hostile SPIR-V and the shader is refused instead of renumbered.
static constexpr uint32_t MAX_LOCATION = 64;

struct StageVaryings {
	std::vector<uint32_t> location_words; // Indices of the Location literals to renumber.
	std::set<uint32_t> occupied; // Locations those variables cover.
	bool has_member_locations = false;
};

// Finds the stage's inter-stage variables: vertex outputs, fragment inputs. The SPIR-V may come
// from user content (guest programs pass raw SPIR-V), so every word read is bounds-checked and
// sizes are capped; on anything malformed it returns false and the shader is refused.
inline bool find_varyings(const std::vector<uint32_t> &p_words, StageVaryings &r_out) {
	if (p_words.size() < 5 || p_words[0] != 0x07230203) {
		return false;
	}
	uint32_t model = UINT32_MAX;
	std::map<uint32_t, uint32_t> width; // Type id -> locations it covers (capped at MAX_LOCATION).
	std::map<uint32_t, uint32_t> constant; // Constant id -> value (array lengths).
	std::map<uint32_t, uint32_t> ptr_type; // Pointer type id -> pointee type id.
	std::map<uint32_t, std::pair<uint32_t, uint32_t>> vars; // Var id -> (storage class, pointee).
	std::map<uint32_t, uint32_t> loc_word; // Var id -> index of its Location literal.
	for (size_t i = 5; i < p_words.size();) {
		const uint32_t count = p_words[i] >> 16;
		const uint32_t op = p_words[i] & 0xffff;
		if (count == 0 || count > p_words.size() - i) {
			return false;
		}
		const uint32_t *w = &p_words[i];
		switch (op) {
			case SPV_OP_ENTRY_POINT:
				if (count < 3) {
					return false;
				}
				if (model != UINT32_MAX && model != w[1]) {
					return false; // Godot stages have one entry point; mixed models are not ours.
				}
				model = w[1];
				break;
			case SPV_OP_TYPE_INT:
			case SPV_OP_TYPE_FLOAT:
			case SPV_OP_TYPE_VECTOR:
				if (count < 2) {
					return false;
				}
				width[w[1]] = 1;
				break;
			case SPV_OP_TYPE_MATRIX:
				if (count < 4) {
					return false;
				}
				width[w[1]] = w[3] < MAX_LOCATION ? w[3] : MAX_LOCATION;
				break;
			case SPV_OP_TYPE_ARRAY: {
				if (count < 4) {
					return false;
				}
				auto el = width.find(w[2]);
				auto len = constant.find(w[3]);
				if (el != width.end() && len != constant.end()) {
					const uint64_t n = uint64_t(el->second) * uint64_t(len->second);
					width[w[1]] = n < MAX_LOCATION ? uint32_t(n) : MAX_LOCATION;
				}
			} break;
			case SPV_OP_TYPE_POINTER:
				if (count < 4) {
					return false;
				}
				ptr_type[w[1]] = w[3];
				break;
			case SPV_OP_CONSTANT:
				if (count < 4) {
					return false;
				}
				constant[w[2]] = w[3];
				break;
			case SPV_OP_VARIABLE: {
				if (count < 4) {
					return false;
				}
				auto pt = ptr_type.find(w[1]);
				if (pt != ptr_type.end()) {
					vars[w[2]] = { w[3], pt->second };
				}
			} break;
			case SPV_OP_DECORATE:
				if (count < 3) {
					return false;
				}
				if (w[2] == SPV_DECORATION_LOCATION) {
					if (count < 4) {
						return false;
					}
					loc_word[w[1]] = uint32_t(i + 3);
				}
				break;
			case SPV_OP_MEMBER_DECORATE:
				if (count < 4 || (w[3] == SPV_DECORATION_LOCATION && count < 5)) {
					return false;
				}
				r_out.has_member_locations |= w[3] == SPV_DECORATION_LOCATION;
				break;
			default:
				break;
		}
		i += count;
	}
	uint32_t storage;
	if (model == SPV_MODEL_VERTEX) {
		storage = SPV_STORAGE_OUTPUT;
	} else if (model == SPV_MODEL_FRAGMENT) {
		storage = SPV_STORAGE_INPUT;
	} else {
		return true; // Compute: nothing to renumber.
	}
	for (const std::pair<const uint32_t, uint32_t> &e : loc_word) {
		auto var = vars.find(e.first);
		if (var == vars.end() || var->second.first != storage) {
			continue;
		}
		auto wid = width.find(var->second.second);
		const uint32_t n = wid == width.end() ? 1 : wid->second;
		const uint32_t base = p_words[e.second];
		if (base >= MAX_LOCATION || n > MAX_LOCATION - base) {
			return false;
		}
		for (uint32_t k = 0; k < n; k++) {
			r_out.occupied.insert(base + k);
		}
		r_out.location_words.push_back(e.second);
	}
	return true;
}

inline bool compact(std::vector<std::vector<uint32_t>> &r_stages) {
	std::vector<StageVaryings> found(r_stages.size());
	std::set<uint32_t> occupied;
	for (size_t s = 0; s < r_stages.size(); s++) {
		if (!find_varyings(r_stages[s], found[s])) {
			return false;
		}
		occupied.insert(found[s].occupied.begin(), found[s].occupied.end());
	}
	for (const StageVaryings &stage : found) {
		// shortcut: keep interface-block locations unchanged until member-aware compaction is needed.
		if (stage.has_member_locations) {
			return true;
		}
	}
	std::map<uint32_t, uint32_t> rank;
	uint32_t next = 0;
	for (uint32_t loc : occupied) {
		rank[loc] = next++;
	}
	for (size_t s = 0; s < r_stages.size(); s++) {
		for (uint32_t idx : found[s].location_words) {
			r_stages[s][idx] = rank[r_stages[s][idx]];
		}
	}
	return true;
}

} // namespace webgpu_varyings
