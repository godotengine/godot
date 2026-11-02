/**************************************************************************/
/*  shader_portability_test.cpp                                         */
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

#include "drivers/webgpu/shader_metadata_webgpu.h"
#include "drivers/webgpu/tests/check.h"
#include "drivers/webgpu/webgpu_varying_locations.h"

#include <algorithm>
#include <initializer_list>

static void emit(std::vector<uint32_t> &r_words, uint32_t p_op, std::initializer_list<uint32_t> p_args) {
	r_words.push_back(((uint32_t(p_args.size()) + 1) << 16) | p_op);
	r_words.insert(r_words.end(), p_args);
}

static std::vector<uint32_t> stage(uint32_t p_model, uint32_t p_location) {
	std::vector<uint32_t> words = { 0x07230203, 0x00010300, 0, 100, 0 };
	emit(words, 15, { p_model, 99, 0x6e69616d, 0 });
	emit(words, 22, { 1, 32 });
	emit(words, 23, { 2, 1, 4 });
	emit(words, 24, { 3, 2, 3 });
	emit(words, 21, { 4, 32, 0 });
	emit(words, 43, { 4, 5, 2 });
	emit(words, 28, { 6, 3, 5 }); // An array of two mat3x4 values occupies six locations.
	const uint32_t storage = p_model == 0 ? 3 : 1;
	emit(words, 32, { 7, storage, 6 });
	emit(words, 32, { 8, storage, 1 });
	emit(words, 59, { 7, 9, storage });
	emit(words, 59, { 8, 10, storage });
	emit(words, 71, { 9, 30, p_location });
	emit(words, 71, { 10, 30, 30 });
	// Vertex inputs and fragment outputs keep their externally defined locations.
	emit(words, 32, { 11, p_model == 0 ? 1u : 3u, 1 });
	emit(words, 59, { 11, 12, p_model == 0 ? 1u : 3u });
	emit(words, 71, { 12, 30, 7 });
	return words;
}

int main() {
	std::vector<std::vector<uint32_t>> stages = { stage(0, 15), stage(4, 15) };
	CHECK(webgpu_varyings::compact(stages));
	for (const auto &words : stages) {
		webgpu_varyings::StageVaryings found;
		CHECK(webgpu_varyings::find_varyings(words, found));
		CHECK(found.occupied.size() == 7);
		CHECK(*found.occupied.rbegin() == 6);
		CHECK(words[words.size() - 1] == 7);
	}
	std::reverse(stages.begin(), stages.end());
	CHECK(webgpu_varyings::compact(stages));
	stages = { stage(0, 15), stage(4, 15) };
	emit(stages[0], 72, { 90, 0, 30, 0 });
	const auto interface_blocks = stages;
	CHECK(webgpu_varyings::compact(stages));
	CHECK(stages == interface_blocks);
	stages = { stage(0, 63) };
	CHECK(!webgpu_varyings::compact(stages));
	stages = { stage(0, 15) };
	stages[0].pop_back();
	CHECK(!webgpu_varyings::compact(stages));
	stages = { { 0, 0, 0, 0, 0 } };
	CHECK(!webgpu_varyings::compact(stages));

	webgpu_metadata::Shader shader;
	shader.has_overrides = true;
	shader.overrides = { 3, 65535 };
	webgpu_metadata::Resource resource;
	resource.group = 2;
	resource.binding = 17;
	resource.kind = webgpu_metadata::READ_WRITE_STORAGE_TEXTURE;
	resource.dimension = webgpu_metadata::D2_ARRAY;
	resource.sample_type = webgpu_metadata::UINT;
	resource.format = "rgba8uint";
	shader.resources.push_back(resource);
	resource.binding = 18;
	resource.used = false;
	shader.resources.push_back(resource);
	const std::string encoded = webgpu_metadata::encode(shader);
	webgpu_metadata::Shader decoded;
	CHECK(!webgpu_metadata::parse(nullptr, decoded));
	CHECK(webgpu_metadata::parse((encoded + "@compute fn main() {}\n").c_str(), decoded));
	CHECK(webgpu_metadata::encode(decoded) == encoded);
	for (size_t i = 0; i < encoded.size(); i++) {
		CHECK(!webgpu_metadata::parse(encoded.substr(0, i).c_str(), decoded));
	}
	std::string bad = encoded;
	bad.replace(bad.find("//BINDING:2"), 11, "//BINDING:4294967296");
	CHECK(!webgpu_metadata::parse(bad.c_str(), decoded));
	CHECK(!webgpu_metadata::parse("//RESOURCE_USAGE\n//WEBGPU_METADATA:99,0\n", decoded));
	CHECK(webgpu_metadata::parse(webgpu_metadata::encode({}).c_str(), decoded));
	std::puts("PASS: cross-stage varying compaction and versioned shader metadata");
}
