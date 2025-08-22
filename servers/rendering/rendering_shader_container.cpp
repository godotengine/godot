/**************************************************************************/
/*  rendering_shader_container.cpp                                        */
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

#include "rendering_shader_container.h"

#include "core/io/compression.h"

static inline uint32_t aligned_to(uint32_t p_size, uint32_t p_alignment) {
	if (p_size % p_alignment) {
		return p_size + (p_alignment - (p_size % p_alignment));
	} else {
		return p_size;
	}
}

uint32_t RenderingShaderContainer::_from_bytes_header_extra_data(const uint8_t *p_bytes) {
	return 0;
}

uint32_t RenderingShaderContainer::_from_bytes_reflection_extra_data(const uint8_t *p_bytes) {
	return 0;
}

uint32_t RenderingShaderContainer::_from_bytes_reflection_binding_uniform_extra_data_start(const uint8_t *p_bytes) {
	return 0;
}

uint32_t RenderingShaderContainer::_from_bytes_reflection_binding_uniform_extra_data(const uint8_t *p_bytes, uint32_t p_index) {
	return 0;
}

uint32_t RenderingShaderContainer::_from_bytes_reflection_specialization_extra_data_start(const uint8_t *p_bytes) {
	return 0;
}

uint32_t RenderingShaderContainer::_from_bytes_reflection_specialization_extra_data(const uint8_t *p_bytes, uint32_t p_index) {
	return 0;
}

uint32_t RenderingShaderContainer::_from_bytes_shader_extra_data_start(const uint8_t *p_bytes) {
	return 0;
}

uint32_t RenderingShaderContainer::_from_bytes_shader_extra_data(const uint8_t *p_bytes, uint32_t p_index) {
	return 0;
}

uint32_t RenderingShaderContainer::_from_bytes_footer_extra_data(const uint8_t *p_bytes) {
	return 0;
}

uint32_t RenderingShaderContainer::_to_bytes_header_extra_data(uint8_t *) const {
	return 0;
}

uint32_t RenderingShaderContainer::_to_bytes_reflection_extra_data(uint8_t *) const {
	return 0;
}

uint32_t RenderingShaderContainer::_to_bytes_reflection_binding_uniform_extra_data(uint8_t *, uint32_t) const {
	return 0;
}

uint32_t RenderingShaderContainer::_to_bytes_reflection_specialization_extra_data(uint8_t *, uint32_t) const {
	return 0;
}

uint32_t RenderingShaderContainer::_to_bytes_shader_extra_data(uint8_t *, uint32_t) const {
	return 0;
}

uint32_t RenderingShaderContainer::_to_bytes_footer_extra_data(uint8_t *) const {
	return 0;
}

void RenderingShaderContainer::_set_from_shader_reflection_post(const String &p_shader_name, const RenderingDeviceCommons::ShaderReflection &p_reflection) {
	// Do nothing.
}

void RenderingShaderContainer::set_from_shader_reflection(const String &p_shader_name, const RenderingDeviceCommons::ShaderReflection &p_reflection) {
	reflection_binding_set_uniforms_count.clear();
	reflection_binding_set_uniforms_data.clear();
	reflection_specialization_data.clear();
	reflection_shader_stages.clear();

	shader_name = p_shader_name.utf8();

	reflection_data.vertex_input_mask = p_reflection.vertex_input_mask;
	reflection_data.fragment_output_mask = p_reflection.fragment_output_mask;
	reflection_data.specialization_constants_count = p_reflection.specialization_constants.size();
	reflection_data.is_compute = p_reflection.is_compute;
	reflection_data.has_multiview = p_reflection.has_multiview;
	reflection_data.compute_local_size[0] = p_reflection.compute_local_size[0];
	reflection_data.compute_local_size[1] = p_reflection.compute_local_size[1];
	reflection_data.compute_local_size[2] = p_reflection.compute_local_size[2];
	reflection_data.set_count = p_reflection.uniform_sets.size();
	reflection_data.push_constant_size = p_reflection.push_constant_size;
	reflection_data.push_constant_stages_mask = uint32_t(p_reflection.push_constant_stages);
	reflection_data.shader_name_len = shader_name.length();

	ReflectionBindingData binding_data;
	for (const Vector<RenderingDeviceCommons::ShaderUniform> &uniform_set : p_reflection.uniform_sets) {
		for (const RenderingDeviceCommons::ShaderUniform &uniform : uniform_set) {
			binding_data.type = uint32_t(uniform.type);
			binding_data.binding = uniform.binding;
			binding_data.stages = uint32_t(uniform.stages);
			binding_data.length = uniform.length;
			binding_data.writable = uint32_t(uniform.writable);
			reflection_binding_set_uniforms_data.push_back(binding_data);
		}

		reflection_binding_set_uniforms_count.push_back(uniform_set.size());
	}

	ReflectionSpecializationData specialization_data;
	for (const RenderingDeviceCommons::ShaderSpecializationConstant &spec : p_reflection.specialization_constants) {
		specialization_data.type = uint32_t(spec.type);
		specialization_data.constant_id = spec.constant_id;
		specialization_data.int_value = spec.int_value;
		specialization_data.stage_flags = uint32_t(spec.stages);
		reflection_specialization_data.push_back(specialization_data);
	}

	for (uint32_t i = 0; i < RenderingDeviceCommons::SHADER_STAGE_MAX; i++) {
		if (p_reflection.stages_bits.has_flag(RenderingDeviceCommons::ShaderStage(1U << i))) {
			reflection_shader_stages.push_back(RenderingDeviceCommons::ShaderStage(i));
		}
	}

	reflection_data.stage_count = reflection_shader_stages.size();

	_set_from_shader_reflection_post(p_shader_name, p_reflection);
}

bool RenderingShaderContainer::set_code_from_spirv(const Vector<RenderingDeviceCommons::ShaderStageSPIRVData> &p_spirv) {
	return _set_code_from_spirv(p_spirv);
}

RenderingDeviceCommons::ShaderReflection RenderingShaderContainer::get_shader_reflection() const {
	RenderingDeviceCommons::ShaderReflection shader_refl;
	shader_refl.push_constant_size = reflection_data.push_constant_size;
	shader_refl.push_constant_stages = reflection_data.push_constant_stages_mask;
	shader_refl.vertex_input_mask = reflection_data.vertex_input_mask;
	shader_refl.fragment_output_mask = reflection_data.fragment_output_mask;
	shader_refl.is_compute = reflection_data.is_compute;
	shader_refl.has_multiview = reflection_data.has_multiview;
	shader_refl.compute_local_size[0] = reflection_data.compute_local_size[0];
	shader_refl.compute_local_size[1] = reflection_data.compute_local_size[1];
	shader_refl.compute_local_size[2] = reflection_data.compute_local_size[2];
	shader_refl.uniform_sets.resize(reflection_data.set_count);
	shader_refl.specialization_constants.resize(reflection_data.specialization_constants_count);
	shader_refl.stages_vector.resize(reflection_data.stage_count);

	DEV_ASSERT(reflection_binding_set_uniforms_count.size() == reflection_data.set_count && "The amount of elements in the reflection and the shader container can't be different.");
	uint32_t uniform_index = 0;
	for (uint32_t i = 0; i < reflection_data.set_count; i++) {
		Vector<RenderingDeviceCommons::ShaderUniform> &uniform_set = shader_refl.uniform_sets.ptrw()[i];
		uint32_t uniforms_count = reflection_binding_set_uniforms_count[i];
		uniform_set.resize(uniforms_count);
		for (uint32_t j = 0; j < uniforms_count; j++) {
			const ReflectionBindingData &binding = reflection_binding_set_uniforms_data[uniform_index++];
			RenderingDeviceCommons::ShaderUniform &uniform = uniform_set.ptrw()[j];
			uniform.type = RenderingDeviceCommons::UniformType(binding.type);
			uniform.writable = binding.writable;
			uniform.length = binding.length;
			uniform.binding = binding.binding;
			uniform.stages = binding.stages;
		}
	}

	shader_refl.specialization_constants.resize(reflection_data.specialization_constants_count);
	for (uint32_t i = 0; i < reflection_data.specialization_constants_count; i++) {
		const ReflectionSpecializationData &spec = reflection_specialization_data[i];
		RenderingDeviceCommons::ShaderSpecializationConstant &sc = shader_refl.specialization_constants.ptrw()[i];
		sc.type = RenderingDeviceCommons::PipelineSpecializationConstantType(spec.type);
		sc.constant_id = spec.constant_id;
		sc.int_value = spec.int_value;
		sc.stages = spec.stage_flags;
	}

	shader_refl.stages_vector.resize(reflection_data.stage_count);
	for (uint32_t i = 0; i < reflection_data.stage_count; i++) {
		shader_refl.stages_vector.set(i, reflection_shader_stages[i]);
		shader_refl.stages_bits.set_flag(RenderingDeviceCommons::ShaderStage(1U << reflection_shader_stages[i]));
	}

	return shader_refl;
}

bool RenderingShaderContainer::from_bytes(const PackedByteArray &p_bytes) {
	const uint64_t alignment = sizeof(uint32_t);
	const uint8_t *bytes_ptr = p_bytes.ptr();
	uint64_t bytes_offset = 0;

	// Read container header.
	ERR_FAIL_COND_V_MSG(int64_t(bytes_offset + sizeof(ContainerHeader)) > p_bytes.size(), false, "Not enough bytes for a container header in shader container.");
	const ContainerHeader &container_header = *(const ContainerHeader *)(&bytes_ptr[bytes_offset]);
	bytes_offset += sizeof(ContainerHeader);
	bytes_offset += _from_bytes_header_extra_data(&bytes_ptr[bytes_offset]);

	ERR_FAIL_COND_V_MSG(container_header.magic_number != CONTAINER_MAGIC_NUMBER, false, "Incorrect magic number in shader container.");
	ERR_FAIL_COND_V_MSG(container_header.version > CONTAINER_VERSION, false, "Unsupported version in shader container.");
	ERR_FAIL_COND_V_MSG(container_header.format != _format(), false, "Incorrect format in shader container.");
	ERR_FAIL_COND_V_MSG(container_header.format_version > _format_version(), false, "Unsupported format version in shader container.");

	// Adjust shaders to the size indicated by the container header.
	shaders.resize(container_header.shader_count);

	// Read reflection data.
	ERR_FAIL_COND_V_MSG(int64_t(bytes_offset + sizeof(ReflectionData)) > p_bytes.size(), false, "Not enough bytes for reflection data in shader container.");
	reflection_data = *(const ReflectionData *)(&bytes_ptr[bytes_offset]);
	bytes_offset += sizeof(ReflectionData);
	bytes_offset += _from_bytes_reflection_extra_data(&bytes_ptr[bytes_offset]);

	// Read shader name.
	ERR_FAIL_COND_V_MSG(int64_t(bytes_offset + reflection_data.shader_name_len) > p_bytes.size(), false, "Not enough bytes for shader name in shader container.");
	if (reflection_data.shader_name_len > 0) {
		String shader_name_str;
		shader_name_str.append_utf8((const char *)(&bytes_ptr[bytes_offset]), reflection_data.shader_name_len);
		shader_name = shader_name_str.utf8();
		bytes_offset = aligned_to(bytes_offset + reflection_data.shader_name_len, alignment);
	} else {
		shader_name = CharString();
	}

	reflection_binding_set_uniforms_count.resize(reflection_data.set_count);
	reflection_binding_set_uniforms_data.clear();

	uint32_t uniform_index = 0;
	for (uint32_t i = 0; i < reflection_data.set_count; i++) {
		ERR_FAIL_COND_V_MSG(int64_t(bytes_offset + sizeof(uint32_t)) > p_bytes.size(), false, "Not enough bytes for uniform set count in shader container.");
		uint32_t uniforms_count = *(uint32_t *)(&bytes_ptr[bytes_offset]);
		reflection_binding_set_uniforms_count.ptrw()[i] = uniforms_count;
		bytes_offset += sizeof(uint32_t);

		reflection_binding_set_uniforms_data.resize(reflection_binding_set_uniforms_data.size() + uniforms_count);
		bytes_offset += _from_bytes_reflection_binding_uniform_extra_data_start(&bytes_ptr[bytes_offset]);

		for (uint32_t j = 0; j < uniforms_count; j++) {
			ERR_FAIL_COND_V_MSG(int64_t(bytes_offset + sizeof(ReflectionBindingData)) > p_bytes.size(), false, "Not enough bytes for uniform in shader container.");
			memcpy(&reflection_binding_set_uniforms_data.ptrw()[uniform_index], &bytes_ptr[bytes_offset], sizeof(ReflectionBindingData));
			bytes_offset += sizeof(ReflectionBindingData);
			bytes_offset += _from_bytes_reflection_binding_uniform_extra_data(&bytes_ptr[bytes_offset], uniform_index);
			uniform_index++;
		}
	}

	reflection_specialization_data.resize(reflection_data.specialization_constants_count);
	bytes_offset += _from_bytes_reflection_specialization_extra_data_start(&bytes_ptr[bytes_offset]);

	for (uint32_t i = 0; i < reflection_data.specialization_constants_count; i++) {
		ERR_FAIL_COND_V_MSG(int64_t(bytes_offset + sizeof(ReflectionSpecializationData)) > p_bytes.size(), false, "Not enough bytes for specialization in shader container.");
		memcpy(&reflection_specialization_data.ptrw()[i], &bytes_ptr[bytes_offset], sizeof(ReflectionSpecializationData));
		bytes_offset += sizeof(ReflectionSpecializationData);
		bytes_offset += _from_bytes_reflection_specialization_extra_data(&bytes_ptr[bytes_offset], i);
	}

	const uint32_t stage_count = reflection_data.stage_count;
	if (stage_count > 0) {
		ERR_FAIL_COND_V_MSG(int64_t(bytes_offset + stage_count * sizeof(RenderingDeviceCommons::ShaderStage)) > p_bytes.size(), false, "Not enough bytes for stages in shader container.");
		reflection_shader_stages.resize(stage_count);
		bytes_offset += _from_bytes_shader_extra_data_start(&bytes_ptr[bytes_offset]);
		memcpy(reflection_shader_stages.ptrw(), &bytes_ptr[bytes_offset], stage_count * sizeof(RenderingDeviceCommons::ShaderStage));
		bytes_offset += stage_count * sizeof(RenderingDeviceCommons::ShaderStage);
	}

	// Read shaders.
	for (int64_t i = 0; i < shaders.size(); i++) {
		ERR_FAIL_COND_V_MSG(int64_t(bytes_offset + sizeof(ShaderHeader)) > p_bytes.size(), false, "Not enough bytes for shader header in shader container.");
		const ShaderHeader &header = *(const ShaderHeader *)(&bytes_ptr[bytes_offset]);
		bytes_offset += sizeof(ShaderHeader);

		ERR_FAIL_COND_V_MSG(int64_t(bytes_offset + header.code_compressed_size) > p_bytes.size(), false, "Not enough bytes for a shader in shader container.");
		Shader &shader = shaders.ptrw()[i];
		shader.shader_stage = RenderingDeviceCommons::ShaderStage(header.shader_stage);
		shader.code_compression_flags = header.code_compression_flags;
		shader.code_decompressed_size = header.code_decompressed_size;
		shader.code_compressed_bytes.resize(header.code_compressed_size);
		memcpy(shader.code_compressed_bytes.ptrw(), &bytes_ptr[bytes_offset], header.code_compressed_size);
		bytes_offset = aligned_to(bytes_offset + header.code_compressed_size, alignment);
		bytes_offset += _from_bytes_shader_extra_data(&bytes_ptr[bytes_offset], i);
	}

	bytes_offset += _from_bytes_footer_extra_data(&bytes_ptr[bytes_offset]);

	ERR_FAIL_COND_V_MSG(bytes_offset != (uint64_t)p_bytes.size(), false, "Amount of bytes in the container does not match the amount of bytes read.");
	return true;
}

PackedByteArray RenderingShaderContainer::to_bytes() const {
	// Compute the exact size the container will require for writing everything out.
	const uint64_t alignment = sizeof(uint32_t);
	uint64_t total_size = 0;
	total_size += sizeof(ContainerHeader) + _to_bytes_header_extra_data(nullptr);
	total_size += sizeof(ReflectionData) + _to_bytes_reflection_extra_data(nullptr);
	total_size += aligned_to(reflection_data.shader_name_len, alignment);
	total_size += reflection_binding_set_uniforms_count.size() * sizeof(uint32_t);
	total_size += reflection_binding_set_uniforms_data.size() * sizeof(ReflectionBindingData);
	total_size += reflection_specialization_data.size() * sizeof(ReflectionSpecializationData);
	total_size += reflection_shader_stages.size() * sizeof(RenderingDeviceCommons::ShaderStage);

	for (uint32_t i = 0; i < reflection_binding_set_uniforms_data.size(); i++) {
		total_size += _to_bytes_reflection_binding_uniform_extra_data(nullptr, i);
	}

	for (uint32_t i = 0; i < reflection_specialization_data.size(); i++) {
		total_size += _to_bytes_reflection_specialization_extra_data(nullptr, i);
	}

	for (uint32_t i = 0; i < shaders.size(); i++) {
		total_size += sizeof(ShaderHeader);
		total_size += shaders[i].code_compressed_bytes.size();
		total_size = aligned_to(total_size, alignment);
		total_size += _to_bytes_shader_extra_data(nullptr, i);
	}

	total_size += _to_bytes_footer_extra_data(nullptr);

	// Create the array that will hold all of the data.
	PackedByteArray bytes;
	bytes.resize_initialized(total_size);

	// Write out the data to the array.
	uint64_t bytes_offset = 0;
	uint8_t *bytes_ptr = bytes.ptrw();
	ContainerHeader &container_header = *(ContainerHeader *)(&bytes_ptr[bytes_offset]);
	container_header.magic_number = CONTAINER_MAGIC_NUMBER;
	container_header.version = CONTAINER_VERSION;
	container_header.format = _format();
	container_header.format_version = _format_version();
	container_header.shader_count = shaders.size();
	bytes_offset += sizeof(ContainerHeader);
	bytes_offset += _to_bytes_header_extra_data(&bytes_ptr[bytes_offset]);

	memcpy(&bytes_ptr[bytes_offset], &reflection_data, sizeof(ReflectionData));
	bytes_offset += sizeof(ReflectionData);
	bytes_offset += _to_bytes_reflection_extra_data(&bytes_ptr[bytes_offset]);

	if (shader_name.size() > 0) {
		memcpy(&bytes_ptr[bytes_offset], shader_name.ptr(), reflection_data.shader_name_len);
		bytes_offset = aligned_to(bytes_offset + reflection_data.shader_name_len, alignment);
	}

	uint32_t uniform_index = 0;
	for (uint32_t uniform_count : reflection_binding_set_uniforms_count) {
		memcpy(&bytes_ptr[bytes_offset], &uniform_count, sizeof(uniform_count));
		bytes_offset += sizeof(uint32_t);

		for (uint32_t i = 0; i < uniform_count; i++) {
			memcpy(&bytes_ptr[bytes_offset], &reflection_binding_set_uniforms_data[uniform_index], sizeof(ReflectionBindingData));
			bytes_offset += sizeof(ReflectionBindingData);
			bytes_offset += _to_bytes_reflection_binding_uniform_extra_data(&bytes_ptr[bytes_offset], uniform_index);
			uniform_index++;
		}
	}

	for (uint32_t i = 0; i < reflection_specialization_data.size(); i++) {
		memcpy(&bytes_ptr[bytes_offset], &reflection_specialization_data.ptr()[i], sizeof(ReflectionSpecializationData));
		bytes_offset += sizeof(ReflectionSpecializationData);
		bytes_offset += _to_bytes_reflection_specialization_extra_data(&bytes_ptr[bytes_offset], i);
	}

	if (!reflection_shader_stages.is_empty()) {
		uint32_t stage_count = reflection_shader_stages.size();
		memcpy(&bytes_ptr[bytes_offset], reflection_shader_stages.ptr(), stage_count * sizeof(RenderingDeviceCommons::ShaderStage));
		bytes_offset += stage_count * sizeof(RenderingDeviceCommons::ShaderStage);
	}

	for (uint32_t i = 0; i < shaders.size(); i++) {
		const Shader &shader = shaders[i];
		ShaderHeader &header = *(ShaderHeader *)(&bytes.ptr()[bytes_offset]);
		header.shader_stage = shader.shader_stage;
		header.code_compressed_size = uint32_t(shader.code_compressed_bytes.size());
		header.code_compression_flags = shader.code_compression_flags;
		header.code_decompressed_size = shader.code_decompressed_size;
		bytes_offset += sizeof(ShaderHeader);
		memcpy(&bytes.ptrw()[bytes_offset], shader.code_compressed_bytes.ptr(), shader.code_compressed_bytes.size());
		bytes_offset = aligned_to(bytes_offset + shader.code_compressed_bytes.size(), alignment);
		bytes_offset += _to_bytes_shader_extra_data(&bytes_ptr[bytes_offset], i);
	}

	bytes_offset += _to_bytes_footer_extra_data(&bytes_ptr[bytes_offset]);

	ERR_FAIL_COND_V_MSG(bytes_offset != total_size, PackedByteArray(), "Amount of bytes written does not match the amount of bytes reserved for the container.");
	return bytes;
}

bool RenderingShaderContainer::compress_code(const uint8_t *p_decompressed_bytes, uint32_t p_decompressed_size, uint8_t *p_compressed_bytes, uint32_t *r_compressed_size, uint32_t *r_compressed_flags) const {
	DEV_ASSERT(p_decompressed_bytes != nullptr);
	DEV_ASSERT(p_decompressed_size > 0);
	DEV_ASSERT(p_compressed_bytes != nullptr);
	DEV_ASSERT(r_compressed_size != nullptr);
	DEV_ASSERT(r_compressed_flags != nullptr);

	*r_compressed_flags = 0;

	PackedByteArray zstd_bytes;
	const int64_t zstd_max_bytes = Compression::get_max_compressed_buffer_size(p_decompressed_size, Compression::MODE_ZSTD);
	zstd_bytes.resize(zstd_max_bytes);

	const int64_t zstd_size = Compression::compress(zstd_bytes.ptrw(), p_decompressed_bytes, p_decompressed_size, Compression::MODE_ZSTD);
	if (zstd_size > 0 && (uint32_t)(zstd_size) < p_decompressed_size) {
		// Only choose Zstd if it results in actual compression.
		memcpy(p_compressed_bytes, zstd_bytes.ptr(), zstd_size);
		*r_compressed_size = zstd_size;
		*r_compressed_flags |= COMPRESSION_FLAG_ZSTD;
	} else {
		// Just copy the input to the output directly.
		memcpy(p_compressed_bytes, p_decompressed_bytes, p_decompressed_size);
		*r_compressed_size = p_decompressed_size;
	}

	return true;
}

bool RenderingShaderContainer::decompress_code(const uint8_t *p_compressed_bytes, uint32_t p_compressed_size, uint32_t p_compressed_flags, uint8_t *p_decompressed_bytes, uint32_t p_decompressed_size) const {
	DEV_ASSERT(p_compressed_bytes != nullptr);
	DEV_ASSERT(p_compressed_size > 0);
	DEV_ASSERT(p_decompressed_bytes != nullptr);
	DEV_ASSERT(p_decompressed_size > 0);

	bool uses_zstd = p_compressed_flags & COMPRESSION_FLAG_ZSTD;
	if (uses_zstd) {
		if (!Compression::decompress(p_decompressed_bytes, p_decompressed_size, p_compressed_bytes, p_compressed_size, Compression::MODE_ZSTD)) {
			ERR_FAIL_V_MSG(false, "Malformed zstd input for decompressing shader code.");
		}
	} else {
		memcpy(p_decompressed_bytes, p_compressed_bytes, MIN(p_compressed_size, p_decompressed_size));
	}

	return true;
}

RenderingShaderContainer::RenderingShaderContainer() {}

RenderingShaderContainer::~RenderingShaderContainer() {}
