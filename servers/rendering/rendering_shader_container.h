/**************************************************************************/
/*  rendering_shader_container.h                                          */
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

#include "core/object/ref_counted.h"
#include "servers/rendering/rendering_device_commons.h"

class RenderingShaderContainer : public RefCounted {
	GDSOFTCLASS(RenderingShaderContainer, RefCounted);

public:
	static const uint32_t CONTAINER_MAGIC_NUMBER = 0x43535247;
	static const uint32_t CONTAINER_VERSION = 2;

protected:
	struct ContainerHeader {
		uint32_t magic_number = 0;
		uint32_t version = 0;
		uint32_t format = 0;
		uint32_t format_version = 0;
		uint32_t shader_count = 0;
	};

	struct ReflectionData {
		uint64_t vertex_input_mask = 0;
		uint32_t fragment_output_mask = 0;
		uint32_t specialization_constants_count = 0;
		uint32_t is_compute = 0;
		uint32_t has_multiview = 0;
		uint32_t compute_local_size[3] = {};
		uint32_t set_count = 0;
		uint32_t push_constant_size = 0;
		uint32_t push_constant_stages_mask = 0;
		uint32_t stage_count = 0;
		uint32_t shader_name_len = 0;
	};

	struct ReflectionBindingData {
		uint32_t type = 0;
		uint32_t binding = 0;
		uint32_t stages = 0;
		uint32_t length = 0; // Size of arrays (in total elements), or UBOs (in bytes * total elements).
		uint32_t writable = 0;

		bool operator<(const ReflectionBindingData &p_other) const {
			return binding < p_other.binding;
		}
	};

	struct ReflectionSpecializationData {
		uint32_t type = 0;
		uint32_t constant_id = 0;
		uint32_t int_value = 0;
		uint32_t stage_flags = 0;
	};

	struct ShaderHeader {
		uint32_t shader_stage = 0;
		uint32_t code_compressed_size = 0;
		uint32_t code_compression_flags = 0;
		uint32_t code_decompressed_size = 0;
	};

	ReflectionData reflection_data;
	Vector<uint32_t> reflection_binding_set_uniforms_count;
	Vector<ReflectionBindingData> reflection_binding_set_uniforms_data;
	Vector<ReflectionSpecializationData> reflection_specialization_data;
	Vector<RenderingDeviceCommons::ShaderStage> reflection_shader_stages;

	virtual uint32_t _format() const = 0;
	virtual uint32_t _format_version() const = 0;

	// These methods will always be called with a valid pointer.
	virtual uint32_t _from_bytes_header_extra_data(const uint8_t *p_bytes);
	virtual uint32_t _from_bytes_reflection_extra_data(const uint8_t *p_bytes);
	virtual uint32_t _from_bytes_reflection_binding_uniform_extra_data_start(const uint8_t *p_bytes);
	virtual uint32_t _from_bytes_reflection_binding_uniform_extra_data(const uint8_t *p_bytes, uint32_t p_index);
	virtual uint32_t _from_bytes_reflection_specialization_extra_data_start(const uint8_t *p_bytes);
	virtual uint32_t _from_bytes_reflection_specialization_extra_data(const uint8_t *p_bytes, uint32_t p_index);
	virtual uint32_t _from_bytes_shader_extra_data_start(const uint8_t *p_bytes);
	virtual uint32_t _from_bytes_shader_extra_data(const uint8_t *p_bytes, uint32_t p_index);
	virtual uint32_t _from_bytes_footer_extra_data(const uint8_t *p_bytes);

	// These methods will be called with a nullptr to retrieve the size of the data.
	virtual uint32_t _to_bytes_header_extra_data(uint8_t *p_bytes) const;
	virtual uint32_t _to_bytes_reflection_extra_data(uint8_t *p_bytes) const;
	virtual uint32_t _to_bytes_reflection_binding_uniform_extra_data(uint8_t *p_bytes, uint32_t p_index) const;
	virtual uint32_t _to_bytes_reflection_specialization_extra_data(uint8_t *p_bytes, uint32_t p_index) const;
	virtual uint32_t _to_bytes_shader_extra_data(uint8_t *p_bytes, uint32_t p_index) const;
	virtual uint32_t _to_bytes_footer_extra_data(uint8_t *p_bytes) const;

	// This method will be called when set_from_shader_reflection() is finished. Used to update internal structures to match the reflection if necessary.
	virtual void _set_from_shader_reflection_post(const String &p_shader_name, const RenderingDeviceCommons::ShaderReflection &p_reflection);

	// This method will be called when set_code_from_spirv() is called.
	virtual bool _set_code_from_spirv(const Vector<RenderingDeviceCommons::ShaderStageSPIRVData> &p_spirv) = 0;

public:
	enum CompressionFlags {
		COMPRESSION_FLAG_ZSTD = 0x1,
	};

	struct Shader {
		RenderingDeviceCommons::ShaderStage shader_stage = RenderingDeviceCommons::SHADER_STAGE_MAX;
		PackedByteArray code_compressed_bytes;
		uint32_t code_compression_flags = 0;
		uint32_t code_decompressed_size = 0;
	};

	CharString shader_name;
	Vector<Shader> shaders;

	void set_from_shader_reflection(const String &p_shader_name, const RenderingDeviceCommons::ShaderReflection &p_reflection);
	bool set_code_from_spirv(const Vector<RenderingDeviceCommons::ShaderStageSPIRVData> &p_spirv);
	RenderingDeviceCommons::ShaderReflection get_shader_reflection() const;
	bool from_bytes(const PackedByteArray &p_bytes);
	PackedByteArray to_bytes() const;
	bool compress_code(const uint8_t *p_decompressed_bytes, uint32_t p_decompressed_size, uint8_t *p_compressed_bytes, uint32_t *r_compressed_size, uint32_t *r_compressed_flags) const;
	bool decompress_code(const uint8_t *p_compressed_bytes, uint32_t p_compressed_size, uint32_t p_compressed_flags, uint8_t *p_decompressed_bytes, uint32_t p_decompressed_size) const;
	RenderingShaderContainer();
	virtual ~RenderingShaderContainer();
};

class RenderingShaderContainerFormat : public RenderingDeviceCommons {
public:
	virtual Ref<RenderingShaderContainer> create_container() const = 0;
	virtual ShaderLanguageVersion get_shader_language_version() const = 0;
	virtual ShaderSpirvVersion get_shader_spirv_version() const = 0;
	virtual String get_customization_configuration_info() const { return ""; }
};
