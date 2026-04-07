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

struct SpvReflectShaderModule;
struct SpvReflectDescriptorBinding;
struct SpvReflectSpecializationConstant;

class RenderingShaderContainer : public RefCounted {
	GDSOFTCLASS(RenderingShaderContainer, RefCounted);

public:
	static const uint32_t CONTAINER_MAGIC_NUMBER = 0x43535247;
	static const uint32_t CONTAINER_VERSION = 2;

protected:
	using RDC = RenderingDeviceCommons;

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
		RDC::PipelineType pipeline_type = RDC::PIPELINE_TYPE_RASTERIZATION;
		uint32_t has_multiview = 0;
		uint32_t has_dynamic_buffers = 0;
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
	Vector<RDC::ShaderStage> reflection_shader_stages;

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

	template <class T>
	struct ReflectSymbol {
		static constexpr uint32_t STAGE_INDEX[RDC::SHADER_STAGE_MAX] = {
			0, // SHADER_STAGE_VERTEX
			1, // SHADER_STAGE_FRAGMENT
			0, // SHADER_STAGE_TESSELATION_CONTROL
			1, // SHADER_STAGE_TESSELATION_EVALUATION
			0, // SHADER_STAGE_COMPUTE
		};

		BitField<RDC::ShaderStage> stages = {};

	private:
		const T *_spv_reflect[2] = { nullptr };

	public:
		_FORCE_INLINE_ constexpr uint32_t get_index_for_stage(RDC::ShaderStage p_stage) const {
			DEV_ASSERT(stages.has_flag((1 << p_stage)));
			return STAGE_INDEX[p_stage];
		}

		const T &get_spv_reflect(RDC::ShaderStage p_stage) const;

		/*! Returns the first valid stage if multiple stages are set.
		 *
		 * Crashes if no stages are set.
		 */
		const T &get_spv_reflect() const {
			for (const T *d : _spv_reflect) {
				if (d != nullptr) {
					return *d;
				}
			}
			CRASH_NOW_MSG("No stages set in ReflectSymbol");
		}
		void set_spv_reflect(RDC::ShaderStage p_stage, const T *p_spv);
	};

	struct ReflectImageTraits {
		RDC::DataFormat format = RDC::DATA_FORMAT_MAX;
	};

	struct ReflectUniform : ReflectSymbol<SpvReflectDescriptorBinding> {
		RDC::UniformType type = RDC::UniformType::UNIFORM_TYPE_MAX;
		uint32_t binding = 0;

		ReflectImageTraits image;

		uint32_t length = 0; // Size of arrays (in total elements), or ubos (in bytes * total elements).
		bool writable = false;

		bool operator<(const ReflectUniform &p_other) const {
			if (binding != p_other.binding) {
				return binding < p_other.binding;
			}
			if (type != p_other.type) {
				return type < p_other.type;
			}
			if (writable != p_other.writable) {
				return writable < p_other.writable;
			}
			if (stages != p_other.stages) {
				return stages < p_other.stages;
			}
			if (length != p_other.length) {
				return length < p_other.length;
			}
			return false;
		}
	};

	struct ReflectSpecializationConstant : ReflectSymbol<SpvReflectSpecializationConstant> {
		RDC::PipelineSpecializationConstantType type = {};
		uint32_t constant_id = 0xffffffff;
		union {
			uint32_t int_value = 0;
			float float_value;
			bool bool_value;
		};

		bool operator<(const ReflectSpecializationConstant &p_other) const { return constant_id < p_other.constant_id; }
	};

	class ReflectShaderStage {
		friend class RenderingShaderContainer;

		Vector<uint8_t> _spirv_data;
		SpvReflectShaderModule *_module = nullptr;

	public:
		RDC::ShaderStage shader_stage = RDC::SHADER_STAGE_MAX;
		const SpvReflectShaderModule &module() const;
		const Span<uint32_t> spirv() const;
		const Vector<uint8_t> spirv_data() const { return _spirv_data; }

		ReflectShaderStage();
		~ReflectShaderStage();
	};

	typedef LocalVector<ReflectUniform> ReflectDescriptorSet;

	struct ReflectShader {
		uint64_t vertex_input_mask = 0;
		uint32_t fragment_output_mask = 0;
		uint32_t compute_local_size[3] = {};
		uint32_t push_constant_size = 0;
		bool has_multiview = false;
		bool has_dynamic_buffers = false;
		RDC::PipelineType pipeline_type = RDC::PIPELINE_TYPE_RASTERIZATION;

		LocalVector<ReflectShaderStage> shader_stages;
		LocalVector<ReflectDescriptorSet> uniform_sets;
		LocalVector<ReflectSymbol<SpvReflectDescriptorBinding>> reflect_uniforms;
		LocalVector<ReflectSpecializationConstant> specialization_constants;
		LocalVector<ReflectSymbol<SpvReflectSpecializationConstant>> reflect_specialization_constants;
		LocalVector<RDC::ShaderStage> stages_vector;
		BitField<RDC::ShaderStage> stages_bits = {};
		BitField<RDC::ShaderStage> push_constant_stages = {};

		_FORCE_INLINE_ bool is_compute() const {
			return stages_bits.has_flag(RDC::SHADER_STAGE_COMPUTE_BIT);
		}

		/*! Returns the uniform at the specified global index.
		 *
		 * This is a flattened view of all uniform sets.
		 */
		ReflectUniform &uniform_at(uint32_t p_index) {
			for (LocalVector<ReflectUniform> &set : uniform_sets) {
				if (p_index < set.size()) {
					return set[p_index];
				}
				p_index -= set.size();
			}
			CRASH_NOW_MSG(vformat("Uniform index %d out of range (total %d)", p_index, uniform_count()));
		}

		uint32_t uniform_count() const {
			uint32_t count = 0;
			for (const LocalVector<ReflectUniform> &set : uniform_sets) {
				count += set.size();
			}
			return count;
		}
	};

	// This method will be called when set_from_shader_reflection() is finished. Used to update internal structures to match the reflection if necessary.
	virtual void _set_from_shader_reflection_post(const ReflectShader &p_shader);

	// This method will be called when set_code_from_spirv() is called.
	virtual bool _set_code_from_spirv(const ReflectShader &p_shader) = 0;

	void set_from_shader_reflection(const ReflectShader &p_reflection);
	Error reflect_spirv(const String &p_shader_name, Span<RDC::ShaderStageSPIRVData> p_spirv, ReflectShader &r_shader);

public:
	enum CompressionFlags {
		COMPRESSION_FLAG_ZSTD = 0x1,
	};

	struct Shader {
		RDC::ShaderStage shader_stage = RDC::SHADER_STAGE_MAX;
		PackedByteArray code_compressed_bytes;
		uint32_t code_compression_flags = 0;
		uint32_t code_decompressed_size = 0;
	};

	CharString shader_name;
	Vector<Shader> shaders;

	bool set_code_from_spirv(const String &p_shader_name, Span<RDC::ShaderStageSPIRVData> p_spirv);
	RDC::ShaderReflection get_shader_reflection() const;
	bool from_bytes(const PackedByteArray &p_bytes);
	PackedByteArray to_bytes() const;
	bool compress_code(const uint8_t *p_decompressed_bytes, uint32_t p_decompressed_size, uint8_t *p_compressed_bytes, uint32_t *r_compressed_size, uint32_t *r_compressed_flags) const;
	bool decompress_code(const uint8_t *p_compressed_bytes, uint32_t p_compressed_size, uint32_t p_compressed_flags, uint8_t *p_decompressed_bytes, uint32_t p_decompressed_size) const;
	RenderingShaderContainer();
	virtual ~RenderingShaderContainer();
};

class RenderingShaderContainerFormat : public RenderingDeviceCommons {
	GDSOFTCLASS(RenderingShaderContainerFormat, RenderingDeviceCommons);

public:
	virtual Ref<RenderingShaderContainer> create_container() const = 0;
	virtual ShaderLanguageVersion get_shader_language_version() const = 0;
	virtual ShaderSpirvVersion get_shader_spirv_version() const = 0;
};
