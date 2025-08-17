/**************************************************************************/
/*  rendering_shader_container_metal.h                                    */
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

#import "sha256_digest.h"

#import "servers/rendering/rendering_device_driver.h"
#import "servers/rendering/rendering_shader_container.h"

constexpr uint32_t R32UI_ALIGNMENT_CONSTANT_ID = 65535;
/// Metal buffer index for the view mask when rendering multi-view.
const uint32_t VIEW_MASK_BUFFER_INDEX = 24;

class RenderingShaderContainerFormatMetal;

/// @brief A minimal structure that defines a device profile for Metal.
///
/// This structure is used by the `RenderingShaderContainerMetal` class to
/// determine options for compiling SPIR-V to Metal source. It currently only
/// contains the minimum properties required to transform shaders from SPIR-V to Metal
/// and potentially compile to a `.metallib`.
struct MetalDeviceProfile {
	enum class Platform : uint32_t {
		macOS = 0,
		iOS = 1,
	};

	/// @brief The GPU family.
	enum class GPU : uint32_t {
		Apple1,
		Apple2,
		Apple3,
		Apple4,
		Apple5,
		Apple6,
		Apple7,
		Apple8,
		Apple9,
	};

	enum class ArgumentBuffersTier : uint32_t {
		Tier1 = 0,
		Tier2 = 1,
	};

	struct Features {
		uint32_t mslVersionMajor = 0;
		uint32_t mslVersionMinor = 0;
		ArgumentBuffersTier argument_buffers_tier = ArgumentBuffersTier::Tier1;
		bool simdPermute = false;
	};

	Platform platform = Platform::macOS;
	GPU gpu = GPU::Apple4;
	Features features;

	static const MetalDeviceProfile *get_profile(Platform p_platform, GPU p_gpu);

	MetalDeviceProfile() = default;

private:
	static Mutex profiles_lock; ///< Mutex to protect access to the profiles map.
	static HashMap<uint32_t, MetalDeviceProfile> profiles;
};

class RenderingShaderContainerMetal : public RenderingShaderContainer {
	GDSOFTCLASS(RenderingShaderContainerMetal, RenderingShaderContainer);

public:
	struct HeaderData {
		enum Flags : uint32_t {
			NONE = 0,
			NEEDS_VIEW_MASK_BUFFER = 1 << 0,
			USES_ARGUMENT_BUFFERS = 1 << 1,
		};

		/// The base profile that was used to generate this shader.
		MetalDeviceProfile profile;

		/// The Metal language version specified when compiling SPIR-V to MSL.
		/// Format is major * 10000 + minor * 100 + patch.
		uint32_t msl_version = UINT32_MAX;
		uint32_t flags = NONE;

		/// @brief Returns `true` if the shader is compiled with multi-view support.
		bool needs_view_mask_buffer() const {
			return flags & NEEDS_VIEW_MASK_BUFFER;
		}

		void set_needs_view_mask_buffer(bool p_value) {
			if (p_value) {
				flags |= NEEDS_VIEW_MASK_BUFFER;
			} else {
				flags &= ~NEEDS_VIEW_MASK_BUFFER;
			}
		}

		/// @brief Returns `true` if the shader was compiled with argument buffer support.
		bool uses_argument_buffers() const {
			return flags & USES_ARGUMENT_BUFFERS;
		}

		void set_uses_argument_buffers(bool p_value) {
			if (p_value) {
				flags |= USES_ARGUMENT_BUFFERS;
			} else {
				flags &= ~USES_ARGUMENT_BUFFERS;
			}
		}
	};

	struct StageData {
		uint32_t vertex_input_binding_mask = 0;
		uint32_t is_position_invariant = 0; ///< <c>true</c> if the position output is invariant
		uint32_t supports_fast_math = 0;
		SHA256Digest hash; ///< SHA 256 hash of the shader code
		uint32_t source_size = 0; ///< size of the source code in the returned bytes
		uint32_t library_size = 0; ///< size of the compiled library in the returned bytes, 0 if it is not compiled
		uint32_t push_constant_binding = UINT32_MAX; ///< Metal binding slot for the push constant data
	};

	struct BindingInfoData {
		uint32_t shader_stage = UINT32_MAX; ///< The shader stage this binding is used in, or UINT32_MAX if not used.
		uint32_t data_type = 0; // MTLDataTypeNone
		uint32_t index = 0;
		uint32_t access = 0; // MTLBindingAccessReadOnly
		uint32_t usage = 0; // MTLResourceUsage (none)
		uint32_t texture_type = 2; // MTLTextureType2D
		uint32_t image_format = 0;
		uint32_t array_length = 0;
		uint32_t is_multisampled = 0;
	};

	struct UniformData {
		/// Specifies the index into the `bindings` array for the shader stage.
		///
		/// For example, a vertex and fragment shader use slots 0 and 1 of the bindings and bindings_secondary arrays.
		static constexpr uint32_t STAGE_INDEX[RenderingDeviceCommons::SHADER_STAGE_MAX] = {
			0, // SHADER_STAGE_VERTEX
			1, // SHADER_STAGE_FRAGMENT
			0, // SHADER_STAGE_TESSELATION_CONTROL
			1, // SHADER_STAGE_TESSELATION_EVALUATION
			0, // SHADER_STAGE_COMPUTE
		};

		/// Specifies the stages the uniform data is
		/// used by the Metal shader.
		uint32_t active_stages = 0;
		/// The primary binding information for the uniform data.
		///
		/// A maximum of two stages is expected for any given pipeline, such as a vertex and fragment, so
		/// the array size is fixed to 2.
		BindingInfoData bindings[2];
		/// The secondary binding information for the uniform data.
		///
		/// This is typically a sampler for an image-sampler uniform
		BindingInfoData bindings_secondary[2];

		_FORCE_INLINE_ constexpr uint32_t get_index_for_stage(RenderingDeviceCommons::ShaderStage p_stage) const {
			return STAGE_INDEX[p_stage];
		}

		_FORCE_INLINE_ BindingInfoData &get_binding_for_stage(RenderingDeviceCommons::ShaderStage p_stage) {
			BindingInfoData &info = bindings[get_index_for_stage(p_stage)];
			DEV_ASSERT(info.shader_stage == UINT32_MAX || info.shader_stage == p_stage); // make sure this uniform isn't used in the other stage
			info.shader_stage = p_stage;
			return info;
		}

		_FORCE_INLINE_ BindingInfoData &get_secondary_binding_for_stage(RenderingDeviceCommons::ShaderStage p_stage) {
			BindingInfoData &info = bindings_secondary[get_index_for_stage(p_stage)];
			DEV_ASSERT(info.shader_stage == UINT32_MAX || info.shader_stage == p_stage); // make sure this uniform isn't used in the other stage
			info.shader_stage = p_stage;
			return info;
		}
	};

	struct SpecializationData {
		uint32_t used_stages = 0;
	};

	HeaderData mtl_reflection_data; // compliment to reflection_data
	Vector<StageData> mtl_shaders; // compliment to shaders

private:
	const MetalDeviceProfile *device_profile = nullptr;
	bool export_mode = false;

	Vector<UniformData> mtl_reflection_binding_set_uniforms_data; // compliment to reflection_binding_set_uniforms_data
	Vector<SpecializationData> mtl_reflection_specialization_data; // compliment to reflection_specialization_data

	Error compile_metal_source(const char *p_source, const StageData &p_stage_data, Vector<uint8_t> &r_binary_data);

public:
	static constexpr uint32_t FORMAT_VERSION = 1;

	void set_export_mode(bool p_export_mode) { export_mode = p_export_mode; }
	void set_device_profile(const MetalDeviceProfile *p_device_profile) { device_profile = p_device_profile; }

	struct MetalShaderReflection {
		Vector<Vector<UniformData>> uniform_sets;
		Vector<SpecializationData> specialization_constants;
	};

	MetalShaderReflection get_metal_shader_reflection() const;

protected:
	virtual uint32_t _from_bytes_reflection_extra_data(const uint8_t *p_bytes) override;
	virtual uint32_t _from_bytes_reflection_binding_uniform_extra_data_start(const uint8_t *p_bytes) override;
	virtual uint32_t _from_bytes_reflection_binding_uniform_extra_data(const uint8_t *p_bytes, uint32_t p_index) override;
	virtual uint32_t _from_bytes_reflection_specialization_extra_data_start(const uint8_t *p_bytes) override;
	virtual uint32_t _from_bytes_reflection_specialization_extra_data(const uint8_t *p_bytes, uint32_t p_index) override;
	virtual uint32_t _from_bytes_shader_extra_data_start(const uint8_t *p_bytes) override;
	virtual uint32_t _from_bytes_shader_extra_data(const uint8_t *p_bytes, uint32_t p_index) override;

	virtual uint32_t _to_bytes_reflection_extra_data(uint8_t *p_bytes) const override;
	virtual uint32_t _to_bytes_reflection_binding_uniform_extra_data(uint8_t *p_bytes, uint32_t p_index) const override;
	virtual uint32_t _to_bytes_reflection_specialization_extra_data(uint8_t *p_bytes, uint32_t p_index) const override;
	virtual uint32_t _to_bytes_shader_extra_data(uint8_t *p_bytes, uint32_t p_index) const override;

	virtual uint32_t _format() const override;
	virtual uint32_t _format_version() const override;
	virtual bool _set_code_from_spirv(const Vector<RenderingDeviceCommons::ShaderStageSPIRVData> &p_spirv) override;
};

class RenderingShaderContainerFormatMetal : public RenderingShaderContainerFormat {
	bool export_mode = false;

	const MetalDeviceProfile *device_profile = nullptr;

public:
	virtual Ref<RenderingShaderContainer> create_container() const override;
	virtual ShaderLanguageVersion get_shader_language_version() const override;
	virtual ShaderSpirvVersion get_shader_spirv_version() const override;
	RenderingShaderContainerFormatMetal(const MetalDeviceProfile *p_device_profile, bool p_export = false);
	virtual ~RenderingShaderContainerFormatMetal() = default;
};
