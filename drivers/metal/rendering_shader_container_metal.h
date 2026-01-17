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

#import "metal_device_profile.h"
#import "sha256_digest.h"

#import "servers/rendering/rendering_device_driver.h"
#import "servers/rendering/rendering_shader_container.h"

constexpr uint32_t R32UI_ALIGNMENT_CONSTANT_ID = 65535;
/// Metal buffer index for the view mask when rendering multi-view.
const uint32_t VIEW_MASK_BUFFER_INDEX = 24;

class RenderingShaderContainerFormatMetal;

class RenderingShaderContainerMetal : public RenderingShaderContainer {
	GDSOFTCLASS(RenderingShaderContainerMetal, RenderingShaderContainer);

public:
	struct HeaderData {
		enum Flags : uint32_t {
			NONE = 0,
			NEEDS_VIEW_MASK_BUFFER = 1 << 0,
			USES_ARGUMENT_BUFFERS = 1 << 1,
			NEEDS_DEBUG_LOGGING = 1 << 2,
		};

		/// The base profile that was used to generate this shader.
		MetalDeviceProfile profile;

		/// The Metal language version specified when compiling SPIR-V to MSL.
		/// Format is major * 10000 + minor * 100 + patch.
		uint32_t msl_version = UINT32_MAX;
		/*! @brief The minimum supported OS version for shaders baked to a `.metallib`.
		 *
		 * NOTE: This property is only valid when shaders are baked to a .metalllib
		 *
		 * Format is major * 10000 + minor * 100 + patch.
		 */
		MinOsVersion os_min_version;
		uint32_t flags = NONE;
		uint32_t push_constant_binding = UINT32_MAX; ///< Metal binding slot for the push constant data

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

		/// Returns `true` if the shader was compiled with the GL_EXT_debug_printf extension enabled.
		bool needs_debug_logging() const {
			return flags & NEEDS_DEBUG_LOGGING;
		}

		void set_needs_debug_logging(bool p_value) {
			if (p_value) {
				flags |= NEEDS_DEBUG_LOGGING;
			} else {
				flags &= ~NEEDS_DEBUG_LOGGING;
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
	};

	struct UniformData {
		uint32_t active_stages = 0;
		uint32_t uniform_type = 0; // UniformType
		uint32_t data_type = 0; // MTLDataTypeNone
		uint32_t access = 0; // MTLBindingAccessReadOnly
		uint32_t usage = 0; // MTLResourceUsage (none)
		uint32_t texture_type = 2; // MTLTextureType2D
		uint32_t image_format = 0;
		uint32_t array_length = 0;
		uint32_t is_multisampled = 0;

		struct Indexes {
			uint32_t buffer = UINT32_MAX;
			uint32_t texture = UINT32_MAX;
			uint32_t sampler = UINT32_MAX;
		};
		Indexes slot;
		Indexes arg_buffer;

		enum class IndexType {
			SLOT,
			ARG,
		};

		_FORCE_INLINE_ Indexes &get_indexes(IndexType p_type) {
			switch (p_type) {
				case IndexType::SLOT:
					return slot;
				case IndexType::ARG:
					return arg_buffer;
			}
		}
	};

	HeaderData mtl_reflection_data; // compliment to reflection_data
	Vector<StageData> mtl_shaders; // compliment to shaders

private:
	struct ToolchainProperties {
		MinOsVersion os_version_min_required;
		uint32_t metal_version = UINT32_MAX;

		_FORCE_INLINE_ bool is_null() const { return os_version_min_required.is_null() || metal_version == UINT32_MAX; }
		_FORCE_INLINE_ bool is_valid() const { return !is_null(); }
	};

	ToolchainProperties compiler_props;

	void _initialize_toolchain_properties();

private:
	const MetalDeviceProfile *device_profile = nullptr;
	bool export_mode = false;

	Vector<UniformData> mtl_reflection_binding_set_uniforms_data; // compliment to reflection_binding_set_uniforms_data

	Error compile_metal_source(const char *p_source, const StageData &p_stage_data, Vector<uint8_t> &r_binary_data);

public:
	static constexpr uint32_t FORMAT_VERSION = 2;

	void set_export_mode(bool p_export_mode) { export_mode = p_export_mode; }
	void set_device_profile(const MetalDeviceProfile *p_device_profile) { device_profile = p_device_profile; }

	struct MetalShaderReflection {
		Vector<Vector<UniformData>> uniform_sets;
	};

	MetalShaderReflection get_metal_shader_reflection() const;

protected:
	virtual uint32_t _from_bytes_reflection_extra_data(const uint8_t *p_bytes) override;
	virtual uint32_t _from_bytes_reflection_binding_uniform_extra_data_start(const uint8_t *p_bytes) override;
	virtual uint32_t _from_bytes_reflection_binding_uniform_extra_data(const uint8_t *p_bytes, uint32_t p_index) override;
	virtual uint32_t _from_bytes_shader_extra_data_start(const uint8_t *p_bytes) override;
	virtual uint32_t _from_bytes_shader_extra_data(const uint8_t *p_bytes, uint32_t p_index) override;

	virtual uint32_t _to_bytes_reflection_extra_data(uint8_t *p_bytes) const override;
	virtual uint32_t _to_bytes_reflection_binding_uniform_extra_data(uint8_t *p_bytes, uint32_t p_index) const override;
	virtual uint32_t _to_bytes_shader_extra_data(uint8_t *p_bytes, uint32_t p_index) const override;

	virtual uint32_t _format() const override;
	virtual uint32_t _format_version() const override;
	virtual bool _set_code_from_spirv(const ReflectShader &p_shader) override;
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
