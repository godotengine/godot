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

struct ShaderCacheEntry;
class MetalDeviceProperties;

constexpr uint32_t R32UI_ALIGNMENT_CONSTANT_ID = 65535;

class RenderingShaderContainerFormatMetal;

class RenderingShaderContainerMetal : public RenderingShaderContainer {
	GDCLASS(RenderingShaderContainerMetal, RenderingShaderContainer);

	RenderingShaderContainerFormatMetal *owner = nullptr;
	bool export_mode = false;

	bool shader_compile_binary_from_spirv(const Vector<RenderingDeviceCommons::ShaderStageSPIRVData> &p_spirv);

public:
	static constexpr uint32_t FORMAT_VERSION = 1;

	struct HeaderData {
		enum Flags : uint32_t {
			NONE = 0,
			NEEDS_VIEW_MASK_BUFFER = 1 << 0,
			USES_ARGUMENT_BUFFERS = 1 << 1,
		};

		// The Metal language version specified when compiling SPIR-V to MSL.
		// Format is major * 10000 + minor * 100 + patch.
		uint32_t msl_version = UINT32_MAX;
		uint32_t flags = NONE;

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

	struct ShaderData {

	};

	RDD::ShaderID create_shader(const Vector<RDD::ImmutableSampler> &p_immutable_samplers);

	void set_owner(const RenderingShaderContainerFormatMetal *p_owner) { owner = (RenderingShaderContainerFormatMetal *)p_owner; }
	void set_export_mode(bool p_export_mode) { export_mode = p_export_mode; }

protected:
	virtual uint32_t _format() const override;
	virtual uint32_t _format_version() const override;
	virtual bool _set_code_from_spirv(const Vector<RenderingDeviceCommons::ShaderStageSPIRVData> &p_spirv) override;

#pragma mark - Serialisation

};

class RenderingShaderContainerFormatMetal : public RenderingShaderContainerFormat {
	friend class RenderingShaderContainerMetal;

	bool export_mode = false;

	MetalDeviceProperties *device_properties = nullptr;

public:
	virtual Ref<RenderingShaderContainer> create_container() const override;
	virtual ShaderLanguageVersion get_shader_language_version() const override;
	virtual ShaderSpirvVersion get_shader_spirv_version() const override;
	RenderingShaderContainerFormatMetal(bool p_export = false);
	virtual ~RenderingShaderContainerFormatMetal() = default;
};
