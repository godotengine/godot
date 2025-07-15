/**************************************************************************/
/*  rendering_shader_container_vulkan.h                                   */
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

#include "servers/rendering/rendering_shader_container.h"

#define RENDERING_SHADER_CONTAINER_VULKAN_COMPRESSION 1
#define RENDERING_SHADER_CONTAINER_VULKAN_SMOLV 1

class RenderingShaderContainerVulkan : public RenderingShaderContainer {
	GDSOFTCLASS(RenderingShaderContainerVulkan, RenderingShaderContainer);

public:
	static const uint32_t FORMAT_VERSION;

protected:
	virtual uint32_t _format() const override;
	virtual uint32_t _format_version() const override;
	virtual bool _set_code_from_spirv(const Vector<RenderingDeviceCommons::ShaderStageSPIRVData> &p_spirv) override;
};

class RenderingShaderContainerFormatVulkan : public RenderingShaderContainerFormat {
public:
	virtual Ref<RenderingShaderContainer> create_container() const override;
	virtual ShaderLanguageVersion get_shader_language_version() const override;
	virtual ShaderSpirvVersion get_shader_spirv_version() const override;
	RenderingShaderContainerFormatVulkan();
	virtual ~RenderingShaderContainerFormatVulkan();
};
