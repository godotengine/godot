/**************************************************************************/
/*  shader_baker_export_plugin_platform_d3d12.cpp                         */
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

#include "shader_baker_export_plugin_platform_d3d12.h"

#include "drivers/d3d12/rendering_shader_container_d3d12.h"

#include <windows.h>

RenderingShaderContainerFormat *ShaderBakerExportPluginPlatformD3D12::create_shader_container_format(const Ref<EditorExportPlatform> &p_platform, const Ref<EditorExportPreset> &p_preset) {
	if (lib_d3d12 == nullptr) {
		lib_d3d12 = LoadLibraryW(L"D3D12.dll");
		ERR_FAIL_NULL_V_MSG(lib_d3d12, nullptr, "Unable to load D3D12.dll.");
	}

	// Shader Model 6.2 is required to export shaders that have FP16 variants.
	RenderingShaderContainerFormatD3D12 *shader_container_format_d3d12 = memnew(RenderingShaderContainerFormatD3D12);
	shader_container_format_d3d12->set_lib_d3d12(lib_d3d12);
	return shader_container_format_d3d12;
}

bool ShaderBakerExportPluginPlatformD3D12::matches_driver(const String &p_driver) {
	return p_driver == "d3d12";
}

ShaderBakerExportPluginPlatformD3D12 ::~ShaderBakerExportPluginPlatformD3D12() {
	if (lib_d3d12 != nullptr) {
		FreeLibrary((HMODULE)(lib_d3d12));
	}
}
