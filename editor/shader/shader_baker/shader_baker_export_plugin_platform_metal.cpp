/**************************************************************************/
/*  shader_baker_export_plugin_platform_metal.cpp                         */
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

#include "shader_baker_export_plugin_platform_metal.h"

#include "drivers/metal/rendering_shader_container_metal.h"

RenderingShaderContainerFormat *ShaderBakerExportPluginPlatformMetal::create_shader_container_format(const Ref<EditorExportPlatform> &p_platform, const Ref<EditorExportPreset> &p_preset) {
	const String &os_name = p_platform->get_os_name();
	const MetalDeviceProfile *profile;
	MinOsVersion min_os_version;

	if (os_name == U"macOS") {
		min_os_version = (String)p_preset->get("application/min_macos_version_arm64");
		// Godot metal doesn't support x86_64 mac so no need to worry about that version
		profile = MetalDeviceProfile::get_profile(MetalDeviceProfile::Platform::macOS, MetalDeviceProfile::GPU::Apple7, min_os_version);
	} else if (os_name == U"iOS") {
		min_os_version = (String)p_preset->get("application/min_ios_version");
		profile = MetalDeviceProfile::get_profile(MetalDeviceProfile::Platform::iOS, MetalDeviceProfile::GPU::Apple7, min_os_version);
	} else if (os_name == U"visionOS") {
		min_os_version = (String)p_preset->get("application/min_visionos_version");
		profile = MetalDeviceProfile::get_profile(MetalDeviceProfile::Platform::visionOS, MetalDeviceProfile::GPU::Apple8, min_os_version);
	} else {
		ERR_FAIL_V_MSG(nullptr, vformat("Unsupported platform: %s", os_name));
	}
	ERR_FAIL_NULL_V(profile, nullptr);
	return memnew(RenderingShaderContainerFormatMetal(profile, true));
}

bool ShaderBakerExportPluginPlatformMetal::matches_driver(const String &p_driver) {
	return p_driver == "metal";
}
