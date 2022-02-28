/*************************************************************************/
/*  editor_property_name_processor.cpp                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "editor_property_name_processor.h"

#include "editor_settings.h"

EditorPropertyNameProcessor *EditorPropertyNameProcessor::singleton = nullptr;

String EditorPropertyNameProcessor::_capitalize_name(const String &p_name) const {
	String capitalized_string = p_name.capitalize();

	// Fix the casing of a few strings commonly found in editor property/setting names.
	for (Map<String, String>::Element *E = capitalize_string_remaps.front(); E; E = E->next()) {
		capitalized_string = capitalized_string.replace(E->key(), E->value());
	}

	return capitalized_string;
}

String EditorPropertyNameProcessor::process_name(const String &p_name) const {
	const String capitalized_string = _capitalize_name(p_name);
	if (EDITOR_GET("interface/editor/translate_properties")) {
		return TTRGET(capitalized_string);
	}
	return capitalized_string;
}

String EditorPropertyNameProcessor::make_tooltip_for_name(const String &p_name) const {
	const String capitalized_string = _capitalize_name(p_name);
	if (EDITOR_GET("interface/editor/translate_properties")) {
		return capitalized_string;
	}
	return TTRGET(capitalized_string);
}

EditorPropertyNameProcessor::EditorPropertyNameProcessor() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;

	// The following initialization is parsed in `editor/translations/extract.py` with a regex.
	// The map name and value definition format should be kept synced with the regex.
	capitalize_string_remaps["2d"] = "2D";
	capitalize_string_remaps["3d"] = "3D";
	capitalize_string_remaps["Adb"] = "ADB";
	capitalize_string_remaps["Bptc"] = "BPTC";
	capitalize_string_remaps["Bvh"] = "BVH";
	capitalize_string_remaps["Csg"] = "CSG";
	capitalize_string_remaps["Cpu"] = "CPU";
	capitalize_string_remaps["Db"] = "dB";
	capitalize_string_remaps["Dof"] = "DoF";
	capitalize_string_remaps["Dpi"] = "DPI";
	capitalize_string_remaps["Etc"] = "ETC";
	capitalize_string_remaps["Fbx"] = "FBX";
	capitalize_string_remaps["Fps"] = "FPS";
	capitalize_string_remaps["Fov"] = "FOV";
	capitalize_string_remaps["Fs"] = "FS";
	capitalize_string_remaps["Fxaa"] = "FXAA";
	capitalize_string_remaps["Ggx"] = "GGX";
	capitalize_string_remaps["Gdscript"] = "GDScript";
	capitalize_string_remaps["Gles 2"] = "GLES2";
	capitalize_string_remaps["Gles 3"] = "GLES3";
	capitalize_string_remaps["Gi Probe"] = "GI Probe";
	capitalize_string_remaps["Hdr"] = "HDR";
	capitalize_string_remaps["Hidpi"] = "hiDPI";
	capitalize_string_remaps["Ik"] = "IK";
	capitalize_string_remaps["Ios"] = "iOS";
	capitalize_string_remaps["Kb"] = "KB";
	capitalize_string_remaps["Msaa"] = "MSAA";
	capitalize_string_remaps["Macos"] = "macOS";
	capitalize_string_remaps["Opentype"] = "OpenType";
	capitalize_string_remaps["Png"] = "PNG";
	capitalize_string_remaps["Pvs"] = "PVS";
	capitalize_string_remaps["Pvrtc"] = "PVRTC";
	capitalize_string_remaps["S 3 Tc"] = "S3TC";
	capitalize_string_remaps["Sdfgi"] = "SDFGI";
	capitalize_string_remaps["Srgb"] = "sRGB";
	capitalize_string_remaps["Ssao"] = "SSAO";
	capitalize_string_remaps["Ssl"] = "SSL";
	capitalize_string_remaps["Ssh"] = "SSH";
	capitalize_string_remaps["Sdk"] = "SDK";
	capitalize_string_remaps["Tcp"] = "TCP";
	capitalize_string_remaps["Uv 1"] = "UV1";
	capitalize_string_remaps["Uv 2"] = "UV2";
	capitalize_string_remaps["Vram"] = "VRAM";
	capitalize_string_remaps["Vsync"] = "V-Sync";
	capitalize_string_remaps["Vector 2"] = "Vector2";
	capitalize_string_remaps["Webrtc"] = "WebRTC";
	capitalize_string_remaps["Websocket"] = "WebSocket";
}

EditorPropertyNameProcessor::~EditorPropertyNameProcessor() {
	singleton = nullptr;
}
