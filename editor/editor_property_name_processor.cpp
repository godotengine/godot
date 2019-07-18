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

EditorPropertyNameProcessor::Style EditorPropertyNameProcessor::get_default_inspector_style() {
	const Style style = (Style)EDITOR_GET("interface/inspector/default_property_name_style").operator int();
	if (style == STYLE_LOCALIZED && !is_localization_available()) {
		return STYLE_CAPITALIZED;
	}
	return style;
}

EditorPropertyNameProcessor::Style EditorPropertyNameProcessor::get_settings_style() {
	const bool translate = EDITOR_GET("interface/editor/localize_settings");
	return translate ? STYLE_LOCALIZED : STYLE_CAPITALIZED;
}

EditorPropertyNameProcessor::Style EditorPropertyNameProcessor::get_tooltip_style(Style p_style) {
	return p_style == STYLE_LOCALIZED ? STYLE_CAPITALIZED : STYLE_LOCALIZED;
}

bool EditorPropertyNameProcessor::is_localization_available() {
	const Vector<String> forbidden = String("en").split(",");
	return forbidden.find(EDITOR_GET("interface/editor/editor_language")) == -1;
}

String EditorPropertyNameProcessor::_capitalize_name(const String &p_name) const {
	const Map<String, String>::Element *cached = capitalize_string_cache.find(p_name);
	if (cached) {
		return cached->value();
	}

	Vector<String> parts = p_name.split("_", false);
	for (int i = 0; i < parts.size(); i++) {
		// Articles/conjunctions/prepositions which should only be capitalized when not at beginning and end.
		if (i > 0 && i + 1 < parts.size() && stop_words.find(parts[i]) != -1) {
			continue;
		}
		const Map<String, String>::Element *remap = capitalize_string_remaps.find(parts[i]);
		if (remap) {
			parts.write[i] = remap->get();
		} else {
			parts.write[i] = parts[i].capitalize();
		}
	}
	const String capitalized = String(" ").join(parts);

	capitalize_string_cache[p_name] = capitalized;
	return capitalized;
}

String EditorPropertyNameProcessor::process_name(const String &p_name, Style p_style) const {
	switch (p_style) {
		case STYLE_RAW: {
			return p_name;
		} break;

		case STYLE_CAPITALIZED: {
			return _capitalize_name(p_name);
		} break;

		case STYLE_LOCALIZED: {
			return TTRGET(_capitalize_name(p_name));
		} break;
	}
	ERR_FAIL_V_MSG(p_name, "Unexpected property name style.");
}

EditorPropertyNameProcessor::EditorPropertyNameProcessor() {
	ERR_FAIL_COND(singleton != nullptr);
	singleton = this;

	// The following initialization is parsed in `editor/translations/extract.py` with a regex.
	// The map name and value definition format should be kept synced with the regex.
	capitalize_string_remaps["2d"] = "2D";
	capitalize_string_remaps["3d"] = "3D";
	capitalize_string_remaps["aa"] = "AA";
	capitalize_string_remaps["aabb"] = "AABB";
	capitalize_string_remaps["adb"] = "ADB";
	capitalize_string_remaps["ao"] = "AO";
	capitalize_string_remaps["api"] = "API";
	capitalize_string_remaps["apk"] = "APK";
	capitalize_string_remaps["arm64-v8a"] = "arm64-v8a";
	capitalize_string_remaps["armeabi-v7a"] = "armeabi-v7a";
	capitalize_string_remaps["arvr"] = "ARVR";
	capitalize_string_remaps["bg"] = "BG";
	capitalize_string_remaps["bidi"] = "BiDi";
	capitalize_string_remaps["bp"] = "BP";
	capitalize_string_remaps["bpc"] = "BPC";
	capitalize_string_remaps["bptc"] = "BPTC";
	capitalize_string_remaps["bvh"] = "BVH";
	capitalize_string_remaps["ca"] = "CA";
	capitalize_string_remaps["cd"] = "CD";
	capitalize_string_remaps["commentfocus"] = "Comment Focus";
	capitalize_string_remaps["cpu"] = "CPU";
	capitalize_string_remaps["csg"] = "CSG";
	capitalize_string_remaps["db"] = "dB";
	capitalize_string_remaps["defaultfocus"] = "Default Focus";
	capitalize_string_remaps["defaultframe"] = "Default Frame";
	capitalize_string_remaps["dof"] = "DoF";
	capitalize_string_remaps["dpi"] = "DPI";
	capitalize_string_remaps["dtls"] = "DTLS";
	capitalize_string_remaps["erp"] = "ERP";
	capitalize_string_remaps["etc"] = "ETC";
	capitalize_string_remaps["etc2"] = "ETC2";
	capitalize_string_remaps["fbx"] = "FBX";
	capitalize_string_remaps["fft"] = "FFT";
	capitalize_string_remaps["fg"] = "FG";
	capitalize_string_remaps["filesystem"] = "FileSystem";
	capitalize_string_remaps["fov"] = "FOV";
	capitalize_string_remaps["fps"] = "FPS";
	capitalize_string_remaps["fs"] = "FS";
	capitalize_string_remaps["fsr"] = "FSR";
	capitalize_string_remaps["fxaa"] = "FXAA";
	capitalize_string_remaps["gdscript"] = "GDScript";
	capitalize_string_remaps["ggx"] = "GGX";
	capitalize_string_remaps["gi"] = "GI";
	capitalize_string_remaps["gl"] = "GL";
	capitalize_string_remaps["glb"] = "GLB";
	capitalize_string_remaps["gles2"] = "GLES2";
	capitalize_string_remaps["gles3"] = "GLES3";
	capitalize_string_remaps["gpu"] = "GPU";
	capitalize_string_remaps["gui"] = "GUI";
	capitalize_string_remaps["guid"] = "GUID";
	capitalize_string_remaps["hdr"] = "HDR";
	capitalize_string_remaps["hidpi"] = "hiDPI";
	capitalize_string_remaps["hipass"] = "High-pass";
	capitalize_string_remaps["hseparation"] = "H Separation";
	capitalize_string_remaps["hsv"] = "HSV";
	capitalize_string_remaps["html"] = "HTML";
	capitalize_string_remaps["http"] = "HTTP";
	capitalize_string_remaps["id"] = "ID";
	capitalize_string_remaps["ids"] = "IDs";
	capitalize_string_remaps["igd"] = "IGD";
	capitalize_string_remaps["ik"] = "IK";
	capitalize_string_remaps["image@2x"] = "Image @2x";
	capitalize_string_remaps["image@3x"] = "Image @3x";
	capitalize_string_remaps["iod"] = "IOD";
	capitalize_string_remaps["ios"] = "iOS";
	capitalize_string_remaps["ip"] = "IP";
	capitalize_string_remaps["ipad"] = "iPad";
	capitalize_string_remaps["iphone"] = "iPhone";
	capitalize_string_remaps["ipv6"] = "IPv6";
	capitalize_string_remaps["ir"] = "IR";
	capitalize_string_remaps["itunes"] = "iTunes";
	capitalize_string_remaps["jit"] = "JIT";
	capitalize_string_remaps["k1"] = "K1";
	capitalize_string_remaps["k2"] = "K2";
	capitalize_string_remaps["kb"] = "(KB)"; // Unit.
	capitalize_string_remaps["ldr"] = "LDR";
	capitalize_string_remaps["lod"] = "LOD";
	capitalize_string_remaps["lowpass"] = "Low-pass";
	capitalize_string_remaps["macos"] = "macOS";
	capitalize_string_remaps["mb"] = "(MB)"; // Unit.
	capitalize_string_remaps["mms"] = "MMS";
	capitalize_string_remaps["ms"] = "(ms)"; // Unit
	capitalize_string_remaps["msaa"] = "MSAA";
	capitalize_string_remaps["msdf"] = "MSDF";
	// Not used for now as AudioEffectReverb has a `msec` property.
	//capitalize_string_remaps["msec"] = "(msec)"; // Unit.
	capitalize_string_remaps["navmesh"] = "NavMesh";
	capitalize_string_remaps["nfc"] = "NFC";
	capitalize_string_remaps["normalmap"] = "Normal Map";
	capitalize_string_remaps["ofs"] = "Offset";
	capitalize_string_remaps["ok"] = "OK";
	capitalize_string_remaps["opengl"] = "OpenGL";
	capitalize_string_remaps["opentype"] = "OpenType";
	capitalize_string_remaps["openxr"] = "OpenXR";
	capitalize_string_remaps["osslsigncode"] = "osslsigncode";
	capitalize_string_remaps["pck"] = "PCK";
	capitalize_string_remaps["png"] = "PNG";
	capitalize_string_remaps["po2"] = "(Power of 2)"; // Unit.
	capitalize_string_remaps["pvrtc"] = "PVRTC";
	capitalize_string_remaps["pvs"] = "PVS";
	capitalize_string_remaps["rcedit"] = "rcedit";
	capitalize_string_remaps["rcodesign"] = "rcodesign";
	capitalize_string_remaps["rgb"] = "RGB";
	capitalize_string_remaps["rid"] = "RID";
	capitalize_string_remaps["rmb"] = "RMB";
	capitalize_string_remaps["rpc"] = "RPC";
	capitalize_string_remaps["s3tc"] = "S3TC";
	capitalize_string_remaps["sdf"] = "SDF";
	capitalize_string_remaps["sdfgi"] = "SDFGI";
	capitalize_string_remaps["sdk"] = "SDK";
	capitalize_string_remaps["sec"] = "(sec)"; // Unit.
	capitalize_string_remaps["selectedframe"] = "Selected Frame";
	capitalize_string_remaps["signtool"] = "signtool";
	capitalize_string_remaps["sms"] = "SMS";
	capitalize_string_remaps["srgb"] = "sRGB";
	capitalize_string_remaps["ssao"] = "SSAO";
	capitalize_string_remaps["ssh"] = "SSH";
	capitalize_string_remaps["ssil"] = "SSIL";
	capitalize_string_remaps["ssl"] = "SSL";
	capitalize_string_remaps["stderr"] = "stderr";
	capitalize_string_remaps["stdout"] = "stdout";
	capitalize_string_remaps["sv"] = "SV";
	capitalize_string_remaps["svg"] = "SVG";
	capitalize_string_remaps["tcp"] = "TCP";
	capitalize_string_remaps["tls"] = "TLS";
	capitalize_string_remaps["ui"] = "UI";
	capitalize_string_remaps["uri"] = "URI";
	capitalize_string_remaps["url"] = "URL";
	capitalize_string_remaps["urls"] = "URLs";
	capitalize_string_remaps["us"] = String::utf8("(µs)"); // Unit.
	capitalize_string_remaps["usb"] = "USB";
	capitalize_string_remaps["usec"] = String::utf8("(µsec)"); // Unit.
	capitalize_string_remaps["uuid"] = "UUID";
	capitalize_string_remaps["uv"] = "UV";
	capitalize_string_remaps["uv1"] = "UV1";
	capitalize_string_remaps["uv2"] = "UV2";
	capitalize_string_remaps["uwp"] = "UWP";
	capitalize_string_remaps["vadjust"] = "V Adjust";
	capitalize_string_remaps["valign"] = "V Align";
	capitalize_string_remaps["vector2"] = "Vector2";
	capitalize_string_remaps["vpn"] = "VPN";
	capitalize_string_remaps["vram"] = "VRAM";
	capitalize_string_remaps["vseparation"] = "V Separation";
	capitalize_string_remaps["vsync"] = "V-Sync";
	capitalize_string_remaps["wap"] = "WAP";
	capitalize_string_remaps["webp"] = "WebP";
	capitalize_string_remaps["webrtc"] = "WebRTC";
	capitalize_string_remaps["websocket"] = "WebSocket";
	capitalize_string_remaps["wine"] = "wine";
	capitalize_string_remaps["wifi"] = "Wi-Fi";
	capitalize_string_remaps["x86"] = "x86";
	capitalize_string_remaps["xr"] = "XR";
	capitalize_string_remaps["xray"] = "X-Ray";
	capitalize_string_remaps["xy"] = "XY";
	capitalize_string_remaps["xz"] = "XZ";
	capitalize_string_remaps["yz"] = "YZ";

	// Articles, conjunctions, prepositions.
	// The following initialization is parsed in `editor/translations/extract.py` with a regex.
	// The word definition format should be kept synced with the regex.
	stop_words.push_back("a");
	stop_words.push_back("an");
	stop_words.push_back("and");
	stop_words.push_back("as");
	stop_words.push_back("at");
	stop_words.push_back("by");
	stop_words.push_back("for");
	stop_words.push_back("in");
	stop_words.push_back("not");
	stop_words.push_back("of");
	stop_words.push_back("on");
	stop_words.push_back("or");
	stop_words.push_back("over");
	stop_words.push_back("per");
	stop_words.push_back("the");
	stop_words.push_back("then");
	stop_words.push_back("to");
}

EditorPropertyNameProcessor::~EditorPropertyNameProcessor() {
	singleton = nullptr;
}
