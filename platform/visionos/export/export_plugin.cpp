/**************************************************************************/
/*  export_plugin.cpp                                                     */
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

#include "export_plugin.h"

#include "logo_svg.gen.h"
#include "run_icon_svg.gen.h"

#include "editor/editor_node.h"

Vector<String> EditorExportPlatformVisionOS::device_types({ "realityDevice" });

void EditorExportPlatformVisionOS::initialize() {
	if (EditorNode::get_singleton()) {
		EditorExportPlatformAppleEmbedded::_initialize(_visionos_logo_svg, _visionos_run_icon_svg);
#ifdef MACOS_ENABLED
		_start_remote_device_poller_thread();
#endif
	}
}

EditorExportPlatformVisionOS::~EditorExportPlatformVisionOS() {
#ifdef MACOS_ENABLED
	_stop_remote_device_poller_thread();
#endif
}

void EditorExportPlatformVisionOS::get_export_options(List<ExportOption> *r_options) const {
	EditorExportPlatformAppleEmbedded::get_export_options(r_options);

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/min_visionos_version"), get_minimum_deployment_target()));

	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "application/app_role", PROPERTY_HINT_ENUM, "Window,Immersive"), 0));
	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "application/immersion_style", PROPERTY_HINT_ENUM, "Full,Mixed"), 1));

	// Front layer falls back to the project icon; middle/back use a black placeholder when unset.
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "icons/icon_front_layer_1024x1024", PROPERTY_HINT_FILE_PATH, "*.svg,*.png,*.webp,*.jpg,*.jpeg"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "icons/icon_middle_layer_1024x1024", PROPERTY_HINT_FILE_PATH, "*.svg,*.png,*.webp,*.jpg,*.jpeg"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "icons/icon_back_layer_1024x1024", PROPERTY_HINT_FILE_PATH, "*.svg,*.png,*.webp,*.jpg,*.jpeg"), ""));
}

Vector<EditorExportPlatformAppleEmbedded::IconInfo> EditorExportPlatformVisionOS::get_icon_infos() const {
	// Layered icons don't fit IconInfo; _export_icons emits them directly.
	return Vector<EditorExportPlatformAppleEmbedded::IconInfo>();
}

Error EditorExportPlatformVisionOS::_export_icons(const Ref<EditorExportPreset> &p_preset, const String &p_iconset_dir) {
	// AppIcon.solidimagestack/<Layer>.solidimagestacklayer/Content.imageset/<layer>.png

	struct LayerInfo {
		const char *name;
		const char *preset_key;
		bool fallback_to_project_icon;
	};
	const LayerInfo layers[] = {
		{ "Front", "icons/icon_front_layer_1024x1024", true },
		{ "Middle", "icons/icon_middle_layer_1024x1024", false },
		{ "Back", "icons/icon_back_layer_1024x1024", false },
	};
	constexpr int LAYER_SIDE = 1024;

	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (da.is_null()) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), TTR("Could not access the filesystem."));
		return ERR_CANT_CREATE;
	}

	String stack_json = "{\"info\":{\"author\":\"xcode\",\"version\":1},\"layers\":[";
	for (int i = 0; i < 3; i++) {
		if (i > 0) {
			stack_json += ",";
		}
		stack_json += String("{\"filename\":\"") + layers[i].name + ".solidimagestacklayer\"}";
	}
	stack_json += "]}";

	{
		Ref<FileAccess> f = FileAccess::open(p_iconset_dir + "Contents.json", FileAccess::WRITE);
		if (f.is_null()) {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat(TTR("Could not write to a file at path \"%s\"."), p_iconset_dir + "Contents.json"));
			return ERR_CANT_CREATE;
		}
		CharString utf8 = stack_json.utf8();
		f->store_buffer((const uint8_t *)utf8.get_data(), utf8.length());
	}

	const String layer_metadata_json = "{\"info\":{\"author\":\"xcode\",\"version\":1}}";
	const Image::Interpolation interpolation = (Image::Interpolation)(p_preset->get("application/icon_interpolation").operator int());

	for (int i = 0; i < 3; i++) {
		const String layer_dir = p_iconset_dir + layers[i].name + ".solidimagestacklayer/";
		const String imageset_dir = layer_dir + "Content.imageset/";
		const String png_name = String(layers[i].name).to_lower() + ".png";

		Error err = da->make_dir_recursive(imageset_dir);
		if (err != OK) {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat(TTR("Could not create a directory at path \"%s\"."), imageset_dir));
			return err;
		}

		// Source resolution: layer-specific path; front also falls back to the project icon.
		String icon_path = p_preset->get(layers[i].preset_key);
		bool warn_on_resize = true;
		if (icon_path.is_empty() && layers[i].fallback_to_project_icon) {
			icon_path = p_preset->get("icons/icon_1024x1024");
			warn_on_resize = false;
		}
		if (icon_path.is_empty() && layers[i].fallback_to_project_icon) {
			icon_path = get_project_setting(p_preset, "application/config/icon");
			warn_on_resize = false;
		}

		Ref<Image> img;
		if (icon_path.is_empty()) {
			// Opaque-black placeholder, full layer size.
			img = Image::create_empty(LAYER_SIDE, LAYER_SIDE, false, Image::FORMAT_RGBA8);
			img->fill(Color(0, 0, 0, 1));
		} else {
			img = _load_icon_or_splash_image(icon_path, &err);
			if (err != OK || img.is_null() || img->is_empty()) {
				add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat("Invalid icon (%s): '%s'.", layers[i].preset_key, icon_path));
				return ERR_UNCONFIGURED;
			}
			if (img->get_width() != LAYER_SIDE || img->get_height() != LAYER_SIDE) {
				if (warn_on_resize) {
					add_message(EXPORT_MESSAGE_WARNING, TTR("Export Icons"), vformat("Icon (%s): '%s' has incorrect size %s and was automatically resized to %s.", layers[i].preset_key, icon_path, img->get_size(), Vector2i(LAYER_SIDE, LAYER_SIDE)));
				}
				img->resize(LAYER_SIDE, LAYER_SIDE, interpolation);
			}
		}

		err = img->save_png(imageset_dir + png_name);
		if (err != OK) {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat("Failed to export icon (%s): '%s'.", layers[i].preset_key, icon_path));
			return err;
		}

		{
			Ref<FileAccess> f = FileAccess::open(layer_dir + "Contents.json", FileAccess::WRITE);
			if (f.is_null()) {
				add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat(TTR("Could not write to a file at path \"%s\"."), layer_dir + "Contents.json"));
				return ERR_CANT_CREATE;
			}
			CharString utf8 = layer_metadata_json.utf8();
			f->store_buffer((const uint8_t *)utf8.get_data(), utf8.length());
		}

		{
			String imageset_json = "{\"images\":[{\"filename\":\"" + png_name + "\",\"idiom\":\"vision\",\"scale\":\"2x\"}],\"info\":{\"author\":\"xcode\",\"version\":1}}";
			Ref<FileAccess> f = FileAccess::open(imageset_dir + "Contents.json", FileAccess::WRITE);
			if (f.is_null()) {
				add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat(TTR("Could not write to a file at path \"%s\"."), imageset_dir + "Contents.json"));
				return ERR_CANT_CREATE;
			}
			CharString utf8 = imageset_json.utf8();
			f->store_buffer((const uint8_t *)utf8.get_data(), utf8.length());
		}
	}

	return OK;
}

String EditorExportPlatformVisionOS::_process_config_file_line(const Ref<EditorExportPreset> &p_preset, const String &p_line, const AppleEmbeddedConfigData &p_config, bool p_debug, const CodeSigningDetails &p_code_signing) {
	// Do visionOS specific processing first, and call super implementation if there are no matches

	String strnew;

	// Supported Destinations
	if (p_line.contains("$targeted_device_family")) {
		strnew += p_line.replace("$targeted_device_family", "7") + "\n";

		// MoltenVK Framework not used on visionOS
	} else if (p_line.contains("$moltenvk_buildfile")) {
		strnew += p_line.replace("$moltenvk_buildfile", "") + "\n";
	} else if (p_line.contains("$moltenvk_fileref")) {
		strnew += p_line.replace("$moltenvk_fileref", "") + "\n";
	} else if (p_line.contains("$moltenvk_buildphase")) {
		strnew += p_line.replace("$moltenvk_buildphase", "") + "\n";
	} else if (p_line.contains("$moltenvk_buildgrp")) {
		strnew += p_line.replace("$moltenvk_buildgrp", "") + "\n";

		// Launch Storyboard
	} else if (p_line.contains("$plist_launch_screen_name")) {
		strnew += p_line.replace("$plist_launch_screen_name", "") + "\n";
	} else if (p_line.contains("$pbx_launch_screen_file_reference")) {
		strnew += p_line.replace("$pbx_launch_screen_file_reference", "") + "\n";
	} else if (p_line.contains("$pbx_launch_screen_copy_files")) {
		strnew += p_line.replace("$pbx_launch_screen_copy_files", "") + "\n";
	} else if (p_line.contains("$pbx_launch_screen_build_phase")) {
		strnew += p_line.replace("$pbx_launch_screen_build_phase", "") + "\n";
	} else if (p_line.contains("$pbx_launch_screen_build_reference")) {
		strnew += p_line.replace("$pbx_launch_screen_build_reference", "") + "\n";

		// OS Deployment Target
	} else if (p_line.contains("$os_deployment_target")) {
		String min_version = p_preset->get("application/min_" + get_platform_name() + "_version");
		String value = "XROS_DEPLOYMENT_TARGET = " + min_version + ";";
		strnew += p_line.replace("$os_deployment_target", value) + "\n";

		// Valid Archs
	} else if (p_line.contains("$valid_archs")) {
		strnew += p_line.replace("$valid_archs", "arm64") + "\n";

		// Application Scene Manifest - Default Session Role
	} else if (p_line.contains("$application_scene_manifest_default_session_role")) {
		int app_role_enum = (int)p_preset->get("application/app_role");
		if (app_role_enum == 0) {
			// Windowed mode, not needed
			strnew += p_line.replace("$application_scene_manifest_default_session_role", "") + "\n";
			return strnew;
		}

		String value =
				"<key>UIApplicationPreferredDefaultSceneSessionRole</key>\n"
				"<string>CPSceneSessionRoleImmersiveSpaceApplication</string>";

		strnew += p_line.replace("$application_scene_manifest_default_session_role", value) + "\n";

		// Application Scene Manifest - Immersive Configuration
	} else if (p_line.contains("$application_scene_manifest_immersive_configuration")) {
		int app_role_enum = (int)p_preset->get("application/app_role");
		if (app_role_enum == 0) {
			// Windowed mode, not needed
			strnew += p_line.replace("$application_scene_manifest_immersive_configuration", "") + "\n";
			return strnew;
		}

		String initial_immersion_style;
		switch ((int)p_preset->get("application/immersion_style")) {
			case 0: // Full
				initial_immersion_style = "UIImmersionStyleFull";
				break;
			case 1: // Mixed
				initial_immersion_style = "UIImmersionStyleMixed";
				break;
		}

		String value =
				"<key>UISceneSessionRoleImmersiveSpaceApplication</key>\n"
				"<array>\n"
				"	<dict>\n"
				"		<key>UISceneInitialImmersionStyle</key>\n"
				"		<string>" +
				initial_immersion_style + "</string>\n"
										  "	</dict>\n"
										  "</array>";

		strnew += p_line.replace("$application_scene_manifest_immersive_configuration", value) + "\n";

		// Apple Embedded common
	} else {
		strnew += EditorExportPlatformAppleEmbedded::_process_config_file_line(p_preset, p_line, p_config, p_debug, p_code_signing);
	}
	return strnew;
}
