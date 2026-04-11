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
#include "main/splash.gen.h"

Vector<String> EditorExportPlatformTVOS::device_types({ "appleTV" });

void EditorExportPlatformTVOS::initialize() {
	if (EditorNode::get_singleton()) {
		EditorExportPlatformAppleEmbedded::_initialize(_tvos_logo_svg, _tvos_run_icon_svg);
#ifdef MACOS_ENABLED
		_start_remote_device_poller_thread();
#endif
	}
}

EditorExportPlatformTVOS::~EditorExportPlatformTVOS() {
#ifdef MACOS_ENABLED
	_stop_remote_device_poller_thread();
#endif
}

void EditorExportPlatformTVOS::get_export_options(List<ExportOption> *r_options) const {
	EditorExportPlatformAppleEmbedded::get_export_options(r_options);

	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/min_tvos_version"), get_minimum_deployment_target()));
}

bool EditorExportPlatformTVOS::has_valid_export_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates, bool p_debug) const {
	bool valid = EditorExportPlatformAppleEmbedded::has_valid_export_configuration(p_preset, r_error, r_missing_templates, p_debug);

	String err;
	String rendering_method = get_project_setting(p_preset, "rendering/renderer/rendering_method.mobile");
	String rendering_driver = get_project_setting(p_preset, "rendering/rendering_device/driver." + get_platform_name());
	if ((rendering_method == "forward_plus" || rendering_method == "mobile") && rendering_driver == "metal") {
		float version = p_preset->get("application/min_tvos_version").operator String().to_float();
		if (version < 14.0) {
			err += TTR("Metal renderer require tvOS 14+.") + "\n";
		}
	}

	if (!err.is_empty()) {
		if (!r_error.is_empty()) {
			r_error += err;
		} else {
			r_error = err;
		}
	}

	return valid;
}

HashMap<String, Variant> EditorExportPlatformTVOS::get_custom_project_settings(const Ref<EditorExportPreset> &p_preset) const {
	return HashMap<String, Variant>();
}

Error EditorExportPlatformTVOS::_export_loading_screen_file(const Ref<EditorExportPreset> &p_preset, const String &p_dest_dir) {
	return OK;
}

Vector<EditorExportPlatformAppleEmbedded::IconInfo> EditorExportPlatformTVOS::get_icon_infos() const {
	return {
		// tvOS App Icons (flat, v1)
		{ PNAME("icons/tvos_small_app_icon"), "tv", "AppIcon-400x240", "400", "1x", "400x240", true },
		{ PNAME("icons/tvos_large_app_icon"), "tv", "AppIcon-1280x768", "1280", "1x", "1280x768", true },
		{ PNAME("icons/tvos_top_shelf"), "tv", "TopShelf-1920x720", "1920", "1x", "1920x720", false },
		{ PNAME("icons/tvos_top_shelf_wide"), "tv", "TopShelfWide-2320x720", "2320", "1x", "2320x720", false },
	};
}

Error EditorExportPlatformTVOS::_export_icons(const Ref<EditorExportPreset> &p_preset, const String &p_iconset_dir) {
	String json_description = "{\"images\":[";
	String sizes;

	Ref<DirAccess> da = DirAccess::open(p_iconset_dir);
	if (da.is_null()) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat(TTR("Could not open a directory at path \"%s\"."), p_iconset_dir));
		return ERR_CANT_OPEN;
	}

	Color boot_bg_color = get_project_setting(p_preset, "application/boot_splash/bg_color");

	struct TVOSIconSize {
		int width;
		int height;
	};

	// tvOS icons are not square, so we need width and height for each.
	const TVOSIconSize tvos_icon_sizes[] = {
		{ 400, 240 },
		{ 1280, 768 },
		{ 1920, 720 },
		{ 2320, 720 },
	};

	Vector<IconInfo> icon_infos = get_icon_infos();
	bool first_icon = true;
	for (int i = 0; i < icon_infos.size(); ++i) {
		IconInfo info = icon_infos[i];
		int target_width = tvos_icon_sizes[i].width;
		int target_height = tvos_icon_sizes[i].height;
		String key = info.preset_key;
		String exp_name = String(info.export_name) + ".png";
		String icon_path = p_preset->get(key);
		bool resize_warning = true;
		if (icon_path.is_empty()) {
			// Try generic 1024x1024 icon.
			key = "icons/icon_1024x1024";
			icon_path = p_preset->get(key);
			resize_warning = false;
		}
		if (icon_path.is_empty()) {
			// Resize main app icon.
			icon_path = get_project_setting(p_preset, "application/config/icon");
			Error err = OK;
			Ref<Image> img = _load_icon_or_splash_image(icon_path, &err);
			if (err != OK || img.is_null() || img->is_empty()) {
				add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat("Invalid icon (%s): '%s'.", info.preset_key, icon_path));
				return ERR_UNCONFIGURED;
			} else if (info.force_opaque && img->detect_alpha() != Image::ALPHA_NONE) {
				img->resize(target_width, target_height, (Image::Interpolation)(p_preset->get("application/icon_interpolation").operator int()));
				Ref<Image> new_img = Image::create_empty(target_width, target_height, false, Image::FORMAT_RGBA8);
				new_img->fill(boot_bg_color);
				_blend_and_rotate(new_img, img, false);
				err = new_img->save_png(p_iconset_dir + exp_name);
			} else {
				img->resize(target_width, target_height, (Image::Interpolation)(p_preset->get("application/icon_interpolation").operator int()));
				err = img->save_png(p_iconset_dir + exp_name);
			}
			if (err) {
				add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat("Failed to export icon (%s): '%s'.", info.preset_key, icon_path));
				return err;
			}
		} else {
			// Load custom icon and resize if required.
			Error err = OK;
			Ref<Image> img = _load_icon_or_splash_image(icon_path, &err);
			if (err != OK || img.is_null() || img->is_empty()) {
				add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat("Invalid icon (%s): '%s'.", info.preset_key, icon_path));
				return ERR_UNCONFIGURED;
			} else if (info.force_opaque && img->detect_alpha() != Image::ALPHA_NONE) {
				if (resize_warning) {
					add_message(EXPORT_MESSAGE_WARNING, TTR("Export Icons"), vformat("Icon (%s) must be opaque.", info.preset_key));
				}
				img->resize(target_width, target_height, (Image::Interpolation)(p_preset->get("application/icon_interpolation").operator int()));
				Ref<Image> new_img = Image::create_empty(target_width, target_height, false, Image::FORMAT_RGBA8);
				new_img->fill(boot_bg_color);
				_blend_and_rotate(new_img, img, false);
				err = new_img->save_png(p_iconset_dir + exp_name);
			} else if (img->get_width() != target_width || img->get_height() != target_height) {
				if (resize_warning) {
					add_message(EXPORT_MESSAGE_WARNING, TTR("Export Icons"), vformat("Icon (%s): '%s' has incorrect size %s and was automatically resized to %s.", info.preset_key, icon_path, img->get_size(), Vector2i(target_width, target_height)));
				}
				img->resize(target_width, target_height, (Image::Interpolation)(p_preset->get("application/icon_interpolation").operator int()));
				err = img->save_png(p_iconset_dir + exp_name);
			} else if (!icon_path.ends_with(".png")) {
				err = img->save_png(p_iconset_dir + exp_name);
			} else {
				err = da->copy(icon_path, p_iconset_dir + exp_name);
			}

			if (err) {
				add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat("Failed to export icon (%s): '%s'.", info.preset_key, icon_path));
				return err;
			}
		}
		sizes += String(info.actual_size_side) + "\n";
		if (first_icon) {
			first_icon = false;
		} else {
			json_description += ",";
		}
		json_description += String("{");
		json_description += String("\"idiom\":") + "\"" + info.idiom + "\",";
		json_description += String("\"platform\":\"" + get_platform_name() + "\",");
		json_description += String("\"size\":") + "\"" + info.unscaled_size + "\",";
		if (String(info.scale) != "1x") {
			json_description += String("\"scale\":") + "\"" + info.scale + "\",";
		}
		json_description += String("\"filename\":") + "\"" + exp_name + "\"";
		json_description += String("}");
	}
	json_description += "],\"info\":{\"author\":\"xcode\",\"version\":1}}";

	Ref<FileAccess> json_file = FileAccess::open(p_iconset_dir + "Contents.json", FileAccess::WRITE);
	if (json_file.is_null()) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat(TTR("Could not write to a file at path \"%s\"."), p_iconset_dir + "Contents.json"));
		return ERR_CANT_CREATE;
	}

	CharString json_utf8 = json_description.utf8();
	json_file->store_buffer((const uint8_t *)json_utf8.get_data(), json_utf8.length());

	Ref<FileAccess> sizes_file = FileAccess::open(p_iconset_dir + "sizes", FileAccess::WRITE);
	if (sizes_file.is_null()) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat(TTR("Could not write to a file at path \"%s\"."), p_iconset_dir + "sizes"));
		return ERR_CANT_CREATE;
	}

	CharString sizes_utf8 = sizes.utf8();
	sizes_file->store_buffer((const uint8_t *)sizes_utf8.get_data(), sizes_utf8.length());

	return OK;
}

String EditorExportPlatformTVOS::_process_config_file_line(const Ref<EditorExportPreset> &p_preset, const String &p_line, const AppleEmbeddedConfigData &p_config, bool p_debug, const CodeSigningDetails &p_code_signing) {
	// Do tvOS specific processing first, and call super implementation if there are no matches

	String strnew;

	// Supported Destinations
	if (p_line.contains("$targeted_device_family")) {
		strnew += p_line.replace("$targeted_device_family", "3") + "\n";

		// MoltenVK Framework not used on tvOS
	} else if (p_line.contains("$moltenvk_buildfile")) {
		strnew += p_line.replace("$moltenvk_buildfile", "") + "\n";
	} else if (p_line.contains("$moltenvk_fileref")) {
		strnew += p_line.replace("$moltenvk_fileref", "") + "\n";
	} else if (p_line.contains("$moltenvk_buildphase")) {
		strnew += p_line.replace("$moltenvk_buildphase", "") + "\n";
	} else if (p_line.contains("$moltenvk_buildgrp")) {
		strnew += p_line.replace("$moltenvk_buildgrp", "") + "\n";

		// Launch Storyboard (not used on tvOS)
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
	} else if (p_line.contains("$launch_screen_image_mode")) {
		strnew += p_line.replace("$launch_screen_image_mode", "") + "\n";
	} else if (p_line.contains("$launch_screen_background_color")) {
		strnew += p_line.replace("$launch_screen_background_color", "") + "\n";

		// OS Deployment Target
	} else if (p_line.contains("$os_deployment_target")) {
		String min_version = p_preset->get("application/min_" + get_platform_name() + "_version");
		String value = "TVOS_DEPLOYMENT_TARGET = " + min_version + ";";
		strnew += p_line.replace("$os_deployment_target", value) + "\n";

		// Valid Archs
	} else if (p_line.contains("$valid_archs")) {
		strnew += p_line.replace("$valid_archs", "arm64 x86_64") + "\n";

		// Application Scene Manifest - Default Session Role
	} else if (p_line.contains("$application_scene_manifest_default_session_role")) {
		strnew += p_line.replace("$application_scene_manifest_default_session_role", "") + "\n";

		// Application Scene Manifest - Immersive Configuration
	} else if (p_line.contains("$application_scene_manifest_immersive_configuration")) {
		strnew += p_line.replace("$application_scene_manifest_immersive_configuration", "") + "\n";

		// Apple Embedded common
	} else {
		strnew += EditorExportPlatformAppleEmbedded::_process_config_file_line(p_preset, p_line, p_config, p_debug, p_code_signing);
	}

	return strnew;
}
