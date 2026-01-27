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

Vector<String> EditorExportPlatformIOS::device_types({ "iPhone", "iPad" });

void EditorExportPlatformIOS::initialize() {
	if (EditorNode::get_singleton()) {
		EditorExportPlatformAppleEmbedded::_initialize(_ios_logo_svg, _ios_run_icon_svg);
#ifdef MACOS_ENABLED
		_start_remote_device_poller_thread();
#endif
	}
}

EditorExportPlatformIOS::~EditorExportPlatformIOS() {
#ifdef MACOS_ENABLED
	_stop_remote_device_poller_thread();
#endif
}

void EditorExportPlatformIOS::get_export_options(List<ExportOption> *r_options) const {
	EditorExportPlatformAppleEmbedded::get_export_options(r_options);

	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "application/targeted_device_family", PROPERTY_HINT_ENUM, "iPhone,iPad,iPhone & iPad"), 2));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "application/min_ios_version"), get_minimum_deployment_target()));

	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "storyboard/image_scale_mode", PROPERTY_HINT_ENUM, "Same as Logo,Center,Scale to Fit,Scale to Fill,Scale"), 0));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "storyboard/custom_image@2x", PROPERTY_HINT_FILE_PATH, "*.png,*.jpg,*.jpeg"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "storyboard/custom_image@3x", PROPERTY_HINT_FILE_PATH, "*.png,*.jpg,*.jpeg"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "storyboard/use_custom_bg_color"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::COLOR, "storyboard/custom_bg_color"), Color()));
}

bool EditorExportPlatformIOS::has_valid_export_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates, bool p_debug) const {
	bool valid = EditorExportPlatformAppleEmbedded::has_valid_export_configuration(p_preset, r_error, r_missing_templates, p_debug);

	String err;
	String rendering_method = get_project_setting(p_preset, "rendering/renderer/rendering_method.mobile");
	String rendering_driver = get_project_setting(p_preset, "rendering/rendering_device/driver." + get_platform_name());
	if ((rendering_method == "forward_plus" || rendering_method == "mobile") && rendering_driver == "metal") {
		float version = p_preset->get("application/min_ios_version").operator String().to_float();
		if (version < 14.0) {
			err += TTR("Metal renderer require iOS 14+.") + "\n";
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

HashMap<String, Variant> EditorExportPlatformIOS::get_custom_project_settings(const Ref<EditorExportPreset> &p_preset) const {
	HashMap<String, Variant> settings;

	int image_scale_mode = p_preset->get("storyboard/image_scale_mode");
	String value;

	switch (image_scale_mode) {
		case 0: {
			String logo_path = get_project_setting(p_preset, "application/boot_splash/image");
			RenderingServer::SplashStretchMode stretch_mode = get_project_setting(p_preset, "application/boot_splash/stretch_mode");
			// If custom logo is not specified, Godot does not scale default one, so we should do the same.
			if (logo_path.is_empty()) {
				value = "center";
			} else {
				switch (stretch_mode) {
					case RenderingServer::SplashStretchMode::SPLASH_STRETCH_MODE_DISABLED: {
						value = "center";
					} break;
					case RenderingServer::SplashStretchMode::SPLASH_STRETCH_MODE_KEEP: {
						value = "scaleAspectFit";
					} break;
					case RenderingServer::SplashStretchMode::SPLASH_STRETCH_MODE_KEEP_WIDTH: {
						value = "scaleAspectFit";
					} break;
					case RenderingServer::SplashStretchMode::SPLASH_STRETCH_MODE_KEEP_HEIGHT: {
						value = "scaleAspectFit";
					} break;
					case RenderingServer::SplashStretchMode::SPLASH_STRETCH_MODE_COVER: {
						value = "scaleAspectFill";
					} break;
					case RenderingServer::SplashStretchMode::SPLASH_STRETCH_MODE_IGNORE: {
						value = "scaleToFill";
					} break;
				}
			}
		} break;
		default: {
			value = storyboard_image_scale_mode[image_scale_mode - 1];
		}
	}
	settings["ios/launch_screen_image_mode"] = value;
	return settings;
}

Error EditorExportPlatformIOS::_export_loading_screen_file(const Ref<EditorExportPreset> &p_preset, const String &p_dest_dir) {
	const String custom_launch_image_2x = p_preset->get("storyboard/custom_image@2x");
	const String custom_launch_image_3x = p_preset->get("storyboard/custom_image@3x");

	if (custom_launch_image_2x.length() > 0 && custom_launch_image_3x.length() > 0) {
		String image_path = p_dest_dir.path_join("splash@2x.png");
		Error err = OK;
		Ref<Image> image = _load_icon_or_splash_image(custom_launch_image_2x, &err);

		if (err != OK || image.is_null() || image->is_empty()) {
			return err;
		}

		if (image->save_png(image_path) != OK) {
			return ERR_FILE_CANT_WRITE;
		}

		image_path = p_dest_dir.path_join("splash@3x.png");
		image = _load_icon_or_splash_image(custom_launch_image_3x, &err);

		if (err != OK || image.is_null() || image->is_empty()) {
			return err;
		}

		if (image->save_png(image_path) != OK) {
			return ERR_FILE_CANT_WRITE;
		}
	} else {
		Error err = OK;
		Ref<Image> splash;

		const String splash_path = get_project_setting(p_preset, "application/boot_splash/image");

		if (!splash_path.is_empty()) {
			splash = _load_icon_or_splash_image(splash_path, &err);
		}

		if (err != OK || splash.is_null() || splash->is_empty()) {
			splash.instantiate(boot_splash_png);
		}

		// Using same image for both @2x and @3x
		// because Godot's own boot logo uses single image for all resolutions.
		// Also not using @1x image, because devices using this image variant
		// are not supported by iOS 9, which is minimal target.
		const String splash_png_path_2x = p_dest_dir.path_join("splash@2x.png");
		const String splash_png_path_3x = p_dest_dir.path_join("splash@3x.png");

		if (splash->save_png(splash_png_path_2x) != OK) {
			return ERR_FILE_CANT_WRITE;
		}

		if (splash->save_png(splash_png_path_3x) != OK) {
			return ERR_FILE_CANT_WRITE;
		}
	}

	return OK;
}

Vector<EditorExportPlatformAppleEmbedded::IconInfo> EditorExportPlatformIOS::get_icon_infos() const {
	Vector<EditorExportPlatformAppleEmbedded::IconInfo> icon_infos;
	return {
		// Settings on iPhone, iPad Pro, iPad, iPad mini
		{ PNAME("icons/settings_58x58"), "universal", "Icon-58", "58", "2x", "29x29", false },
		{ PNAME("icons/settings_87x87"), "universal", "Icon-87", "87", "3x", "29x29", false },

		// Notifications on iPhone, iPad Pro, iPad, iPad mini
		{ PNAME("icons/notification_40x40"), "universal", "Icon-40", "40", "2x", "20x20", false },
		{ PNAME("icons/notification_60x60"), "universal", "Icon-60", "60", "3x", "20x20", false },
		{ PNAME("icons/notification_76x76"), "universal", "Icon-76", "76", "2x", "38x38", false },
		{ PNAME("icons/notification_114x114"), "universal", "Icon-114", "114", "3x", "38x38", false },

		// Spotlight on iPhone, iPad Pro, iPad, iPad mini
		{ PNAME("icons/spotlight_80x80"), "universal", "Icon-80", "80", "2x", "40x40", false },
		{ PNAME("icons/spotlight_120x120"), "universal", "Icon-120", "120", "3x", "40x40", false },

		// Home Screen on iPhone
		{ PNAME("icons/iphone_120x120"), "universal", "Icon-120-1", "120", "2x", "60x60", false },
		{ PNAME("icons/iphone_180x180"), "universal", "Icon-180", "180", "3x", "60x60", false },

		// Home Screen on iPad Pro
		{ PNAME("icons/ipad_167x167"), "universal", "Icon-167", "167", "2x", "83.5x83.5", false },

		// Home Screen on iPad, iPad mini
		{ PNAME("icons/ipad_152x152"), "universal", "Icon-152", "152", "2x", "76x76", false },

		{ PNAME("icons/ios_128x128"), "universal", "Icon-128", "128", "2x", "64x64", false },
		{ PNAME("icons/ios_192x192"), "universal", "Icon-192", "192", "3x", "64x64", false },

		{ PNAME("icons/ios_136x136"), "universal", "Icon-136", "136", "2x", "68x68", false },

		// App Store
		{ PNAME("icons/app_store_1024x1024"), "universal", "Icon-1024", "1024", "1x", "1024x1024", true },
	};
}

Error EditorExportPlatformIOS::_export_icons(const Ref<EditorExportPreset> &p_preset, const String &p_iconset_dir) {
	String json_description = "{\"images\":[";
	String sizes;

	Ref<DirAccess> da = DirAccess::open(p_iconset_dir);
	if (da.is_null()) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat(TTR("Could not open a directory at path \"%s\"."), p_iconset_dir));
		return ERR_CANT_OPEN;
	}

	Color boot_bg_color = get_project_setting(p_preset, "application/boot_splash/bg_color");

	enum IconColorMode {
		ICON_NORMAL,
		ICON_DARK,
		ICON_TINTED,
		ICON_MAX,
	};

	Vector<IconInfo> icon_infos = get_icon_infos();
	bool first_icon = true;
	for (int i = 0; i < icon_infos.size(); ++i) {
		for (int color_mode = ICON_NORMAL; color_mode < ICON_MAX; color_mode++) {
			IconInfo info = icon_infos[i];
			int side_size = String(info.actual_size_side).to_int();
			String key = info.preset_key;
			String exp_name = info.export_name;
			if (color_mode == ICON_DARK) {
				key += "_dark";
				exp_name += "_dark";
			} else if (color_mode == ICON_TINTED) {
				key += "_tinted";
				exp_name += "_tinted";
			}
			exp_name += ".png";
			String icon_path = p_preset->get(key);
			bool resize_waning = true;
			if (icon_path.is_empty()) {
				// Load and resize base icon.
				key = "icons/icon_1024x1024";
				if (color_mode == ICON_DARK) {
					key += "_dark";
				} else if (color_mode == ICON_TINTED) {
					key += "_tinted";
				}
				icon_path = p_preset->get(key);
				resize_waning = false;
			}
			if (icon_path.is_empty()) {
				if (color_mode != ICON_NORMAL) {
					continue;
				}
				// Resize main app icon.
				icon_path = get_project_setting(p_preset, "application/config/icon");
				Error err = OK;
				Ref<Image> img = _load_icon_or_splash_image(icon_path, &err);
				if (err != OK || img.is_null() || img->is_empty()) {
					add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat("Invalid icon (%s): '%s'.", info.preset_key, icon_path));
					return ERR_UNCONFIGURED;
				} else if (info.force_opaque && img->detect_alpha() != Image::ALPHA_NONE) {
					img->resize(side_size, side_size, (Image::Interpolation)(p_preset->get("application/icon_interpolation").operator int()));
					Ref<Image> new_img = Image::create_empty(side_size, side_size, false, Image::FORMAT_RGBA8);
					new_img->fill(boot_bg_color);
					_blend_and_rotate(new_img, img, false);
					err = new_img->save_png(p_iconset_dir + exp_name);
				} else {
					img->resize(side_size, side_size, (Image::Interpolation)(p_preset->get("application/icon_interpolation").operator int()));
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
					if (resize_waning) {
						add_message(EXPORT_MESSAGE_WARNING, TTR("Export Icons"), vformat("Icon (%s) must be opaque.", info.preset_key));
					}
					img->resize(side_size, side_size, (Image::Interpolation)(p_preset->get("application/icon_interpolation").operator int()));
					Ref<Image> new_img = Image::create_empty(side_size, side_size, false, Image::FORMAT_RGBA8);
					new_img->fill(boot_bg_color);
					_blend_and_rotate(new_img, img, false);
					err = new_img->save_png(p_iconset_dir + exp_name);
				} else if (img->get_width() != side_size || img->get_height() != side_size) {
					if (resize_waning) {
						add_message(EXPORT_MESSAGE_WARNING, TTR("Export Icons"), vformat("Icon (%s): '%s' has incorrect size %s and was automatically resized to %s.", info.preset_key, icon_path, img->get_size(), Vector2i(side_size, side_size)));
					}
					img->resize(side_size, side_size, (Image::Interpolation)(p_preset->get("application/icon_interpolation").operator int()));
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
			if (color_mode != ICON_NORMAL) {
				json_description += String("\"appearances\":[{");
				json_description += String("\"appearance\":\"luminosity\",");
				if (color_mode == ICON_DARK) {
					json_description += String("\"value\":\"dark\"");
				} else if (color_mode == ICON_TINTED) {
					json_description += String("\"value\":\"tinted\"");
				}
				json_description += String("}],");
			}
			json_description += String("\"idiom\":") + "\"" + info.idiom + "\",";
			json_description += String("\"platform\":\"" + get_platform_name() + "\",");
			json_description += String("\"size\":") + "\"" + info.unscaled_size + "\",";
			if (String(info.scale) != "1x") {
				json_description += String("\"scale\":") + "\"" + info.scale + "\",";
			}
			json_description += String("\"filename\":") + "\"" + exp_name + "\"";
			json_description += String("}");
		}
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

String EditorExportPlatformIOS::_process_config_file_line(const Ref<EditorExportPreset> &p_preset, const String &p_line, const AppleEmbeddedConfigData &p_config, bool p_debug, const CodeSigningDetails &p_code_signing) {
	// Do iOS specific processing first, and call super implementation if there are no matches

	String strnew;

	// Supported Destinations
	if (p_line.contains("$targeted_device_family")) {
		String xcode_value;
		switch ((int)p_preset->get("application/targeted_device_family")) {
			case 0: // iPhone
				xcode_value = "1";
				break;
			case 1: // iPad
				xcode_value = "2";
				break;
			case 2: // iPhone & iPad
				xcode_value = "1,2";
				break;
		}
		strnew += p_line.replace("$targeted_device_family", xcode_value) + "\n";

		// MoltenVK Framework
	} else if (p_line.contains("$moltenvk_buildfile")) {
		String value = "9039D3BE24C093AC0020482C /* MoltenVK.xcframework in Frameworks */ = {isa = PBXBuildFile; fileRef = 9039D3BD24C093AC0020482C /* MoltenVK.xcframework */; };";
		strnew += p_line.replace("$moltenvk_buildfile", value) + "\n";
	} else if (p_line.contains("$moltenvk_fileref")) {
		String value = "9039D3BD24C093AC0020482C /* MoltenVK.xcframework */ = {isa = PBXFileReference; lastKnownFileType = wrapper.xcframework; name = MoltenVK; path = MoltenVK.xcframework; sourceTree = \"<group>\"; };";
		strnew += p_line.replace("$moltenvk_fileref", value) + "\n";
	} else if (p_line.contains("$moltenvk_buildphase")) {
		String value = "9039D3BE24C093AC0020482C /* MoltenVK.xcframework in Frameworks */,";
		strnew += p_line.replace("$moltenvk_buildphase", value) + "\n";
	} else if (p_line.contains("$moltenvk_buildgrp")) {
		String value = "9039D3BD24C093AC0020482C /* MoltenVK.xcframework */,";
		strnew += p_line.replace("$moltenvk_buildgrp", value) + "\n";

		// Launch Storyboard
	} else if (p_line.contains("$plist_launch_screen_name")) {
		String value = "<key>UILaunchStoryboardName</key>\n<string>Launch Screen</string>";
		strnew += p_line.replace("$plist_launch_screen_name", value) + "\n";
	} else if (p_line.contains("$pbx_launch_screen_file_reference")) {
		String value = "90DD2D9D24B36E8000717FE1 = {isa = PBXFileReference; fileEncoding = 4; lastKnownFileType = file.storyboard; path = \"Launch Screen.storyboard\"; sourceTree = \"<group>\"; };";
		strnew += p_line.replace("$pbx_launch_screen_file_reference", value) + "\n";
	} else if (p_line.contains("$pbx_launch_screen_copy_files")) {
		String value = "90DD2D9D24B36E8000717FE1 /* Launch Screen.storyboard */,";
		strnew += p_line.replace("$pbx_launch_screen_copy_files", value) + "\n";
	} else if (p_line.contains("$pbx_launch_screen_build_phase")) {
		String value = "90DD2D9E24B36E8000717FE1 /* Launch Screen.storyboard in Resources */,";
		strnew += p_line.replace("$pbx_launch_screen_build_phase", value) + "\n";
	} else if (p_line.contains("$pbx_launch_screen_build_reference")) {
		String value = "90DD2D9E24B36E8000717FE1 /* Launch Screen.storyboard in Resources */ = {isa = PBXBuildFile; fileRef = 90DD2D9D24B36E8000717FE1 /* Launch Screen.storyboard */; };";
		strnew += p_line.replace("$pbx_launch_screen_build_reference", value) + "\n";

		// Launch Storyboard customization
	} else if (p_line.contains("$launch_screen_image_mode")) {
		int image_scale_mode = p_preset->get("storyboard/image_scale_mode");
		String value;

		switch (image_scale_mode) {
			case 0: {
				String logo_path = get_project_setting(p_preset, "application/boot_splash/image");
				bool is_on = get_project_setting(p_preset, "application/boot_splash/fullsize");
				// If custom logo is not specified, Godot does not scale default one, so we should do the same.
				value = (is_on && logo_path.length() > 0) ? "scaleAspectFit" : "center";
			} break;
			default: {
				value = storyboard_image_scale_mode[image_scale_mode - 1];
			}
		}

		strnew += p_line.replace("$launch_screen_image_mode", value) + "\n";
	} else if (p_line.contains("$launch_screen_background_color")) {
		bool use_custom = p_preset->get("storyboard/use_custom_bg_color");
		Color color = use_custom ? p_preset->get("storyboard/custom_bg_color") : get_project_setting(p_preset, "application/boot_splash/bg_color");
		const String value_format = "red=\"$red\" green=\"$green\" blue=\"$blue\" alpha=\"$alpha\"";

		Dictionary value_dictionary;
		value_dictionary["red"] = color.r;
		value_dictionary["green"] = color.g;
		value_dictionary["blue"] = color.b;
		value_dictionary["alpha"] = color.a;
		String value = value_format.format(value_dictionary, "$_");

		strnew += p_line.replace("$launch_screen_background_color", value) + "\n";

		// OS Deployment Target
	} else if (p_line.contains("$os_deployment_target")) {
		String min_version = p_preset->get("application/min_" + get_platform_name() + "_version");
		String value = "IPHONEOS_DEPLOYMENT_TARGET = " + min_version + ";";
		strnew += p_line.replace("$os_deployment_target", value) + "\n";

		// Valid Archs
	} else if (p_line.contains("$valid_archs")) {
		strnew += p_line.replace("$valid_archs", "arm64 x86_64") + "\n";

		// Apple Embedded common
	} else {
		strnew += EditorExportPlatformAppleEmbedded::_process_config_file_line(p_preset, p_line, p_config, p_debug, p_code_signing);
	}

	return strnew;
}
