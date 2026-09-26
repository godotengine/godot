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

	// Each tvOS app icon is a parallax image stack with its own layers. The back layer
	// falls back to the project icon; unset middle and front layers are emitted transparent.
	static const char *icon_layer_keys[] = {
		"icons/tvos_small_app_icon_back_layer",
		"icons/tvos_small_app_icon_middle_layer",
		"icons/tvos_small_app_icon_front_layer",
		"icons/tvos_large_app_icon_back_layer",
		"icons/tvos_large_app_icon_middle_layer",
		"icons/tvos_large_app_icon_front_layer",
	};
	for (const char *key : icon_layer_keys) {
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, key, PROPERTY_HINT_FILE_PATH, "*.svg,*.png,*.webp,*.jpg,*.jpeg"), ""));
	}

	r_options->push_back(ExportOption(PropertyInfo(Variant::INT, "storyboard/image_scale_mode", PROPERTY_HINT_ENUM, "Same as Logo,Center,Scale to Fit,Scale to Fill,Scale"), 0));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "storyboard/custom_image@2x", PROPERTY_HINT_FILE_PATH, "*.png,*.jpg,*.jpeg"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "storyboard/custom_image@3x", PROPERTY_HINT_FILE_PATH, "*.png,*.jpg,*.jpeg"), ""));
	r_options->push_back(ExportOption(PropertyInfo(Variant::BOOL, "storyboard/use_custom_bg_color"), false));
	r_options->push_back(ExportOption(PropertyInfo(Variant::COLOR, "storyboard/custom_bg_color"), Color()));
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
	HashMap<String, Variant> settings;

	int image_scale_mode = p_preset->get("storyboard/image_scale_mode");
	String value;

	switch (image_scale_mode) {
		case 0: {
			String logo_path = get_project_setting(p_preset, "application/boot_splash/image");
			RSE::SplashStretchMode stretch_mode = get_project_setting(p_preset, "application/boot_splash/stretch_mode");
			// If custom logo is not specified, Godot does not scale default one, so we should do the same.
			if (logo_path.is_empty()) {
				value = "center";
			} else {
				switch (stretch_mode) {
					case RSE::SplashStretchMode::SPLASH_STRETCH_MODE_DISABLED: {
						value = "center";
					} break;
					case RSE::SplashStretchMode::SPLASH_STRETCH_MODE_KEEP: {
						value = "scaleAspectFit";
					} break;
					case RSE::SplashStretchMode::SPLASH_STRETCH_MODE_KEEP_WIDTH: {
						value = "scaleAspectFit";
					} break;
					case RSE::SplashStretchMode::SPLASH_STRETCH_MODE_KEEP_HEIGHT: {
						value = "scaleAspectFit";
					} break;
					case RSE::SplashStretchMode::SPLASH_STRETCH_MODE_COVER: {
						value = "scaleAspectFill";
					} break;
					case RSE::SplashStretchMode::SPLASH_STRETCH_MODE_IGNORE: {
						value = "scaleToFill";
					} break;
				}
			}
		} break;
		default: {
			value = storyboard_image_scale_mode[image_scale_mode - 1];
		}
	}
	settings["tvos/launch_screen_image_mode"] = value;
	return settings;
}

Error EditorExportPlatformTVOS::_export_loading_screen_file(const Ref<EditorExportPreset> &p_preset, const String &p_dest_dir) {
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

Vector<EditorExportPlatformAppleEmbedded::IconInfo> EditorExportPlatformTVOS::get_icon_infos() const {
	// The app icons are layered image stacks that don't fit IconInfo; their layer options
	// are declared in `get_export_options()` and emitted by `_export_icons()`. Only the flat
	// Top Shelf images are described here.
	return {
		{ PNAME("icons/tvos_top_shelf"), "tv", "TopShelf-1920x720", "1920", "1x", "1920x720", false },
		{ PNAME("icons/tvos_top_shelf_wide"), "tv", "TopShelfWide-2320x720", "2320", "1x", "2320x720", false },
	};
}

Error EditorExportPlatformTVOS::_export_icons(const Ref<EditorExportPreset> &p_preset, const String &p_iconset_dir) {
	// tvOS ignores the flat `.appiconset` the shared exporter writes for iOS. It needs a
	// Brand Assets catalog holding layered `.imagestack` app icons and `.imageset` Top Shelf
	// images. `_get_iconset_dir_name()` already points the base class at
	// `AppIcon.brandassets`, so `p_iconset_dir` is that catalog and the
	// `ASSETCATALOG_COMPILER_APPICON_NAME` build setting ("AppIcon") still resolves to it.
	const String info_blob = "\"info\":{\"author\":\"xcode\",\"version\":1}";

	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (da.is_null()) {
		add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), TTR("Could not access the filesystem."));
		return ERR_CANT_CREATE;
	}

	const Color boot_bg_color = get_project_setting(p_preset, "application/boot_splash/bg_color");
	const Image::Interpolation interpolation = (Image::Interpolation)(p_preset->get("application/icon_interpolation").operator int());

	auto write_json = [&](const String &p_path, const String &p_json) -> Error {
		Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
		if (f.is_null()) {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat(TTR("Could not write to a file at path \"%s\"."), p_path));
			return ERR_CANT_CREATE;
		}
		f->store_string(p_json);
		return OK;
	};

	auto make_dir = [&](const String &p_dir) -> Error {
		Error err = da->make_dir_recursive(p_dir);
		if (err != OK && err != ERR_ALREADY_EXISTS) {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat(TTR("Could not create a directory at path \"%s\"."), p_dir));
			return err;
		}
		return OK;
	};

	// Loads an export option's image and writes it out at the requested size.
	auto save_png = [&](const String &p_key, const String &p_icon_path, int p_width, int p_height, bool p_force_opaque, bool p_warn_on_resize, const String &p_out_png) -> Error {
		Error err = OK;
		Ref<Image> img = _load_icon_or_splash_image(p_icon_path, &err);
		if (err != OK || img.is_null() || img->is_empty()) {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat(TTR("Invalid icon (%s): '%s'."), p_key, p_icon_path));
			return ERR_UNCONFIGURED;
		}
		if (p_force_opaque && img->detect_alpha() != Image::ALPHA_NONE) {
			if (p_warn_on_resize) {
				add_message(EXPORT_MESSAGE_WARNING, TTR("Export Icons"), vformat(TTR("Icon (%s) must be opaque."), p_key));
			}
			img->resize(p_width, p_height, interpolation);
			Ref<Image> opaque = Image::create_empty(p_width, p_height, false, Image::FORMAT_RGBA8);
			opaque->fill(boot_bg_color);
			_blend_and_rotate(opaque, img, false);
			err = opaque->save_png(p_out_png);
		} else {
			if (img->get_width() != p_width || img->get_height() != p_height) {
				if (p_warn_on_resize) {
					add_message(EXPORT_MESSAGE_WARNING, TTR("Export Icons"), vformat(TTR("Icon (%s): '%s' has incorrect size %s and was automatically resized to %s."), p_key, p_icon_path, img->get_size(), Vector2i(p_width, p_height)));
				}
				img->resize(p_width, p_height, interpolation);
			}
			err = img->save_png(p_out_png);
		}
		if (err != OK) {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat(TTR("Failed to export icon (%s): '%s'."), p_key, p_icon_path));
		}
		return err;
	};

	// Fully transparent stand-in for an unset image stack layer.
	auto save_transparent_png = [&](int p_width, int p_height, const String &p_out_png) -> Error {
		Ref<Image> img = Image::create_empty(p_width, p_height, false, Image::FORMAT_RGBA8);
		img->fill(Color(0, 0, 0, 0));
		Error err = img->save_png(p_out_png);
		if (err != OK) {
			add_message(EXPORT_MESSAGE_ERROR, TTR("Export Icons"), vformat(TTR("Could not write to a file at path \"%s\"."), p_out_png));
		}
		return err;
	};

	// Emits an `.imageset` from a single export option, at @1x and, when p_hidpi, @2x. A
	// `_dark` variant becomes a luminosity appearance when that option is set.
	auto build_flat_imageset = [&](const String &p_dir, const String &p_preset_key, const String &p_png_base, int p_width, int p_height, bool p_hidpi, bool p_force_opaque) -> Error {
		Error err = make_dir(p_dir);
		if (err != OK) {
			return err;
		}

		static const char *appearance_suffix[] = { "", "_dark" };
		static const char *appearance_value[] = { nullptr, "dark" };

		String images_json;
		for (int scale = 1; scale <= (p_hidpi ? 2 : 1); scale++) {
			const String scale_str = itos(scale) + "x";
			for (uint64_t mode = 0; mode < std_size(appearance_suffix); mode++) {
				const String suffix = appearance_suffix[mode];
				String icon_path = p_preset->get(p_preset_key + suffix);
				bool warn_on_resize = true;
				if (icon_path.is_empty()) {
					// Fall back to the generic 1024x1024 icon of the same appearance.
					icon_path = p_preset->get("icons/icon_1024x1024" + suffix);
					warn_on_resize = false;
				}
				if (icon_path.is_empty()) {
					// Only the default appearance falls back to the main app icon; an unset
					// dark variant is simply not emitted.
					if (appearance_value[mode]) {
						continue;
					}
					icon_path = get_project_setting(p_preset, "application/config/icon");
					warn_on_resize = false;
				}

				const String png_name = p_png_base + suffix + "@" + scale_str + ".png";
				err = save_png(p_preset_key + suffix, icon_path, p_width * scale, p_height * scale, p_force_opaque, warn_on_resize, p_dir.path_join(png_name));
				if (err != OK) {
					return err;
				}

				if (!images_json.is_empty()) {
					images_json += ",";
				}
				images_json += "{";
				if (appearance_value[mode]) {
					images_json += vformat("\"appearances\":[{\"appearance\":\"luminosity\",\"value\":\"%s\"}],", appearance_value[mode]);
				}
				images_json += vformat("\"filename\":\"%s\",\"idiom\":\"tv\",\"scale\":\"%s\"}", png_name, scale_str);
			}
		}

		return write_json(p_dir.path_join("Contents.json"), "{\"images\":[" + images_json + "]," + info_blob + "}");
	};

	// Emits one `<Layer>.imagestacklayer/Content.imageset`. The back layer carries the icon
	// and falls back to the project icon; the middle and front layers default to transparent
	// so an unconfigured stack still satisfies the two-layer minimum tvOS enforces.
	auto build_layer_imageset = [&](const String &p_dir, const String &p_preset_key, const String &p_png_base, int p_width, int p_height, bool p_hidpi, bool p_is_base) -> Error {
		Error err = make_dir(p_dir);
		if (err != OK) {
			return err;
		}

		String icon_path = p_preset->get(p_preset_key);
		bool warn_on_resize = true;
		if (icon_path.is_empty() && p_is_base) {
			icon_path = p_preset->get("icons/icon_1024x1024");
			warn_on_resize = false;
		}
		if (icon_path.is_empty() && p_is_base) {
			icon_path = get_project_setting(p_preset, "application/config/icon");
			warn_on_resize = false;
		}

		String images_json;
		for (int scale = 1; scale <= (p_hidpi ? 2 : 1); scale++) {
			const String scale_str = itos(scale) + "x";
			const String png_name = p_png_base + "@" + scale_str + ".png";
			const String png_path = p_dir.path_join(png_name);

			if (icon_path.is_empty()) {
				err = save_transparent_png(p_width * scale, p_height * scale, png_path);
			} else {
				// Only the back layer is composited over the splash background; the layers
				// above it are overlays and must keep their alpha.
				err = save_png(p_preset_key, icon_path, p_width * scale, p_height * scale, p_is_base, warn_on_resize, png_path);
			}
			if (err != OK) {
				return err;
			}

			if (!images_json.is_empty()) {
				images_json += ",";
			}
			images_json += vformat("{\"filename\":\"%s\",\"idiom\":\"tv\",\"scale\":\"%s\"}", png_name, scale_str);
		}

		return write_json(p_dir.path_join("Contents.json"), "{\"images\":[" + images_json + "]," + info_blob + "}");
	};

	struct LayerInfo {
		const char *name;
		const char *preset_suffix;
		bool is_base;
	};
	// Listed front-to-back, the order the stack's Contents.json expects.
	const LayerInfo layers[] = {
		{ "Front", "_front_layer", false },
		{ "Middle", "_middle_layer", false },
		{ "Back", "_back_layer", true },
	};

	struct StackInfo {
		const char *dir_name;
		const char *size;
		const char *preset_prefix;
		const char *png_base;
		int width;
		int height;
		bool hidpi;
	};
	// The App Store icon is @1x only; the one shown on the home screen also needs @2x.
	const StackInfo stacks[] = {
		{ "App Icon.imagestack", "400x240", "icons/tvos_small_app_icon", "AppIcon-400x240", 400, 240, true },
		{ "App Icon - App Store.imagestack", "1280x768", "icons/tvos_large_app_icon", "AppIcon-1280x768", 1280, 768, false },
	};

	struct TopShelfInfo {
		const char *dir_name;
		const char *role;
		const char *size;
		const char *preset_key;
		const char *png_base;
		int width;
		int height;
	};
	const TopShelfInfo top_shelves[] = {
		{ "Top Shelf Image.imageset", "top-shelf-image", "1920x720", "icons/tvos_top_shelf", "TopShelf-1920x720", 1920, 720 },
		{ "Top Shelf Image Wide.imageset", "top-shelf-image-wide", "2320x720", "icons/tvos_top_shelf_wide", "TopShelfWide-2320x720", 2320, 720 },
	};

	String assets_json;

	for (const StackInfo &stack : stacks) {
		const String stack_dir = p_iconset_dir.path_join(stack.dir_name);

		String layers_json;
		for (const LayerInfo &layer : layers) {
			const String layer_dir = stack_dir.path_join(String(layer.name) + ".imagestacklayer");

			Error err = build_layer_imageset(layer_dir.path_join("Content.imageset"), String(stack.preset_prefix) + layer.preset_suffix, String(stack.png_base) + "-" + layer.name, stack.width, stack.height, stack.hidpi, layer.is_base);
			if (err != OK) {
				return err;
			}
			err = write_json(layer_dir.path_join("Contents.json"), "{" + info_blob + "}");
			if (err != OK) {
				return err;
			}

			if (!layers_json.is_empty()) {
				layers_json += ",";
			}
			layers_json += vformat("{\"filename\":\"%s.imagestacklayer\"}", layer.name);
		}

		Error err = write_json(stack_dir.path_join("Contents.json"), "{\"layers\":[" + layers_json + "]," + info_blob + "}");
		if (err != OK) {
			return err;
		}

		if (!assets_json.is_empty()) {
			assets_json += ",";
		}
		assets_json += vformat("{\"filename\":\"%s\",\"idiom\":\"tv\",\"role\":\"primary-app-icon\",\"size\":\"%s\"}", stack.dir_name, stack.size);
	}

	for (const TopShelfInfo &top_shelf : top_shelves) {
		Error err = build_flat_imageset(p_iconset_dir.path_join(top_shelf.dir_name), top_shelf.preset_key, top_shelf.png_base, top_shelf.width, top_shelf.height, true, false);
		if (err != OK) {
			return err;
		}

		if (!assets_json.is_empty()) {
			assets_json += ",";
		}
		assets_json += vformat("{\"filename\":\"%s\",\"idiom\":\"tv\",\"role\":\"%s\",\"size\":\"%s\"}", top_shelf.dir_name, top_shelf.role, top_shelf.size);
	}

	return write_json(p_iconset_dir.path_join("Contents.json"), "{\"assets\":[" + assets_json + "]," + info_blob + "}");
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
		String value = "TVOS_DEPLOYMENT_TARGET = " + min_version + ";";
		strnew += p_line.replace("$os_deployment_target", value) + "\n";

		// Required Device Capabilities
	} else if (p_line.contains("$required_device_capabilities")) {
		String capabilities;
		for (const String &capability : p_config.capabilities) {
			capabilities += "<string>" + capability + "</string>\n";
		}
		for (const String &capability : p_preset->get("capabilities/additional").operator PackedStringArray()) {
			capabilities += "<string>" + capability + "</string>\n";
		}
		strnew += p_line.replace("$required_device_capabilities", capabilities) + "\n";

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
