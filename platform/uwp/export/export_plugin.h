/*************************************************************************/
/*  export_plugin.h                                                      */
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

#ifndef UWP_EXPORT_PLUGIN_H
#define UWP_EXPORT_PLUGIN_H

#include "core/config/project_settings.h"
#include "core/crypto/crypto_core.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/marshalls.h"
#include "core/io/zip_io.h"
#include "core/object/class_db.h"
#include "core/version.h"
#include "editor/editor_export.h"
#include "editor/editor_node.h"

#include "thirdparty/minizip/unzip.h"
#include "thirdparty/minizip/zip.h"

#include "app_packager.h"

#include <zlib.h>

// Capabilities
static const char *uwp_capabilities[] = {
	"allJoyn",
	"codeGeneration",
	"internetClient",
	"internetClientServer",
	"privateNetworkClientServer",
	nullptr
};
static const char *uwp_uap_capabilities[] = {
	"appointments",
	"blockedChatMessages",
	"chat",
	"contacts",
	"enterpriseAuthentication",
	"musicLibrary",
	"objects3D",
	"picturesLibrary",
	"phoneCall",
	"removableStorage",
	"sharedUserCertificates",
	"userAccountInformation",
	"videosLibrary",
	"voipCall",
	nullptr
};
static const char *uwp_device_capabilities[] = {
	"bluetooth",
	"location",
	"microphone",
	"proximity",
	"webcam",
	nullptr
};

class EditorExportPlatformUWP : public EditorExportPlatform {
	GDCLASS(EditorExportPlatformUWP, EditorExportPlatform);

	Ref<ImageTexture> logo;

	enum Platform {
		ARM,
		X86,
		X64
	};

	bool _valid_resource_name(const String &p_name) const {
		if (p_name.is_empty()) {
			return false;
		}
		if (p_name.ends_with(".")) {
			return false;
		}

		static const char *invalid_names[] = {
			"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7",
			"COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
			nullptr
		};

		const char **t = invalid_names;
		while (*t) {
			if (p_name == *t) {
				return false;
			}
			t++;
		}

		return true;
	}

	bool _valid_guid(const String &p_guid) const {
		Vector<String> parts = p_guid.split("-");

		if (parts.size() != 5) {
			return false;
		}
		if (parts[0].length() != 8) {
			return false;
		}
		for (int i = 1; i < 4; i++) {
			if (parts[i].length() != 4) {
				return false;
			}
		}
		if (parts[4].length() != 12) {
			return false;
		}

		return true;
	}

	bool _valid_bgcolor(const String &p_color) const {
		if (p_color.is_empty()) {
			return true;
		}
		if (p_color.begins_with("#") && p_color.is_valid_html_color()) {
			return true;
		}

		// Colors from https://msdn.microsoft.com/en-us/library/windows/apps/dn934817.aspx
		static const char *valid_colors[] = {
			"aliceBlue", "antiqueWhite", "aqua", "aquamarine", "azure", "beige",
			"bisque", "black", "blanchedAlmond", "blue", "blueViolet", "brown",
			"burlyWood", "cadetBlue", "chartreuse", "chocolate", "coral", "cornflowerBlue",
			"cornsilk", "crimson", "cyan", "darkBlue", "darkCyan", "darkGoldenrod",
			"darkGray", "darkGreen", "darkKhaki", "darkMagenta", "darkOliveGreen", "darkOrange",
			"darkOrchid", "darkRed", "darkSalmon", "darkSeaGreen", "darkSlateBlue", "darkSlateGray",
			"darkTurquoise", "darkViolet", "deepPink", "deepSkyBlue", "dimGray", "dodgerBlue",
			"firebrick", "floralWhite", "forestGreen", "fuchsia", "gainsboro", "ghostWhite",
			"gold", "goldenrod", "gray", "green", "greenYellow", "honeydew",
			"hotPink", "indianRed", "indigo", "ivory", "khaki", "lavender",
			"lavenderBlush", "lawnGreen", "lemonChiffon", "lightBlue", "lightCoral", "lightCyan",
			"lightGoldenrodYellow", "lightGreen", "lightGray", "lightPink", "lightSalmon", "lightSeaGreen",
			"lightSkyBlue", "lightSlateGray", "lightSteelBlue", "lightYellow", "lime", "limeGreen",
			"linen", "magenta", "maroon", "mediumAquamarine", "mediumBlue", "mediumOrchid",
			"mediumPurple", "mediumSeaGreen", "mediumSlateBlue", "mediumSpringGreen", "mediumTurquoise", "mediumVioletRed",
			"midnightBlue", "mintCream", "mistyRose", "moccasin", "navajoWhite", "navy",
			"oldLace", "olive", "oliveDrab", "orange", "orangeRed", "orchid",
			"paleGoldenrod", "paleGreen", "paleTurquoise", "paleVioletRed", "papayaWhip", "peachPuff",
			"peru", "pink", "plum", "powderBlue", "purple", "red",
			"rosyBrown", "royalBlue", "saddleBrown", "salmon", "sandyBrown", "seaGreen",
			"seaShell", "sienna", "silver", "skyBlue", "slateBlue", "slateGray",
			"snow", "springGreen", "steelBlue", "tan", "teal", "thistle",
			"tomato", "transparent", "turquoise", "violet", "wheat", "white",
			"whiteSmoke", "yellow", "yellowGreen",
			nullptr
		};

		const char **color = valid_colors;

		while (*color) {
			if (p_color == *color) {
				return true;
			}
			color++;
		}

		return false;
	}

	bool _valid_image(const StreamTexture2D *p_image, int p_width, int p_height) const {
		if (!p_image) {
			return false;
		}

		// TODO: Add resource creation or image rescaling to enable other scales:
		// 1.25, 1.5, 2.0
		return p_width == p_image->get_width() && p_height == p_image->get_height();
	}

	Vector<uint8_t> _fix_manifest(const Ref<EditorExportPreset> &p_preset, const Vector<uint8_t> &p_template, bool p_give_internet) const {
		String result = String::utf8((const char *)p_template.ptr(), p_template.size());

		result = result.replace("$godot_version$", VERSION_FULL_NAME);

		result = result.replace("$identity_name$", p_preset->get("package/unique_name"));
		result = result.replace("$publisher$", p_preset->get("package/publisher"));

		result = result.replace("$product_guid$", p_preset->get("identity/product_guid"));
		result = result.replace("$publisher_guid$", p_preset->get("identity/publisher_guid"));

		String version = itos(p_preset->get("version/major")) + "." + itos(p_preset->get("version/minor")) + "." + itos(p_preset->get("version/build")) + "." + itos(p_preset->get("version/revision"));
		result = result.replace("$version_string$", version);

		Platform arch = (Platform)(int)p_preset->get("architecture/target");
		String architecture = arch == ARM ? "arm" : (arch == X86 ? "x86" : "x64");
		result = result.replace("$architecture$", architecture);

		result = result.replace("$display_name$", String(p_preset->get("package/display_name")).is_empty() ? (String)ProjectSettings::get_singleton()->get("application/config/name") : String(p_preset->get("package/display_name")));

		result = result.replace("$publisher_display_name$", p_preset->get("package/publisher_display_name"));
		result = result.replace("$app_description$", p_preset->get("package/description"));
		result = result.replace("$bg_color$", p_preset->get("images/background_color"));
		result = result.replace("$short_name$", p_preset->get("package/short_name"));

		String name_on_tiles = "";
		if ((bool)p_preset->get("tiles/show_name_on_square150x150")) {
			name_on_tiles += "          <uap:ShowOn Tile=\"square150x150Logo\" />\n";
		}
		if ((bool)p_preset->get("tiles/show_name_on_wide310x150")) {
			name_on_tiles += "          <uap:ShowOn Tile=\"wide310x150Logo\" />\n";
		}
		if ((bool)p_preset->get("tiles/show_name_on_square310x310")) {
			name_on_tiles += "          <uap:ShowOn Tile=\"square310x310Logo\" />\n";
		}

		String show_name_on_tiles = "";
		if (!name_on_tiles.is_empty()) {
			show_name_on_tiles = "<uap:ShowNameOnTiles>\n" + name_on_tiles + "        </uap:ShowNameOnTiles>";
		}

		result = result.replace("$name_on_tiles$", name_on_tiles);

		String rotations = "";
		if ((bool)p_preset->get("orientation/landscape")) {
			rotations += "          <uap:Rotation Preference=\"landscape\" />\n";
		}
		if ((bool)p_preset->get("orientation/portrait")) {
			rotations += "          <uap:Rotation Preference=\"portrait\" />\n";
		}
		if ((bool)p_preset->get("orientation/landscape_flipped")) {
			rotations += "          <uap:Rotation Preference=\"landscapeFlipped\" />\n";
		}
		if ((bool)p_preset->get("orientation/portrait_flipped")) {
			rotations += "          <uap:Rotation Preference=\"portraitFlipped\" />\n";
		}

		String rotation_preference = "";
		if (!rotations.is_empty()) {
			rotation_preference = "<uap:InitialRotationPreference>\n" + rotations + "        </uap:InitialRotationPreference>";
		}

		result = result.replace("$rotation_preference$", rotation_preference);

		String capabilities_elements = "";
		const char **basic = uwp_capabilities;
		while (*basic) {
			if ((bool)p_preset->get("capabilities/" + String(*basic))) {
				capabilities_elements += "    <Capability Name=\"" + String(*basic) + "\" />\n";
			}
			basic++;
		}
		const char **uap = uwp_uap_capabilities;
		while (*uap) {
			if ((bool)p_preset->get("capabilities/" + String(*uap))) {
				capabilities_elements += "    <uap:Capability Name=\"" + String(*uap) + "\" />\n";
			}
			uap++;
		}
		const char **device = uwp_device_capabilities;
		while (*device) {
			if ((bool)p_preset->get("capabilities/" + String(*device))) {
				capabilities_elements += "    <DeviceCapability Name=\"" + String(*device) + "\" />\n";
			}
			device++;
		}

		if (!((bool)p_preset->get("capabilities/internetClient")) && p_give_internet) {
			capabilities_elements += "    <Capability Name=\"internetClient\" />\n";
		}

		String capabilities_string = "<Capabilities />";
		if (!capabilities_elements.is_empty()) {
			capabilities_string = "<Capabilities>\n" + capabilities_elements + "  </Capabilities>";
		}

		result = result.replace("$capabilities_place$", capabilities_string);

		Vector<uint8_t> r_ret;
		r_ret.resize(result.length());

		for (int i = 0; i < result.length(); i++) {
			r_ret.write[i] = result.utf8().get(i);
		}

		return r_ret;
	}

	Vector<uint8_t> _get_image_data(const Ref<EditorExportPreset> &p_preset, const String &p_path) {
		Vector<uint8_t> data;
		StreamTexture2D *texture = nullptr;

		if (p_path.find("StoreLogo") != -1) {
			texture = p_preset->get("images/store_logo").is_zero() ? nullptr : Object::cast_to<StreamTexture2D>(((Object *)p_preset->get("images/store_logo")));
		} else if (p_path.find("Square44x44Logo") != -1) {
			texture = p_preset->get("images/square44x44_logo").is_zero() ? nullptr : Object::cast_to<StreamTexture2D>(((Object *)p_preset->get("images/square44x44_logo")));
		} else if (p_path.find("Square71x71Logo") != -1) {
			texture = p_preset->get("images/square71x71_logo").is_zero() ? nullptr : Object::cast_to<StreamTexture2D>(((Object *)p_preset->get("images/square71x71_logo")));
		} else if (p_path.find("Square150x150Logo") != -1) {
			texture = p_preset->get("images/square150x150_logo").is_zero() ? nullptr : Object::cast_to<StreamTexture2D>(((Object *)p_preset->get("images/square150x150_logo")));
		} else if (p_path.find("Square310x310Logo") != -1) {
			texture = p_preset->get("images/square310x310_logo").is_zero() ? nullptr : Object::cast_to<StreamTexture2D>(((Object *)p_preset->get("images/square310x310_logo")));
		} else if (p_path.find("Wide310x150Logo") != -1) {
			texture = p_preset->get("images/wide310x150_logo").is_zero() ? nullptr : Object::cast_to<StreamTexture2D>(((Object *)p_preset->get("images/wide310x150_logo")));
		} else if (p_path.find("SplashScreen") != -1) {
			texture = p_preset->get("images/splash_screen").is_zero() ? nullptr : Object::cast_to<StreamTexture2D>(((Object *)p_preset->get("images/splash_screen")));
		} else {
			ERR_PRINT("Unable to load logo");
		}

		if (!texture) {
			return data;
		}

		String tmp_path = EditorPaths::get_singleton()->get_cache_dir().plus_file("uwp_tmp_logo.png");

		Error err = texture->get_image()->save_png(tmp_path);

		if (err != OK) {
			String err_string = "Couldn't save temp logo file.";

			EditorNode::add_io_error(err_string);
			ERR_FAIL_V_MSG(data, err_string);
		}

		FileAccess *f = FileAccess::open(tmp_path, FileAccess::READ, &err);

		if (err != OK) {
			String err_string = "Couldn't open temp logo file.";
			// Cleanup generated file.
			DirAccess::remove_file_or_error(tmp_path);
			EditorNode::add_io_error(err_string);
			ERR_FAIL_V_MSG(data, err_string);
		}

		data.resize(f->get_length());
		f->get_buffer(data.ptrw(), data.size());

		f->close();
		memdelete(f);
		DirAccess::remove_file_or_error(tmp_path);

		return data;
	}

	static bool _should_compress_asset(const String &p_path, const Vector<uint8_t> &p_data) {
		/* TODO: This was copied verbatim from Android export. It should be
		 * refactored to the parent class and also be used for .zip export.
		 */

		/*
		 *  By not compressing files with little or not benefit in doing so,
		 *  a performance gain is expected at runtime. Moreover, if the APK is
		 *  zip-aligned, assets stored as they are can be efficiently read by
		 *  Android by memory-mapping them.
		 */

		// -- Unconditional uncompress to mimic AAPT plus some other

		static const char *unconditional_compress_ext[] = {
			// From https://github.com/android/platform_frameworks_base/blob/master/tools/aapt/Package.cpp
			// These formats are already compressed, or don't compress well:
			".jpg", ".jpeg", ".png", ".gif",
			".wav", ".mp2", ".mp3", ".ogg", ".aac",
			".mpg", ".mpeg", ".mid", ".midi", ".smf", ".jet",
			".rtttl", ".imy", ".xmf", ".mp4", ".m4a",
			".m4v", ".3gp", ".3gpp", ".3g2", ".3gpp2",
			".amr", ".awb", ".wma", ".wmv",
			// Godot-specific:
			".webp", // Same reasoning as .png
			".cfb", // Don't let small config files slow-down startup
			".scn", // Binary scenes are usually already compressed
			".stex", // Streamable textures are usually already compressed
			// Trailer for easier processing
			nullptr
		};

		for (const char **ext = unconditional_compress_ext; *ext; ++ext) {
			if (p_path.to_lower().ends_with(String(*ext))) {
				return false;
			}
		}

		// -- Compressed resource?

		if (p_data.size() >= 4 && p_data[0] == 'R' && p_data[1] == 'S' && p_data[2] == 'C' && p_data[3] == 'C') {
			// Already compressed
			return false;
		}

		// --- TODO: Decide on texture resources according to their image compression setting

		return true;
	}

	static Error save_appx_file(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total, const Vector<String> &p_enc_in_filters, const Vector<String> &p_enc_ex_filters, const Vector<uint8_t> &p_key) {
		AppxPackager *packager = (AppxPackager *)p_userdata;
		String dst_path = p_path.replace_first("res://", "game/");

		return packager->add_file(dst_path, p_data.ptr(), p_data.size(), p_file, p_total, _should_compress_asset(p_path, p_data));
	}

public:
	virtual String get_name() const override;
	virtual String get_os_name() const override;

	virtual List<String> get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const override;

	virtual Ref<Texture2D> get_logo() const override;

	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) override;

	virtual void get_export_options(List<ExportOption> *r_options) override;

	virtual bool can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const override;

	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0) override;

	virtual void get_platform_features(List<String> *r_features) override;

	virtual void resolve_platform_feature_priorities(const Ref<EditorExportPreset> &p_preset, Set<String> &p_features) override;

	EditorExportPlatformUWP();
};

#endif
