/*************************************************************************/
/*  export.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "export.h"

#include "editor/editor_export.h"
#include "platform/sailfish/logo.gen.h"
#include "scene/resources/texture.h"
#include "core/io/marshalls.h"
#include "core/io/zip_io.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "core/version.h"
#include "editor/editor_export.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"

class EditorExportPlatformSailfish : public EditorExportPlatform {
	GDCLASS(EditorExportPlatformSailfish, EditorExportPlatform)

	
	struct Device {
		String address;
		String name;
		String arch;
	};
public:
	EditorExportPlatformSailfish() {
		// Ref<Image> img = memnew(Image(_sailfish_logo));
		// logo.instance();
		// logo->create_from_image(img);
	}

	void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) override {

	}

	String get_os_name() const override {
		return "SailfishOS";
	}

	String get_name() const override {	
		return "SailfishOS";
	}

	void set_logo(Ref<Texture> logo) {
		this->logo = logo;
	}

	Ref<Texture> get_logo() const override {
		return logo;
	}

	void get_export_options(List<ExportOption> *r_options) override {
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "sailfish_sdk/sdk_path", PROPERTY_HINT_GLOBAL_DIR), ""));
		// r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "sailfish_sdk/arm_target", PROPERTY_HINT_ENUM), ""));
		// r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "sailfish_sdk/x86_target", PROPERTY_HINT_ENUM), ""));

		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_binary/arm", PROPERTY_HINT_GLOBAL_DIR), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_binary/arm_debug", PROPERTY_HINT_GLOBAL_DIR), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_binary/x86", PROPERTY_HINT_GLOBAL_DIR), ""));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "custom_binary/x86_debug", PROPERTY_HINT_GLOBAL_DIR), ""));

		r_options->push_back(ExportOption(PropertyInfo(Variant::INT,    "version/release", PROPERTY_HINT_RANGE, "1,40096,1,or_greater"), 1));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "version/string", PROPERTY_HINT_PLACEHOLDER_TEXT, "1.0.0"), "1.0.0"));

		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/name", PROPERTY_HINT_PLACEHOLDER_TEXT, "harbour-$genname"), "harbour-$genname"));
		r_options->push_back(ExportOption(PropertyInfo(Variant::STRING, "package/game_name", PROPERTY_HINT_PLACEHOLDER_TEXT, "Game Name [default if blank]"), ""));
	}

	bool can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const override {
		return true;
	}
	List<String> get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const override {
		List<String> ext;
		return ext;
	}
	Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0) override {
		return Error::OK;
	}
	void get_platform_features(List<String> *r_features)override {

	}
	void resolve_platform_feature_priorities(const Ref<EditorExportPreset> &p_preset, Set<String> &p_features) override {

	}
protected:
	Ref<ImageTexture> logo;
};

void register_sailfish_exporter() {

	Ref<EditorExportPlatformSailfish> platform;
	Ref<EditorExportPlatformPC> p;
	platform.instance();

	Ref<Image> img = memnew(Image(_sailfish_logo));
	Ref<ImageTexture> logo;
	logo.instance();
	logo->create_from_image(img);
	platform->set_logo(logo);
	// platform->set_name("SailfishOS/SDL");
	// p->set_extension("arm", "binary_format/arm");
	// platform->set_extension("x86", "binary_format/i486");
	// platform->set_release_32("godot.sailfish.opt.arm");
	// platform->set_debug_32("godot.sailfish.opt.debug.arm");
	// platform->set_release_64("godot.sailfish.opt.x86");
	// platform->set_debug_64("godot.sailfish.opt.debug.x86");
	// platform->set_os_name("SailfishOS");
	// platform->set_chmod_flags(0755);

	EDITOR_DEF("export/sailfish/sdk_path", "");
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::STRING, "export/sailfish/sdk_path", PROPERTY_HINT_GLOBAL_DIR));

	EditorExport::get_singleton()->add_export_platform(platform);
}
