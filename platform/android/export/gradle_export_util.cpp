/**************************************************************************/
/*  gradle_export_util.cpp                                                */
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

#include "gradle_export_util.h"

#include "core/string/translation_server.h"
#include "modules/regex/regex.h"

int _get_android_orientation_value(DisplayServer::ScreenOrientation screen_orientation) {
	switch (screen_orientation) {
		case DisplayServer::SCREEN_PORTRAIT:
			return 1;
		case DisplayServer::SCREEN_REVERSE_LANDSCAPE:
			return 8;
		case DisplayServer::SCREEN_REVERSE_PORTRAIT:
			return 9;
		case DisplayServer::SCREEN_SENSOR_LANDSCAPE:
			return 11;
		case DisplayServer::SCREEN_SENSOR_PORTRAIT:
			return 12;
		case DisplayServer::SCREEN_SENSOR:
			return 13;
		case DisplayServer::SCREEN_LANDSCAPE:
		default:
			return 0;
	}
}

String _get_android_orientation_label(DisplayServer::ScreenOrientation screen_orientation) {
	switch (screen_orientation) {
		case DisplayServer::SCREEN_PORTRAIT:
			return "portrait";
		case DisplayServer::SCREEN_REVERSE_LANDSCAPE:
			return "reverseLandscape";
		case DisplayServer::SCREEN_REVERSE_PORTRAIT:
			return "reversePortrait";
		case DisplayServer::SCREEN_SENSOR_LANDSCAPE:
			return "userLandscape";
		case DisplayServer::SCREEN_SENSOR_PORTRAIT:
			return "userPortrait";
		case DisplayServer::SCREEN_SENSOR:
			return "fullUser";
		case DisplayServer::SCREEN_LANDSCAPE:
		default:
			return "landscape";
	}
}

int _get_app_category_value(int category_index) {
	switch (category_index) {
		case APP_CATEGORY_ACCESSIBILITY:
			return 8;
		case APP_CATEGORY_AUDIO:
			return 1;
		case APP_CATEGORY_IMAGE:
			return 3;
		case APP_CATEGORY_MAPS:
			return 6;
		case APP_CATEGORY_NEWS:
			return 5;
		case APP_CATEGORY_PRODUCTIVITY:
			return 7;
		case APP_CATEGORY_SOCIAL:
			return 4;
		case APP_CATEGORY_UNDEFINED:
			return -1;
		case APP_CATEGORY_VIDEO:
			return 2;
		case APP_CATEGORY_GAME:
		default:
			return 0;
	}
}

String _get_app_category_label(int category_index) {
	switch (category_index) {
		case APP_CATEGORY_ACCESSIBILITY:
			return "accessibility";
		case APP_CATEGORY_AUDIO:
			return "audio";
		case APP_CATEGORY_IMAGE:
			return "image";
		case APP_CATEGORY_MAPS:
			return "maps";
		case APP_CATEGORY_NEWS:
			return "news";
		case APP_CATEGORY_PRODUCTIVITY:
			return "productivity";
		case APP_CATEGORY_SOCIAL:
			return "social";
		case APP_CATEGORY_VIDEO:
			return "video";
		case APP_CATEGORY_GAME:
		default:
			return "game";
	}
}

// Utility method used to create a directory.
Error create_directory(const String &p_dir) {
	if (!DirAccess::exists(p_dir)) {
		Ref<DirAccess> filesystem_da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		ERR_FAIL_COND_V_MSG(filesystem_da.is_null(), ERR_CANT_CREATE, "Cannot create directory '" + p_dir + "'.");
		Error err = filesystem_da->make_dir_recursive(p_dir);
		ERR_FAIL_COND_V_MSG(err, ERR_CANT_CREATE, "Cannot create directory '" + p_dir + "'.");
	}
	return OK;
}

// Writes p_data into a file at p_path, creating directories if necessary.
// Note: this will overwrite the file at p_path if it already exists.
Error store_file_at_path(const String &p_path, const Vector<uint8_t> &p_data) {
	String dir = p_path.get_base_dir();
	Error err = create_directory(dir);
	if (err != OK) {
		return err;
	}
	Ref<FileAccess> fa = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(fa.is_null(), ERR_CANT_CREATE, "Cannot create file '" + p_path + "'.");
	fa->store_buffer(p_data.ptr(), p_data.size());
	return OK;
}

// Writes string p_data into a file at p_path, creating directories if necessary.
// Note: this will overwrite the file at p_path if it already exists.
Error store_string_at_path(const String &p_path, const String &p_data) {
	String dir = p_path.get_base_dir();
	Error err = create_directory(dir);
	if (err != OK) {
		if (OS::get_singleton()->is_stdout_verbose()) {
			print_error("Unable to write data into " + p_path);
		}
		return err;
	}
	Ref<FileAccess> fa = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(fa.is_null(), ERR_CANT_CREATE, "Cannot create file '" + p_path + "'.");
	fa->store_string(p_data);
	return OK;
}

// Implementation of EditorExportPlatform::SaveFileFunction.
// This method will only be called as an input to export_project_files.
// It is used by the export_project_files method to save all the asset files into the gradle project.
// It's functionality mirrors that of the method save_apk_file.
// This method will be called ONLY when gradle build is enabled.
Error rename_and_store_file_in_gradle_project(const Ref<EditorExportPreset> &p_preset, void *p_userdata, const EditorExportPlatform::SaveFileInfo &p_info, const Vector<uint8_t> &p_data) {
	CustomExportData *export_data = static_cast<CustomExportData *>(p_userdata);

	const String simplified_path = EditorExportPlatform::simplify_path(p_info.path);

	Vector<uint8_t> enc_data;
	EditorExportPlatform::SavedData sd;
	Error err = _store_temp_file(p_preset, simplified_path, p_data, enc_data, sd);
	if (err != OK) {
		return err;
	}

	const String dst_path = export_data->assets_directory + String("/") + simplified_path.trim_prefix("res://");
	print_verbose("Saving project files from " + simplified_path + " into " + dst_path);
	err = store_file_at_path(dst_path, enc_data);

	export_data->pd.file_ofs.push_back(sd);
	return err;
}

String _android_xml_escape(const String &p_string) {
	// Android XML requires strings to be both valid XML (`xml_escape()`) but also
	// to escape characters which are valid XML but have special meaning in Android XML.
	// https://developer.android.com/guide/topics/resources/string-resource.html#FormattingAndStyling
	// Note: Didn't handle U+XXXX unicode chars, could be done if needed.
	return p_string
			.replace("@", "\\@")
			.replace("?", "\\?")
			.replace("'", "\\'")
			.replace("\"", "\\\"")
			.replace("\n", "\\n")
			.replace("\t", "\\t")
			.xml_escape(false);
}

// Creates strings.xml files inside the gradle project for different locales.
Error _create_project_name_strings_files(const Ref<EditorExportPreset> &p_preset, const String &p_project_name, const String &p_gradle_build_dir, const Dictionary &p_appnames) {
	print_verbose("Creating strings resources for supported locales for project " + p_project_name);
	// Stores the string into the default values directory.
	String processed_default_xml_string = vformat(GODOT_PROJECT_NAME_XML_STRING, _android_xml_escape(p_project_name));
	store_string_at_path(p_gradle_build_dir.path_join("res/values/godot_project_name_string.xml"), processed_default_xml_string);

	// Searches the Gradle project res/ directory to find all supported locales
	Ref<DirAccess> da = DirAccess::open(p_gradle_build_dir.path_join("res"));
	if (da.is_null()) {
		if (OS::get_singleton()->is_stdout_verbose()) {
			print_error("Unable to open Android resources directory.");
		}
		return ERR_CANT_OPEN;
	}

	// Setup a temporary translation domain to translate the project name.
	const StringName domain_name = "godot.project_name_localization";
	Ref<TranslationDomain> domain = TranslationServer::get_singleton()->get_or_add_domain(domain_name);
	TranslationServer::get_singleton()->load_project_translations(domain);

	da->list_dir_begin();
	while (true) {
		String file = da->get_next();
		if (file.is_empty()) {
			break;
		}
		if (!file.begins_with("values-")) {
			// NOTE: This assumes all directories that start with "values-" are for localization.
			continue;
		}
		String locale = file.replace("values-", "").replace("-r", "_");
		String locale_directory = p_gradle_build_dir.path_join("res/" + file + "/godot_project_name_string.xml");

		String locale_project_name;
		if (p_appnames.is_empty()) {
			domain->set_locale_override(locale);
			locale_project_name = domain->translate(p_project_name, String());
		} else {
			locale_project_name = p_appnames.get(locale, p_project_name);
		}
		if (locale_project_name != p_project_name) {
			String processed_xml_string = vformat(GODOT_PROJECT_NAME_XML_STRING, _android_xml_escape(locale_project_name));
			print_verbose("Storing project name for locale " + locale + " under " + locale_directory);
			store_string_at_path(locale_directory, processed_xml_string);
		} else {
			// TODO: Once the legacy build system is deprecated we don't need to have xml files for this else branch
			store_string_at_path(locale_directory, processed_default_xml_string);
		}
	}
	da->list_dir_end();

	TranslationServer::get_singleton()->remove_domain(domain_name);

	return OK;
}

String bool_to_string(bool v) {
	return v ? "true" : "false";
}

String _get_gles_tag() {
	return "    <uses-feature android:glEsVersion=\"0x00030000\" android:required=\"true\" />\n";
}

String _get_screen_sizes_tag(const Ref<EditorExportPreset> &p_preset) {
	String manifest_screen_sizes = "    <supports-screens \n        tools:node=\"replace\"";
	String sizes[] = { "small", "normal", "large", "xlarge" };
	constexpr size_t num_sizes = std_size(sizes);
	for (size_t i = 0; i < num_sizes; i++) {
		String feature_name = vformat("screen/support_%s", sizes[i]);
		String feature_support = bool_to_string(p_preset->get(feature_name));
		String xml_entry = vformat("\n        android:%sScreens=\"%s\"", sizes[i], feature_support);
		manifest_screen_sizes += xml_entry;
	}
	manifest_screen_sizes += " />\n";
	return manifest_screen_sizes;
}

String _get_activity_tag(const Ref<EditorExportPlatform> &p_export_platform, const Ref<EditorExportPreset> &p_preset, bool p_debug) {
	String export_plugins_activity_element_contents;
	Vector<Ref<EditorExportPlugin>> export_plugins = EditorExport::get_singleton()->get_export_plugins();
	for (int i = 0; i < export_plugins.size(); i++) {
		if (export_plugins[i]->supports_platform(p_export_platform)) {
			const String contents = export_plugins[i]->get_android_manifest_activity_element_contents(p_export_platform, p_debug);
			if (!contents.is_empty()) {
				export_plugins_activity_element_contents += contents;
				export_plugins_activity_element_contents += "\n";
			}
		}
	}

	// Update the GodotApp activity tag.
	String orientation = _get_android_orientation_label(DisplayServer::ScreenOrientation(int(p_export_platform->get_project_setting(p_preset, "display/window/handheld/orientation"))));
	String manifest_activity_text = vformat(
			"        <activity android:name=\".GodotApp\" "
			"tools:replace=\"android:screenOrientation,android:excludeFromRecents,android:resizeableActivity\" "
			"tools:node=\"mergeOnlyAttributes\" "
			"android:excludeFromRecents=\"%s\" "
			"android:screenOrientation=\"%s\" "
			"android:resizeableActivity=\"%s\">\n",
			bool_to_string(p_preset->get("package/exclude_from_recents")),
			orientation,
			bool_to_string(bool(p_export_platform->get_project_setting(p_preset, "display/window/size/resizable"))));

	// *LAUNCHER and *HOME categories should only go to the activity-alias.
	Ref<RegEx> activity_content_to_remove_regex = RegEx::create_from_string(R"delim(<category\s+android:name\s*=\s*"\S+(LAUNCHER|HOME)"\s*\/>)delim");
	String updated_export_plugins_activity_element_contents = activity_content_to_remove_regex->sub(export_plugins_activity_element_contents, "", true);
	manifest_activity_text += updated_export_plugins_activity_element_contents;

	manifest_activity_text += "        </activity>\n";

	// Update the GodotAppLauncher activity tag.
	manifest_activity_text += "        <activity-alias\n"
							  "            tools:node=\"mergeOnlyAttributes\"\n"
							  "            android:name=\".GodotAppLauncher\"\n"
							  "            android:targetActivity=\".GodotApp\"\n"
							  "            android:exported=\"true\">\n";

	manifest_activity_text += "            <intent-filter>\n"
							  "                <action android:name=\"android.intent.action.MAIN\" />\n"
							  "                <category android:name=\"android.intent.category.DEFAULT\" />\n";

	bool show_in_app_library = p_preset->get("package/show_in_app_library");
	if (show_in_app_library) {
		manifest_activity_text += "                <category android:name=\"android.intent.category.LAUNCHER\" />\n";
	}

	bool uses_leanback_category = p_preset->get("package/show_in_android_tv");
	if (uses_leanback_category) {
		manifest_activity_text += "                <category android:name=\"android.intent.category.LEANBACK_LAUNCHER\" />\n";
	}

	bool uses_home_category = p_preset->get("package/show_as_launcher_app");
	if (uses_home_category) {
		manifest_activity_text += "                <category android:name=\"android.intent.category.HOME\" />\n";
	}

	manifest_activity_text += "            </intent-filter>\n";

	// Hybrid categories should only go to the actual 'GodotApp' activity.
	Ref<RegEx> activity_alias_content_to_remove_regex = RegEx::create_from_string(R"delim(<category\s+android:name\s*=\s*"org.godotengine.xr.hybrid.(IMMERSIVE|PANEL)"\s*\/>)delim");
	String updated_export_plugins_activity_alias_element_contents = activity_alias_content_to_remove_regex->sub(export_plugins_activity_element_contents, "", true);
	manifest_activity_text += updated_export_plugins_activity_alias_element_contents;

	manifest_activity_text += "        </activity-alias>\n";
	return manifest_activity_text;
}

String _get_application_tag(const Ref<EditorExportPlatform> &p_export_platform, const Ref<EditorExportPreset> &p_preset, bool p_has_read_write_storage_permission, bool p_debug, const Vector<MetadataInfo> &p_metadata) {
	int app_category_index = (int)(p_preset->get("package/app_category"));
	bool is_game = app_category_index == APP_CATEGORY_GAME;

	String manifest_application_text = vformat(
			"    <application android:label=\"@string/godot_project_name_string\"\n"
			"        android:allowBackup=\"%s\"\n"
			"        android:icon=\"@mipmap/icon\"\n"
			"        android:isGame=\"%s\"\n"
			"        android:hasFragileUserData=\"%s\"\n"
			"        android:requestLegacyExternalStorage=\"%s\"\n",
			bool_to_string(p_preset->get("user_data_backup/allow")),
			bool_to_string(is_game),
			bool_to_string(p_preset->get("package/retain_data_on_uninstall")),
			bool_to_string(p_has_read_write_storage_permission));
	if (app_category_index != APP_CATEGORY_UNDEFINED) {
		manifest_application_text += vformat("        android:appCategory=\"%s\"\n", _get_app_category_label(app_category_index));
		manifest_application_text += "        tools:replace=\"android:allowBackup,android:appCategory,android:isGame,android:hasFragileUserData,android:requestLegacyExternalStorage\"\n";
	} else {
		manifest_application_text += "        tools:remove=\"android:appCategory\"\n";
		manifest_application_text += "        tools:replace=\"android:allowBackup,android:isGame,android:hasFragileUserData,android:requestLegacyExternalStorage\"\n";
	}
	manifest_application_text += "        tools:ignore=\"GoogleAppIndexingWarning\">\n\n";

	for (int i = 0; i < p_metadata.size(); i++) {
		manifest_application_text += vformat("        <meta-data tools:node=\"replace\" android:name=\"%s\" android:value=\"%s\" />\n", p_metadata[i].name, p_metadata[i].value);
	}

	Vector<Ref<EditorExportPlugin>> export_plugins = EditorExport::get_singleton()->get_export_plugins();
	for (int i = 0; i < export_plugins.size(); i++) {
		if (export_plugins[i]->supports_platform(p_export_platform)) {
			const String contents = export_plugins[i]->get_android_manifest_application_element_contents(p_export_platform, p_debug);
			if (!contents.is_empty()) {
				manifest_application_text += contents;
				manifest_application_text += "\n";
			}
		}
	}

	manifest_application_text += _get_activity_tag(p_export_platform, p_preset, p_debug);
	manifest_application_text += "    </application>\n";
	return manifest_application_text;
}

Error _store_temp_file(const Ref<EditorExportPreset> &p_preset, const String &p_simplified_path, const Vector<uint8_t> &p_data, Vector<uint8_t> &r_enc_data, EditorExportPlatform::SavedData &r_sd) {
	Error err = OK;
	Ref<FileAccess> ftmp = FileAccess::create_temp(FileAccess::WRITE_READ, "export", "tmp", false, &err);
	if (err != OK) {
		return err;
	}
	r_sd.path_utf8 = p_simplified_path.trim_prefix("res://").utf8();
	r_sd.ofs = 0;
	r_sd.size = p_data.size();
	err = EditorExportPlatform::_encrypt_and_store_data(ftmp, p_preset, p_simplified_path, p_data, r_sd.encrypted);
	if (err != OK) {
		return err;
	}

	r_enc_data.resize(ftmp->get_length());
	ftmp->seek(0);
	ftmp->get_buffer(r_enc_data.ptrw(), r_enc_data.size());
	ftmp.unref();

	// Store MD5 of original file.
	{
		unsigned char hash[16];
		CryptoCore::md5(p_data.ptr(), p_data.size(), hash);
		r_sd.md5.resize(16);
		for (int i = 0; i < 16; i++) {
			r_sd.md5.write[i] = hash[i];
		}
	}
	return OK;
}
