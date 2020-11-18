/*************************************************************************/
/*  gradle_export_util.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GODOT_GRADLE_EXPORT_UTIL_H
#define GODOT_GRADLE_EXPORT_UTIL_H

#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/zip_io.h"
#include "core/os/os.h"
#include "editor/editor_export.h"

const String godot_project_name_xml_string = R"(<?xml version="1.0" encoding="utf-8"?>
<!--WARNING: THIS FILE WILL BE OVERWRITTEN AT BUILD TIME-->
<resources>
	<string name="godot_project_name_string">%s</string>
</resources>
)";

struct CustomExportData {
	String assets_directory;
	bool debug;
	Vector<String> libs;
};

int _get_android_orientation_value(DisplayServer::ScreenOrientation screen_orientation);

String _get_android_orientation_label(DisplayServer::ScreenOrientation screen_orientation);

// Utility method used to create a directory.
Error create_directory(const String &p_dir);

// Writes p_data into a file at p_path, creating directories if necessary.
// Note: this will overwrite the file at p_path if it already exists.
Error store_file_at_path(const String &p_path, const Vector<uint8_t> &p_data);

// Writes string p_data into a file at p_path, creating directories if necessary.
// Note: this will overwrite the file at p_path if it already exists.
Error store_string_at_path(const String &p_path, const String &p_data);

// Implementation of EditorExportSaveFunction.
// This method will only be called as an input to export_project_files.
// It is used by the export_project_files method to save all the asset files into the gradle project.
// It's functionality mirrors that of the method save_apk_file.
// This method will be called ONLY when custom build is enabled.
Error rename_and_store_file_in_gradle_project(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total, const Vector<String> &p_enc_in_filters, const Vector<String> &p_enc_ex_filters, const Vector<uint8_t> &p_key);

// Creates strings.xml files inside the gradle project for different locales.
Error _create_project_name_strings_files(const Ref<EditorExportPreset> &p_preset, const String &project_name);
//Error _create_project_name_strings_files(const Ref<EditorExportPreset> &p_preset, const String &project_name) {
//	// Stores the string into the default values directory.
//	String processed_default_xml_string = vformat(godot_project_name_xml_string, project_name.xml_escape(true));
//	store_string_at_path("res://android/build/res/values/godot_project_name_string.xml", processed_default_xml_string);

//	// Searches the Gradle project res/ directory to find all supported locales
//	DirAccessRef da = DirAccess::open("res://android/build/res");
//	if (!da) {
//		return ERR_CANT_OPEN;
//	}
//	da->list_dir_begin();
//	while (true) {
//		String file = da->get_next();
//		if (file == "") {
//			break;
//		}
//		if (!file.begins_with("values-")) {
//			// NOTE: This assumes all directories that start with "values-" are for localization.
//			continue;
//		}
//		String locale = file.replace("values-", "").replace("-r", "_");
//		String property_name = "application/config/name_" + locale;
//		String locale_directory = "res://android/build/res/" + file + "/godot_project_name_string.xml";
//		if (ProjectSettings::get_singleton()->has_setting(property_name)) {
//			String locale_project_name = ProjectSettings::get_singleton()->get(property_name);
//			String processed_xml_string = vformat(godot_project_name_xml_string, locale_project_name.xml_escape(true));
//			store_string_at_path(locale_directory, processed_xml_string);
//		} else {
//			// TODO: Once the legacy build system is deprecated we don't need to have xml files for this else branch
//			store_string_at_path(locale_directory, processed_default_xml_string);
//		}
//	}
//	da->list_dir_end();
//	return OK;
//}

//String bool_to_string(bool v) {
//	return v ? "true" : "false";
//}

//String _get_gles_tag() {
//	bool min_gles3 = ProjectSettings::get_singleton()->get("rendering/driver/driver_name") == "GLES3" &&
//					 !ProjectSettings::get_singleton()->get("rendering/quality/driver/fallback_to_gles2");
//	return min_gles3 ? "    <uses-feature android:glEsVersion=\"0x00030000\" android:required=\"true\" />\n" : "";
//}

String bool_to_string(bool v);

String _get_gles_tag();

String _get_screen_sizes_tag(const Ref<EditorExportPreset> &p_preset);

String _get_xr_features_tag(const Ref<EditorExportPreset> &p_preset);

String _get_instrumentation_tag(const Ref<EditorExportPreset> &p_preset);

String _get_activity_tag(const Ref<EditorExportPreset> &p_preset);

String _get_application_tag(const Ref<EditorExportPreset> &p_preset, bool p_has_storage_permission);

#endif //GODOT_GRADLE_EXPORT_UTIL_H
