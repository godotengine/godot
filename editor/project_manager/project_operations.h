/**************************************************************************/
/*  project_operations.h                                                  */
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

#pragma once

#include "core/config/project_settings.h"
#include "core/error/error_list.h"
#include "core/io/config_file.h"
#include "core/string/ustring.h"
#include "editor/version_control/editor_vcs_interface.h"

class ProjectOperations {
public:
	static String resolve_origin_path(const String &p_path, bool &r_is_zip);
	static Error validate_origin_zip(const String &p_path, String &r_zip_root);
	static Error validate_origin_folder(const String &p_path);
	static Error validate_destination(const String &p_path, bool &r_is_folder_empty, const bool p_create_folder = false, bool *r_subdir_exists = nullptr);
	static bool is_destination_inside_origin(const String &p_destination_path, const String &p_origin_path);

	static Error create_project(const String &p_path, const HashMap<String, Variant> &p_settings, const EditorVCSInterface::VCSMetadata &p_vcs_metadata = EditorVCSInterface::VCSMetadata::NONE);
	static Error duplicate_project(const String &p_origin_path, const String &p_destination_path);
	static Error install_zip_project(const String &p_origin_path, const String &p_destination_path, const String &p_root_path, Vector<String> &r_failed_files);

	static Error create_project_dir(const String &p_path);
#ifdef ANDROID_ENABLED
	static void create_nomedia_file(const String &p_path);
#endif
	static void merge_project_settings(const HashMap<String, Variant> &p_added_settings, ProjectSettings::CustomMap &r_base_settings);
	static Error edit_existing_project_settings(const HashMap<String, Variant> &p_settings, const String &p_path, Ref<ConfigFile> p_project_godot);
};
