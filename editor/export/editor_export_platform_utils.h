/**************************************************************************/
/*  editor_export_platform_utils.h                                        */
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

#include "core/io/dir_access.h"
#include "editor/export/editor_export_platform_data.h"
#include "editor/export/editor_export_preset.h"

class EditorExportPlatformUtils {
public:
	struct AsyncPckFileDependenciesState {
	private:
		void _get_file_dependencies_of(const String &p_file, HashMap<String, const HashSet<String> *> &p_dependencies);

	public:
		List<HashSet<String>> file_dependencies_lists;
		HashMap<String, HashSet<String> *> file_dependencies;

		void add_to_file_dependencies(const String &p_file);
		void add_to_file_dependencies(const HashSet<String> &p_file_set);
		HashMap<String, const HashSet<String> *> get_file_dependencies_of(const HashSet<String> &p_file_set);
		HashMap<String, const HashSet<String> *> get_file_dependencies_of(const String &p_file);

		void clear() { file_dependencies.clear(); }
	};

	// Get the
	static String get_path_from_dependency(const String &p_dependency);

	static int get_pad(int p_alignment, int p_n);

	static Variant get_project_setting(const Ref<EditorExportPreset> &p_preset, const StringName &p_name);

	static bool encrypt_and_store_directory(Ref<FileAccess> p_fd, EditorExportPlatformData::PackData &p_pack_data, const Vector<uint8_t> &p_key, uint64_t p_seed, uint64_t p_file_base);
	static Error encrypt_and_store_data(Ref<FileAccess> p_fd, const String &p_path, const Vector<uint8_t> &p_data, const Vector<String> &p_enc_in_filters, const Vector<String> &p_enc_ex_filters, const Vector<uint8_t> &p_key, uint64_t p_seed, bool &r_encrypt);

	static Error store_temp_file(const String &p_simplified_path, const PackedByteArray &p_data, const PackedStringArray &p_encoded_included_filters, const PackedStringArray &p_encoded_excluded_filters, const Vector<uint8_t> &p_key, uint64_t p_seed, bool p_delta, PackedByteArray &r_encoded_data, EditorExportPlatformData::SavedData &r_saved_data);

	// Utility method used to create a directory.
	static Error create_directory(const String &p_dir);

	// Writes p_data into a file at p_path, creating directories if necessary.
	// Note: this will overwrite the file at p_path if it already exists.
	static Error store_file_at_path(const String &p_path, const PackedByteArray &p_data);

	// Writes string p_data into a file at p_path, creating directories if necessary.
	// Note: this will overwrite the file at p_path if it already exists.
	static Error store_string_at_path(const String &p_path, const String &p_data);

	// Converts script encryption key to bytes.
	static PackedByteArray convert_string_encryption_key_to_bytes(const String &p_encryption_key);

	// Finds resource files from a file system directory.
	static void export_find_resources(EditorFileSystemDirectory *p_dir, HashSet<String> &p_paths);

	//
	static void export_find_customized_resources(const Ref<EditorExportPreset> &p_preset, EditorFileSystemDirectory *p_dir, EditorExportPreset::FileExportMode p_mode, HashSet<String> &p_paths);

	//
	static void export_find_dependencies(const String &p_path, HashSet<String> &p_paths);

	static void export_find_preset_resources(const Ref<EditorExportPreset> &p_preset, HashSet<String> &p_paths);

	static void edit_files_with_filter(Ref<DirAccess> &da, const Vector<String> &p_filters, HashSet<String> &r_list, bool exclude);

	static void edit_filter_list(HashSet<String> &r_list, const String &p_filter, bool exclude);

	static Vector<uint8_t> filter_extension_list_config_file(const String &p_config_path, const HashSet<String> &p_paths);

	static Vector<String> get_forced_export_files(const Ref<EditorExportPreset> &p_preset);

	static HashMap<String, PackedByteArray> get_internal_export_files(const Ref<EditorExportPlatform> &p_editor_export_platform, const Ref<EditorExportPreset> &p_preset, bool p_debug);
};
