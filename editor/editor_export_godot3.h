/*************************************************************************/
/*  editor_export_godot3.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef EDITOR_EXPORT_GODOT3_H
#define EDITOR_EXPORT_GODOT3_H

#include "editor/editor_file_system.h"
#include "io/export_data.h"
#include "scene/main/node.h"

class EditorExportGodot3 {

	Map<String, int> pack_names;
	HashMap<Variant, int, VariantHasher> pack_values;

	int _pack_name(const String &p_name) {
		if (pack_names.has(p_name)) {
			return pack_names[p_name];
		}

		int idx = pack_names.size();
		pack_names[p_name] = idx;
		return idx;
	}

	int _pack_value(const Variant &p_value) {
		if (pack_values.has(p_value)) {
			return pack_values[p_value];
		}

		int idx = pack_values.size();
		pack_values[p_value] = idx;
		return idx;
	}

	Map<String, String> globals_rename_map;
	Map<String, String> prop_rename_map;
	Map<String, String> type_rename_map;
	Map<String, String> signal_rename_map;

	Map<String, String> resource_replace_map;

	String _replace_resource(const String &p_res) {
		if (resource_replace_map.has(p_res))
			return resource_replace_map[p_res];
		else
			return p_res;
	}

	Error _get_property_as_text(const Variant &p_variant, String &p_string);

	void _save_text(const String &p_path, ExportData &resource);

	void _save_binary_property(const Variant &p_property, FileAccess *f);

	void _save_binary(const String &p_path, ExportData &resource);
	void _save_config(const String &p_path);

	void _rename_properties(const String &p_type, List<ExportData::PropertyData> *p_props);
	void _add_new_properties(const String &p_type, List<ExportData::PropertyData> *p_props);
	void _convert_resources(ExportData &resource);
	void _unpack_packed_scene(ExportData &resource);
	void _pack_packed_scene(ExportData &resource);

	Error _convert_script(const String &p_path, const String &p_target_path, bool mark_converted_lines);

	void _find_files(EditorFileSystemDirectory *p_dir, List<String> *r_files);

public:
	Error export_godot3(const String &p_path, bool convert_scripts, bool mark_converted_lines);

	EditorExportGodot3();
};

#endif // EDITOR_EXPORT_GODOT3_H
