#ifndef EDITOR_EXPORT_GODOT3_H
#define EDITOR_EXPORT_GODOT3_H

#include "scene/main/node.h"
#include "tools/editor/editor_file_system.h"
#include "io/export_data.h"

class EditorExportGodot3 {

	Map<String,int> pack_names;
	HashMap<Variant,int,VariantHasher> pack_values;

	int _pack_name(const String& p_name) {
		if (pack_names.has(p_name)) {
			return pack_names[p_name];
		}

		int idx = pack_names.size();
		pack_names[p_name]=idx;
		return idx;
	}

	int _pack_value(const Variant& p_value) {
		if (pack_values.has(p_value)) {
			return pack_values[p_value];
		}

		int idx = pack_values.size();
		pack_values[p_value]=idx;
		return idx;
	}

	Map<String,String>  prop_rename_map;
	Map<String,String>  type_rename_map;
	Map<String,String>  signal_rename_map;

	Map<String,String> resource_replace_map;

	String _replace_resource(const String& p_res) {
		if (resource_replace_map.has(p_res))
			return resource_replace_map[p_res];
		else
			return p_res;
	}

	Error _get_property_as_text(const Variant& p_variant,String&p_string);

	void _save_text(const String& p_path,ExportData &resource);

	void _save_binary_property(const Variant& p_property,FileAccess *f);

	void _save_binary(const String& p_path,ExportData &resource);
	void _save_config(const String &p_path);

	void _rename_properties(const String& p_type,List<ExportData::PropertyData> *p_props);
	void _convert_resources(ExportData &resource);
	void _unpack_packed_scene(ExportData &resource);
	void _pack_packed_scene(ExportData &resource);

	void _find_files(EditorFileSystemDirectory *p_dir, List<String> * r_files);
public:


	Error export_godot3(const String& p_path);

	EditorExportGodot3();
};

#endif // EDITOR_EXPORT_GODOT3_H
