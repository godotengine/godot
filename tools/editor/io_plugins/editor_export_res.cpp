/*************************************************************************/
/*  editor_export_res.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "editor_export_res.h"

#include "core/io/resource_saver.h"

#include "tools/editor/editor_node.h"
#include "tools/editor/editor_settings.h"

void EditorExportResources::initialize() {

	String json_path = EditorSettings::get_singleton()->get_settings_path()+"/tmp/tmpexport.json";

	Vector<uint8_t> data = FileAccess::get_file_as_array(json_path);
	if (data.empty())
		return;

	cache_map.parse_json((const char *) data.ptr());
}

void EditorExportResources::finalize() {

	String json_path = EditorSettings::get_singleton()->get_settings_path()+"/tmp/tmpexport.json";

	FileAccess *f=FileAccess::open(json_path,FileAccess::READ_WRITE);
	if (f) {

		String json = cache_map.to_json();
		CharString utf8 = json.utf8();
		f->store_buffer((const uint8_t *) utf8.ptr(), utf8.length());
		memdelete(f);
	}
}

Vector<uint8_t> EditorExportResources::custom_export(String& p_path,const Ref<EditorExportPlatform> &p_platform) {

	// save xml resouces to binary format
	if (p_path.ends_with(".xml")) {

		if (cache_map.has(p_path)) {

			Dictionary info=cache_map[p_path];
			String md5 = info["md5"];
			String base_extension = info["base_extension"];

			String cache_path = EditorSettings::get_singleton()->get_settings_path()+"/tmp/"+md5+"."+base_extension;

			Vector<uint8_t> data = FileAccess::get_file_as_array(cache_path);
			if (!data.empty()) {

				print_line("CACHED: " + p_path);
				p_path = p_path.replace("xml", base_extension);
				return data;
			}
		}

		Ref<Resource> res = ResourceLoader::load(p_path, "", true);
		if(res.is_null())
			return Vector<uint8_t>();

		// Use file checksum(md5) for cache exported binary file
		String md5 = FileAccess::get_md5(p_path);
		String new_path = EditorSettings::get_singleton()->get_settings_path()+"/tmp/";
		new_path += md5 + "." + res->get_base_extension();

		int flg=0;
		if (EditorSettings::get_singleton()->get("on_save/compress_binary_resources"))
			flg|=ResourceSaver::FLAG_COMPRESS;
		if (EditorSettings::get_singleton()->get("on_save/save_paths_as_relative"))
			flg|=ResourceSaver::FLAG_RELATIVE_PATHS;

		if(ResourceSaver::save(new_path, res, flg) != OK)
			return Vector<uint8_t>();

		// Add cache information
		Dictionary d;
		d["md5"] = md5;
		d["base_extension"] = res->get_base_extension();
		cache_map[p_path] = d;

		Vector<uint8_t> data = FileAccess::get_file_as_array(new_path);
		p_path = p_path.replace("xml", res->get_base_extension());
		return data;
	}
	return Vector<uint8_t>();
}


