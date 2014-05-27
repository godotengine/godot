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

Vector<uint8_t> EditorExportResources::custom_export(String& p_path,const Ref<EditorExportPlatform> &p_platform) {

	// save xml resouces to binary format
	if (p_path.ends_with(".xml")) {
		Ref<Resource> res = ResourceLoader::load(p_path, "", true);
		if(res.is_null())
			return Vector<uint8_t>();

		int flg=0;
		if (EditorSettings::get_singleton()->get("on_save/compress_binary_resources"))
			flg|=ResourceSaver::FLAG_COMPRESS;
		if (EditorSettings::get_singleton()->get("on_save/save_paths_as_relative"))
			flg|=ResourceSaver::FLAG_RELATIVE_PATHS;

		String new_path = EditorSettings::get_singleton()->get_settings_path()+"/tmp/";
		DirAccess *dir = DirAccess::open(new_path);
		new_path += "tmp_export_res." + res->get_base_extension();

		if(ResourceSaver::save(new_path, res, flg) == OK) {

			Vector<uint8_t> data = FileAccess::get_file_as_array(new_path);

			dir->remove(new_path);
			memdelete(dir);

			p_path = p_path.replace("xml", res->get_base_extension());
			return data;
		}
	}
	return Vector<uint8_t>();
}


