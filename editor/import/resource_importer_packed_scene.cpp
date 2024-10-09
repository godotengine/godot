/**************************************************************************/
/*  resource_importer_packed_scene.cpp                                    */
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

#include "resource_importer_packed_scene.h"

String ResourceImporterPackedScene::get_importer_name() const {
	return "PackedScene";
}

String ResourceImporterPackedScene::get_visible_name() const {
	return "PackedScene";
}

void ResourceImporterPackedScene::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("tscn");
}

String ResourceImporterPackedScene::get_save_extension() const {
	return "rtscn";
}

String ResourceImporterPackedScene::get_resource_type() const {
	return "PackedScene";
}

bool ResourceImporterPackedScene::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	return true;
}

bool ResourceImporterPackedScene::is_read_only() const {
	return false;
}

int ResourceImporterPackedScene::get_import_order() const {
	return ResourceFormatImporter::IMPORT_ORDER_LOW_PRIORITY;
}

int ResourceImporterPackedScene::get_preset_count() const {
	return 0;
}

String ResourceImporterPackedScene::get_preset_name(int p_idx) const {
	return String();
}

void ResourceImporterPackedScene::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
}

Error ResourceImporterPackedScene::import(const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	//make an empty import file in .godot/imported folder (won't be used during export)
	Error err;
	Ref<FileAccess> f = FileAccess::open(p_save_path + ".rtscn", FileAccess::WRITE, &err);
	ERR_FAIL_COND_V_MSG(err, ERR_CANT_OPEN, "Cannot save packed scene import file '" + p_save_path + "'.rtscn");

	return OK;
}

ResourceImporterPackedScene::ResourceImporterPackedScene() {
}
