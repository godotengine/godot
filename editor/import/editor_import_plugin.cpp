/*************************************************************************/
/*  editor_import_plugin.cpp                                             */
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

#include "editor_import_plugin.h"
#include "core/object/script_language.h"

EditorImportPlugin::EditorImportPlugin() {
}

String EditorImportPlugin::get_importer_name() const {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("_get_importer_name")), "");
	return get_script_instance()->call("_get_importer_name");
}

String EditorImportPlugin::get_visible_name() const {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("_get_visible_name")), "");
	return get_script_instance()->call("_get_visible_name");
}

void EditorImportPlugin::get_recognized_extensions(List<String> *p_extensions) const {
	ERR_FAIL_COND(!(get_script_instance() && get_script_instance()->has_method("_get_recognized_extensions")));
	Array extensions = get_script_instance()->call("_get_recognized_extensions");
	for (int i = 0; i < extensions.size(); i++) {
		p_extensions->push_back(extensions[i]);
	}
}

String EditorImportPlugin::get_preset_name(int p_idx) const {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("_get_preset_name")), "");
	return get_script_instance()->call("_get_preset_name", p_idx);
}

int EditorImportPlugin::get_preset_count() const {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("_get_preset_count")), 0);
	return get_script_instance()->call("_get_preset_count");
}

String EditorImportPlugin::get_save_extension() const {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("_get_save_extension")), "");
	return get_script_instance()->call("_get_save_extension");
}

String EditorImportPlugin::get_resource_type() const {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("_get_resource_type")), "");
	return get_script_instance()->call("_get_resource_type");
}

float EditorImportPlugin::get_priority() const {
	if (!(get_script_instance() && get_script_instance()->has_method("_get_priority"))) {
		return ResourceImporter::get_priority();
	}
	return get_script_instance()->call("_get_priority");
}

int EditorImportPlugin::get_import_order() const {
	if (!(get_script_instance() && get_script_instance()->has_method("_get_import_order"))) {
		return ResourceImporter::get_import_order();
	}
	return get_script_instance()->call("_get_import_order");
}

void EditorImportPlugin::get_import_options(List<ResourceImporter::ImportOption> *r_options, int p_preset) const {
	ERR_FAIL_COND(!(get_script_instance() && get_script_instance()->has_method("_get_import_options")));
	Array needed;
	needed.push_back("name");
	needed.push_back("default_value");
	Array options = get_script_instance()->call("_get_import_options", p_preset);
	for (int i = 0; i < options.size(); i++) {
		Dictionary d = options[i];
		ERR_FAIL_COND(!d.has_all(needed));
		String name = d["name"];
		Variant default_value = d["default_value"];

		PropertyHint hint = PROPERTY_HINT_NONE;
		if (d.has("property_hint")) {
			hint = (PropertyHint)d["property_hint"].operator int64_t();
		}

		String hint_string;
		if (d.has("hint_string")) {
			hint_string = d["hint_string"];
		}

		uint32_t usage = PROPERTY_USAGE_DEFAULT;
		if (d.has("usage")) {
			usage = d["usage"];
		}

		ImportOption option(PropertyInfo(default_value.get_type(), name, hint, hint_string, usage), default_value);
		r_options->push_back(option);
	}
}

bool EditorImportPlugin::get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("_get_option_visibility")), true);
	Dictionary d;
	Map<StringName, Variant>::Element *E = p_options.front();
	while (E) {
		d[E->key()] = E->get();
		E = E->next();
	}
	return get_script_instance()->call("_get_option_visibility", p_option, d);
}

Error EditorImportPlugin::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	ERR_FAIL_COND_V(!(get_script_instance() && get_script_instance()->has_method("_import")), ERR_UNAVAILABLE);
	Dictionary options;
	Array platform_variants, gen_files;

	Map<StringName, Variant>::Element *E = p_options.front();
	while (E) {
		options[E->key()] = E->get();
		E = E->next();
	}
	Error err = (Error)get_script_instance()->call("_import", p_source_file, p_save_path, options, platform_variants, gen_files).operator int64_t();

	for (int i = 0; i < platform_variants.size(); i++) {
		r_platform_variants->push_back(platform_variants[i]);
	}
	for (int i = 0; i < gen_files.size(); i++) {
		r_gen_files->push_back(gen_files[i]);
	}
	return err;
}

void EditorImportPlugin::_bind_methods() {
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_importer_name"));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_visible_name"));
	BIND_VMETHOD(MethodInfo(Variant::INT, "_get_preset_count"));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_preset_name", PropertyInfo(Variant::INT, "preset")));
	BIND_VMETHOD(MethodInfo(Variant::ARRAY, "_get_recognized_extensions"));
	BIND_VMETHOD(MethodInfo(Variant::ARRAY, "_get_import_options", PropertyInfo(Variant::INT, "preset")));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_save_extension"));
	BIND_VMETHOD(MethodInfo(Variant::STRING, "_get_resource_type"));
	BIND_VMETHOD(MethodInfo(Variant::FLOAT, "_get_priority"));
	BIND_VMETHOD(MethodInfo(Variant::INT, "_get_import_order"));
	BIND_VMETHOD(MethodInfo(Variant::BOOL, "_get_option_visibility", PropertyInfo(Variant::STRING, "option"), PropertyInfo(Variant::DICTIONARY, "options")));
	BIND_VMETHOD(MethodInfo(Variant::INT, "_import", PropertyInfo(Variant::STRING, "source_file"), PropertyInfo(Variant::STRING, "save_path"), PropertyInfo(Variant::DICTIONARY, "options"), PropertyInfo(Variant::ARRAY, "platform_variants"), PropertyInfo(Variant::ARRAY, "gen_files")));
}
