/*************************************************************************/
/*  csharp_project.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "csharp_project.h"

#include "core/io/json.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/os/os.h"
#include "core/project_settings.h"

#include "../csharp_script.h"
#include "../mono_gd/gd_mono_class.h"
#include "../mono_gd/gd_mono_marshal.h"
#include "../utils/string_utils.h"
#include "script_class_parser.h"

namespace CSharpProject {

String generate_core_api_project(const String &p_dir, const Vector<String> &p_files) {

	_GDMONO_SCOPE_DOMAIN_(TOOLS_DOMAIN)

	GDMonoClass *klass = GDMono::get_singleton()->get_editor_tools_assembly()->get_class("GodotSharpTools.Project", "ProjectGenerator");

	Variant dir = p_dir;
	Variant compile_items = p_files;
	const Variant *args[2] = { &dir, &compile_items };
	MonoException *exc = NULL;
	MonoObject *ret = klass->get_method("GenCoreApiProject", 2)->invoke(NULL, args, &exc);

	if (exc) {
		GDMonoUtils::debug_print_unhandled_exception(exc);
		ERR_FAIL_V(String());
	}

	return ret ? GDMonoMarshal::mono_string_to_godot((MonoString *)ret) : String();
}

String generate_editor_api_project(const String &p_dir, const String &p_core_proj_path, const Vector<String> &p_files) {

	_GDMONO_SCOPE_DOMAIN_(TOOLS_DOMAIN)

	GDMonoClass *klass = GDMono::get_singleton()->get_editor_tools_assembly()->get_class("GodotSharpTools.Project", "ProjectGenerator");

	Variant dir = p_dir;
	Variant core_proj_path = p_core_proj_path;
	Variant compile_items = p_files;
	const Variant *args[3] = { &dir, &core_proj_path, &compile_items };
	MonoException *exc = NULL;
	MonoObject *ret = klass->get_method("GenEditorApiProject", 3)->invoke(NULL, args, &exc);

	if (exc) {
		GDMonoUtils::debug_print_unhandled_exception(exc);
		ERR_FAIL_V(String());
	}

	return ret ? GDMonoMarshal::mono_string_to_godot((MonoString *)ret) : String();
}

String generate_game_project(const String &p_dir, const String &p_name, const Vector<String> &p_files) {

	_GDMONO_SCOPE_DOMAIN_(TOOLS_DOMAIN)

	GDMonoClass *klass = GDMono::get_singleton()->get_editor_tools_assembly()->get_class("GodotSharpTools.Project", "ProjectGenerator");

	Variant dir = p_dir;
	Variant name = p_name;
	Variant compile_items = p_files;
	const Variant *args[3] = { &dir, &name, &compile_items };
	MonoException *exc = NULL;
	MonoObject *ret = klass->get_method("GenGameProject", 3)->invoke(NULL, args, &exc);

	if (exc) {
		GDMonoUtils::debug_print_unhandled_exception(exc);
		ERR_FAIL_V(String());
	}

	return ret ? GDMonoMarshal::mono_string_to_godot((MonoString *)ret) : String();
}

void add_item(const String &p_project_path, const String &p_item_type, const String &p_include) {

	if (!GLOBAL_DEF("mono/project/auto_update_project", true))
		return;

	_GDMONO_SCOPE_DOMAIN_(TOOLS_DOMAIN)

	GDMonoClass *klass = GDMono::get_singleton()->get_editor_tools_assembly()->get_class("GodotSharpTools.Project", "ProjectUtils");

	Variant project_path = p_project_path;
	Variant item_type = p_item_type;
	Variant include = p_include;
	const Variant *args[3] = { &project_path, &item_type, &include };
	MonoException *exc = NULL;
	klass->get_method("AddItemToProjectChecked", 3)->invoke(NULL, args, &exc);

	if (exc) {
		GDMonoUtils::debug_print_unhandled_exception(exc);
		ERR_FAIL();
	}
}

Error generate_scripts_metadata(const String &p_project_path, const String &p_output_path) {

	_GDMONO_SCOPE_DOMAIN_(TOOLS_DOMAIN)

	if (FileAccess::exists(p_output_path)) {
		DirAccessRef da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		Error rm_err = da->remove(p_output_path);

		ERR_EXPLAIN("Failed to remove old scripts metadata file");
		ERR_FAIL_COND_V(rm_err != OK, rm_err);
	}

	GDMonoClass *project_utils = GDMono::get_singleton()->get_editor_tools_assembly()->get_class("GodotSharpTools.Project", "ProjectUtils");

	void *args[2] = {
		GDMonoMarshal::mono_string_from_godot(p_project_path),
		GDMonoMarshal::mono_string_from_godot("Compile")
	};

	MonoException *exc = NULL;
	MonoArray *ret = (MonoArray *)project_utils->get_method("GetIncludeFiles", 2)->invoke_raw(NULL, args, &exc);

	if (exc) {
		GDMonoUtils::debug_print_unhandled_exception(exc);
		ERR_FAIL_V(FAILED);
	}

	PoolStringArray project_files = GDMonoMarshal::mono_array_to_PoolStringArray(ret);
	PoolStringArray::Read r = project_files.read();

	Dictionary old_dict = CSharpLanguage::get_singleton()->get_scripts_metadata();
	Dictionary new_dict;

	for (int i = 0; i < project_files.size(); i++) {
		const String &project_file = ("res://" + r[i]).simplify_path();

		uint64_t modified_time = FileAccess::get_modified_time(project_file);

		const Variant *old_file_var = old_dict.getptr(project_file);
		if (old_file_var) {
			Dictionary old_file_dict = old_file_var->operator Dictionary();

			if (old_file_dict["modified_time"].operator uint64_t() == modified_time) {
				// No changes so no need to parse again
				new_dict[project_file] = old_file_dict;
				continue;
			}
		}

		ScriptClassParser scp;
		Error err = scp.parse_file(project_file);
		if (err != OK) {
			ERR_PRINTS("Parse error: " + scp.get_error());
			ERR_EXPLAIN("Failed to determine namespace and class for script: " + project_file);
			ERR_FAIL_V(err);
		}

		Vector<ScriptClassParser::ClassDecl> classes = scp.get_classes();

		bool found = false;
		Dictionary class_dict;

		String search_name = project_file.get_file().get_basename();

		for (int j = 0; j < classes.size(); j++) {
			const ScriptClassParser::ClassDecl &class_decl = classes[j];

			if (class_decl.base.size() == 0)
				continue; // Does not inherit nor implement anything, so it can't be a script class

			String class_cmp;

			if (class_decl.nested) {
				class_cmp = class_decl.name.get_slice(".", class_decl.name.get_slice_count(".") - 1);
			} else {
				class_cmp = class_decl.name;
			}

			if (class_cmp != search_name)
				continue;

			class_dict["namespace"] = class_decl.namespace_;
			class_dict["class_name"] = class_decl.name;
			class_dict["nested"] = class_decl.nested;

			found = true;
			break;
		}

		if (found) {
			Dictionary file_dict;
			file_dict["modified_time"] = modified_time;
			file_dict["class"] = class_dict;
			new_dict[project_file] = file_dict;
		}
	}

	if (new_dict.size()) {
		String json = JSON::print(new_dict, "", false);

		Error ferr;
		FileAccess *f = FileAccess::open(p_output_path, FileAccess::WRITE, &ferr);
		ERR_EXPLAIN("Cannot open file for writing: " + p_output_path);
		ERR_FAIL_COND_V(ferr != OK, ferr);
		f->store_string(json);
		f->flush();
		f->close();
		memdelete(f);
	}

	return OK;
}

} // namespace CSharpProject
