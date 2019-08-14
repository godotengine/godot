/*************************************************************************/
/*  csharp_project.cpp                                                   */
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

bool generate_api_solution_impl(const String &p_solution_dir, const String &p_core_proj_dir, const Vector<String> &p_core_compile_items,
		const String &p_editor_proj_dir, const Vector<String> &p_editor_compile_items,
		GDMonoAssembly *p_tools_project_editor_assembly) {

	GDMonoClass *klass = p_tools_project_editor_assembly->get_class("GodotTools.ProjectEditor", "ApiSolutionGenerator");

	Variant solution_dir = p_solution_dir;
	Variant core_proj_dir = p_core_proj_dir;
	Variant core_compile_items = p_core_compile_items;
	Variant editor_proj_dir = p_editor_proj_dir;
	Variant editor_compile_items = p_editor_compile_items;
	const Variant *args[5] = { &solution_dir, &core_proj_dir, &core_compile_items, &editor_proj_dir, &editor_compile_items };
	MonoException *exc = NULL;
	klass->get_method("GenerateApiSolution", 5)->invoke(NULL, args, &exc);

	if (exc) {
		GDMonoUtils::debug_print_unhandled_exception(exc);
		ERR_FAIL_V(false);
	}

	return true;
}

bool generate_api_solution(const String &p_solution_dir, const String &p_core_proj_dir, const Vector<String> &p_core_compile_items,
		const String &p_editor_proj_dir, const Vector<String> &p_editor_compile_items) {

	if (GDMono::get_singleton()->get_tools_project_editor_assembly()) {
		return generate_api_solution_impl(p_solution_dir, p_core_proj_dir, p_core_compile_items,
				p_editor_proj_dir, p_editor_compile_items,
				GDMono::get_singleton()->get_tools_project_editor_assembly());
	} else {
		MonoDomain *temp_domain = GDMonoUtils::create_domain("GodotEngine.ApiSolutionGenerationDomain");
		CRASH_COND(temp_domain == NULL);
		_GDMONO_SCOPE_EXIT_DOMAIN_UNLOAD_(temp_domain);

		_GDMONO_SCOPE_DOMAIN_(temp_domain);

		GDMonoAssembly *tools_project_editor_asm = NULL;

		bool assembly_loaded = GDMono::get_singleton()->load_assembly(TOOLS_PROJECT_EDITOR_ASM_NAME, &tools_project_editor_asm);
		ERR_FAIL_COND_V_MSG(!assembly_loaded, false, "Failed to load assembly: '" TOOLS_PROJECT_EDITOR_ASM_NAME "'.");

		return generate_api_solution_impl(p_solution_dir, p_core_proj_dir, p_core_compile_items,
				p_editor_proj_dir, p_editor_compile_items,
				tools_project_editor_asm);
	}
}

void add_item(const String &p_project_path, const String &p_item_type, const String &p_include) {

	if (!GLOBAL_DEF("mono/project/auto_update_project", true))
		return;

	GDMonoAssembly *tools_project_editor_assembly = GDMono::get_singleton()->get_tools_project_editor_assembly();

	GDMonoClass *klass = tools_project_editor_assembly->get_class("GodotTools.ProjectEditor", "ProjectUtils");

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

} // namespace CSharpProject
