/*************************************************************************/
/*  standalone_tools.cpp                                                 */
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

#include "standalone_tools.h"

#ifdef TOOLS_ENABLED
#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/os/dir_access.h"

#include "editor/doc_data_class_path.gen.h"
#include "editor/doc_tools.h"

void run_doc_tool(const String &p_doc_tool_path, bool p_doc_base_types) {
	// Needed to instance editor-only classes for their default values
	Engine::get_singleton()->set_editor_hint(true);

#ifndef MODULE_MONO_ENABLED
	// Hack to define Mono-specific project settings even on non-Mono builds,
	// so that we don't lose their descriptions and default values in DocData.
	// Default values should be synced with mono_gd/gd_mono.cpp.
	GLOBAL_DEF("mono/debugger_agent/port", 23685);
	GLOBAL_DEF("mono/debugger_agent/wait_for_debugger", false);
	GLOBAL_DEF("mono/debugger_agent/wait_timeout", 3000);
	GLOBAL_DEF("mono/profiler/args", "log:calls,alloc,sample,output=output.mlpd");
	GLOBAL_DEF("mono/profiler/enabled", false);
	GLOBAL_DEF("mono/unhandled_exception_policy", 0);
	// From editor/csharp_project.cpp.
	GLOBAL_DEF("mono/project/auto_update_project", true);
#endif

	DocTools doc;
	doc.generate(p_doc_base_types);

	DocTools docsrc;
	Map<String, String> doc_data_classes;
	Set<String> checked_paths;

	print_line("Loading docs...");
	for (int i = 0; i < _doc_data_class_path_count; i++) {
		// Custom modules are always located by absolute path.
		String path = _doc_data_class_paths[i].path;
		if (path.is_rel_path()) {
			path = p_doc_tool_path.plus_file(path);
		}
		String name = _doc_data_class_paths[i].name;
		doc_data_classes[name] = path;
		if (!checked_paths.has(path)) {
			checked_paths.insert(path);

			// Create the module documentation directory if it doesn't exist
			DirAccess *da = DirAccess::create_for_path(path);
			da->make_dir_recursive(path);
			memdelete(da);

			docsrc.load_classes(path);
			print_line("Loading docs from: " + path);
		}
	}

	String index_path = p_doc_tool_path.plus_file("doc/classes");
	// Create the main documentation directory if it doesn't exist
	DirAccess *da = DirAccess::create_for_path(index_path);
	da->make_dir_recursive(index_path);
	memdelete(da);

	docsrc.load_classes(index_path);
	checked_paths.insert(index_path);
	print_line("Loading docs from: " + index_path);

	print_line("Merging docs...");
	doc.merge_from(docsrc);
	for (Set<String>::Element *E = checked_paths.front(); E; E = E->next()) {
		print_line("Erasing old docs at: " + E->get());
		DocTools::erase_classes(E->get());
	}

	print_line("Generating new docs...");
	doc.save_classes(index_path, doc_data_classes);
}
#endif // TOOLS_ENABLED
