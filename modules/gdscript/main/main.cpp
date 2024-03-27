/**************************************************************************/
/*  main.cpp                                                              */
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

#include "main.h"

#include "core/config/project_settings.h"
#include "core/register_core_types.h"
#include "modules/gdscript/gdscript.h"
#include "modules/gdscript/gdscript_analyzer.h"
#include "modules/gdscript/gdscript_compiler.h"
#include "modules/gdscript/gdscript_parser.h"
#include "modules/register_module_types.h"

static Vector<String> args;
static Engine *engine;
static ProjectSettings *globals;

int Main::test_entrypoint(int p_argc, char *p_argv[], bool &r_tests_need_run) {
	return 0;
}

Error Main::setup(const char *p_execpath, int p_argc, char *p_argv[], bool p_second_phase) {
	if (p_argc < 1)
		return ERR_BUG;
	for (int i = 0; i < p_argc; i++) {
		args.push_back(String::utf8(p_argv[i]));
	}

	Thread::make_main_thread();
	OS::get_singleton()->initialize();
	engine = memnew(Engine);
	register_core_types();
	globals = memnew(ProjectSettings);
	register_core_settings();
	initialize_modules(MODULE_INITIALIZATION_LEVEL_CORE);
	register_core_extensions();
	register_core_singletons();

	initialize_modules(MODULE_INITIALIZATION_LEVEL_SERVERS);

	ScriptServer::init_languages();
	return OK;
}

int Main::start() {
	Error err = OK;
	Ref<GDScript> script;
	script.instantiate();
	const String &source_file = args[0];
	script->set_path(source_file);
	err = script->load_source_code(source_file);
	ERR_FAIL_COND_V_MSG(err != OK, err, "Could not load source code for: '" + source_file + "'");
	GDScriptParser parser;
	err = parser.parse(script->get_source_code(), source_file, false);
	if (err != OK) {
		const List<GDScriptParser::ParserError> &errors = parser.get_errors();
		if (!errors.is_empty()) {
			print_line(errors[0].message);
		}
		return err;
	}
	GDScriptAnalyzer analyzer(&parser);
	err = analyzer.analyze();
	if (err != OK) {
		const List<GDScriptParser::ParserError> &errors = parser.get_errors();
		if (!errors.is_empty()) {
			print_line(errors[0].message);
		}
		return err;
	}

	GDScriptCompiler compiler;
	err = compiler.compile(&parser, script.ptr(), false);
	if (err != OK) {
		print_line(compiler.get_error());
		return err;
	}
	const HashMap<StringName, GDScriptFunction *>::ConstIterator main_func = script->get_member_functions().find("main");
	ERR_FAIL_COND_V_MSG(!main_func, ERR_CANT_RESOLVE, "Could not find main function in: '" + source_file + "'");

	err = script->reload();
	ERR_FAIL_COND_V_MSG(err != OK, err, "Could not reload script: '" + source_file + "'");

	Object *obj = ClassDB::instantiate(script->get_native()->get_name());
	Ref<RefCounted> obj_ref;
	if (obj->is_ref_counted()) {
		obj_ref = Ref<RefCounted>(Object::cast_to<RefCounted>(obj));
	}
	obj->set_script(script);
	GDScriptInstance *instance = static_cast<GDScriptInstance *>(obj->get_script_instance());

	Callable::CallError call_err;
	instance->callp("main", nullptr, 0, call_err);

	ERR_FAIL_COND_V_MSG(call_err.error != Callable::CallError::CALL_OK, ERR_SCRIPT_FAILED, "Could not call main function in: '" + source_file + "'");

	if (obj_ref.is_null()) {
		memdelete(obj);
	}
	return 0;
}

void Main::cleanup(bool p_force) {
	OS::get_singleton()->delete_main_loop();
	ScriptServer::finish_languages();
	uninitialize_modules(MODULE_INITIALIZATION_LEVEL_SERVERS);
	OS::get_singleton()->finalize();
	if (globals) {
		memdelete(globals);
	}
	if (engine) {
		memdelete(engine);
	}
	unregister_core_extensions();
	uninitialize_modules(MODULE_INITIALIZATION_LEVEL_CORE);
	unregister_core_types();
	OS::get_singleton()->finalize_core();
}

bool Main::iteration() {
	return true;
}
