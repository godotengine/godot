/*************************************************************************/
/*  api.cpp                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "api.h"
#include "core/engine.h"
#include "javascript_singleton.h"
#include "javascript_tools_editor_plugin.h"

static JavaScript *javascript_eval;

void register_javascript_api() {
	JavaScriptToolsEditorPlugin::initialize();
	ClassDB::register_virtual_class<JavaScriptObject>();
	ClassDB::register_virtual_class<JavaScript>();
	javascript_eval = memnew(JavaScript);
	Engine::get_singleton()->add_singleton(Engine::Singleton("JavaScript", javascript_eval));
}

void unregister_javascript_api() {
	memdelete(javascript_eval);
}

JavaScript *JavaScript::singleton = nullptr;

JavaScript *JavaScript::get_singleton() {
	return singleton;
}

JavaScript::JavaScript() {
	ERR_FAIL_COND_MSG(singleton != nullptr, "JavaScript singleton already exist.");
	singleton = this;
}

JavaScript::~JavaScript() {}

void JavaScript::_bind_methods() {
	ClassDB::bind_method(D_METHOD("eval", "code", "use_global_execution_context"), &JavaScript::eval, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_interface", "interface"), &JavaScript::get_interface);
	ClassDB::bind_method(D_METHOD("create_callback", "object", "method"), &JavaScript::create_callback);
	{
		MethodInfo mi;
		mi.name = "create_object";
		mi.arguments.push_back(PropertyInfo(Variant::STRING, "object"));
		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "create_object", &JavaScript::_create_object_bind, mi);
	}
	ClassDB::bind_method(D_METHOD("download_buffer", "buffer", "name", "mime"), &JavaScript::download_buffer, DEFVAL("application/octet-stream"));
}

#if !defined(JAVASCRIPT_ENABLED) || !defined(JAVASCRIPT_EVAL_ENABLED)
Variant JavaScript::eval(const String &p_code, bool p_use_global_exec_context) {
	return Variant();
}

Ref<JavaScriptObject> JavaScript::get_interface(const String &p_interface) {
	return Ref<JavaScriptObject>();
}

Ref<JavaScriptObject> JavaScript::create_callback(Object *p_ref, const StringName &p_method) {
	return Ref<JavaScriptObject>();
}

Variant JavaScript::_create_object_bind(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	if (p_argcount < 1) {
		r_error.error = Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument = 0;
		return Ref<JavaScriptObject>();
	}
	if (p_args[0]->get_type() != Variant::STRING) {
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 0;
		r_error.expected = Variant::STRING;
		return Ref<JavaScriptObject>();
	}
	return Ref<JavaScriptObject>();
}
#endif
#if !defined(JAVASCRIPT_ENABLED)
void JavaScript::download_buffer(Vector<uint8_t> p_arr, const String &p_name, const String &p_mime) {
}
#endif
