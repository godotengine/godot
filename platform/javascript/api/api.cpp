/*************************************************************************/
/*  api.cpp                                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "javascript_eval.h"
#include "javascript_tools_editor_plugin.h"

static JavaScript *javascript_eval;

void register_javascript_api() {
	JavaScriptToolsEditorPlugin::initialize();
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
}

#if !defined(JAVASCRIPT_ENABLED) || !defined(JAVASCRIPT_EVAL_ENABLED)
Variant JavaScript::eval(const String &p_code, bool p_use_global_exec_context) {
	return Variant();
}
#endif
