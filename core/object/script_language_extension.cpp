/*************************************************************************/
/*  script_language_extension.cpp                                        */
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

#include "script_language_extension.h"

#include "core/debugger/engine_debugger.h"
#include "core/debugger/script_debugger.h"
#include "core/object/script_language_internals.h"

class ThreadContextExtensionRef : public ThreadContextRefBase {
	Ref<ScriptLanguageThreadContext> _context;

public:
	bool is_valid() const {
		return _context.is_valid();
	}

	ScriptLanguageThreadContext &context() {
		return *_context.ptr();
	}

	// To be called by main thread on language exit; because thread local storage for the
	// main thread is not cleared until after Godot checks for memory leaks.
	void free_context() {
		_context.unref();
	}

	void set_context(ScriptLanguageThreadContext *p_context) {
		if (_detect_infinite_loop()) {
			// We cannot continue, so just leave _context uninitialized to force a crash.
			_context.unref();
			return;
		}
		_context = Ref<ScriptLanguageThreadContext>(p_context);
		_end_detect_infinite_loop();
	}
};

// These get initialized for each thread.  If debugging is enabled, the entry for
// the current script language extension (by language_index from ScriptServer) will
// hold a thread local storage context for that thread.
thread_local ThreadContextExtensionRef t_script_extension_current_thread[ScriptServer::MAX_LANGUAGES];

// Dummy context to use if the extension declined to create a context for some or all threads.
class EmptyThreadContext : public ScriptLanguageThreadContext {
	ScriptLanguage *language;
	DebugThreadID debug_thread_id;

public:
	ScriptLanguage *get_language() const override {
		return language;
	}

	DebugThreadID debug_get_thread_id() const override {
		return debug_thread_id;
	}

	String debug_get_error() const override {
		return "";
	}

	Severity debug_get_error_severity() const override {
		return SEVERITY_NONE;
	}

	int debug_get_stack_level_count() const override {
		return 0;
	}

	int debug_get_stack_level_line(int) const override {
		return 0;
	}

	String debug_get_stack_level_function(int) const override {
		return "";
	}

	String debug_get_stack_level_source(int) const override {
		return "";
	}

	void debug_get_stack_level_locals(int, List<String> *, List<Variant> *, int, int) const override {
		// No code.
	}

	void debug_get_stack_level_members(int, List<String> *, List<Variant> *, int, int) const override {
		// No code.
	}

	String debug_parse_stack_level_expression(int, const String &, int, int) const override {
		return "";
	}

	bool is_main_thread() const override {
		return false;
	}

	EmptyThreadContext(ScriptLanguage *p_language, const DebugThreadID &p_debug_thread_id) :
			language(p_language),
			debug_thread_id(p_debug_thread_id) {
		// No code.
	}
};

ScriptLanguageThreadContext &ScriptLanguageExtension::current_thread() {
	ThreadContextExtensionRef &ref = t_script_extension_current_thread[language_index];
	if (unlikely(!ref.is_valid())) {
		ref.set_context(create_thread_context());
	}
	return ref.context();
}

void ScriptExtension::_bind_methods() {
	GDVIRTUAL_BIND(_editor_can_reload_from_file);
	GDVIRTUAL_BIND(_placeholder_erased, "placeholder");

	GDVIRTUAL_BIND(_can_instantiate);
	GDVIRTUAL_BIND(_get_base_script);
	GDVIRTUAL_BIND(_inherits_script, "script");

	GDVIRTUAL_BIND(_get_instance_base_type);
	GDVIRTUAL_BIND(_instance_create, "for_object");
	GDVIRTUAL_BIND(_placeholder_instance_create, "for_object");

	GDVIRTUAL_BIND(_instance_has, "object");

	GDVIRTUAL_BIND(_has_source_code);
	GDVIRTUAL_BIND(_get_source_code);

	GDVIRTUAL_BIND(_set_source_code, "code");
	GDVIRTUAL_BIND(_reload, "keep_state");

	GDVIRTUAL_BIND(_get_documentation);

	GDVIRTUAL_BIND(_has_method, "method");
	GDVIRTUAL_BIND(_get_method_info, "method");

	GDVIRTUAL_BIND(_is_tool);
	GDVIRTUAL_BIND(_is_valid);
	GDVIRTUAL_BIND(_get_language);

	GDVIRTUAL_BIND(_has_script_signal, "signal");
	GDVIRTUAL_BIND(_get_script_signal_list);

	GDVIRTUAL_BIND(_has_property_default_value, "property");
	GDVIRTUAL_BIND(_get_property_default_value, "property");

	GDVIRTUAL_BIND(_update_exports);
	GDVIRTUAL_BIND(_get_script_method_list);
	GDVIRTUAL_BIND(_get_script_property_list);

	GDVIRTUAL_BIND(_get_member_line, "member");

	GDVIRTUAL_BIND(_get_constants);
	GDVIRTUAL_BIND(_get_members);
	GDVIRTUAL_BIND(_is_placeholder_fallback_enabled);

	GDVIRTUAL_BIND(_get_rpc_config);
}

void ScriptLanguageThreadContextExtension::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_parent"), &ScriptLanguageThreadContextExtension::get_parent);

	GDVIRTUAL_BIND(_debug_get_error);
	GDVIRTUAL_BIND(_debug_get_stack_level_count);

	GDVIRTUAL_BIND(_debug_get_stack_level_line, "level");
	GDVIRTUAL_BIND(_debug_get_stack_level_function, "level");
	GDVIRTUAL_BIND(_debug_get_stack_level_locals, "level", "max_subitems", "max_depth");
	GDVIRTUAL_BIND(_debug_get_stack_level_members, "level", "max_subitems", "max_depth");
	GDVIRTUAL_BIND(_debug_get_stack_level_instance, "level");
	GDVIRTUAL_BIND(_debug_parse_stack_level_expression, "level", "expression", "max_subitems", "max_depth");
	GDVIRTUAL_BIND(_debug_get_current_stack_info);
}

// virtual
ScriptLanguage *ScriptLanguageThreadContextExtension::get_language() const {
	return parent;
}

void ScriptLanguageThreadContextExtension::debug_get_stack_level_locals(int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) const {
	Dictionary ret;
	GDVIRTUAL_REQUIRED_CALL(_debug_get_stack_level_locals, p_level, p_max_subitems, p_max_depth, ret);
	if (ret.size() == 0) {
		return;
	}
	if (p_locals != nullptr && ret.has("locals")) {
		PackedStringArray strings = ret["locals"];
		for (int i = 0; i < strings.size(); i++) {
			p_locals->push_back(strings[i]);
		}
	}
	if (p_values != nullptr && ret.has("values")) {
		Array values = ret["values"];
		for (int i = 0; i < values.size(); i++) {
			p_values->push_back(values[i]);
		}
	}
}

void ScriptLanguageThreadContextExtension::debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems, int p_max_depth) const {
	Dictionary ret;
	GDVIRTUAL_REQUIRED_CALL(_debug_get_stack_level_members, p_level, p_max_subitems, p_max_depth, ret);
	if (ret.size() == 0) {
		return;
	}
	if (p_members != nullptr && ret.has("members")) {
		PackedStringArray strings = ret["members"];
		for (int i = 0; i < strings.size(); i++) {
			p_members->push_back(strings[i]);
		}
	}
	if (p_values != nullptr && ret.has("values")) {
		Array values = ret["values"];
		for (int i = 0; i < values.size(); i++) {
			p_values->push_back(values[i]);
		}
	}
}

ScriptInstance *ScriptLanguageThreadContextExtension::debug_get_stack_level_instance(int p_level) const {
	GDNativePtr<void> ret = nullptr;
	GDVIRTUAL_REQUIRED_CALL(_debug_get_stack_level_instance, p_level, ret);
	return reinterpret_cast<ScriptInstance *>(ret.operator void *());
}

Vector<ScriptLanguageThreadContext::StackInfo> ScriptLanguageThreadContextExtension::debug_get_current_stack_info() const {
	TypedArray<Dictionary> ret;
	GDVIRTUAL_REQUIRED_CALL(_debug_get_current_stack_info, ret);
	Vector<StackInfo> sret;
	for (int i = 0; i < ret.size(); i++) {
		StackInfo si;
		Dictionary d = ret[i];
		ERR_CONTINUE(!d.has("file"));
		ERR_CONTINUE(!d.has("func"));
		ERR_CONTINUE(!d.has("line"));
		si.file = d["file"];
		si.func = d["func"];
		si.line = d["line"];
		sret.push_back(si);
	}
	return sret;
}

ScriptLanguageThreadContext::DebugThreadID ScriptLanguageThreadContextExtension::debug_get_thread_id() const {
	PackedByteArray ret;
	GDVIRTUAL_REQUIRED_CALL(_debug_get_thread_id, ret);
	return ret;
}

ScriptLanguageThreadContext::Severity ScriptLanguageThreadContextExtension::debug_get_error_severity() const {
	int ret = 0;
	GDVIRTUAL_REQUIRED_CALL(_debug_get_error_severity, ret);
	ERR_FAIL_COND_V_MSG(ret < SEVERITY_NONE, SEVERITY_NONE, vformat("severity code must be at least %d", SEVERITY_NONE));
	ERR_FAIL_COND_V_MSG(ret >= SEVERITY_NUM_VALUES, static_cast<Severity>(SEVERITY_NUM_VALUES - 1), vformat("severity code must be at most %d", SEVERITY_NUM_VALUES - 1));
	return static_cast<Severity>(ret);
}

bool ScriptLanguageThreadContextExtension::is_main_thread() const {
	bool ret = false;
	GDVIRTUAL_REQUIRED_CALL(_is_main_thread, ret);
	return ret;
}

void ScriptLanguageExtension::_bind_methods() {
	ClassDB::bind_method(D_METHOD("debug_break"), &ScriptLanguageExtension::debug_break);
	ClassDB::bind_method(D_METHOD("debug_step"), &ScriptLanguageExtension::debug_step);
	ClassDB::bind_method(D_METHOD("debug_request_break"), &ScriptLanguageExtension::debug_request_break);
	ClassDB::bind_method(D_METHOD("create_unique_debug_thread_id"), &ScriptLanguageExtension::create_unique_debug_thread_id);

	GDVIRTUAL_BIND(_create_thread_context);
	GDVIRTUAL_BIND(_get_name);
	GDVIRTUAL_BIND(_init);
	GDVIRTUAL_BIND(_get_type);
	GDVIRTUAL_BIND(_get_extension);
	GDVIRTUAL_BIND(_execute_file, "path");
	GDVIRTUAL_BIND(_finish);

	GDVIRTUAL_BIND(_get_reserved_words);
	GDVIRTUAL_BIND(_is_control_flow_keyword, "keyword");
	GDVIRTUAL_BIND(_get_comment_delimiters);
	GDVIRTUAL_BIND(_get_string_delimiters);
	GDVIRTUAL_BIND(_make_template, "template", "class_name", "base_class_name");
	GDVIRTUAL_BIND(_get_built_in_templates, "object");
	GDVIRTUAL_BIND(_is_using_templates);
	GDVIRTUAL_BIND(_validate, "script", "path", "validate_functions", "validate_errors", "validate_warnings", "validate_safe_lines");

	GDVIRTUAL_BIND(_validate_path, "path");
	GDVIRTUAL_BIND(_create_script);
	GDVIRTUAL_BIND(_has_named_classes);
	GDVIRTUAL_BIND(_supports_builtin_mode);
	GDVIRTUAL_BIND(_supports_documentation);
	GDVIRTUAL_BIND(_can_inherit_from_file);
	GDVIRTUAL_BIND(_find_function, "class_name", "function_name");
	GDVIRTUAL_BIND(_make_function, "class_name", "function_name", "function_args");
	GDVIRTUAL_BIND(_open_in_external_editor, "script", "line", "column");
	GDVIRTUAL_BIND(_overrides_external_editor);

	GDVIRTUAL_BIND(_complete_code, "code", "path", "owner");
	GDVIRTUAL_BIND(_lookup_code, "code", "symbol", "path", "owner");
	GDVIRTUAL_BIND(_auto_indent_code, "code", "from_line", "to_line");

	GDVIRTUAL_BIND(_add_global_constant, "name", "value");
	GDVIRTUAL_BIND(_add_named_global_constant, "name", "value");
	GDVIRTUAL_BIND(_remove_named_global_constant, "name");

	GDVIRTUAL_BIND(_thread_enter);
	GDVIRTUAL_BIND(_thread_exit);

	GDVIRTUAL_BIND(_debug_get_globals, "max_subitems", "max_depth");

	GDVIRTUAL_BIND(_reload_all_scripts);
	GDVIRTUAL_BIND(_reload_tool_script, "script", "soft_reload");

	GDVIRTUAL_BIND(_get_recognized_extensions);
	GDVIRTUAL_BIND(_get_public_functions);
	GDVIRTUAL_BIND(_get_public_constants);
	GDVIRTUAL_BIND(_get_public_annotations);

	GDVIRTUAL_BIND(_profiling_start);
	GDVIRTUAL_BIND(_profiling_stop);

	GDVIRTUAL_BIND(_profiling_get_accumulated_data, "info_array", "info_max");
	GDVIRTUAL_BIND(_profiling_get_frame_data, "info_array", "info_max");

	GDVIRTUAL_BIND(_alloc_instance_binding_data, "object");
	GDVIRTUAL_BIND(_free_instance_binding_data, "data");

	GDVIRTUAL_BIND(_refcount_incremented_instance_binding, "object");
	GDVIRTUAL_BIND(_refcount_decremented_instance_binding, "object");

	GDVIRTUAL_BIND(_frame);

	GDVIRTUAL_BIND(_handles_global_class_type, "type");
	GDVIRTUAL_BIND(_get_global_class_name, "path");

	BIND_ENUM_CONSTANT(LOOKUP_RESULT_SCRIPT_LOCATION);
	BIND_ENUM_CONSTANT(LOOKUP_RESULT_CLASS);
	BIND_ENUM_CONSTANT(LOOKUP_RESULT_CLASS_CONSTANT);
	BIND_ENUM_CONSTANT(LOOKUP_RESULT_CLASS_PROPERTY);
	BIND_ENUM_CONSTANT(LOOKUP_RESULT_CLASS_METHOD);
	BIND_ENUM_CONSTANT(LOOKUP_RESULT_CLASS_SIGNAL);
	BIND_ENUM_CONSTANT(LOOKUP_RESULT_CLASS_ENUM);
	BIND_ENUM_CONSTANT(LOOKUP_RESULT_CLASS_TBD_GLOBALSCOPE);
	BIND_ENUM_CONSTANT(LOOKUP_RESULT_CLASS_ANNOTATION);
	BIND_ENUM_CONSTANT(LOOKUP_RESULT_MAX);

	BIND_ENUM_CONSTANT(LOCATION_LOCAL);
	BIND_ENUM_CONSTANT(LOCATION_PARENT_MASK);
	BIND_ENUM_CONSTANT(LOCATION_OTHER_USER_CODE);
	BIND_ENUM_CONSTANT(LOCATION_OTHER);

	BIND_ENUM_CONSTANT(CODE_COMPLETION_KIND_CLASS);
	BIND_ENUM_CONSTANT(CODE_COMPLETION_KIND_FUNCTION);
	BIND_ENUM_CONSTANT(CODE_COMPLETION_KIND_SIGNAL);
	BIND_ENUM_CONSTANT(CODE_COMPLETION_KIND_VARIABLE);
	BIND_ENUM_CONSTANT(CODE_COMPLETION_KIND_MEMBER);
	BIND_ENUM_CONSTANT(CODE_COMPLETION_KIND_ENUM);
	BIND_ENUM_CONSTANT(CODE_COMPLETION_KIND_CONSTANT);
	BIND_ENUM_CONSTANT(CODE_COMPLETION_KIND_NODE_PATH);
	BIND_ENUM_CONSTANT(CODE_COMPLETION_KIND_FILE_PATH);
	BIND_ENUM_CONSTANT(CODE_COMPLETION_KIND_PLAIN_TEXT);
	BIND_ENUM_CONSTANT(CODE_COMPLETION_KIND_MAX);
}

// virtual
void ScriptLanguageExtension::init(int p_language_index) {
	ScriptLanguage::init(p_language_index);
	GDVIRTUAL_REQUIRED_CALL(_init);
}

// virtual
void ScriptLanguageExtension::finish() {
	GDVIRTUAL_REQUIRED_CALL(_finish);

	// Free main thread context so that it does not trigger ObjectDB leak detection.
	t_script_extension_current_thread[language_index].free_context();
}

PackedByteArray ScriptLanguageExtension::create_unique_debug_thread_id() const {
	return ScriptLanguageInternals::create_thread_id(*this, Thread::get_caller_id());
}

void ScriptLanguageExtension::debug_break() {
	if (EngineDebugger::is_active()) {
		EngineDebugger::get_script_debugger()->debug(current_thread());
	}
}

void ScriptLanguageExtension::debug_step() {
	if (EngineDebugger::is_active()) {
		EngineDebugger::get_script_debugger()->step(current_thread());
	}
}

void ScriptLanguageExtension::debug_request_break() {
	if (EngineDebugger::is_active()) {
		EngineDebugger::get_script_debugger()->debug_request_break();
	}
}

ScriptLanguageThreadContext *ScriptLanguageExtension::create_thread_context() {
	if (GDVIRTUAL_IS_OVERRIDDEN(_create_thread_context)) {
		ScriptLanguageThreadContextExtension *ret = nullptr;
		GDVIRTUAL_CALL(_create_thread_context, ret);
		if (ret != nullptr) {
			ret->parent = this;
			return ret;
		}
	}
	return memnew(EmptyThreadContext(this, create_unique_debug_thread_id()));
}
