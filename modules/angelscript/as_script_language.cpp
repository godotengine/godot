/**************************************************************************/
/*  as_script_language.cpp                                                */
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

#include "as_script_language.h"

#include "as_debugger.h"
#include "as_engine.h"
#include "as_script.h"

#include "core/config/project_settings.h"
#include "core/debugger/engine_debugger.h"
#include "core/debugger/script_debugger.h"
#include "core/object/class_db.h"
#include "core/object/editor_language.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"

#include <angelscript.h>

ASScriptLanguage *ASScriptLanguage::singleton = nullptr;

ASScriptLanguage::ASScriptLanguage() {
	singleton = this;
}

ASScriptLanguage::~ASScriptLanguage() {
	singleton = nullptr;
}

void ASScriptLanguage::init() {
	// ASEngine 自身也是懒初始化的，这里只是让引擎在语言被正式启用时提前就绪
	// （--test / headless 路径不会走这里，靠 ASEngine::get_singleton() 兜底）。
	ASEngine::get_singleton()->ensure_initialized();

	// M3：宿主节流回收的间隔（秒）。0 表示关闭宿主侧兜底，只保留 AS 自身的回收。
	GLOBAL_DEF_BASIC("angel_script/gc/interval_seconds", 5.0);
	ASEngine::get_singleton()->set_gc_interval_seconds(GLOBAL_GET("angel_script/gc/interval_seconds"));
}

void ASScriptLanguage::frame() {
	ASEngine *as_engine = ASEngine::get_singleton();
	if (as_engine != nullptr) {
		as_engine->maybe_collect_garbage();
	}

#ifdef DEBUG_ENABLED
	if (profiling) {
		MutexLock lock(profile_mutex);
		// 把本帧计数滚动为「上一帧」并清零，供 profiling_get_frame_data() 读取；
		// 累计数据（call_count/total_time）保持不动。frame 与采样写入同锁，避免撕裂。
		for (KeyValue<StringName, ProfileEntry> &kv : profile_data) {
			kv.value.last_frame_call_count = kv.value.frame_call_count;
			kv.value.last_frame_total_time = kv.value.frame_total_time;
			kv.value.frame_call_count = 0;
			kv.value.frame_total_time = 0;
		}
	}
#endif
}

void ASScriptLanguage::finish() {
	ASEngine *as_engine = ASEngine::get_singleton();
	if (as_engine != nullptr) {
		as_engine->collect_garbage(); // 退出前最后收一次，减少退出期泄漏告警
	}
	ASEngine::get_singleton()->shutdown();
}

Vector<String> ASScriptLanguage::get_reserved_words() const {
	static const char *words[] = {
		"and", "abstract", "auto", "bool", "break", "case", "cast", "catch", "class",
		"const", "continue", "default", "do", "double", "else", "enum", "explicit",
		"external", "false",
		"final", "float", "for", "from", "funcdef", "get", "if", "import", "in", "inout",
		"int", "int8", "int16", "int32", "int64", "interface", "is", "mixin",
		"namespace", "not", "null", "or", "out", "override", "private", "protected",
		"return", "set", "shared", "string", "switch", "true", "try", "typedef", "uint",
		"uint8", "uint16", "uint32", "uint64", "void", "while", "xor", nullptr
	};
	Vector<String> out;
	for (int i = 0; words[i] != nullptr; i++) {
		out.push_back(words[i]);
	}
	return out;
}

bool ASScriptLanguage::is_control_flow_keyword(const String &p_string) const {
	return p_string == "break" || p_string == "case" || p_string == "continue" ||
			p_string == "default" || p_string == "do" || p_string == "else" ||
			p_string == "for" || p_string == "if" || p_string == "return" ||
			p_string == "switch" || p_string == "while";
}

Vector<String> ASScriptLanguage::get_comment_delimiters() const {
	Vector<String> out;
	out.push_back("//");
	out.push_back("/* */");
	return out;
}

Vector<String> ASScriptLanguage::get_doc_comment_delimiters() const {
	return Vector<String>();
}

Vector<String> ASScriptLanguage::get_string_delimiters() const {
	Vector<String> out;
	out.push_back("\" \"");
	out.push_back("' '");
	return out;
}

String ASScriptLanguage::make_function(const String &p_class, const String &p_name, const PackedStringArray &p_args) const {
	return String();
}

Ref<Script> ASScriptLanguage::make_template(const String &p_template, const String &p_class_name, const String &p_base_class_name) const {
	Ref<ASScript> scr;
	scr.instantiate();

	// 本语言未开启编辑器模板系统（is_using_templates() 为 false，对话框传入的 p_template
	// 恒为空），因此这里按模块约定合成最小可用骨架：`// godot_base:` 基类指令 +
	// 与文件名同名的空脚本类。缺了这一步，基类返回的空引用会让
	// ScriptCreateDialog::_create_new() 在 scr->set_path() 处空指针崩溃。
	String base = p_base_class_name.strip_edges().unquote();
	if (base.is_empty() || !ClassDB::class_exists(base)) {
		base = "Object";
	}

	scr->set_source_code(vformat("// godot_base: %s\n\nclass %s {\n}\n", base, p_class_name.strip_edges()));
	return scr;
}

bool ASScriptLanguage::debug_break(const String &p_error, bool p_allow_continue, bool p_is_error_breakpoint) {
	if (!EngineDebugger::is_active()) {
		return false;
	}
	EngineDebugger::get_script_debugger()->debug(this, p_allow_continue, p_is_error_breakpoint);
	return true;
}

Vector<ScriptLanguage::StackInfo> ASScriptLanguage::debug_get_current_stack_info() {
	asIScriptContext *ctx = ASDebugger::get_break_context();
	if (ctx == nullptr) {
		// 不在断点期间时退回到「当前线程正在执行的上下文」，这样即使是 AS 调用的某个
		// Godot API 报错，AS 的栈帧也能并进那条错误的回溯里。
		ctx = asGetActiveContext();
	}
	return ASDebugger::build_stack_info(ctx);
}

String ASScriptLanguage::debug_get_error() const {
	return ASDebugger::get_error();
}

int ASScriptLanguage::debug_get_stack_level_count() const {
	return ASDebugger::get_stack_level_count();
}

int ASScriptLanguage::debug_get_stack_level_line(int p_level) const {
	return ASDebugger::get_stack_level_line(p_level);
}

String ASScriptLanguage::debug_get_stack_level_function(int p_level) const {
	return ASDebugger::get_stack_level_function(p_level);
}

String ASScriptLanguage::debug_get_stack_level_source(int p_level) const {
	return ASDebugger::get_stack_level_source(p_level);
}

void ASScriptLanguage::debug_get_stack_level_locals(int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	ASDebugger::get_stack_level_locals(p_level, p_locals, p_values, p_max_subitems, p_max_depth);
}

void ASScriptLanguage::debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	ASDebugger::get_stack_level_members(p_level, p_members, p_values, p_max_subitems, p_max_depth);
}

void ASScriptLanguage::debug_get_globals(List<String> *p_globals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	ASDebugger::get_globals(p_globals, p_values, p_max_subitems, p_max_depth);
}

String ASScriptLanguage::debug_parse_stack_level_expression(int p_level, const String &p_expression, int p_max_subitems, int p_max_depth) {
	return ASDebugger::parse_stack_level_expression(p_level, p_expression);
}

void ASScriptLanguage::reload_all_scripts() {
}

void ASScriptLanguage::reload_scripts(const Array &p_scripts) {
#ifdef TOOLS_ENABLED
	for (int i = 0; i < p_scripts.size(); i++) {
		Object *obj = p_scripts[i];
		Ref<ASScript> scr = Object::cast_to<ASScript>(obj);
		if (scr.is_null()) {
			continue;
		}
		// 上游已移除 soft/hard 重启参数；AS 实例持有模块内的 asIScriptObject，
		// 重编译会 DiscardModule，必须先摘除实例再重编译，并保留属性状态。
		scr->reload_with_instances(true);
	}
#endif
}

void ASScriptLanguage::reload_tool_script(const Ref<Script> &p_script) {
	Array scripts;
	scripts.push_back(p_script);
	reload_scripts(scripts);
}

#ifdef TOOLS_ENABLED
namespace {

bool _as_is_identifier_char(char32_t p_char) {
	return (p_char >= 'A' && p_char <= 'Z') || (p_char >= 'a' && p_char <= 'z') || (p_char >= '0' && p_char <= '9') || p_char == '_';
}

// 编辑器语言服务：validate 复用 AS 编译器诊断；complete_code 对引擎已注册的
// 全局符号做前缀匹配。两者都不改动引擎状态（validate 用临时模块，编译后即弃）。
class ASEditorLanguage : public EditorLanguage {
public:
	bool validate(const String &p_code, const String &p_path, List<ScriptError> *r_errors, List<Warning> *r_warnings, List<String> *r_functions, HashSet<int> *) const override {
		Vector<ASEngine::ASMessage> messages;
		Vector<String> functions;
		const bool ok = ASEngine::get_singleton()->compile_for_diagnostics("__as_editor_validate", p_code, &messages, &functions);

		// Build() 成功仍可能带告警；错误类型以消息回调的 type 为准。
		bool has_error = !ok;
		for (const ASEngine::ASMessage &m : messages) {
			if (m.type == asMSGTYPE_ERROR) {
				has_error = true;
				if (r_errors != nullptr) {
					ScriptError e;
					e.path = p_path;
					e.start_line = m.row;
					e.start_column = m.column;
					e.end_line = m.row;
					e.end_column = m.column + 1;
					e.message = m.message;
					r_errors->push_back(e);
				}
			} else if (m.type == asMSGTYPE_WARNING) {
				if (r_warnings != nullptr) {
					Warning w;
					w.start_line = m.row;
					w.start_column = m.column;
					w.end_line = m.row;
					w.end_column = m.column + 1;
					w.message = m.message;
					r_warnings->push_back(w);
				}
			}
		}

		if (ok && r_functions != nullptr) {
			for (const String &f : functions) {
				r_functions->push_back(f);
			}
		}
		return !has_error;
	}

	Error complete_code(const String &p_code, const String &, Object *, List<CompletionOption> *r_options, bool &r_force, String &) override {
		r_force = false;
		if (r_options == nullptr) {
			return OK;
		}
		// 光标处总是哨兵字符 0xFFFF；取其前面的标识符作为补全前缀。
		const int caret = p_code.find(String::chr(0xFFFF));
		if (caret < 0) {
			return OK;
		}
		int start = caret;
		while (start > 0 && _as_is_identifier_char(p_code[start - 1])) {
			start--;
		}
		const String prefix = p_code.substr(start, caret - start);
		if (prefix.is_empty()) {
			return OK;
		}

		asIScriptEngine *engine = ASEngine::get_singleton()->get_engine();
		if (engine == nullptr) {
			return OK;
		}

		HashMap<String, CompletionKind> kinds;
		Vector<String> names;
		auto consider = [&](const char *p_name, CompletionKind p_kind) {
			if (p_name == nullptr) {
				return;
			}
			const String name = String(p_name);
			if (!name.begins_with(prefix) || kinds.has(name)) {
				return;
			}
			kinds.insert(name, p_kind);
			names.push_back(name);
		};

		for (asUINT i = 0; i < engine->GetObjectTypeCount(); i++) {
			consider(engine->GetObjectTypeByIndex(i)->GetName(), CompletionKind::CLASS);
		}
		for (asUINT i = 0; i < engine->GetEnumCount(); i++) {
			consider(engine->GetEnumByIndex(i)->GetName(), CompletionKind::ENUM);
		}
		for (asUINT i = 0; i < engine->GetGlobalFunctionCount(); i++) {
			consider(engine->GetGlobalFunctionByIndex(i)->GetName(), CompletionKind::FUNCTION);
		}
		for (asUINT i = 0; i < engine->GetGlobalPropertyCount(); i++) {
			const char *name = nullptr;
			engine->GetGlobalPropertyByIndex(i, &name);
			consider(name, CompletionKind::CONSTANT);
		}

		// 确定性输出：字母序，并限量以防超大 API 淹没编辑器。
		names.sort();
		for (int i = 0; i < names.size() && i < 64; i++) {
			r_options->push_back(CompletionOption(names[i], kinds[names[i]], CompletionLocation::OTHER, String()));
		}
		return OK;
	}

	int32_t find_function(const String &p_function, const String &p_code) const override {
		if (p_function.is_empty()) {
			return -1;
		}
		const Vector<String> lines = p_code.split("\n");
		for (int i = 0; i < lines.size(); i++) {
			String line = lines[i];
			const int comment = line.find("//");
			if (comment >= 0) {
				line = line.substr(0, comment);
			}
			const int idx = line.find(p_function);
			if (idx < 0) {
				continue;
			}
			if (idx > 0 && _as_is_identifier_char(line[idx - 1])) {
				continue;
			}
			int j = idx + p_function.length();
			while (j < line.length() && (line[j] == ' ' || line[j] == '\t')) {
				j++;
			}
			if (j < line.length() && line[j] == '(') {
				return i;
			}
		}
		return -1;
	}
};

} // namespace

EditorLanguage *ASScriptLanguage::get_editor_language() {
	static ASEditorLanguage editor_language;
	return &editor_language;
}
#endif

void ASScriptLanguage::get_public_functions(List<MethodInfo> *p_functions) const {
}

void ASScriptLanguage::get_public_constants(List<Pair<String, Variant>> *p_constants) const {
}

void ASScriptLanguage::get_public_annotations(List<MethodInfo> *p_annotations) const {
}

void ASScriptLanguage::profiling_start() {
#ifdef DEBUG_ENABLED
	MutexLock lock(profile_mutex);
	profile_data.clear();
	profiling = true;
#endif
}

void ASScriptLanguage::profiling_stop() {
#ifdef DEBUG_ENABLED
	MutexLock lock(profile_mutex);
	profiling = false;
#endif
}

bool ASScriptLanguage::is_profiling() const {
#ifdef DEBUG_ENABLED
	return profiling;
#else
	return false;
#endif
}

void ASScriptLanguage::profile_function(const StringName &p_signature, uint64_t p_usec) {
#ifdef DEBUG_ENABLED
	MutexLock lock(profile_mutex);
	// 采样已停止时不得再累加：call_function 路径已用 is_profiling() 预筛，
	// 但直接调用（含测试）仍需此处兜底，否则会污染停止后的历史数据。
	if (!profiling) {
		return;
	}
	ProfileEntry &entry = profile_data[p_signature];
	entry.call_count++;
	entry.total_time += p_usec;
	entry.frame_call_count++;
	entry.frame_total_time += p_usec;
#endif
}

int ASScriptLanguage::profiling_get_accumulated_data(ProfilingInfo *p_info_arr, int p_info_max) {
	int current = 0;
#ifdef DEBUG_ENABLED
	MutexLock lock(profile_mutex);
	for (const KeyValue<StringName, ProfileEntry> &kv : profile_data) {
		if (current >= p_info_max) {
			break;
		}
		ProfilingInfo &info = p_info_arr[current];
		info.signature = kv.key;
		info.call_count = kv.value.call_count;
		info.total_time = kv.value.total_time;
		// v1 不拆分嵌套 AS→AS 调用，自身耗时即总耗时。
		info.self_time = kv.value.total_time;
		// v1 不统计 AS→Godot 原生调用。
		info.internal_time = 0;
		current++;
	}
#endif
	return current;
}

int ASScriptLanguage::profiling_get_frame_data(ProfilingInfo *p_info_arr, int p_info_max) {
	int current = 0;
#ifdef DEBUG_ENABLED
	MutexLock lock(profile_mutex);
	for (const KeyValue<StringName, ProfileEntry> &kv : profile_data) {
		if (current >= p_info_max) {
			break;
		}
		if (kv.value.last_frame_call_count == 0) {
			continue;
		}
		ProfilingInfo &info = p_info_arr[current];
		info.signature = kv.key;
		info.call_count = kv.value.last_frame_call_count;
		info.total_time = kv.value.last_frame_total_time;
		info.self_time = kv.value.last_frame_total_time;
		info.internal_time = 0;
		current++;
	}
#endif
	return current;
}
