/**************************************************************************/
/*  as_script_language.h                                                  */
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

#pragma once

#include "core/object/script_language.h"
#include "core/os/mutex.h"
#include "core/templates/hash_map.h"

class ASScriptLanguage : public ScriptLanguage {
	static ASScriptLanguage *singleton;

public:
	static ASScriptLanguage *get_singleton() { return singleton; }

	ASScriptLanguage();
	virtual ~ASScriptLanguage() override;

	virtual String get_name() const override { return "AngelScript"; }
	virtual void init() override;
	virtual void frame() override;
	virtual String get_type() const override { return "AngelScript"; }
	virtual String get_extension() const override { return "as"; }
	virtual void finish() override;

	virtual Vector<String> get_reserved_words() const override;
	virtual bool is_control_flow_keyword(const String &p_string) const override;
	virtual Vector<String> get_comment_delimiters() const override;
	virtual Vector<String> get_doc_comment_delimiters() const override;
	virtual Vector<String> get_string_delimiters() const override;

#ifdef TOOLS_ENABLED
	// 编辑器语言服务已从 ScriptLanguage 迁往 EditorLanguage；AS 暂未实现，返回默认实现。
	virtual EditorLanguage *get_editor_language() override;
#endif

	virtual bool supports_builtin_mode() const override { return false; }
	virtual String make_function(const String &p_class, const String &p_name, const PackedStringArray &p_args) const override;
	virtual Ref<Script> make_template(const String &p_template, const String &p_class_name, const String &p_base_class_name) const override;
	virtual void add_global_constant(const StringName &p_variable, const Variant &p_value) override {}

	// 照搬 GDScript 的模式（gdscript_editor.cpp:273-287）：调试器不活跃时直接返回 false。
	bool debug_break(const String &p_error, bool p_allow_continue, bool p_is_error_breakpoint = true);
	virtual Vector<StackInfo> debug_get_current_stack_info() override;

	virtual String debug_get_error() const override;
	virtual int debug_get_stack_level_count() const override;
	virtual int debug_get_stack_level_line(int p_level) const override;
	virtual String debug_get_stack_level_function(int p_level) const override;
	virtual String debug_get_stack_level_source(int p_level) const override;
	virtual void debug_get_stack_level_locals(int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems = -1, int p_max_depth = -1) override;
	virtual void debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems = -1, int p_max_depth = -1) override;
	virtual void debug_get_globals(List<String> *p_globals, List<Variant> *p_values, int p_max_subitems = -1, int p_max_depth = -1) override;
	virtual String debug_parse_stack_level_expression(int p_level, const String &p_expression, int p_max_subitems = -1, int p_max_depth = -1) override;

	virtual void reload_all_scripts() override;
	virtual void reload_scripts(const Array &p_scripts) override;
	virtual void reload_tool_script(const Ref<Script> &p_script) override;

	virtual void get_public_functions(List<MethodInfo> *p_functions) const override;
	virtual void get_public_constants(List<Pair<String, Variant>> *p_constants) const override;
	virtual void get_public_annotations(List<MethodInfo> *p_annotations) const override;

	virtual void profiling_start() override;
	virtual void profiling_stop() override;
	virtual void profiling_set_save_native_calls(bool p_enable) override {}
	virtual int profiling_get_accumulated_data(ProfilingInfo *p_info_arr, int p_info_max) override;
	virtual int profiling_get_frame_data(ProfilingInfo *p_info_arr, int p_info_max) override;

	// 性能采样内部接口。恒声明以保持调用点无需条件编译；
	// release 下 is_profiling() 恒 false、profile_function() 空操作。
	bool is_profiling() const;
	void profile_function(const StringName &p_signature, uint64_t p_usec);

private:
#ifdef DEBUG_ENABLED
	// 性能采样条目（时间单位：微秒）。与 GDScript 一致只在调试构建中存在：
	// release 下这些成员无任何读写，保留会触发 clang 的 -Wunused-private-field，
	// 并让每个实例白白构造一个 Mutex 与 HashMap。
	struct ProfileEntry {
		uint64_t call_count = 0;
		uint64_t total_time = 0;
		uint64_t frame_call_count = 0;
		uint64_t frame_total_time = 0;
		uint64_t last_frame_call_count = 0;
		uint64_t last_frame_total_time = 0;
	};

	bool profiling = false;
	Mutex profile_mutex;
	HashMap<StringName, ProfileEntry> profile_data;
#endif
};

// 语言单例的全局实例指针，供 register_types.cpp 与测试使用（定义在 as_script_language.cpp）。
extern ASScriptLanguage *script_language_as;
