/**************************************************************************/
/*  as_debugger.h                                                         */
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
#include "core/templates/hash_set.h"
#include "core/templates/list.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"

class asIScriptContext;
class asIScriptEngine;

// AngelScript <-> Godot 调试器的唯一桥接层。
//
// 为什么所有查询都直读「活着的」上下文而不做快照：AS 的调用栈在 Execute() 返回后
// 立刻被展开，栈帧与局部变量地址随之失效。因此异常上报与断点挂起都必须在
// SetExceptionCallback / SetLineCallback 回调仍在栈上的那一刻完成。
//
// 每个查询都提供「显式 ctx 重载 + 无参包装」两个版本：无参版读 thread_local 的
// break_context（生产路径，由编辑器在断点期间反查），带 ctx 版供测试在自己的
// 异常回调内直接调用——保证被测代码就是被用的代码。
class ASDebugger {
public:
	// 唯一的回调安装入口。本模块自己创建的上下文经 ASEngine::create_context 装配；
	// AS 内部经 RequestContext() 创建的上下文由 ASEngine 注册的上下文回调兜住。
	static void attach(asIScriptContext *p_ctx);

	// Exception attachment is permanent; line callbacks follow debugger activity.
	static bool is_attached(asIScriptContext *p_ctx);
	static bool is_line_enabled(asIScriptContext *p_ctx);
	static void sync_line_callback(asIScriptContext *p_ctx);

	static void on_exception(asIScriptContext *p_ctx, void *p_user);
	static void on_line(asIScriptContext *p_ctx, void *p_user);

	// One scope per Execute, including reentrant host calls and AS-owned contexts.
	static void enter_execution(asIScriptContext *p_ctx);
	static void leave_execution();
	class ExecutionScope {
	public:
		explicit ExecutionScope(asIScriptContext *p_ctx) { enter_execution(p_ctx); }
		~ExecutionScope() { leave_execution(); }
		ExecutionScope(const ExecutionScope &) = delete;
		ExecutionScope &operator=(const ExecutionScope &) = delete;
	};

	static void get_globals(List<String> *p_names, List<Variant> *p_values, int p_max_subitems, int p_max_depth);

	// 这个异常是否值得打断执行：脚本自己 try/catch 接住的异常不是故障。
	// 单独抽成谓词是为了可测——headless 下 EngineDebugger 不活跃，
	// on_exception 整体会提前返回，只有这个谓词能被独立断言。
	static bool is_unhandled_exception(asIScriptContext *p_ctx);

	// 本次异常是否已由调试器以富信息形式上报；读后清零。
	// call_function 用它决定要不要再打那条干巴巴的 ERR_PRINT，避免同一异常出现两条。
	// 标志按上下文记录而非一个裸 bool：上报点（on_exception）与消费点（call_function）
	// 并非一一对应——ASScriptInstance 的构造期异常只置位、永不消费。若不比对上下文，
	// 这面陈旧的旗子会把后续别处合法的 ERR_PRINT 静默吃掉。
	static void mark_exception_reported(asIScriptContext *p_ctx);
	static bool consume_exception_reported(asIScriptContext *p_ctx);

	static asIScriptContext *get_break_context();
	static String get_error();

	static Vector<ScriptLanguage::StackInfo> build_stack_info(asIScriptContext *p_ctx);

	static int get_stack_level_count();
	static int get_stack_level_count(asIScriptContext *p_ctx);
	static int get_stack_level_line(int p_level);
	static int get_stack_level_line(asIScriptContext *p_ctx, int p_level);
	static String get_stack_level_function(int p_level);
	static String get_stack_level_function(asIScriptContext *p_ctx, int p_level);
	static String get_stack_level_source(int p_level);
	static String get_stack_level_source(asIScriptContext *p_ctx, int p_level);

	static void get_stack_level_locals(int p_level, List<String> *p_names, List<Variant> *p_values, int p_max_subitems, int p_max_depth);
	static void get_stack_level_locals(asIScriptContext *p_ctx, int p_level, List<String> *p_names, List<Variant> *p_values, int p_max_subitems, int p_max_depth);

	static void get_stack_level_members(int p_level, List<String> *p_names, List<Variant> *p_values, int p_max_subitems, int p_max_depth);
	static void get_stack_level_members(asIScriptContext *p_ctx, int p_level, List<String> *p_names, List<Variant> *p_values, int p_max_subitems, int p_max_depth);

	// 只支持「表达式恰好是当前帧的某个局部变量名或 this 成员名」的查表式取值。
	// AngelScript 没有运行时求值 API，真正的表达式求值需要临时编译函数，属另一个特性。
	static String parse_stack_level_expression(int p_level, const String &p_expression);
	static String parse_stack_level_expression(asIScriptContext *p_ctx, int p_level, const String &p_expression);

	// p_addr 的语义见实现里的「一次解引用规则」注释：非句柄类型它直指数据本体，
	// 句柄类型它指向句柄槽（槽值才是编码后的对象）。
	static Variant decode_var(void *p_addr, int p_type_id, asIScriptEngine *p_engine, int p_depth, int p_max_depth, int p_max_subitems, HashSet<const void *> &r_seen);
};
