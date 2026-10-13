/**************************************************************************/
/*  as_engine.h                                                           */
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

#include "binding/as_binding_plan.h"
#include "core/error/error_list.h"
#include "core/string/ustring.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"

#include <angelscript.h>

// AngelScript 引擎的唯一持有者：负责生命周期、模块编译与函数执行。
// 主线程独占（spec §4）；ScriptServer::init_languages() 在 --test / headless 路径下
// 不会被调用，因此这里采用懒初始化。
class ASEngine {
	asIScriptEngine *engine = nullptr;
	String last_error;
	// 运行期绑定计划：与 --dump-angelscript-api 消费同一份内省结果（spec §3.8.2 方案 A）。
	ASBindingPlan binding_plan;

	// M3：宿主侧兜底垃圾回收的节流状态。AS 自身的 autoGarbageCollect 仍开启
	// （增量回收），这里只在间隔到期或显式请求时做一次完整周期。
	double gc_interval_seconds = 5.0; // 0.0 表示关闭宿主节流（只保留 AS 自身回收）。
	bool gc_pending = false;
	uint64_t last_gc_usec = 0;
	int gc_count = 0;

	static void _message_callback(const asSMessageInfo *p_msg, void *p_param);
	// 编辑器诊断专用的消息回调：把结构化消息写入 p_param（Vector<ASMessage>*）。
	static void _diag_message_callback(const asSMessageInfo *p_msg, void *p_param);

	// AS 内部有一类上下文不经 create_context()：脚本全局变量初始化与脚本对象的
	// 拷贝构造 / opEquals / GC 枚举都走 asIScriptEngine::RequestContext()。接管这对
	// 回调，才能让这些路径上的异常也装上调试器回调（否则它们对调试器完全不可见）。
	static asIScriptContext *_request_context(asIScriptEngine *p_engine, void *p_param);
	static void _return_context(asIScriptEngine *p_engine, asIScriptContext *p_ctx, void *p_param);

	// 注册阶段一的最小内建 API（当前只有 as_log_int），由 ensure_initialized() 调用一次。
	void _register_builtins();
	// 内省 ClassDB 并注册绑定层（值类型 / 对象类型 / 枚举 / @GlobalScope 工具函数）。
	void _initialize_binding();

	// 递归收集 res:// 下的 .as 源路径（供离线编译工具定位输入文件）。
	static void _collect_project_script_paths_recursive(const String &p_dir, Vector<String> &r_paths, int p_depth);

	// 编译期消息缓冲：只在 _capture_messages 为真时收集，供阶段 2 提取缺失符号。
	Vector<String> compile_messages;
	bool capture_messages = false;
	// 阶段 2 重试上限：超过即全量兜底（防止无限循环）。
	static const int MAX_BINDING_RETRIES = 8;
	// 工程脚本递归扫描深度上限，避免符号链接环导致无限递归。
	static const int MAX_SCAN_DEPTH = 16;

public:
	static ASEngine *get_singleton();

	bool ensure_initialized();
	bool is_initialized() const { return engine != nullptr; }
	asIScriptEngine *get_engine() const { return engine; }

	// 上下文创建的唯一入口：内部会立刻安装调试回调，避免新增创建点时漏装。
	asIScriptContext *create_context();

	// 运行期绑定计划（ensure_initialized() 之后有效）。
	const ASBindingPlan &get_binding_plan() const { return binding_plan; }

	// 收集 res:// 下全部 .as 源文件的路径（保留 res:// 前缀，供编译工具定位输入并推导输出路径）。
	static void collect_project_script_paths(const String &p_dir, Vector<String> &r_paths);

	// 编译一个独立模块；同名模块会被整体替换。失败时把 AngelScript 诊断写入 r_error。
	bool compile_module(const String &p_name, const String &p_source, String *r_error);

	// 编辑器诊断：在一次性模块上编译源码，回传结构化消息（类型/行/列/文本）。
	// 与 compile_module 不同：不参与缺失符号补绑定、不污染 compile_messages，
	// 编译结束后临时模块被丢弃、消息回调恢复为运行期回调。Build 成功返回 true。
	struct ASMessage {
		int type = 0; // asMSGTYPE_ERROR / asMSGTYPE_WARNING / asMSGTYPE_INFORMATION
		int row = 0;
		int column = 0;
		String section;
		String message;
	};
	bool compile_for_diagnostics(const String &p_name, const String &p_source, Vector<ASMessage> *r_messages, Vector<String> *r_functions);

	// 执行一个无参全局 AS 函数并把整型返回值写入 r_ret（测试/宿主辅助入口）。
	// 内部复用 call_function()，因此与回调通道共享参数编组、返回值映射与异常日志。
	static Error execute(asIScriptEngine *p_engine, asIScriptFunction *p_func, int *r_ret);

	// 缓存跨界调用的标量参数/返回布局，避免每次派发重复 GetParam/GetReturnTypeId。
	enum ASScalarKind : uint8_t { AS_VOID, AS_BOOL, AS_INT32, AS_INT64, AS_FLOAT32, AS_FLOAT64, AS_OTHER };
	struct ASCallPlan {
		asIScriptFunction *function = nullptr;
		asUINT argument_count = 0;
		ASScalarKind arguments[4] = { AS_VOID, AS_VOID, AS_VOID, AS_VOID };
		ASScalarKind return_kind = AS_VOID;
		bool fast_scalar_eligible = false;
	};
	static ASCallPlan make_call_plan(asIScriptFunction *p_func);

	// 以 Variant 编组调用 AS 函数（ScriptInstance::callp 的通道）。只支持
	// bool / int(int64) / float(double) 三类参数；返回值按 AS 返回类型映射为 Variant。
	static Error call_function(asIScriptContext *p_context, asIScriptFunction *p_func, asIScriptObject *p_object, const Variant **p_args, int p_argc, Variant *r_ret, const ASCallPlan *p_plan = nullptr);

	void shutdown();

	// M3：宿主侧兜底垃圾回收。request_gc() 打一个待回收标记（实例析构时调用）；
	// maybe_collect_garbage() 由主循环每帧调用，在到期或有待处理请求时回收一次；
	// collect_garbage() 立即执行一次完整周期，供退出收尾使用。
	void request_gc();
	void maybe_collect_garbage();
	void collect_garbage();
	void set_gc_interval_seconds(double p_seconds);
	double get_gc_interval_seconds() const;
	int get_gc_count() const;
};
