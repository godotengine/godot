/**************************************************************************/
/*  as_engine.cpp                                                         */
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

#include "as_engine.h"

#include "as_debugger.h"
#include "as_host_bridge.h"
#include "as_script_instance.h"
#include "as_script_language.h"
#include "binding/as_binding_decl.h"
#include "binding/as_binding_error_mapper.h"
#include "binding/as_binding_lazy.h"
#include "binding/as_binding_object.h"
#include "binding/as_binding_registry.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/object/object.h"
#include "core/os/os.h"
#include "core/string/print_string.h"

ASEngine *ASEngine::get_singleton() {
	static ASEngine engine_singleton;
	return &engine_singleton;
}

void ASEngine::_message_callback(const asSMessageInfo *p_msg, void *p_param) {
	ASEngine *self = static_cast<ASEngine *>(p_param);
	if (self == nullptr || p_msg == nullptr) {
		return;
	}
	if (!self->last_error.is_empty()) {
		self->last_error += "\n";
	}
	self->last_error += String(p_msg->section ? p_msg->section : "") + "(" + itos(p_msg->row) + "," + itos(p_msg->col) + "): " + String(p_msg->message ? p_msg->message : "");
	if (self->capture_messages) {
		self->compile_messages.push_back(String(p_msg->message ? p_msg->message : ""));
	}
}

void ASEngine::_diag_message_callback(const asSMessageInfo *p_msg, void *p_param) {
	Vector<ASMessage> *msgs = static_cast<Vector<ASMessage> *>(p_param);
	if (msgs == nullptr || p_msg == nullptr) {
		return;
	}
	ASMessage m;
	m.type = (int)p_msg->type;
	m.row = (int)p_msg->row;
	m.column = (int)p_msg->col;
	m.section = String(p_msg->section ? p_msg->section : "");
	m.message = String(p_msg->message ? p_msg->message : "");
	msgs->push_back(m);
}

asIScriptContext *ASEngine::_request_context(asIScriptEngine *p_engine, void *p_param) {
	if (p_engine == nullptr) {
		return nullptr;
	}
	asIScriptContext *ctx = p_engine->CreateContext();
	if (ctx != nullptr) {
		ASDebugger::attach(ctx);
		ASDebugger::enter_execution(ctx);
	}
	return ctx;
}

void ASEngine::_return_context(asIScriptEngine *p_engine, asIScriptContext *p_ctx, void *p_param) {
	if (p_ctx == nullptr) {
		return;
	}
	// Unprepare may execute script destructors through another requested context.
	ASDebugger::leave_execution();
	// 不做上下文池化：AS 内部借还很稀疏，缓存收益不抵「上下文被复用时残留状态」的风险。
	p_ctx->Unprepare();
	p_ctx->Release();
}

// 内建函数一律走泛型调用约定（asCALL_GENERIC）：设计 §3 约定绑定层统一使用泛型调用，
// 内建 API 先行遵守，避免 M2 绑定层出现两套调用约定。
static void _as_log_int(asIScriptGeneric *p_generic) {
	// 本阶段只提供 int 版本：字符串类型需要 RegisterStringFactory 与 Godot String 的编组（M2）。
	print_line(itos((int)p_generic->GetArgDWord(0)));
}

// 绑定层完成注册后才存在的调试出口：内建 `print` 是 vararg，AS 2.38 不可表达，
// 因此热更脚本的字符串输出走这里。Godot `String` 值类型以 Variant 为存储，
// 直接复用 AS_KIND_VALUE 编组。
static void _as_log_string(asIScriptGeneric *p_generic) {
	print_line(String(as_binding_marshal_arg(p_generic, 0, AS_KIND_VALUE, Variant::STRING)));
}

// M3 信号链（设计 §5）的三个宿主内建。AS 脚本类不是 Node 子类，脚本内无法引用承载它的
// 节点，因此必须由宿主提供取 owner 的入口，信号才有"发射者"。

// 取当前脚本实例承载的节点（非拥有句柄）。不在脚本执行期时返回 null。
static void _as_self(asIScriptGeneric *p_generic) {
	ASScriptInstance *inst = as_current_script_instance();
	Object *owner = (inst != nullptr) ? inst->get_owner() : nullptr;
	p_generic->SetReturnObject(as_handle_encode(owner, AS_KIND_OBJECT_NONOWNING));
}

// 在指定对象上发射信号；args 是实参的 Array（可为空）。实参存储必须在 emit_signalp
// 返回前保持有效，故先把 Array 拷进本地 Vector，再把指针数组交给引擎。
static void _as_emit_signal(asIScriptGeneric *p_generic) {
	Object *obj = as_handle_decode(p_generic->GetArgObject(0), AS_KIND_OBJECT_NONOWNING);
	if (obj == nullptr) {
		asIScriptContext *ctx = asGetActiveContext();
		if (ctx != nullptr) {
			ctx->SetException("AngelScript: as_emit_signal: target object is null or has been freed");
		}
		return;
	}
	const Variant name_v = as_binding_marshal_arg(p_generic, 1, AS_KIND_VALUE, Variant::STRING);
	const Variant args_v = as_binding_marshal_arg(p_generic, 2, AS_KIND_VALUE, Variant::ARRAY);
	if (args_v.get_type() != Variant::ARRAY) {
		asIScriptContext *ctx = asGetActiveContext();
		if (ctx != nullptr) {
			ctx->SetException("AngelScript: as_emit_signal: args must be an Array");
		}
		return;
	}
	const Array args = args_v;
	Vector<Variant> storage;
	storage.resize(args.size());
	for (int i = 0; i < args.size(); i++) {
		storage.write[i] = args[i];
	}
	Vector<const Variant *> pointers;
	pointers.resize(storage.size());
	for (int i = 0; i < storage.size(); i++) {
		pointers.write[i] = &storage[i];
	}
	obj->emit_signalp(StringName(name_v), pointers.size() > 0 ? pointers.ptrw() : nullptr, pointers.size());
}

// 生成指向宿主对象方法的 Callable，供 Object.connect 接收脚本信号。
static void _as_callable(asIScriptGeneric *p_generic) {
	Object *obj = as_handle_decode(p_generic->GetArgObject(0), AS_KIND_OBJECT_NONOWNING);
	const Variant method_v = as_binding_marshal_arg(p_generic, 1, AS_KIND_VALUE, Variant::STRING);
	const Variant ret = Callable(obj, StringName(method_v));
	as_binding_marshal_return(p_generic, AS_KIND_VALUE, ret, Variant::CALLABLE);
}

// M5 互操作 L2：AS -> 宿主的唯一入口。把 Array 展平为 Variant 数组后交给宿主桥，
// 未安装或宿主报错时返回空 Variant 并记录诊断，绝不崩溃（spec §6.2）。
static void _as_host_call(asIScriptGeneric *p_generic) {
	const int method_id = (int)p_generic->GetArgDWord(0);
	const Variant args_value = as_binding_marshal_arg(p_generic, 1, AS_KIND_VALUE, Variant::ARRAY);

	// storage 必须活到 invoke 返回：宿主按 const Variant* 读取连续元素。
	Vector<Variant> storage;
	if (args_value.get_type() == Variant::ARRAY) {
		const Array args_array = args_value;
		const int count = args_array.size();
		storage.resize(count);
		for (int i = 0; i < count; i++) {
			storage.set(i, args_array[i]);
		}
	}

	Variant ret;
	const Error err = ASHostBridge::get_singleton()->invoke(
			method_id, storage.is_empty() ? nullptr : storage.ptr(), storage.size(), &ret);
	if (err != OK) {
		ERR_PRINT(vformat("as_host_call(%d) failed: the host bridge is not installed or the host reported an error (error %d).", method_id, (int)err));
		Variant empty;
		as_binding_marshal_return(p_generic, AS_KIND_VALUE, empty);
		return;
	}
	as_binding_marshal_return(p_generic, AS_KIND_VALUE, ret);
}

bool ASEngine::ensure_initialized() {
	if (engine != nullptr) {
		return true;
	}

	engine = asCreateScriptEngine(ANGELSCRIPT_VERSION);
	if (engine == nullptr) {
		return false;
	}

	engine->SetMessageCallback(asFUNCTION(_message_callback), this, asCALL_CDECL);
	engine->SetContextCallbacks(_request_context, _return_context, nullptr);

	_register_builtins();

	// 一次性绑定初始化是启动期的大头（惰性注册下 = 值类型 + 全部对象骨架 + core + 工程扫描）。
	// 用 --verbose 记录一次，供性能回归与后续调优对照（spec §7.4）。
	const uint64_t init_begin_ms = OS::get_singleton()->get_ticks_msec();
	_initialize_binding();
	const uint64_t init_ms = OS::get_singleton()->get_ticks_msec() - init_begin_ms;
	print_verbose(vformat("ANGELSCRIPT_INIT_MS=%d", init_ms));
	return true;
}

void ASEngine::_register_builtins() {
	// 阶段一唯一的内建 API：让端到端验收脚本能调用宿主（spec §5、M1 交付项）。
	const int result = engine->RegisterGlobalFunction("void as_log_int(int value)", asFUNCTION(_as_log_int), asCALL_GENERIC);
	ERR_FAIL_COND_MSG(result < 0, vformat("Failed to register the builtin function 'as_log_int' (error %d).", result));
}

// 收集工程 res:// 下的全部 .as 源路径（相对工程根）。编辑器与导出模板都通过
// DirAccess 枚举虚拟文件系统（pck 内容同样可枚举），所以两条路径共用这一实现。
// 跳过隐藏目录（.godot/.git 等）并限制深度，避免符号链接环导致无限递归。
void ASEngine::collect_project_script_paths(const String &p_dir, Vector<String> &r_paths) {
	_collect_project_script_paths_recursive(p_dir, r_paths, 0);
}

void ASEngine::_collect_project_script_paths_recursive(const String &p_dir, Vector<String> &r_paths, int p_depth) {
	if (p_depth > MAX_SCAN_DEPTH) {
		return;
	}
	Ref<DirAccess> dir = DirAccess::open(p_dir);
	if (dir.is_null()) {
		return;
	}
	dir->list_dir_begin();
	String entry = dir->get_next();
	while (!entry.is_empty()) {
		if (entry.begins_with(".")) {
			entry = dir->get_next();
			continue;
		}
		const String full_path = p_dir.path_join(entry);
		if (dir->current_is_dir()) {
			_collect_project_script_paths_recursive(full_path, r_paths, p_depth + 1);
		} else if (entry.get_extension().to_lower() == "as") {
			r_paths.push_back(full_path);
		}
		entry = dir->get_next();
	}
	dir->list_dir_end();
}

void ASEngine::_initialize_binding() {
	// 计划构建会读取 angel_script/class_whitelist 与 class_blacklist（默认全放行）。
	binding_plan.build(ASBindingScope::from_project_settings());

	// 阶段 1：core + 工程扫描成员。scan_project 打开时把 res:// 下全部 .as 源交给词法扫描，
	// 命中类在编译前就被按需注册；未命中的漏网之鱼由阶段 2（编译失败重试/全量兜底）补齐。
	Vector<String> sources;
	if (ASBindingLazyRegistry::scan_project_enabled()) {
		Vector<String> project_paths;
		collect_project_script_paths("res://", project_paths);
		sources.resize(project_paths.size());
		for (int i = 0; i < project_paths.size(); i++) {
			sources.set(i, FileAccess::get_file_as_string(project_paths[i]));
		}
	}
	ASBindingLazyRegistry::get_singleton()->ensure_initialized(binding_plan, engine, sources);

	// as_log_string 的签名用到绑定层的 Godot `String` 值类型，必须等到值类型注册之后
	// 才能注册（否则 AS 会因未知类型报错并把引擎标记为配置错误且不可恢复）。
	const int result = engine->RegisterGlobalFunction("void as_log_string(const String &in value)", asFUNCTION(_as_log_string), asCALL_GENERIC);
	ERR_FAIL_COND_MSG(result < 0, vformat("Failed to register the builtin function 'as_log_string' (error %d).", result));

	// M3 信号链的内建：Object/String/Array/Callable 此时都已由绑定层注册完毕。
	const int result_self = engine->RegisterGlobalFunction("Object @as_self()", asFUNCTION(_as_self), asCALL_GENERIC);
	ERR_FAIL_COND_MSG(result_self < 0, vformat("Failed to register the builtin function 'as_self' (error %d).", result_self));
	const int result_emit = engine->RegisterGlobalFunction("void as_emit_signal(Object @obj, const String &in name, const Array &in args)", asFUNCTION(_as_emit_signal), asCALL_GENERIC);
	ERR_FAIL_COND_MSG(result_emit < 0, vformat("Failed to register the builtin function 'as_emit_signal' (error %d).", result_emit));
	const int result_callable = engine->RegisterGlobalFunction("Callable as_callable(Object @obj, const String &in method)", asFUNCTION(_as_callable), asCALL_GENERIC);
	ERR_FAIL_COND_MSG(result_callable < 0, vformat("Failed to register the builtin function 'as_callable' (error %d).", result_callable));

	// M5 互操作 L2：签名用到 Array/Variant 值类型，必须等绑定层就绪后注册。
	const int result_host_call = engine->RegisterGlobalFunction("Variant as_host_call(int method_id, const Array &in args)", asFUNCTION(_as_host_call), asCALL_GENERIC);
	ERR_FAIL_COND_MSG(result_host_call < 0, vformat("Failed to register the builtin function 'as_host_call' (error %d).", result_host_call));
}

bool ASEngine::compile_module(const String &p_name, const String &p_source, String *r_error) {
	ERR_FAIL_NULL_V(engine, false);

	last_error = String();
	CharString module_cstr = p_name.utf8();
	CharString src = p_source.utf8();

	// 阶段 2：编译失败且缺失绑定可识别时，增量注册后丢弃模块重编；无进展/达上限则全量兜底后重编一次。
	// 用户脚本自身错误在兜底后原样报出（不掩盖、不放大）。
	for (int attempt = 0; attempt <= MAX_BINDING_RETRIES; attempt++) {
		last_error = String();
		compile_messages.clear();
		capture_messages = true;

		asIScriptModule *mod = engine->GetModule(module_cstr.get_data(), asGM_ALWAYS_CREATE);
		if (mod == nullptr) {
			capture_messages = false;
			if (r_error) {
				*r_error = "failed to create AngelScript module: " + p_name;
			}
			return false;
		}
		if (mod->AddScriptSection(module_cstr.get_data(), src.get_data(), src.length()) < 0) {
			capture_messages = false;
			if (r_error) {
				*r_error = "failed to add script section: " + p_name;
			}
			engine->DiscardModule(module_cstr.get_data());
			return false;
		}
		const int build_result = mod->Build();
		capture_messages = false;
		if (build_result >= 0) {
			return true;
		}

		// 失败：尝试按缺失符号做增量补齐。
		const bool full_before = ASBindingLazyRegistry::get_singleton()->is_full();
		if (!full_before) {
			ASBindingMissingSymbols missing;
			if (ASBindingErrorMapper::extract(compile_messages, binding_plan, &missing) && !missing.types.is_empty()) {
				const int64_t added = ASBindingLazyRegistry::get_singleton()->ensure_registered(binding_plan, missing.types, engine);
				if (added > 0) {
					engine->DiscardModule(module_cstr.get_data());
					continue; // 有进展：重编。
				}
			}
			// 无进展或无法识别：全量兜底后重编一次。
			ASBindingLazyRegistry::get_singleton()->register_all(binding_plan, engine);
			engine->DiscardModule(module_cstr.get_data());
			continue;
		}

		// 已兜底仍失败：用户脚本自身错误，原样报出。
		if (r_error) {
			*r_error = last_error.is_empty() ? "AngelScript build failed: " + p_name : last_error;
		}
		engine->DiscardModule(module_cstr.get_data());
		return false;
	}

	// 理论上不可达：循环内每次 continue 前后都有终止条件。
	if (r_error) {
		*r_error = last_error.is_empty() ? "AngelScript build failed: " + p_name : last_error;
	}
	engine->DiscardModule(module_cstr.get_data());
	return false;
}

bool ASEngine::compile_for_diagnostics(const String &p_name, const String &p_source, Vector<ASMessage> *r_messages, Vector<String> *r_functions) {
	if (!ensure_initialized()) {
		if (r_messages != nullptr) {
			ASMessage m;
			m.type = asMSGTYPE_ERROR;
			m.message = "AngelScript engine failed to initialize.";
			r_messages->push_back(m);
		}
		return false;
	}

	// 编辑器诊断只需要结构化消息，不触发惰性补绑定；用独立回调收集后恢复常驻回调。
	Vector<ASMessage> msgs;
	engine->SetMessageCallback(asFUNCTION(_diag_message_callback), &msgs, asCALL_CDECL);

	const CharString name = p_name.utf8();
	const CharString source = p_source.utf8();
	bool ok = false;
	asIScriptModule *mod = engine->GetModule(name.get_data(), asGM_ALWAYS_CREATE);
	if (mod != nullptr) {
		mod->AddScriptSection("editor", source.get_data());
		ok = mod->Build() >= 0;
		if (ok && r_functions != nullptr) {
			const asUINT function_count = mod->GetFunctionCount();
			for (asUINT i = 0; i < function_count; i++) {
				asIScriptFunction *func = mod->GetFunctionByIndex(i);
				// 只列出全局命名空间里的自由函数（供编辑器大纲），排除类方法。
				// 注意：AS 对无命名空间的函数返回「非空指针的空串」而非 nullptr。
				const char *func_ns = func != nullptr ? func->GetNamespace() : nullptr;
				if (func == nullptr || func->GetObjectType() != nullptr || (func_ns != nullptr && func_ns[0] != '\0')) {
					continue;
				}
				r_functions->push_back(String(func->GetName()));
			}
		}
		engine->DiscardModule(name.get_data());
	}

	engine->SetMessageCallback(asFUNCTION(_message_callback), this, asCALL_CDECL);
	if (r_messages != nullptr) {
		r_messages->append_array(msgs);
	}
	return ok;
}

asIScriptContext *ASEngine::create_context() {
	if (engine == nullptr) {
		return nullptr;
	}
	asIScriptContext *ctx = engine->CreateContext();
	if (ctx != nullptr) {
		ASDebugger::attach(ctx);
	}
	return ctx;
}

Error ASEngine::execute(asIScriptEngine *p_engine, asIScriptFunction *p_func, int *r_ret) {
	ERR_FAIL_NULL_V(p_engine, ERR_INVALID_PARAMETER);
	ERR_FAIL_NULL_V(p_func, ERR_INVALID_PARAMETER);

	// 经统一入口创建，保证调试回调一定装上。
	asIScriptContext *ctx = get_singleton()->create_context();
	ERR_FAIL_NULL_V(ctx, ERR_CANT_CREATE);

	// 无参全局函数 = 以空对象、零参调用 call_function，复用其参数编组、返回值映射
	// （bool/int32/int64/float/double）与异常日志；旧实现无条件 GetReturnDWord()，
	// 对 void/bool/double 返回值都会给出错误结果。
	Variant ret;
	const Error err = call_function(ctx, p_func, nullptr, nullptr, 0, &ret);
	ctx->Release();
	if (err != OK) {
		return err;
	}
	if (r_ret != nullptr) {
		*r_ret = (ret.get_type() == Variant::INT) ? (int)(int64_t)ret : 0;
	}
	return OK;
}

void ASEngine::shutdown() {
	if (engine == nullptr) {
		return;
	}
	engine->ShutDownAndRelease();
	engine = nullptr;
	last_error = String();
}

void ASEngine::request_gc() {
	gc_pending = true;
}

void ASEngine::collect_garbage() {
	if (!is_initialized() || engine == nullptr) {
		return;
	}
	// 先清待回收标志，再执行 GC：GarbageCollect 会同步析构 AS 脚本对象，其析构
	// 可能回调宿主并再次 request_gc()（例如脚本实例析构）。若在 GC 之后清标志，
	// 这次新产生的请求会被吞掉；当 interval=0 时该请求将再无自动消费路径。
	gc_pending = false;
	engine->GarbageCollect(asGC_FULL_CYCLE);
	last_gc_usec = OS::get_singleton()->get_ticks_usec();
	gc_count++;
	print_verbose("[AngelScript] garbage collect");
}

void ASEngine::maybe_collect_garbage() {
	if (!is_initialized() || engine == nullptr) {
		return;
	}
	if (gc_pending) {
		collect_garbage();
		return;
	}
	if (gc_interval_seconds <= 0.0) {
		return;
	}
	const uint64_t now = OS::get_singleton()->get_ticks_usec();
	const uint64_t interval_usec = (uint64_t)(gc_interval_seconds * 1000000.0);
	if (now - last_gc_usec >= interval_usec) {
		collect_garbage();
	}
}

void ASEngine::set_gc_interval_seconds(double p_seconds) {
	gc_interval_seconds = p_seconds;
}

double ASEngine::get_gc_interval_seconds() const {
	return gc_interval_seconds;
}

int ASEngine::get_gc_count() const {
	return gc_count;
}

// 构造编辑器 Profiler 面板展示用的签名。面板按 "<path>::<line>::<func>" 三段解析
// （script_editor_debugger.cpp:872-881），段数不对会落进空分支，导致行名空白、
// 源码跳转失效，所以必须与 GDScript 一样补上声明行号这一段。
// 全局/静态函数为 "<module>::<line>::<func>"，对象方法为 "<module>::<line>::<Class>.<func>"。
static StringName _make_profile_signature(asIScriptFunction *p_func) {
	const char *module_name = p_func->GetModuleName();
	String signature = (module_name != nullptr && module_name[0] != '\0') ? String(module_name) : String("<module>");

	// GetDeclaredAt 对非脚本函数返回 asNOT_SUPPORTED 并把 row 置 0，可安全直接取用。
	int row = 0;
	p_func->GetDeclaredAt(nullptr, &row, nullptr);
	signature += "::" + itos(row);

	const char *class_name = nullptr;
	asITypeInfo *type_info = p_func->GetObjectType();
	if (type_info != nullptr) {
		class_name = type_info->GetName();
	}

	const char *func_name = p_func->GetName();
	if (class_name != nullptr) {
		signature += "::" + String(class_name) + "." + String(func_name != nullptr ? func_name : "<anonymous>");
	} else {
		signature += "::" + String(func_name != nullptr ? func_name : "<anonymous>");
	}
	return StringName(signature);
}

static ASEngine::ASScalarKind scalar_kind_for_type(int p_type_id) {
	switch (p_type_id) {
		case asTYPEID_VOID: return ASEngine::AS_VOID;
		case asTYPEID_BOOL: return ASEngine::AS_BOOL;
		case asTYPEID_INT32: case asTYPEID_UINT32: return ASEngine::AS_INT32;
		case asTYPEID_INT64: case asTYPEID_UINT64: return ASEngine::AS_INT64;
		case asTYPEID_FLOAT: return ASEngine::AS_FLOAT32;
		case asTYPEID_DOUBLE: return ASEngine::AS_FLOAT64;
		default: return ASEngine::AS_OTHER;
	}
}

ASEngine::ASCallPlan ASEngine::make_call_plan(asIScriptFunction *p_func) {
	ASCallPlan plan;
	plan.function = p_func;
	if (p_func == nullptr) {
		return plan;
	}
	plan.argument_count = p_func->GetParamCount();
	plan.fast_scalar_eligible = plan.argument_count <= 4;
	for (asUINT i = 0; i < MIN(4u, plan.argument_count); i++) {
		int type_id = asTYPEID_VOID;
		asDWORD flags = 0;
		p_func->GetParam(i, &type_id, &flags);
		plan.arguments[i] = scalar_kind_for_type(type_id);
		if (plan.arguments[i] == AS_OTHER || (flags & asTM_INOUTREF)) {
			plan.fast_scalar_eligible = false;
		}
	}
	asDWORD return_flags = 0;
	plan.return_kind = scalar_kind_for_type(p_func->GetReturnTypeId(&return_flags));
	if (plan.return_kind == AS_OTHER || (return_flags & asTM_INOUTREF)) {
		plan.fast_scalar_eligible = false;
	}
	return plan;
}

Error ASEngine::call_function(asIScriptContext *p_context, asIScriptFunction *p_func, asIScriptObject *p_object, const Variant **p_args, int p_argc, Variant *r_ret, const ASCallPlan *p_plan) {
	ERR_FAIL_NULL_V(p_context, ERR_INVALID_PARAMETER);
	ERR_FAIL_NULL_V(p_func, ERR_INVALID_PARAMETER);
	// Snapshot before Execute: a reentrant call may grow the instance's plan map.
	const ASCallPlan plan = p_plan != nullptr ? *p_plan : ASCallPlan();
	const bool use_plan = plan.function == p_func && plan.fast_scalar_eligible;
	const asUINT expected_argc = use_plan ? plan.argument_count : p_func->GetParamCount();
	ERR_FAIL_COND_V(p_argc < 0 || (asUINT)p_argc != expected_argc, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_argc > 0 && p_args == nullptr, ERR_INVALID_PARAMETER);
	if (!use_plan) {
		asDWORD return_flags = 0;
		p_func->GetReturnTypeId(&return_flags);
		if (return_flags & asTM_INOUTREF) {
			return ERR_INVALID_PARAMETER;
		}
	}

	// M3：信号/回调会在脚本执行过程中再次进入脚本（宿主 Callable → Object::callp →
	// 本函数），此时复用的 asIScriptContext 仍处于 asEXECUTION_ACTIVE，直接 Prepare
	// 会返回 asCONTEXT_ACTIVE 导致调用失败（表现为「信号回调 Method not found」）。
	// 先用 PushState() 把外层执行状态压栈、让上下文回到可 Prepare 的状态，调用结束后
	// 再用 PopState() 还原；PopState() 内部会负责清理本次调用产生的栈。
	const bool nested = p_context->GetState() == asEXECUTION_ACTIVE;
	if (nested && p_context->PushState() < 0) {
		return ERR_CANT_CREATE;
	}
	// 采样仅在性能分析器开启时进行：关闭时只多一次指针判空。
	ASScriptLanguage *profile_lang = ASScriptLanguage::get_singleton();
	uint64_t profile_start = 0;
	if (profile_lang != nullptr && profile_lang->is_profiling()) {
		profile_start = OS::get_singleton()->get_ticks_usec();
	}
	// 统一收尾：嵌套调用的清理交给 PopState()，非嵌套维持原行为（Unprepare）。
	// 采样在此写回，因此所有已进入执行阶段的退出分支都会被采集
	// （前置的 ERR_FAIL_* 与 PushState 失败发生在采样起点之前，脚本未执行，不采样）。
	auto finish = [&](Error p_result) {
		if (profile_start != 0) {
			// 必须先取完耗时再构造签名：StringName 构造要拼 String 并加全局表锁，
			// 开销达微秒级；而函数实参求值顺序未定义，交给编译器排就可能把采样器
			// 自身开销计进被测函数（小函数会被放大数倍）。
			// 只计脚本执行本身（Prepare + 参数编组 + Execute），清理不计入。
			const uint64_t elapsed = OS::get_singleton()->get_ticks_usec() - profile_start;
			profile_lang->profile_function(_make_profile_signature(p_func), elapsed);
		}
		if (nested) {
			p_context->PopState();
		} else {
			p_context->Unprepare();
		}
		return p_result;
	};

	if (p_context->Prepare(p_func) < 0) {
		return finish(ERR_CANT_CREATE);
	}
	if (p_object != nullptr) {
		p_context->SetObject(p_object);
	}

	for (int i = 0; i < p_argc; i++) {
		if (p_args[i] == nullptr) {
			return finish(ERR_INVALID_PARAMETER);
		}
		const Variant &arg = *p_args[i];
		ASScalarKind kind = AS_OTHER;
		if (use_plan) {
			kind = plan.arguments[i];
		} else {
			int type_id = asTYPEID_VOID;
			asDWORD flags = 0;
			p_func->GetParam(i, &type_id, &flags);
			// The Variant bridge has no reference/out storage contract.
			if (flags & asTM_INOUTREF) {
				return finish(ERR_INVALID_PARAMETER);
			}
			kind = scalar_kind_for_type(type_id);
		}
		int set_result = asSUCCESS;
		switch (kind) {
			case AS_BOOL: if (arg.get_type()!=Variant::BOOL) return finish(ERR_INVALID_PARAMETER); set_result=p_context->SetArgByte(i,arg.operator bool()?1:0); break;
			case AS_INT32: if (arg.get_type()!=Variant::INT) return finish(ERR_INVALID_PARAMETER); set_result=p_context->SetArgDWord(i,(asDWORD)(int32_t)arg.operator int64_t()); break;
			case AS_INT64: if (arg.get_type()!=Variant::INT) return finish(ERR_INVALID_PARAMETER); set_result=p_context->SetArgQWord(i,(asQWORD)arg.operator int64_t()); break;
			case AS_FLOAT32: if (arg.get_type()!=Variant::FLOAT) return finish(ERR_INVALID_PARAMETER); set_result=p_context->SetArgFloat(i,(float)arg.operator double()); break;
			case AS_FLOAT64: if (arg.get_type()!=Variant::FLOAT) return finish(ERR_INVALID_PARAMETER); set_result=p_context->SetArgDouble(i,arg.operator double()); break;
			default: return finish(ERR_INVALID_PARAMETER);
		}
		if (set_result < 0) return finish(ERR_INVALID_PARAMETER);
	}

	int executed;
	{
		ASDebugger::ExecutionScope debug_scope(p_context);
		executed = p_context->Execute();
	}
	if (executed != asEXECUTION_FINISHED) {
		// 调试器已经以「富信息 + 调用栈」的形式上报过了，再打一条干巴巴的 ERR_PRINT
		// 会让错误面板出现重复条目。非调试运行时 consume 恒为 false，行为与以前完全一致。
		if (!ASDebugger::consume_exception_reported(p_context)) {
			// 静默吞掉异常会让脚本错误在宿主侧完全不可见（callp 只把它翻译成一个泛化错误码）。
			const char *exception = p_context->GetExceptionString();
			const char *section = nullptr;
			const int line = p_context->GetExceptionLineNumber(nullptr, &section);
			ERR_PRINT(vformat("AngelScript call '%s' failed: %s (%s:%d).",
					p_func->GetName() != nullptr ? p_func->GetName() : "<anonymous>",
					exception != nullptr ? exception : "execution did not finish",
					section != nullptr ? section : "<script>",
					line));
		}
		return finish(FAILED);
	}

	if (r_ret != nullptr) {
		const int return_type = use_plan ? (int)plan.return_kind : (int)scalar_kind_for_type(p_func->GetReturnTypeId());
		switch (return_type) {
			case AS_VOID: *r_ret = Variant(); break;
			case AS_BOOL: *r_ret = Variant(p_context->GetReturnByte() != 0); break;
			case AS_INT32: *r_ret = Variant((int64_t)(int32_t)p_context->GetReturnDWord()); break;
			case AS_INT64: *r_ret = Variant((int64_t)p_context->GetReturnQWord()); break;
			case AS_FLOAT32: *r_ret = Variant((double)p_context->GetReturnFloat()); break;
			case AS_FLOAT64: *r_ret = Variant(p_context->GetReturnDouble()); break;
			default:
				*r_ret = Variant();
				break;
		}
	}

	return finish(OK);
}
