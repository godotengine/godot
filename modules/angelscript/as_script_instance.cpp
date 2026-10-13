/**************************************************************************/
/*  as_script_instance.cpp                                                */
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

#include "as_script_instance.h"

#include "as_engine.h"
#include "as_debugger.h"
#include "as_script.h"
#include "as_script_language.h"
#include "binding/as_binding_decl.h"
#include "binding/as_binding_native_storage.h"
#include "binding/as_binding_object.h"

#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/variant/variant.h"

// 对象句柄属性的拥有/非拥有判据与绑定层、跳板必须完全一致：静态类型派生自
// RefCounted 的句柄由 AS 引用计数保活，其余（含 Object 自身）只是借用。
static ASBindingKind _object_kind_for_type_name(const String &p_type_name) {
	return ClassDB::is_parent_class(p_type_name, "RefCounted")
			? AS_KIND_OBJECT_OWNING
			: AS_KIND_OBJECT_NONOWNING;
}

// M3：脚本执行期间指向“当前正在运行的脚本实例”。AS 脚本类不是 Node 子类，脚本内无法
// 引用承载它的节点；宿主内建 as_self() 靠这个指针取 owner。用 RAII 在每次进入/离开脚本
// 调用时保存与恢复，嵌套调用（脚本回调里再触发脚本调用）也能正确回退到外层实例。
static ASScriptInstance *g_current_script_instance = nullptr;

namespace {
struct ASInstanceScope {
	ASScriptInstance *previous = nullptr;
	explicit ASInstanceScope(ASScriptInstance *p_instance) :
			previous(g_current_script_instance) {
		g_current_script_instance = p_instance;
	}
	~ASInstanceScope() {
		g_current_script_instance = previous;
	}
};
} // namespace

ASScriptInstance *as_current_script_instance() {
	return g_current_script_instance;
}

// M3：`signal_<name>` 方法只是信号声明，不是可调用方法。ASScript::has_method /
// get_script_method_list 已过滤，而 Object::has_method / Object::call 会先问脚本实例，
// 所以实例层也必须过滤，否则会出现"方法列表里没有、却能被 has_method/call 命中"的幻影方法。
static bool _is_signal_declaration(const StringName &p_method) {
	return String(p_method).begins_with("signal_");
}

ASScriptInstance::ASScriptInstance(const Ref<ASScript> &p_script, Object *p_owner) {
	script = p_script;
	owner = p_owner;

	asITypeInfo *type = script->get_type_info();
	ERR_FAIL_NULL_MSG(type, "AngelScript type info is missing for the script instance.");

	asIScriptEngine *engine = ASEngine::get_singleton()->get_engine();
	ERR_FAIL_NULL_MSG(engine, "AngelScript engine is not initialized.");

	context = ASEngine::get_singleton()->create_context();
	ERR_FAIL_NULL_MSG(context, "Failed to create an AngelScript context.");

	// 默认构造（无参 factory）必须在此完成：之后引擎只按名字调方法、按索引读写属性。
	// 不能直接取 GetFactoryByIndex(0)：脚本类若只声明了带参构造，索引 0 就是那个带参
	// factory，Prepare/Execute 会在形参未赋值的情况下跑出一个"看似成功"的错误对象。
	asIScriptFunction *factory = nullptr;
	for (asUINT i = 0; i < type->GetFactoryCount(); i++) {
		asIScriptFunction *candidate = type->GetFactoryByIndex(i);
		if (candidate != nullptr && candidate->GetParamCount() == 0) {
			factory = candidate;
			break;
		}
	}
	if (factory == nullptr) {
		// 具名局部量承载 utf8 缓冲：临时 CharString 的 data() 不能跨语句使用（Task 3 同法）。
		CharString default_decl = (String(type->GetName()) + "@ " + String(type->GetName()) + "()").utf8();
		factory = type->GetFactoryByDecl(default_decl.get_data());
	}
	ERR_FAIL_NULL_MSG(factory, "AngelScript class has no default factory (default constructor).");

	if (context->Prepare(factory) < 0) {
		ERR_FAIL_MSG("Failed to prepare the AngelScript factory.");
	}
	int factory_result;
	{
		ASDebugger::ExecutionScope debug_scope(context);
		factory_result = context->Execute();
	}
	if (factory_result != asEXECUTION_FINISHED) {
		context->Unprepare();
		ERR_FAIL_MSG("AngelScript constructor execution failed.");
	}

	object = *(asIScriptObject **)context->GetAddressOfReturnValue();
	if (object != nullptr) {
		// 先自持一份再 Unprepare：避免上下文释放返回值时把对象一起带走。
		object->AddRef();
	}
	context->Unprepare();

#ifdef TOOLS_ENABLED
	// 注册到脚本：编辑器热重载需枚举所有承载对象，先摘除再重编译，避免模块被丢弃后对象悬垂。
	script->_add_instance(this);
#endif
}

ASScriptInstance::~ASScriptInstance() {
#ifdef TOOLS_ENABLED
	if (script.is_valid()) {
		script->_remove_instance(this);
	}
#endif
	if (object != nullptr) {
		object->Release();
		object = nullptr;
	}
	if (context != nullptr) {
		context->Release();
		context = nullptr;
	}
	// M3：脚本实例销毁意味着可能有一批纯脚本对象失去外部引用，
	// 请求宿主在下一帧做一次完整回收（避免等到节流间隔到期）。
	ASEngine *as_engine = ASEngine::get_singleton();
	if (as_engine != nullptr) {
		as_engine->request_gc();
	}
}

bool ASScriptInstance::_find_property(const StringName &p_name, int *r_index, int *r_type_id) const {
	asITypeInfo *type = script->get_type_info();
	if (type == nullptr) {
		return false;
	}
	for (asUINT i = 0; i < type->GetPropertyCount(); i++) {
		// asITypeInfo 用 GetProperty(index, &name, &type_id) 枚举属性；
		// GetPropertyName/GetPropertyTypeId 是 asIScriptObject（实例接口）的成员。
		const char *prop_name = nullptr;
		int type_id = 0;
		if (type->GetProperty(i, &prop_name, &type_id) < 0 || prop_name == nullptr) {
			continue;
		}
		if (String(p_name) == String(prop_name)) {
			if (r_index != nullptr) {
				*r_index = (int)i;
			}
			if (r_type_id != nullptr) {
				*r_type_id = type_id;
			}
			return true;
		}
	}
	return false;
}

int ASScriptInstance::_expected_param_count(const StringName &p_method) {
	// 引擎只按固定签名派发这几个 GDVIRTUAL 回调（scene/main/node.h:433-437）；
	// 签名不匹配就当方法不存在，否则 call() 会因参数个数校验失败而报错。
	if (p_method == "_ready" || p_method == "_enter_tree" || p_method == "_exit_tree") {
		return 0;
	}
	if (p_method == "_process" || p_method == "_physics_process" || p_method == "_notification") {
		return 1;
	}
	return -1;
}

uint64_t ASScriptInstance::method_resolution_count_value = 0;
uint64_t ASScriptInstance::call_plan_analysis_count_value = 0;

uint64_t ASScriptInstance::method_resolution_count() { return method_resolution_count_value; }
uint64_t ASScriptInstance::call_plan_analysis_count() { return call_plan_analysis_count_value; }
void ASScriptInstance::reset_method_resolution_count() { method_resolution_count_value = 0; call_plan_analysis_count_value = 0; }

const ASEngine::ASCallPlan *ASScriptInstance::_resolve_call_plan(const StringName &p_method) const {
	HashMap<StringName, ASEngine::ASCallPlan>::ConstIterator cached = method_resolution_cache.find(p_method);
	if (cached) return &cached->value;
	method_resolution_count_value++;
	asIScriptFunction *func = nullptr;
	if (!_is_signal_declaration(p_method)) {
		asITypeInfo *type = script->get_type_info();
		if (type != nullptr) {
			CharString name = String(p_method).utf8();
			func = type->GetMethodByName(name.get_data(), true);
			if (func != nullptr) {
				const int expected = _expected_param_count(p_method);
				if (expected >= 0 && (int)func->GetParamCount() != expected) func = nullptr;
			}
		}
	}
	ASEngine::ASCallPlan plan;
	if (func != nullptr) { plan = ASEngine::make_call_plan(func); call_plan_analysis_count_value++; }
	method_resolution_cache.insert(p_method, plan);
	return method_resolution_cache.getptr(p_method);
}

bool ASScriptInstance::set(const StringName &p_name, const Variant &p_value) {
	if (object == nullptr) {
		return false;
	}
	int index = -1;
	int type_id = 0;
	if (!_find_property(p_name, &index, &type_id)) {
		return false;
	}
	void *addr = object->GetAddressOfProperty((asUINT)index);
	if (addr == nullptr) {
		return false;
	}
	asIScriptEngine *engine = ASEngine::get_singleton()->get_engine();
	if (engine == nullptr) {
		return false;
	}

	if (type_id & asTYPEID_OBJHANDLE) {
		// 对象句柄属性：槽是 sizeof(void*) 的句柄，编码方式由静态类型决定。
		// 不能走 engine->AssignScriptObject()：本模块的对象类型只注册了 addref/release
		// （asOBJ_REF，无 opAssign/POD），AssignScriptObject 对引用类型会返回
		// asNOT_SUPPORTED 并不改槽（且无活动上下文时连异常都不抛，会静默失败）。
		// 这里按 AS 赋句柄的语义手工维护计数：先更新槽，再 release 旧对象、addref 新对象。
		asITypeInfo *ti = engine->GetTypeInfoById(type_id);
		if (ti == nullptr) {
			return false;
		}
		const ASBindingKind kind = _object_kind_for_type_name(String(ti->GetName()));
		Object *new_obj = p_value.operator Object *();
		if (new_obj != nullptr && kind == AS_KIND_OBJECT_OWNING && Object::cast_to<RefCounted>(new_obj) == nullptr) {
			// 句柄静态类型派生自 RefCounted，但实参不是引用计数对象（如把 Node 塞进 Resource@）。
			// 继续下去会在非 RefCounted 上调 init_ref()，是未定义行为，直接拒绝。
			return false;
		}
		void **slot = (void **)addr;
		Object *old_obj = as_handle_decode(*slot, kind);
		if (old_obj == new_obj) {
			return true; // 同一对象：槽内容等价，无需调整计数。
		}
		*slot = as_handle_encode(new_obj, kind);
		if (old_obj != nullptr) {
			engine->ReleaseScriptObject(old_obj, ti);
		}
		if (new_obj != nullptr) {
			engine->AddRefScriptObject(new_obj, ti);
		}
		return true;
	}

	if (type_id & asTYPEID_MASK_OBJECT) {
		asITypeInfo *ti = engine->GetTypeInfoById(type_id);
		if (ti == nullptr) {
			return false;
		}
		// 绑定层的值类型槽分两类：原生存储类型（Vector2/3/2i/3i）槽里是原生 T
		// （8/12 字节），其余（String/Array/Rect2…）槽里是 sizeof(Variant) 的 Variant。
		// 小写内建 `string` 例外：它是 sizeof(void*) 的 intern 句柄，不是 Variant 存储，本轮不做往返。
		if (String(ti->GetName()) == "string") {
			return false;
		}
		const Variant::Type native_type = ASBindingDecl::as_name_to_variant_type(String(ti->GetName()));
		if (ASNativeValueStorage::is_native_storage_type(native_type)) {
			ASNativeValueStorage::variant_to_native(native_type, p_value, addr);
			return true;
		}
		*(Variant *)addr = p_value;
		return true;
	}

	switch (type_id & ~asTYPEID_OBJHANDLE) {
		case asTYPEID_BOOL:
			*(bool *)addr = p_value.operator bool();
			return true;
		case asTYPEID_INT32:
			*(int32_t *)addr = (int32_t)p_value.operator int64_t();
			return true;
		case asTYPEID_INT64:
			*(int64_t *)addr = p_value.operator int64_t();
			return true;
		case asTYPEID_FLOAT:
			*(float *)addr = (float)p_value.operator double();
			return true;
		case asTYPEID_DOUBLE:
			*(double *)addr = p_value.operator double();
			return true;
		default:
			// 其它未绑定的引用型属性（如函数指针）不支持。
			return false;
	}
}

bool ASScriptInstance::get(const StringName &p_name, Variant &r_ret) const {
	if (object == nullptr) {
		return false;
	}
	int index = -1;
	int type_id = 0;
	if (!_find_property(p_name, &index, &type_id)) {
		return false;
	}
	void *addr = object->GetAddressOfProperty((asUINT)index);
	if (addr == nullptr) {
		return false;
	}
	asIScriptEngine *engine = ASEngine::get_singleton()->get_engine();
	if (engine == nullptr) {
		return false;
	}

	if (type_id & asTYPEID_OBJHANDLE) {
		asITypeInfo *ti = engine->GetTypeInfoById(type_id);
		if (ti == nullptr) {
			return false;
		}
		const ASBindingKind kind = _object_kind_for_type_name(String(ti->GetName()));
		Object *o = as_handle_decode(*((void **)addr), kind);
		r_ret = Variant(o);
		return true;
	}

	if (type_id & asTYPEID_MASK_OBJECT) {
		asITypeInfo *ti = engine->GetTypeInfoById(type_id);
		if (ti == nullptr) {
			return false;
		}
		// 小写内建 `string` 不是 Variant 存储，不做往返（见 set()）。
		if (String(ti->GetName()) == "string") {
			return false;
		}
		const Variant::Type native_type = ASBindingDecl::as_name_to_variant_type(String(ti->GetName()));
		if (ASNativeValueStorage::is_native_storage_type(native_type)) {
			r_ret = ASNativeValueStorage::native_to_variant(native_type, addr);
			return true;
		}
		r_ret = *(const Variant *)addr;
		return true;
	}

	// AS 的内存布局：bool 1 字节、int 4 字节、float 4 字节；这里按各自宽度读，避免越界。
	switch (type_id & ~asTYPEID_OBJHANDLE) {
		case asTYPEID_BOOL:
			r_ret = Variant(*(const bool *)addr);
			return true;
		case asTYPEID_INT32:
			r_ret = Variant((int64_t)*(const int32_t *)addr);
			return true;
		case asTYPEID_INT64:
			r_ret = Variant((int64_t)*(const int64_t *)addr);
			return true;
		case asTYPEID_FLOAT:
			r_ret = Variant((double)*(const float *)addr);
			return true;
		case asTYPEID_DOUBLE:
			r_ret = Variant(*(const double *)addr);
			return true;
		default:
			return false;
	}
}

void ASScriptInstance::get_property_list(List<PropertyInfo> *p_properties) const {
	script->get_script_property_list(p_properties);
}

Variant::Type ASScriptInstance::get_property_type(const StringName &p_name, bool *r_is_valid) const {
	int index = -1;
	int type_id = 0;
	if (!_find_property(p_name, &index, &type_id)) {
		if (r_is_valid != nullptr) {
			*r_is_valid = false;
		}
		return Variant::NIL;
	}

	if (type_id & asTYPEID_OBJHANDLE) {
		// 对象句柄一律报 OBJECT（拥有与否由 _find_property 的静态类型决定，对外不可见）。
		if (r_is_valid != nullptr) {
			*r_is_valid = true;
		}
		return Variant::OBJECT;
	}

	if (type_id & asTYPEID_MASK_OBJECT) {
		asIScriptEngine *engine = ASEngine::get_singleton()->get_engine();
		asITypeInfo *ti = engine != nullptr ? engine->GetTypeInfoById(type_id) : nullptr;
		if (ti == nullptr || object == nullptr || String(ti->GetName()) == "string") {
			// 小写内建 `string` 不做往返，不能报成"有效 NIL"。
			if (r_is_valid != nullptr) {
				*r_is_valid = false;
			}
			return Variant::NIL;
		}
		void *addr = object->GetAddressOfProperty((asUINT)index);
		if (addr == nullptr) {
			if (r_is_valid != nullptr) {
				*r_is_valid = false;
			}
			return Variant::NIL;
		}
		if (r_is_valid != nullptr) {
			*r_is_valid = true;
		}
		// 原生存储类型的槽是原生 T，静态类型即可直接给出；其余槽是一颗 Variant，取实际类型。
		const Variant::Type native_type = ASBindingDecl::as_name_to_variant_type(String(ti->GetName()));
		if (ASNativeValueStorage::is_native_storage_type(native_type)) {
			return native_type;
		}
		return (*(const Variant *)addr).get_type();
	}

	if (r_is_valid != nullptr) {
		*r_is_valid = true;
	}
	switch (type_id & ~asTYPEID_OBJHANDLE) {
		case asTYPEID_BOOL:
			return Variant::BOOL;
		case asTYPEID_INT32:
		case asTYPEID_INT64:
			return Variant::INT;
		case asTYPEID_FLOAT:
		case asTYPEID_DOUBLE:
			return Variant::FLOAT;
		default:
			// 不支持的类型（对象/字符串等，M2 绑定层）不能报成"有效 NIL"。
			if (r_is_valid != nullptr) {
				*r_is_valid = false;
			}
			return Variant::NIL;
	}
}

void ASScriptInstance::get_method_list(List<MethodInfo> *p_list) const {
	script->get_script_method_list(p_list);
}

bool ASScriptInstance::has_method(const StringName &p_method) const {
	// 与 callp 共用同一份缓存解析：信号声明、无此方法、签名不符一律为 false。
	return _resolve_call_plan(p_method)->function != nullptr;
}

Variant ASScriptInstance::callp(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_OK;

	// 每实例缓存方法解析结果：首次走慢路径，之后仅一次 StringName 哈希查找。
	// 信号声明、无此方法、签名不符统一解析为 nullptr → 非法方法。
	const ASEngine::ASCallPlan *plan = _resolve_call_plan(p_method);
	asIScriptFunction *func = plan->function;
	if (func == nullptr) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
		return Variant();
	}

	// 进入脚本执行前登记当前实例，离开时（正常返回或异常）由 RAII 自动恢复。
	// notification() 内部走 callp()，因此这里是脚本执行的唯一入口。
	ASInstanceScope instance_scope(this);

	Variant ret;
	const Error err = ASEngine::call_function(context, func, object, p_args, p_argcount, &ret, plan);
	if (err == ERR_INVALID_PARAMETER) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		return Variant();
	}
	if (err != OK) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
		return Variant();
	}
	return ret;
}

void ASScriptInstance::notification(int p_notification, bool p_reversed) {
	// 只处理 _notification(int)；_ready/_process 等由引擎经 has_method/callp 按名字调用。
	Variant arg = p_notification;
	const Variant *args[1] = { &arg };
	Callable::CallError error;
	callp("_notification", args, 1, error);
}

ScriptLanguage *ASScriptInstance::get_language() {
	return ASScriptLanguage::get_singleton();
}
