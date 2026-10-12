#pragma once

// 把 ClassDB 里可见的对象类型注册进 AngelScript。
//
// 注册分两遍（必须遵守，见 spec §3.8 M2 规格）：
//   1. register_skeleton()：只建类型骨架（RegisterObjectType + addref/release + 可选 factory），
//      让后续任何签名引用都能解析到类型名。
//   2. register_class()：挂方法/属性/常量/opImplCast。
// 原因是 AS 2.38 里任何一次 Register* 失败都会给引擎置上 configFailed，
// 之后所有编译都会报 "Invalid configuration"，不可恢复。
//
// 对象句柄在 AS 侧的存储是 `sizeof(void*)` 的槽，槽内容由静态类型决定（spec §3）：
//   OWNING     槽 = 裸 Object*（对象派生自 RefCounted，由 AS 的 addref/release 保活）。
//   NONOWNING  槽 = ObjectID（对象可能随时被 free()，每次解引用前必须经 ObjectDB 校验）。
// 所有槽的读写都必须走下面这两个函数，不允许在别处直接强转。
// 向上转换（opImplCast）同样必须按**目标** kind 重新编码：把 `Resource@`（裸指针）
// 赋给 `Object@`（ObjectID 槽）时，槽值必须从指针换成 ID，不能原样透传。

#include "as_binding_plan.h"
#include "core/error/error_list.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"

#include <angelscript.h>

class Object;

// 对象句柄槽 <-> Object* 的唯一转换点（spec §3.2）。
// 解码失败返回 nullptr（对象已释放、槽为 0 哨兵、或 kind 语义不符），调用方负责 set_exception。
Object *as_handle_decode(void *p_slot, ASBindingKind p_kind);
// 编码：p_obj == nullptr 时返回 nullptr（即 null 句柄）。
void *as_handle_encode(Object *p_obj, ASBindingKind p_kind);

class ASBindingObject {
public:
	// 第一遍：建类型骨架。可重复调用（已存在的类型直接跳过）。
	static Error register_skeleton(const ASBindingClass &p_class, asIScriptEngine *p_engine);
	// 第二遍：挂成员（方法/属性/常量/向上转换）。同一类型只挂一次。
	static Error register_class(const ASBindingClass &p_class, asIScriptEngine *p_engine);
	// 注册 ClassDB 枚举。枚举名里的 '.' 会被替换成 '_'（AS 标识符不允许点号），
	// 即脚本侧写作 `Node_ProcessMode::PROCESS_MODE_ALWAYS`。
	static Error register_enums(const Vector<ASBindingEnum> &p_enums, asIScriptEngine *p_engine);

	// 以下为 asCALL_GENERIC 跳板，函数指针交给 Register*。
	static void generic_method_call(asIScriptGeneric *p_gen);
	// 静态方法（注册为全局函数，无 self）：编组实参后以 self=nullptr 调 MethodBind。
	static void generic_static_method_call(asIScriptGeneric *p_gen);
	static void generic_upcast(asIScriptGeneric *p_gen);
	static void generic_instantiate(asIScriptGeneric *p_gen);
	// RefCounted 语义：句柄持有即加引用。
	static void generic_addref(asIScriptGeneric *p_gen);
	static void generic_release(asIScriptGeneric *p_gen);
	// 非 RefCounted（Node 等）：句柄槽里存 ObjectID，不是裸指针。addref/release 有意
	// 为空操作——脚本持有弱句柄不保活对象、不参与引用计数（M3 语义）；且对象 free() 后
	// 槽仍存 ObjectID，空操作跳板绝不触碰已释放内存。
	static void generic_addref_noop(asIScriptGeneric *p_gen);
	static void generic_release_noop(asIScriptGeneric *p_gen);
	static void generic_property_get(asIScriptGeneric *p_gen);
	static void generic_property_set(asIScriptGeneric *p_gen);
};
