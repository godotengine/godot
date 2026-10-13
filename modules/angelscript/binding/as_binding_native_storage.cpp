/**************************************************************************/
/*  as_binding_native_storage.cpp                                         */
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

#include "as_binding_native_storage.h"

// 仅为取得 AS_NATIVE_VALUE_OPS_ENABLED 门控宏（定义于该头文件）。它不是“未使用包含”，
// 删掉会让下方 #if 静默退化为 #else（原生存储整体失效）。
#include "as_binding_native_value_ops.h"

#include "core/error/error_macros.h"
#include "core/math/aabb.h"
#include "core/math/basis.h"
#include "core/math/color.h"
#include "core/math/plane.h"
#include "core/math/projection.h"
#include "core/math/quaternion.h"
#include "core/math/rect2.h"
#include "core/math/rect2i.h"
#include "core/math/transform_2d.h"
#include "core/math/transform_3d.h"
#include "core/math/vector2.h"
#include "core/math/vector2i.h"
#include "core/math/vector3.h"
#include "core/math/vector3i.h"
#include "core/math/vector4.h"
#include "core/math/vector4i.h"

#include <angelscript.h>

#include <new>
#include <type_traits>

#if AS_NATIVE_VALUE_OPS_ENABLED

namespace {

// 4 个原生存储类型的公共生命周期 thunk。它们都是平凡可拷贝/析构的小结构体，
// 无需类型专属逻辑，用模板统一生成原生（asCALL_CDECL_OBJFIRST）行为，
// 避免热路径上每个临时对象析构都走 asCALL_GENERIC 跳板（见 Task6 性能验证）。
template <typename T>
void native_ctor_default(T *self) {
	new (self) T();
}

template <typename T>
void native_ctor_copy(T *self, const T *p_other) {
	new (self) T(*p_other);
}

template <typename T>
void native_dtor(T *self) {
	self->~T();
}

} // namespace

namespace ASNativeValueStorage {

bool is_native_storage_type(Variant::Type p_type) {
	switch (p_type) {
		case Variant::VECTOR2:
		case Variant::VECTOR2I:
		case Variant::VECTOR3:
		case Variant::VECTOR3I:
		case Variant::VECTOR4:
		case Variant::VECTOR4I:
		case Variant::RECT2:
		case Variant::RECT2I:
		case Variant::TRANSFORM2D:
		case Variant::PLANE:
		case Variant::QUATERNION:
		case Variant::AABB:
		case Variant::BASIS:
		case Variant::TRANSFORM3D:
		case Variant::PROJECTION:
		case Variant::COLOR:
			return true;
		default:
			return false;
	}
}

bool is_trivial_probe_type(Variant::Type p_type) {
#ifdef AS_POD_GEOMETRY_PROBE
	static_assert(std::is_trivially_copyable_v<Vector2> && std::is_trivially_destructible_v<Vector2>);
	static_assert(std::is_trivially_copyable_v<Vector3> && std::is_trivially_destructible_v<Vector3>);
	static_assert(std::is_trivially_copyable_v<Color> && std::is_trivially_destructible_v<Color>);
	return p_type == Variant::VECTOR2 || p_type == Variant::VECTOR3 || p_type == Variant::COLOR;
#else
	return false;
#endif
}

bool register_native_lifecycle(asIScriptEngine *p_engine, const String &p_name, Variant::Type p_type) {
	void *fn_ctor_default = nullptr;
	void *fn_ctor_copy = nullptr;
	void *fn_dtor = nullptr;
	// 这些值类型都是平凡可拷贝/析构的小结构体，生命周期行为可由模板统一生成；
	// 新增类型只需在此表加一行（与 register_skeleton / 三个转换函数保持同步）。
#define AS_NATIVE_STORAGE_CASE(m_type, m_variant)     \
	case m_variant:                                   \
		fn_ctor_default = (void *)&native_ctor_default<m_type>; \
		fn_ctor_copy = (void *)&native_ctor_copy<m_type>;       \
		fn_dtor = (void *)&native_dtor<m_type>;                 \
		break;
	switch (p_type) {
		AS_NATIVE_STORAGE_CASE(Vector2, Variant::VECTOR2)
		AS_NATIVE_STORAGE_CASE(Vector2i, Variant::VECTOR2I)
		AS_NATIVE_STORAGE_CASE(Vector3, Variant::VECTOR3)
		AS_NATIVE_STORAGE_CASE(Vector3i, Variant::VECTOR3I)
		AS_NATIVE_STORAGE_CASE(Vector4, Variant::VECTOR4)
		AS_NATIVE_STORAGE_CASE(Vector4i, Variant::VECTOR4I)
		AS_NATIVE_STORAGE_CASE(Rect2, Variant::RECT2)
		AS_NATIVE_STORAGE_CASE(Rect2i, Variant::RECT2I)
		AS_NATIVE_STORAGE_CASE(Transform2D, Variant::TRANSFORM2D)
		AS_NATIVE_STORAGE_CASE(Plane, Variant::PLANE)
		AS_NATIVE_STORAGE_CASE(Quaternion, Variant::QUATERNION)
		AS_NATIVE_STORAGE_CASE(AABB, Variant::AABB)
		AS_NATIVE_STORAGE_CASE(Basis, Variant::BASIS)
		AS_NATIVE_STORAGE_CASE(Transform3D, Variant::TRANSFORM3D)
		AS_NATIVE_STORAGE_CASE(Projection, Variant::PROJECTION)
		AS_NATIVE_STORAGE_CASE(Color, Variant::COLOR)
		default:
			return false; // 非原生类型：调用方走 generic。
	}
#undef AS_NATIVE_STORAGE_CASE
	CharString cname = p_name.utf8();
	const CharString copy_decl = ("void f(const " + p_name + " &in)").utf8();
	if (is_trivial_probe_type(p_type)) {
		// Explicit Vector2(other) still requires its constructor signature. Keep
		// that and semantic defaults (Color alpha=1); omit assign/destructor only.
		return p_engine->RegisterObjectBehaviour(cname.get_data(), asBEHAVE_CONSTRUCT, "void f()", asFUNCTION(fn_ctor_default), asCALL_CDECL_OBJFIRST) >= 0 &&
				p_engine->RegisterObjectBehaviour(cname.get_data(), asBEHAVE_CONSTRUCT, copy_decl.get_data(), asFUNCTION(fn_ctor_copy), asCALL_CDECL_OBJFIRST) >= 0;
	}
	if (p_engine->RegisterObjectBehaviour(cname.get_data(), asBEHAVE_CONSTRUCT, "void f()", asFUNCTION(fn_ctor_default), asCALL_CDECL_OBJFIRST) < 0 ||
			p_engine->RegisterObjectBehaviour(cname.get_data(), asBEHAVE_CONSTRUCT, copy_decl.get_data(), asFUNCTION(fn_ctor_copy), asCALL_CDECL_OBJFIRST) < 0 ||
			p_engine->RegisterObjectBehaviour(cname.get_data(), asBEHAVE_DESTRUCT, "void f()", asFUNCTION(fn_dtor), asCALL_CDECL_OBJFIRST) < 0) {
		ERR_PRINT(vformat("AngelScript: failed to register native lifecycle for '%s'.", p_name));
		return false;
	}
	return true;
}

bool register_skeleton(asIScriptEngine *p_engine, Variant::Type p_type) {
	const char *name = nullptr;
	int size = 0;
	switch (p_type) {
		case Variant::VECTOR2: name = "Vector2"; size = sizeof(Vector2); break;
		case Variant::VECTOR2I: name = "Vector2i"; size = sizeof(Vector2i); break;
		case Variant::VECTOR3: name = "Vector3"; size = sizeof(Vector3); break;
		case Variant::VECTOR3I: name = "Vector3i"; size = sizeof(Vector3i); break;
		case Variant::VECTOR4: name = "Vector4"; size = sizeof(Vector4); break;
		case Variant::VECTOR4I: name = "Vector4i"; size = sizeof(Vector4i); break;
		case Variant::RECT2: name = "Rect2"; size = sizeof(Rect2); break;
		case Variant::RECT2I: name = "Rect2i"; size = sizeof(Rect2i); break;
		case Variant::TRANSFORM2D: name = "Transform2D"; size = sizeof(Transform2D); break;
		case Variant::PLANE: name = "Plane"; size = sizeof(Plane); break;
		case Variant::QUATERNION: name = "Quaternion"; size = sizeof(Quaternion); break;
		case Variant::AABB: name = "AABB"; size = sizeof(AABB); break;
		case Variant::BASIS: name = "Basis"; size = sizeof(Basis); break;
		case Variant::TRANSFORM3D: name = "Transform3D"; size = sizeof(Transform3D); break;
		case Variant::PROJECTION: name = "Projection"; size = sizeof(Projection); break;
		case Variant::COLOR: name = "Color"; size = sizeof(Color); break;
		default: return false;
	}
	if (p_engine->GetTypeInfoByName(name) != nullptr) {
		return true; // 幂等
	}
	asDWORD flags = asOBJ_VALUE | asOBJ_APP_CLASS | asOBJ_APP_CLASS_CONSTRUCTOR |
			asOBJ_APP_CLASS_DESTRUCTOR | asOBJ_APP_CLASS_ASSIGNMENT | asOBJ_APP_CLASS_COPY_CONSTRUCTOR;
	// APP_CLASS flags describe our existing hidden-return-slot thunks. POD
	// describes VM lifetime only; do NOT replace the calling-convention flags.
	if (is_trivial_probe_type(p_type)) {
		flags |= asOBJ_POD;
	}
	if (p_engine->RegisterObjectType(name, size, flags) < 0) {
		ERR_PRINT(vformat("AngelScript: failed to register native value type '%s'.", name));
		return false;
	}
	return true;
}

Variant native_to_variant(Variant::Type p_type, const void *p_slot) {
	switch (p_type) {
		case Variant::VECTOR2: return *(const Vector2 *)p_slot;
		case Variant::VECTOR2I: return *(const Vector2i *)p_slot;
		case Variant::VECTOR3: return *(const Vector3 *)p_slot;
		case Variant::VECTOR3I: return *(const Vector3i *)p_slot;
		case Variant::VECTOR4: return *(const Vector4 *)p_slot;
		case Variant::VECTOR4I: return *(const Vector4i *)p_slot;
		case Variant::RECT2: return *(const Rect2 *)p_slot;
		case Variant::RECT2I: return *(const Rect2i *)p_slot;
		case Variant::TRANSFORM2D: return *(const Transform2D *)p_slot;
		case Variant::PLANE: return *(const Plane *)p_slot;
		case Variant::QUATERNION: return *(const Quaternion *)p_slot;
		case Variant::AABB: return *(const AABB *)p_slot;
		case Variant::BASIS: return *(const Basis *)p_slot;
		case Variant::TRANSFORM3D: return *(const Transform3D *)p_slot;
		case Variant::PROJECTION: return *(const Projection *)p_slot;
		case Variant::COLOR: return *(const Color *)p_slot;
		default: return Variant();
	}
}

void variant_to_native(Variant::Type p_type, const Variant &p_value, void *p_slot) {
	switch (p_type) {
		case Variant::VECTOR2: *(Vector2 *)p_slot = (Vector2)p_value; break;
		case Variant::VECTOR2I: *(Vector2i *)p_slot = (Vector2i)p_value; break;
		case Variant::VECTOR3: *(Vector3 *)p_slot = (Vector3)p_value; break;
		case Variant::VECTOR3I: *(Vector3i *)p_slot = (Vector3i)p_value; break;
		case Variant::VECTOR4: *(Vector4 *)p_slot = (Vector4)p_value; break;
		case Variant::VECTOR4I: *(Vector4i *)p_slot = (Vector4i)p_value; break;
		case Variant::RECT2: *(Rect2 *)p_slot = (Rect2)p_value; break;
		case Variant::RECT2I: *(Rect2i *)p_slot = (Rect2i)p_value; break;
		case Variant::TRANSFORM2D: *(Transform2D *)p_slot = (Transform2D)p_value; break;
		case Variant::PLANE: *(Plane *)p_slot = (Plane)p_value; break;
		case Variant::QUATERNION: *(Quaternion *)p_slot = (Quaternion)p_value; break;
		case Variant::AABB: *(AABB *)p_slot = (AABB)p_value; break;
		case Variant::BASIS: *(Basis *)p_slot = (Basis)p_value; break;
		case Variant::TRANSFORM3D: *(Transform3D *)p_slot = (Transform3D)p_value; break;
		case Variant::PROJECTION: *(Projection *)p_slot = (Projection)p_value; break;
		case Variant::COLOR: *(Color *)p_slot = (Color)p_value; break;
		default: break;
	}
}

bool write_native_return(asIScriptGeneric *p_gen, Variant::Type p_type, const Variant &p_value) {
	switch (p_type) {
		case Variant::VECTOR2: { Vector2 t = (Vector2)p_value; p_gen->SetReturnObject(&t); return true; }
		case Variant::VECTOR2I: { Vector2i t = (Vector2i)p_value; p_gen->SetReturnObject(&t); return true; }
		case Variant::VECTOR3: { Vector3 t = (Vector3)p_value; p_gen->SetReturnObject(&t); return true; }
		case Variant::VECTOR3I: { Vector3i t = (Vector3i)p_value; p_gen->SetReturnObject(&t); return true; }
		case Variant::VECTOR4: { Vector4 t = (Vector4)p_value; p_gen->SetReturnObject(&t); return true; }
		case Variant::VECTOR4I: { Vector4i t = (Vector4i)p_value; p_gen->SetReturnObject(&t); return true; }
		case Variant::RECT2: { Rect2 t = (Rect2)p_value; p_gen->SetReturnObject(&t); return true; }
		case Variant::RECT2I: { Rect2i t = (Rect2i)p_value; p_gen->SetReturnObject(&t); return true; }
		case Variant::TRANSFORM2D: { Transform2D t = (Transform2D)p_value; p_gen->SetReturnObject(&t); return true; }
		case Variant::PLANE: { Plane t = (Plane)p_value; p_gen->SetReturnObject(&t); return true; }
		case Variant::QUATERNION: { Quaternion t = (Quaternion)p_value; p_gen->SetReturnObject(&t); return true; }
		case Variant::AABB: { AABB t = (AABB)p_value; p_gen->SetReturnObject(&t); return true; }
		case Variant::BASIS: { Basis t = (Basis)p_value; p_gen->SetReturnObject(&t); return true; }
		case Variant::TRANSFORM3D: { Transform3D t = (Transform3D)p_value; p_gen->SetReturnObject(&t); return true; }
		case Variant::PROJECTION: { Projection t = (Projection)p_value; p_gen->SetReturnObject(&t); return true; }
		case Variant::COLOR: { Color t = (Color)p_value; p_gen->SetReturnObject(&t); return true; }
		default: return false;
	}
}

} // namespace ASNativeValueStorage

#else // !AS_NATIVE_VALUE_OPS_ENABLED

namespace ASNativeValueStorage {

bool is_trivial_probe_type(Variant::Type p_type) {
	return false;
}

bool is_native_storage_type(Variant::Type p_type) {
	(void)p_type;
	return false;
}

bool register_native_lifecycle(asIScriptEngine *p_engine, const String &p_name, Variant::Type p_type) {
	(void)p_engine;
	(void)p_name;
	(void)p_type;
	return false;
}

bool register_skeleton(asIScriptEngine *p_engine, Variant::Type p_type) {
	(void)p_engine;
	(void)p_type;
	return false;
}

Variant native_to_variant(Variant::Type p_type, const void *p_slot) {
	(void)p_type;
	(void)p_slot;
	return Variant();
}

void variant_to_native(Variant::Type p_type, const Variant &p_value, void *p_slot) {
	(void)p_type;
	(void)p_value;
	(void)p_slot;
}

bool write_native_return(asIScriptGeneric *p_gen, Variant::Type p_type, const Variant &p_value) {
	(void)p_gen;
	(void)p_type;
	(void)p_value;
	return false;
}

} // namespace ASNativeValueStorage

#endif // AS_NATIVE_VALUE_OPS_ENABLED
