/**************************************************************************/
/*  as_binding_decl.cpp                                                   */
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

#include "as_binding_decl.h"

#include "as_binding_native_storage.h"
#include "as_binding_object.h"
#include "core/object/class_db.h"

#include <angelscript.h>

String ASBindingDecl::variant_type_to_as(Variant::Type p_type) {
	switch (p_type) {
		// 标量：AS 原生类型。INT 必须是 64 位，Variant::INT 是 int64，32 位会截断。
		case Variant::BOOL: return "bool";
		case Variant::INT: return "int64";
		case Variant::FLOAT: return "double";
		// 内建值类型：本模块统一以 Variant 为存储，AS 侧按类型名注册。
		case Variant::STRING: return "String";
		case Variant::STRING_NAME: return "StringName";
		case Variant::NODE_PATH: return "NodePath";
		case Variant::RID: return "RID";
		case Variant::VECTOR2: return "Vector2";
		case Variant::VECTOR2I: return "Vector2i";
		case Variant::VECTOR3: return "Vector3";
		case Variant::VECTOR3I: return "Vector3i";
		case Variant::VECTOR4: return "Vector4";
		case Variant::VECTOR4I: return "Vector4i";
		case Variant::RECT2: return "Rect2";
		case Variant::RECT2I: return "Rect2i";
		case Variant::TRANSFORM2D: return "Transform2D";
		case Variant::PLANE: return "Plane";
		case Variant::QUATERNION: return "Quaternion";
		case Variant::AABB: return "AABB";
		case Variant::BASIS: return "Basis";
		case Variant::TRANSFORM3D: return "Transform3D";
		case Variant::PROJECTION: return "Projection";
		case Variant::COLOR: return "Color";
		case Variant::PACKED_BYTE_ARRAY: return "PackedByteArray";
		case Variant::PACKED_INT32_ARRAY: return "PackedInt32Array";
		case Variant::PACKED_INT64_ARRAY: return "PackedInt64Array";
		case Variant::PACKED_FLOAT32_ARRAY: return "PackedFloat32Array";
		case Variant::PACKED_FLOAT64_ARRAY: return "PackedFloat64Array";
		case Variant::PACKED_STRING_ARRAY: return "PackedStringArray";
		case Variant::PACKED_VECTOR2_ARRAY: return "PackedVector2Array";
		case Variant::PACKED_VECTOR3_ARRAY: return "PackedVector3Array";
		case Variant::PACKED_VECTOR4_ARRAY: return "PackedVector4Array";
		case Variant::PACKED_COLOR_ARRAY: return "PackedColorArray";
		case Variant::ARRAY: return "Array";
		case Variant::DICTIONARY: return "Dictionary";
		case Variant::CALLABLE: return "Callable";
		case Variant::SIGNAL: return "Signal";
		default: return String();
	}
}

Variant::Type ASBindingDecl::as_name_to_variant_type(const String &p_as_name) {
	if (p_as_name.is_empty()) {
		return Variant::NIL;
	}
	// 以 variant_type_to_as 为唯一事实来源反查，避免维护两份会漂移的映射表。
	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		const Variant::Type t = (Variant::Type)i;
		if (variant_type_to_as(t) == p_as_name) {
			return t;
		}
	}
	return Variant::NIL; // 对象类型（`Node@` 等）与未知类型：无 Variant::Type 对应。
}

ASBindingType ASBindingDecl::resolve(const PropertyInfo &p_info) {
	ASBindingType out;
	if (p_info.type == Variant::OBJECT) {
		const StringName cls = p_info.class_name;
		// 空 class_name 表示“任意 Object”；Object 自身不是 RefCounted ⇒ 归为非拥有。
		const bool owning = cls != StringName() && ClassDB::is_parent_class(cls, "RefCounted");
		out.valid = true;
		out.as_name = String(cls == StringName() ? StringName("Object") : cls) + "@";
		out.kind = owning ? AS_KIND_OBJECT_OWNING : AS_KIND_OBJECT_NONOWNING;
		return out;
	}
	if (p_info.type == Variant::NIL) {
		// AS 无法表达「void 值」：NIL 作为形参/属性类型只可能是 Variant
		// （上游约定参数不能是 void，故不需要 NIL_IS_VARIANT 标记）。
		// 返回值是否为 void 由 render_signature 在调用 resolve 前判定。
		out.valid = true;
		out.as_name = "Variant";
		out.kind = AS_KIND_VALUE;
		return out;
	}
	String as = variant_type_to_as(p_info.type);
	if (as.is_empty()) {
		// 只写裸类型名，由调用方按场景拼 reason 前缀（return/param/property）。
		out.unbound_reason = String(Variant::get_type_name(p_info.type));
		return out;
	}
	out.valid = true;
	out.as_name = as;
	if (p_info.type == Variant::BOOL) {
		out.kind = AS_KIND_BOOL;
	} else if (p_info.type == Variant::INT) {
		out.kind = AS_KIND_INT64;
	} else if (p_info.type == Variant::FLOAT) {
		out.kind = AS_KIND_DOUBLE;
	} else {
		out.kind = AS_KIND_VALUE;
	}
	return out;
}

String as_binding_render_param(const String &p_as_name) {
	if (p_as_name == "bool" || p_as_name == "int64" || p_as_name == "double") {
		return p_as_name;
	}
	if (p_as_name.ends_with("@")) {
		// 对象句柄必须带 `+`（auto handle）：实参若不带，拥有句柄传给已注册函数时
		// 引擎只做一次 addref 而不在调用后 release，引用计数会永久 +1（Ruling C）。
		// 对非拥有类型（addref/release 为 noop）加 `+` 无害，故可统一处理。
		return p_as_name + "+";
	}
	return "const " + p_as_name + " &in";
}

static String render_param(const ASBindingType &t) {
	return as_binding_render_param(t.as_name);
}

// 渲染返回类型与形参表（method_to_decl / static_method_to_decl 的公共部分）。
static bool render_signature(const MethodInfo &p_info, const String &p_name, bool p_is_const, String *r_decl, String *r_reason) {
	String ret;
	if (p_info.return_val.type == Variant::NIL && !(p_info.return_val.usage & PROPERTY_USAGE_NIL_IS_VARIANT)) {
		// 真 void：NIL 且未标记 NIL_IS_VARIANT。
		ret = "void";
	} else {
		// 真 Variant 返回（NIL + NIL_IS_VARIANT）与其它类型统一走 resolve，
		// AS 侧声明为 Variant，由跳板用已注册的 Variant 拷贝构造写回。
		ASBindingType rt = ASBindingDecl::resolve(p_info.return_val);
		if (!rt.valid) {
			if (r_reason) {
				*r_reason = "unsupported-return-type: " + rt.unbound_reason;
			}
			return false;
		}
		ret = rt.as_name;
	}

	String args;
	for (int i = 0; i < p_info.arguments.size(); i++) {
		ASBindingType at = ASBindingDecl::resolve(p_info.arguments[i]);
		if (!at.valid) {
			if (r_reason) {
				*r_reason = vformat("unsupported-param-type: %s (arg %d)", at.unbound_reason, i);
			}
			return false;
		}
		if (i > 0) {
			args += ", ";
		}
		args += render_param(at);
	}

	String decl;
	decl += ret + " " + p_name + "(" + args + ")";
	if (p_is_const) {
		decl += " const";
	}

	if (r_decl) {
		*r_decl = decl;
	}
	return true;
}

bool ASBindingDecl::method_to_decl(const MethodInfo &p_info, String *r_decl, String *r_reason) {
	if (r_decl) {
		*r_decl = String();
	}
	if (r_reason) {
		*r_reason = String();
	}

	// vararg 无法在 AS 静态类型里表达（AS 只允许 asCALL_GENERIC 的 vararg 函数）。
	if (p_info.flags & METHOD_FLAG_VARARG) {
		if (r_reason) {
			*r_reason = "vararg";
		}
		return false;
	}

	// AS 2.38 的解析器不识别 `static` 关键字：把 "static ..." 交给 RegisterObjectMethod
	// 会返回 asINVALID_DECLARATION 并永久污染引擎（探针验证）。静态方法走
	// static_method_to_decl 注册为全局函数（见该函数注释）。
	if (p_info.flags & METHOD_FLAG_STATIC) {
		if (r_reason) {
			*r_reason = "static-method";
		}
		return false;
	}

	return render_signature(p_info, String(p_info.name), (p_info.flags & METHOD_FLAG_CONST) != 0, r_decl, r_reason);
}

bool ASBindingDecl::static_method_to_decl(const MethodInfo &p_info, const StringName &p_scope, String *r_decl, String *r_reason) {
	if (r_decl) {
		*r_decl = String();
	}
	if (r_reason) {
		*r_reason = String();
	}

	if (p_info.flags & METHOD_FLAG_VARARG) {
		if (r_reason) {
			*r_reason = "vararg";
		}
		return false;
	}
	if (!(p_info.flags & METHOD_FLAG_STATIC)) {
		if (r_reason) {
			*r_reason = "not-static-method";
		}
		return false;
	}

	// 全局函数没有 this：声明不带 const；名字沿用 `<作用域>_<成员名>` 约定。
	const String name = String(p_scope) + "_" + String(p_info.name);
	return render_signature(p_info, name, false, r_decl, r_reason);
}

Variant as_binding_marshal_arg(asIScriptGeneric *p_gen, int p_index, ASBindingKind p_kind, Variant::Type p_stored_type) {
	switch (p_kind) {
		case AS_KIND_BOOL:
			return Variant((bool)p_gen->GetArgByte(p_index));
		case AS_KIND_INT64:
			return Variant((int64_t)p_gen->GetArgQWord(p_index));
		case AS_KIND_DOUBLE:
			return Variant(p_gen->GetArgDouble(p_index));
		case AS_KIND_VALUE: {
			// 原生存储类型：槽里是原生 T；其余仍是本模块统一的 Variant 存储。
			void *p = p_gen->GetArgObject(p_index);
			if (ASNativeValueStorage::is_native_storage_type(p_stored_type)) {
				return p ? ASNativeValueStorage::native_to_variant(p_stored_type, p) : Variant();
			}
			return p ? *(const Variant *)p : Variant();
		}
		case AS_KIND_OBJECT_OWNING:
		case AS_KIND_OBJECT_NONOWNING:
			// 槽的语义由 kind 决定（拥有=裸指针，非拥有=ObjectID），统一经解码单点还原。
			return Variant(as_handle_decode(p_gen->GetArgObject(p_index), p_kind));
		default:
			return Variant();
	}
}

void as_binding_marshal_return(asIScriptGeneric *p_gen, ASBindingKind p_kind, const Variant &p_value, Variant::Type p_return_type) {
	switch (p_kind) {
		case AS_KIND_BOOL:
			*(bool *)p_gen->GetAddressOfReturnLocation() = (bool)p_value;
			break;
		case AS_KIND_INT64:
			*(int64_t *)p_gen->GetAddressOfReturnLocation() = (int64_t)p_value;
			break;
		case AS_KIND_DOUBLE:
			*(double *)p_gen->GetAddressOfReturnLocation() = (double)p_value;
			break;
		case AS_KIND_VALUE:
			// 原生存储类型：以原生 T 写回，AS 用原生拷贝构造复制；其余交给 AS 用注册的 Variant 拷贝构造。
			if (ASNativeValueStorage::is_native_storage_type(p_return_type) &&
					ASNativeValueStorage::write_native_return(p_gen, p_return_type, p_value)) {
				break;
			}
			p_gen->SetReturnObject((void *)&p_value);
			break;
		case AS_KIND_OBJECT_OWNING:
		case AS_KIND_OBJECT_NONOWNING:
			// 返回值同样按 kind 编码：非拥有类型只交出 ObjectID。
			p_gen->SetReturnObject(as_handle_encode(p_value.operator Object *(), p_kind));
			break;
		default:
			break;
	}
}
