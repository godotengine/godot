/*************************************************************************/
/*  gd_mono_marshal.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "gd_mono_marshal.h"

#include "gd_mono.h"
#include "gd_mono_class.h"

namespace GDMonoMarshal {

#define RETURN_BOXED_STRUCT(m_t, m_var_in)                                    \
	{                                                                         \
		const m_t &m_in = m_var_in->operator ::m_t();                         \
		MARSHALLED_OUT(m_t, m_in, raw);                                       \
		return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(m_t), raw); \
	}

#define RETURN_UNBOXED_STRUCT(m_t, m_var_in)               \
	{                                                      \
		float *raw = (float *)mono_object_unbox(m_var_in); \
		MARSHALLED_IN(m_t, raw, ret);                      \
		return ret;                                        \
	}

Variant::Type managed_to_variant_type(const ManagedType &p_type) {
	switch (p_type.type_encoding) {
		case MONO_TYPE_BOOLEAN:
			return Variant::BOOL;

		case MONO_TYPE_I1:
			return Variant::INT;
		case MONO_TYPE_I2:
			return Variant::INT;
		case MONO_TYPE_I4:
			return Variant::INT;
		case MONO_TYPE_I8:
			return Variant::INT;

		case MONO_TYPE_U1:
			return Variant::INT;
		case MONO_TYPE_U2:
			return Variant::INT;
		case MONO_TYPE_U4:
			return Variant::INT;
		case MONO_TYPE_U8:
			return Variant::INT;

		case MONO_TYPE_R4:
			return Variant::REAL;
		case MONO_TYPE_R8:
			return Variant::REAL;

		case MONO_TYPE_STRING: {
			return Variant::STRING;
		} break;

		case MONO_TYPE_VALUETYPE: {
			GDMonoClass *tclass = p_type.type_class;

			if (tclass == CACHED_CLASS(Vector2))
				return Variant::VECTOR2;

			if (tclass == CACHED_CLASS(Rect2))
				return Variant::RECT2;

			if (tclass == CACHED_CLASS(Transform2D))
				return Variant::TRANSFORM2D;

			if (tclass == CACHED_CLASS(Vector3))
				return Variant::VECTOR3;

			if (tclass == CACHED_CLASS(Basis))
				return Variant::BASIS;

			if (tclass == CACHED_CLASS(Quat))
				return Variant::QUAT;

			if (tclass == CACHED_CLASS(Transform))
				return Variant::TRANSFORM;

			if (tclass == CACHED_CLASS(AABB))
				return Variant::AABB;

			if (tclass == CACHED_CLASS(Color))
				return Variant::COLOR;

			if (tclass == CACHED_CLASS(Plane))
				return Variant::PLANE;

			if (mono_class_is_enum(tclass->get_raw()))
				return Variant::INT;
		} break;

		case MONO_TYPE_ARRAY:
		case MONO_TYPE_SZARRAY: {
			MonoArrayType *array_type = mono_type_get_array_type(GDMonoClass::get_raw_type(p_type.type_class));

			if (array_type->eklass == CACHED_CLASS_RAW(MonoObject))
				return Variant::ARRAY;

			if (array_type->eklass == CACHED_CLASS_RAW(uint8_t))
				return Variant::POOL_BYTE_ARRAY;

			if (array_type->eklass == CACHED_CLASS_RAW(int32_t))
				return Variant::POOL_INT_ARRAY;

			if (array_type->eklass == REAL_T_MONOCLASS)
				return Variant::POOL_REAL_ARRAY;

			if (array_type->eklass == CACHED_CLASS_RAW(String))
				return Variant::POOL_STRING_ARRAY;

			if (array_type->eklass == CACHED_CLASS_RAW(Vector2))
				return Variant::POOL_VECTOR2_ARRAY;

			if (array_type->eklass == CACHED_CLASS_RAW(Vector3))
				return Variant::POOL_VECTOR3_ARRAY;

			if (array_type->eklass == CACHED_CLASS_RAW(Color))
				return Variant::POOL_COLOR_ARRAY;
		} break;

		case MONO_TYPE_CLASS: {
			GDMonoClass *type_class = p_type.type_class;

			// GodotObject
			if (CACHED_CLASS(GodotObject)->is_assignable_from(type_class)) {
				return Variant::OBJECT;
			}

			if (CACHED_CLASS(NodePath) == type_class) {
				return Variant::NODE_PATH;
			}

			if (CACHED_CLASS(RID) == type_class) {
				return Variant::_RID;
			}
		} break;

		case MONO_TYPE_GENERICINST: {
			if (CACHED_RAW_MONO_CLASS(Dictionary) == p_type.type_class->get_raw()) {
				return Variant::DICTIONARY;
			}
		} break;

		default: {
		} break;
	}

	// Unknown
	return Variant::NIL;
}

String mono_to_utf8_string(MonoString *p_mono_string) {
	MonoError error;
	char *utf8 = mono_string_to_utf8_checked(p_mono_string, &error);

	ERR_EXPLAIN("Conversion of MonoString to UTF8 failed.");
	ERR_FAIL_COND_V(!mono_error_ok(&error), String());

	String ret = String::utf8(utf8);

	mono_free(utf8);

	return ret;
}

String mono_to_utf16_string(MonoString *p_mono_string) {
	int len = mono_string_length(p_mono_string);
	String ret;

	if (len == 0)
		return ret;

	ret.resize(len + 1);
	ret.set(len, 0);

	CharType *src = (CharType *)mono_string_chars(p_mono_string);
	CharType *dst = &(ret.operator[](0));

	for (int i = 0; i < len; i++) {
		dst[i] = src[i];
	}

	return ret;
}

MonoObject *variant_to_mono_object(const Variant *p_var) {
	ManagedType type;

	type.type_encoding = MONO_TYPE_OBJECT;

	return variant_to_mono_object(p_var, type);
}

MonoObject *variant_to_mono_object(const Variant *p_var, const ManagedType &p_type) {
	switch (p_type.type_encoding) {
		case MONO_TYPE_BOOLEAN: {
			MonoBoolean val = p_var->operator bool();
			return BOX_BOOLEAN(val);
		}

		case MONO_TYPE_I1: {
			char val = p_var->operator signed char();
			return BOX_INT8(val);
		}
		case MONO_TYPE_I2: {
			short val = p_var->operator signed short();
			return BOX_INT16(val);
		}
		case MONO_TYPE_I4: {
			int val = p_var->operator signed int();
			return BOX_INT32(val);
		}
		case MONO_TYPE_I8: {
			int64_t val = p_var->operator int64_t();
			return BOX_INT64(val);
		}

		case MONO_TYPE_U1: {
			char val = p_var->operator unsigned char();
			return BOX_UINT8(val);
		}
		case MONO_TYPE_U2: {
			short val = p_var->operator unsigned short();
			return BOX_UINT16(val);
		}
		case MONO_TYPE_U4: {
			int val = p_var->operator unsigned int();
			return BOX_UINT32(val);
		}
		case MONO_TYPE_U8: {
			uint64_t val = p_var->operator uint64_t();
			return BOX_UINT64(val);
		}

		case MONO_TYPE_R4: {
			float val = p_var->operator float();
			return BOX_FLOAT(val);
		}
		case MONO_TYPE_R8: {
			double val = p_var->operator double();
			return BOX_DOUBLE(val);
		}

		case MONO_TYPE_STRING: {
			return (MonoObject *)mono_string_from_godot(p_var->operator String());
		} break;

		case MONO_TYPE_VALUETYPE: {
			GDMonoClass *tclass = p_type.type_class;

			if (tclass == CACHED_CLASS(Vector2))
				RETURN_BOXED_STRUCT(Vector2, p_var);

			if (tclass == CACHED_CLASS(Rect2))
				RETURN_BOXED_STRUCT(Rect2, p_var);

			if (tclass == CACHED_CLASS(Transform2D))
				RETURN_BOXED_STRUCT(Transform2D, p_var);

			if (tclass == CACHED_CLASS(Vector3))
				RETURN_BOXED_STRUCT(Vector3, p_var);

			if (tclass == CACHED_CLASS(Basis))
				RETURN_BOXED_STRUCT(Basis, p_var);

			if (tclass == CACHED_CLASS(Quat))
				RETURN_BOXED_STRUCT(Quat, p_var);

			if (tclass == CACHED_CLASS(Transform))
				RETURN_BOXED_STRUCT(Transform, p_var);

			if (tclass == CACHED_CLASS(AABB))
				RETURN_BOXED_STRUCT(AABB, p_var);

			if (tclass == CACHED_CLASS(Color))
				RETURN_BOXED_STRUCT(Color, p_var);

			if (tclass == CACHED_CLASS(Plane))
				RETURN_BOXED_STRUCT(Plane, p_var);

			if (mono_class_is_enum(tclass->get_raw())) {
				int val = p_var->operator signed int();
				return BOX_ENUM(tclass->get_raw(), val);
			}
		} break;

		case MONO_TYPE_ARRAY:
		case MONO_TYPE_SZARRAY: {
			MonoArrayType *array_type = mono_type_get_array_type(GDMonoClass::get_raw_type(p_type.type_class));

			if (array_type->eklass == CACHED_CLASS_RAW(MonoObject))
				return (MonoObject *)Array_to_mono_array(p_var->operator Array());

			if (array_type->eklass == CACHED_CLASS_RAW(uint8_t))
				return (MonoObject *)PoolByteArray_to_mono_array(p_var->operator PoolByteArray());

			if (array_type->eklass == CACHED_CLASS_RAW(int32_t))
				return (MonoObject *)PoolIntArray_to_mono_array(p_var->operator PoolIntArray());

			if (array_type->eklass == REAL_T_MONOCLASS)
				return (MonoObject *)PoolRealArray_to_mono_array(p_var->operator PoolRealArray());

			if (array_type->eklass == CACHED_CLASS_RAW(String))
				return (MonoObject *)PoolStringArray_to_mono_array(p_var->operator PoolStringArray());

			if (array_type->eklass == CACHED_CLASS_RAW(Vector2))
				return (MonoObject *)PoolVector2Array_to_mono_array(p_var->operator PoolVector2Array());

			if (array_type->eklass == CACHED_CLASS_RAW(Vector3))
				return (MonoObject *)PoolVector3Array_to_mono_array(p_var->operator PoolVector3Array());

			if (array_type->eklass == CACHED_CLASS_RAW(Color))
				return (MonoObject *)PoolColorArray_to_mono_array(p_var->operator PoolColorArray());

			ERR_EXPLAIN(String() + "Attempted to convert Variant to a managed array of unmarshallable element type.");
			ERR_FAIL_V(NULL);
		} break;

		case MONO_TYPE_CLASS: {
			GDMonoClass *type_class = p_type.type_class;

			// GodotObject
			if (CACHED_CLASS(GodotObject)->is_assignable_from(type_class)) {
				return GDMonoUtils::unmanaged_get_managed(p_var->operator Object *());
			}

			if (CACHED_CLASS(NodePath) == type_class) {
				return GDMonoUtils::create_managed_from(p_var->operator NodePath());
			}

			if (CACHED_CLASS(RID) == type_class) {
				return GDMonoUtils::create_managed_from(p_var->operator RID());
			}
		} break;
		case MONO_TYPE_OBJECT: {
			// Variant
			switch (p_var->get_type()) {
				case Variant::BOOL: {
					MonoBoolean val = p_var->operator bool();
					return BOX_BOOLEAN(val);
				}
				case Variant::INT: {
					int val = p_var->operator signed int();
					return BOX_INT32(val);
				}
				case Variant::REAL: {
#ifdef REAL_T_IS_DOUBLE
					double val = p_var->operator double();
					return BOX_DOUBLE(val);
#else
					float val = p_var->operator float();
					return BOX_FLOAT(val);
#endif
				}
				case Variant::STRING:
					return (MonoObject *)mono_string_from_godot(p_var->operator String());
				case Variant::VECTOR2:
					RETURN_BOXED_STRUCT(Vector2, p_var);
				case Variant::RECT2:
					RETURN_BOXED_STRUCT(Rect2, p_var);
				case Variant::VECTOR3:
					RETURN_BOXED_STRUCT(Vector3, p_var);
				case Variant::TRANSFORM2D:
					RETURN_BOXED_STRUCT(Transform2D, p_var);
				case Variant::PLANE:
					RETURN_BOXED_STRUCT(Plane, p_var);
				case Variant::QUAT:
					RETURN_BOXED_STRUCT(Quat, p_var);
				case Variant::AABB:
					RETURN_BOXED_STRUCT(AABB, p_var);
				case Variant::BASIS:
					RETURN_BOXED_STRUCT(Basis, p_var);
				case Variant::TRANSFORM:
					RETURN_BOXED_STRUCT(Transform, p_var);
				case Variant::COLOR:
					RETURN_BOXED_STRUCT(Color, p_var);
				case Variant::NODE_PATH:
					return GDMonoUtils::create_managed_from(p_var->operator NodePath());
				case Variant::_RID:
					return GDMonoUtils::create_managed_from(p_var->operator RID());
				case Variant::OBJECT: {
					return GDMonoUtils::unmanaged_get_managed(p_var->operator Object *());
				}
				case Variant::DICTIONARY:
					return Dictionary_to_mono_object(p_var->operator Dictionary());
				case Variant::ARRAY:
					return (MonoObject *)Array_to_mono_array(p_var->operator Array());
				case Variant::POOL_BYTE_ARRAY:
					return (MonoObject *)PoolByteArray_to_mono_array(p_var->operator PoolByteArray());
				case Variant::POOL_INT_ARRAY:
					return (MonoObject *)PoolIntArray_to_mono_array(p_var->operator PoolIntArray());
				case Variant::POOL_REAL_ARRAY:
					return (MonoObject *)PoolRealArray_to_mono_array(p_var->operator PoolRealArray());
				case Variant::POOL_STRING_ARRAY:
					return (MonoObject *)PoolStringArray_to_mono_array(p_var->operator PoolStringArray());
				case Variant::POOL_VECTOR2_ARRAY:
					return (MonoObject *)PoolVector2Array_to_mono_array(p_var->operator PoolVector2Array());
				case Variant::POOL_VECTOR3_ARRAY:
					return (MonoObject *)PoolVector3Array_to_mono_array(p_var->operator PoolVector3Array());
				case Variant::POOL_COLOR_ARRAY:
					return (MonoObject *)PoolColorArray_to_mono_array(p_var->operator PoolColorArray());
				default:
					return NULL;
			}
			break;
			case MONO_TYPE_GENERICINST: {
				if (CACHED_RAW_MONO_CLASS(Dictionary) == p_type.type_class->get_raw()) {
					return Dictionary_to_mono_object(p_var->operator Dictionary());
				}
			} break;
		} break;
	}

	ERR_EXPLAIN(String() + "Attempted to convert Variant to an unmarshallable managed type. Name: \'" +
				p_type.type_class->get_name() + "\' Encoding: " + itos(p_type.type_encoding));
	ERR_FAIL_V(NULL);
}

Variant mono_object_to_variant(MonoObject *p_obj) {
	if (!p_obj)
		return Variant();

	GDMonoClass *tclass = GDMono::get_singleton()->get_class(mono_object_get_class(p_obj));
	ERR_FAIL_COND_V(!tclass, Variant());

	MonoType *raw_type = tclass->get_raw_type(tclass);

	ManagedType type;

	type.type_encoding = mono_type_get_type(raw_type);
	type.type_class = tclass;

	return mono_object_to_variant(p_obj, type);
}

Variant mono_object_to_variant(MonoObject *p_obj, const ManagedType &p_type) {
	switch (p_type.type_encoding) {
		case MONO_TYPE_BOOLEAN:
			return (bool)unbox<MonoBoolean>(p_obj);

		case MONO_TYPE_I1:
			return unbox<int8_t>(p_obj);
		case MONO_TYPE_I2:
			return unbox<int16_t>(p_obj);
		case MONO_TYPE_I4:
			return unbox<int32_t>(p_obj);
		case MONO_TYPE_I8:
			return unbox<int64_t>(p_obj);

		case MONO_TYPE_U1:
			return unbox<uint8_t>(p_obj);
		case MONO_TYPE_U2:
			return unbox<uint16_t>(p_obj);
		case MONO_TYPE_U4:
			return unbox<uint32_t>(p_obj);
		case MONO_TYPE_U8:
			return unbox<uint64_t>(p_obj);

		case MONO_TYPE_R4:
			return unbox<float>(p_obj);
		case MONO_TYPE_R8:
			return unbox<double>(p_obj);

		case MONO_TYPE_STRING: {
			String str = mono_string_to_godot((MonoString *)p_obj);
			return str;
		} break;

		case MONO_TYPE_VALUETYPE: {
			GDMonoClass *tclass = p_type.type_class;

			if (tclass == CACHED_CLASS(Vector2))
				RETURN_UNBOXED_STRUCT(Vector2, p_obj);

			if (tclass == CACHED_CLASS(Rect2))
				RETURN_UNBOXED_STRUCT(Rect2, p_obj);

			if (tclass == CACHED_CLASS(Transform2D))
				RETURN_UNBOXED_STRUCT(Transform2D, p_obj);

			if (tclass == CACHED_CLASS(Vector3))
				RETURN_UNBOXED_STRUCT(Vector3, p_obj);

			if (tclass == CACHED_CLASS(Basis))
				RETURN_UNBOXED_STRUCT(Basis, p_obj);

			if (tclass == CACHED_CLASS(Quat))
				RETURN_UNBOXED_STRUCT(Quat, p_obj);

			if (tclass == CACHED_CLASS(Transform))
				RETURN_UNBOXED_STRUCT(Transform, p_obj);

			if (tclass == CACHED_CLASS(AABB))
				RETURN_UNBOXED_STRUCT(AABB, p_obj);

			if (tclass == CACHED_CLASS(Color))
				RETURN_UNBOXED_STRUCT(Color, p_obj);

			if (tclass == CACHED_CLASS(Plane))
				RETURN_UNBOXED_STRUCT(Plane, p_obj);

			if (mono_class_is_enum(tclass->get_raw()))
				return unbox<int32_t>(p_obj);
		} break;

		case MONO_TYPE_ARRAY:
		case MONO_TYPE_SZARRAY: {
			MonoArrayType *array_type = mono_type_get_array_type(GDMonoClass::get_raw_type(p_type.type_class));

			if (array_type->eklass == CACHED_CLASS_RAW(MonoObject))
				return mono_array_to_Array((MonoArray *)p_obj);

			if (array_type->eklass == CACHED_CLASS_RAW(uint8_t))
				return mono_array_to_PoolByteArray((MonoArray *)p_obj);

			if (array_type->eklass == CACHED_CLASS_RAW(int32_t))
				return mono_array_to_PoolIntArray((MonoArray *)p_obj);

			if (array_type->eklass == REAL_T_MONOCLASS)
				return mono_array_to_PoolRealArray((MonoArray *)p_obj);

			if (array_type->eklass == CACHED_CLASS_RAW(String))
				return mono_array_to_PoolStringArray((MonoArray *)p_obj);

			if (array_type->eklass == CACHED_CLASS_RAW(Vector2))
				return mono_array_to_PoolVector2Array((MonoArray *)p_obj);

			if (array_type->eklass == CACHED_CLASS_RAW(Vector3))
				return mono_array_to_PoolVector3Array((MonoArray *)p_obj);

			if (array_type->eklass == CACHED_CLASS_RAW(Color))
				return mono_array_to_PoolColorArray((MonoArray *)p_obj);

			ERR_EXPLAIN(String() + "Attempted to convert a managed array of unmarshallable element type to Variant.");
			ERR_FAIL_V(Variant());
		} break;

		case MONO_TYPE_CLASS: {
			GDMonoClass *type_class = p_type.type_class;

			// GodotObject
			if (CACHED_CLASS(GodotObject)->is_assignable_from(type_class)) {
				Object *ptr = unbox<Object *>(CACHED_FIELD(GodotObject, ptr)->get_value(p_obj));
				return ptr ? Variant(ptr) : Variant();
			}

			if (CACHED_CLASS(NodePath) == type_class) {
				NodePath *ptr = unbox<NodePath *>(CACHED_FIELD(NodePath, ptr)->get_value(p_obj));
				return ptr ? Variant(*ptr) : Variant();
			}

			if (CACHED_CLASS(RID) == type_class) {
				RID *ptr = unbox<RID *>(CACHED_FIELD(RID, ptr)->get_value(p_obj));
				return ptr ? Variant(*ptr) : Variant();
			}
		} break;

		case MONO_TYPE_GENERICINST: {
			if (CACHED_RAW_MONO_CLASS(Dictionary) == p_type.type_class->get_raw()) {
				return mono_object_to_Dictionary(p_obj);
			}
		} break;
	}

	ERR_EXPLAIN(String() + "Attempted to convert an unmarshallable managed type to Variant. Name: \'" +
				p_type.type_class->get_name() + "\' Encoding: " + itos(p_type.type_encoding));
	ERR_FAIL_V(Variant());
}

MonoArray *Array_to_mono_array(const Array &p_array) {
	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(MonoObject), p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		MonoObject *boxed = variant_to_mono_object(p_array[i]);
		mono_array_setref(ret, i, boxed);
	}

	return ret;
}

Array mono_array_to_Array(MonoArray *p_array) {
	Array ret;
	int length = mono_array_length(p_array);

	for (int i = 0; i < length; i++) {
		MonoObject *elem = mono_array_get(p_array, MonoObject *, i);
		ret.push_back(mono_object_to_variant(elem));
	}

	return ret;
}

// TODO Optimize reading/writing from/to PoolArrays

MonoArray *PoolIntArray_to_mono_array(const PoolIntArray &p_array) {
	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(int32_t), p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		mono_array_set(ret, int32_t, i, p_array[i]);
	}

	return ret;
}

PoolIntArray mono_array_to_PoolIntArray(MonoArray *p_array) {
	PoolIntArray ret;
	int length = mono_array_length(p_array);

	for (int i = 0; i < length; i++) {
		int32_t elem = mono_array_get(p_array, int32_t, i);
		ret.push_back(elem);
	}

	return ret;
}

MonoArray *PoolByteArray_to_mono_array(const PoolByteArray &p_array) {
	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(uint8_t), p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		mono_array_set(ret, uint8_t, i, p_array[i]);
	}

	return ret;
}

PoolByteArray mono_array_to_PoolByteArray(MonoArray *p_array) {
	PoolByteArray ret;
	int length = mono_array_length(p_array);

	for (int i = 0; i < length; i++) {
		uint8_t elem = mono_array_get(p_array, uint8_t, i);
		ret.push_back(elem);
	}

	return ret;
}

MonoArray *PoolRealArray_to_mono_array(const PoolRealArray &p_array) {
	MonoArray *ret = mono_array_new(mono_domain_get(), REAL_T_MONOCLASS, p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		mono_array_set(ret, real_t, i, p_array[i]);
	}

	return ret;
}

PoolRealArray mono_array_to_PoolRealArray(MonoArray *p_array) {
	PoolRealArray ret;
	int length = mono_array_length(p_array);

	for (int i = 0; i < length; i++) {
		real_t elem = mono_array_get(p_array, real_t, i);
		ret.push_back(elem);
	}

	return ret;
}

MonoArray *PoolStringArray_to_mono_array(const PoolStringArray &p_array) {
	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(String), p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		MonoString *boxed = mono_string_from_godot(p_array[i]);
		mono_array_set(ret, MonoString *, i, boxed);
	}

	return ret;
}

PoolStringArray mono_array_to_PoolStringArray(MonoArray *p_array) {
	PoolStringArray ret;
	int length = mono_array_length(p_array);

	for (int i = 0; i < length; i++) {
		MonoString *elem = mono_array_get(p_array, MonoString *, i);
		ret.push_back(mono_string_to_godot(elem));
	}

	return ret;
}

MonoArray *PoolColorArray_to_mono_array(const PoolColorArray &p_array) {
	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(Color), p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
#ifdef YOLOCOPY
		mono_array_set(ret, Color, i, p_array[i]);
#else
		real_t *raw = (real_t *)mono_array_addr_with_size(ret, sizeof(real_t) * 4, i);
		const Color &elem = p_array[i];
		raw[0] = elem.r;
		raw[1] = elem.g;
		raw[2] = elem.b;
		raw[3] = elem.a;
#endif
	}

	return ret;
}

PoolColorArray mono_array_to_PoolColorArray(MonoArray *p_array) {
	PoolColorArray ret;
	int length = mono_array_length(p_array);

	for (int i = 0; i < length; i++) {
		real_t *raw_elem = (real_t *)mono_array_addr_with_size(p_array, sizeof(real_t) * 4, i);
		MARSHALLED_IN(Color, raw_elem, elem);
		ret.push_back(elem);
	}

	return ret;
}

MonoArray *PoolVector2Array_to_mono_array(const PoolVector2Array &p_array) {
	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(Vector2), p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
#ifdef YOLOCOPY
		mono_array_set(ret, Vector2, i, p_array[i]);
#else
		real_t *raw = (real_t *)mono_array_addr_with_size(ret, sizeof(real_t) * 2, i);
		const Vector2 &elem = p_array[i];
		raw[0] = elem.x;
		raw[1] = elem.y;
#endif
	}

	return ret;
}

PoolVector2Array mono_array_to_PoolVector2Array(MonoArray *p_array) {
	PoolVector2Array ret;
	int length = mono_array_length(p_array);

	for (int i = 0; i < length; i++) {
		real_t *raw_elem = (real_t *)mono_array_addr_with_size(p_array, sizeof(real_t) * 2, i);
		MARSHALLED_IN(Vector2, raw_elem, elem);
		ret.push_back(elem);
	}

	return ret;
}

MonoArray *PoolVector3Array_to_mono_array(const PoolVector3Array &p_array) {
	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(Vector3), p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
#ifdef YOLOCOPY
		mono_array_set(ret, Vector3, i, p_array[i]);
#else
		real_t *raw = (real_t *)mono_array_addr_with_size(ret, sizeof(real_t) * 3, i);
		const Vector3 &elem = p_array[i];
		raw[0] = elem.x;
		raw[1] = elem.y;
		raw[2] = elem.z;
#endif
	}

	return ret;
}

PoolVector3Array mono_array_to_PoolVector3Array(MonoArray *p_array) {
	PoolVector3Array ret;
	int length = mono_array_length(p_array);

	for (int i = 0; i < length; i++) {
		real_t *raw_elem = (real_t *)mono_array_addr_with_size(p_array, sizeof(real_t) * 3, i);
		MARSHALLED_IN(Vector3, raw_elem, elem);
		ret.push_back(elem);
	}

	return ret;
}

MonoObject *Dictionary_to_mono_object(const Dictionary &p_dict) {
	MonoArray *keys = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(MonoObject), p_dict.size());
	MonoArray *values = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(MonoObject), p_dict.size());

	int i = 0;
	const Variant *dkey = NULL;
	while ((dkey = p_dict.next(dkey))) {
		mono_array_set(keys, MonoObject *, i, variant_to_mono_object(dkey));
		mono_array_set(values, MonoObject *, i, variant_to_mono_object(p_dict[*dkey]));
		i++;
	}

	GDMonoUtils::MarshalUtils_ArraysToDict arrays_to_dict = CACHED_METHOD_THUNK(MarshalUtils, ArraysToDictionary);

	MonoObject *ex = NULL;
	MonoObject *ret = arrays_to_dict(keys, values, &ex);

	if (ex) {
		mono_print_unhandled_exception(ex);
		ERR_FAIL_V(NULL);
	}

	return ret;
}

Dictionary mono_object_to_Dictionary(MonoObject *p_dict) {
	Dictionary ret;

	GDMonoUtils::MarshalUtils_DictToArrays dict_to_arrays = CACHED_METHOD_THUNK(MarshalUtils, DictionaryToArrays);

	MonoArray *keys = NULL;
	MonoArray *values = NULL;
	MonoObject *ex = NULL;
	dict_to_arrays(p_dict, &keys, &values, &ex);

	if (ex) {
		mono_print_unhandled_exception(ex);
		ERR_FAIL_V(Dictionary());
	}

	int length = mono_array_length(keys);

	for (int i = 0; i < length; i++) {
		MonoObject *key_obj = mono_array_get(keys, MonoObject *, i);
		MonoObject *value_obj = mono_array_get(values, MonoObject *, i);

		Variant key = key_obj ? mono_object_to_variant(key_obj) : Variant();
		Variant value = value_obj ? mono_object_to_variant(value_obj) : Variant();

		ret[key] = value;
	}

	return ret;
}
}
