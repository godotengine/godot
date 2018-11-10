/*************************************************************************/
/*  gd_mono_marshal.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

			if (mono_class_is_enum(tclass->get_mono_ptr()))
				return Variant::INT;
		} break;

		case MONO_TYPE_ARRAY:
		case MONO_TYPE_SZARRAY: {
			MonoArrayType *array_type = mono_type_get_array_type(p_type.type_class->get_mono_type());

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

			if (CACHED_CLASS(Dictionary) == type_class) {
				return Variant::DICTIONARY;
			}

			if (CACHED_CLASS(Array) == type_class) {
				return Variant::ARRAY;
			}
		} break;

		case MONO_TYPE_GENERICINST: {
			MonoReflectionType *reftype = mono_type_get_object(SCRIPTS_DOMAIN, p_type.type_class->get_mono_type());

			MonoException *exc = NULL;
			GDMonoUtils::IsDictionaryGenericType type_is_dict = CACHED_METHOD_THUNK(MarshalUtils, IsDictionaryGenericType);
			MonoBoolean is_dict = type_is_dict((MonoObject *)reftype, (MonoObject **)&exc);
			UNLIKELY_UNHANDLED_EXCEPTION(exc);

			if (is_dict) {
				return Variant::DICTIONARY;
			}

			exc = NULL;
			GDMonoUtils::IsArrayGenericType type_is_array = CACHED_METHOD_THUNK(MarshalUtils, IsArrayGenericType);
			MonoBoolean is_array = type_is_array((MonoObject *)reftype, (MonoObject **)&exc);
			UNLIKELY_UNHANDLED_EXCEPTION(exc);

			if (is_array) {
				return Variant::ARRAY;
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
	// type.type_class is not needed when we specify the MONO_TYPE_OBJECT encoding

	return variant_to_mono_object(p_var, type);
}

MonoObject *variant_to_mono_object(const Variant *p_var, const ManagedType &p_type) {
	switch (p_type.type_encoding) {
		case MONO_TYPE_BOOLEAN: {
			MonoBoolean val = p_var->operator bool();
			return BOX_BOOLEAN(val);
		}

		case MONO_TYPE_CHAR: {
			uint16_t val = p_var->operator unsigned short();
			return BOX_UINT16(val);
		}

		case MONO_TYPE_I1: {
			int8_t val = p_var->operator signed char();
			return BOX_INT8(val);
		}
		case MONO_TYPE_I2: {
			int16_t val = p_var->operator signed short();
			return BOX_INT16(val);
		}
		case MONO_TYPE_I4: {
			int32_t val = p_var->operator signed int();
			return BOX_INT32(val);
		}
		case MONO_TYPE_I8: {
			int64_t val = p_var->operator int64_t();
			return BOX_INT64(val);
		}

		case MONO_TYPE_U1: {
			uint8_t val = p_var->operator unsigned char();
			return BOX_UINT8(val);
		}
		case MONO_TYPE_U2: {
			uint16_t val = p_var->operator unsigned short();
			return BOX_UINT16(val);
		}
		case MONO_TYPE_U4: {
			uint32_t val = p_var->operator unsigned int();
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

			if (tclass == CACHED_CLASS(Vector2)) {
				GDMonoMarshal::M_Vector2 from = MARSHALLED_OUT(Vector2, p_var->operator ::Vector2());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Vector2), &from);
			}

			if (tclass == CACHED_CLASS(Rect2)) {
				GDMonoMarshal::M_Rect2 from = MARSHALLED_OUT(Rect2, p_var->operator ::Rect2());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Rect2), &from);
			}

			if (tclass == CACHED_CLASS(Transform2D)) {
				GDMonoMarshal::M_Transform2D from = MARSHALLED_OUT(Transform2D, p_var->operator ::Transform2D());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Transform2D), &from);
			}

			if (tclass == CACHED_CLASS(Vector3)) {
				GDMonoMarshal::M_Vector3 from = MARSHALLED_OUT(Vector3, p_var->operator ::Vector3());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Vector3), &from);
			}

			if (tclass == CACHED_CLASS(Basis)) {
				GDMonoMarshal::M_Basis from = MARSHALLED_OUT(Basis, p_var->operator ::Basis());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Basis), &from);
			}

			if (tclass == CACHED_CLASS(Quat)) {
				GDMonoMarshal::M_Quat from = MARSHALLED_OUT(Quat, p_var->operator ::Quat());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Quat), &from);
			}

			if (tclass == CACHED_CLASS(Transform)) {
				GDMonoMarshal::M_Transform from = MARSHALLED_OUT(Transform, p_var->operator ::Transform());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Transform), &from);
			}

			if (tclass == CACHED_CLASS(AABB)) {
				GDMonoMarshal::M_AABB from = MARSHALLED_OUT(AABB, p_var->operator ::AABB());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(AABB), &from);
			}

			if (tclass == CACHED_CLASS(Color)) {
				GDMonoMarshal::M_Color from = MARSHALLED_OUT(Color, p_var->operator ::Color());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Color), &from);
			}

			if (tclass == CACHED_CLASS(Plane)) {
				GDMonoMarshal::M_Plane from = MARSHALLED_OUT(Plane, p_var->operator ::Plane());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Plane), &from);
			}

			if (mono_class_is_enum(tclass->get_mono_ptr())) {
				MonoType *enum_basetype = mono_class_enum_basetype(tclass->get_mono_ptr());
				MonoClass *enum_baseclass = mono_class_from_mono_type(enum_basetype);
				switch (mono_type_get_type(enum_basetype)) {
					case MONO_TYPE_BOOLEAN: {
						MonoBoolean val = p_var->operator bool();
						return BOX_ENUM(enum_baseclass, val);
					}
					case MONO_TYPE_CHAR: {
						uint16_t val = p_var->operator unsigned short();
						return BOX_ENUM(enum_baseclass, val);
					}
					case MONO_TYPE_I1: {
						int8_t val = p_var->operator signed char();
						return BOX_ENUM(enum_baseclass, val);
					}
					case MONO_TYPE_I2: {
						int16_t val = p_var->operator signed short();
						return BOX_ENUM(enum_baseclass, val);
					}
					case MONO_TYPE_I4: {
						int32_t val = p_var->operator signed int();
						return BOX_ENUM(enum_baseclass, val);
					}
					case MONO_TYPE_I8: {
						int64_t val = p_var->operator int64_t();
						return BOX_ENUM(enum_baseclass, val);
					}
					case MONO_TYPE_U1: {
						uint8_t val = p_var->operator unsigned char();
						return BOX_ENUM(enum_baseclass, val);
					}
					case MONO_TYPE_U2: {
						uint16_t val = p_var->operator unsigned short();
						return BOX_ENUM(enum_baseclass, val);
					}
					case MONO_TYPE_U4: {
						uint32_t val = p_var->operator unsigned int();
						return BOX_ENUM(enum_baseclass, val);
					}
					case MONO_TYPE_U8: {
						uint64_t val = p_var->operator uint64_t();
						return BOX_ENUM(enum_baseclass, val);
					}
					default: {
						ERR_EXPLAIN(String() + "Attempted to convert Variant to a managed enum value of unmarshallable base type.");
						ERR_FAIL_V(NULL);
					}
				}
			}
		} break;

		case MONO_TYPE_ARRAY:
		case MONO_TYPE_SZARRAY: {
			MonoArrayType *array_type = mono_type_get_array_type(p_type.type_class->get_mono_type());

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

			if (CACHED_CLASS(Dictionary) == type_class) {
				return GDMonoUtils::create_managed_from(p_var->operator Dictionary(), CACHED_CLASS(Dictionary));
			}

			if (CACHED_CLASS(Array) == type_class) {
				return GDMonoUtils::create_managed_from(p_var->operator Array(), CACHED_CLASS(Array));
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
					int32_t val = p_var->operator signed int();
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
				case Variant::VECTOR2: {
					GDMonoMarshal::M_Vector2 from = MARSHALLED_OUT(Vector2, p_var->operator ::Vector2());
					return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Vector2), &from);
				}
				case Variant::RECT2: {
					GDMonoMarshal::M_Rect2 from = MARSHALLED_OUT(Rect2, p_var->operator ::Rect2());
					return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Rect2), &from);
				}
				case Variant::VECTOR3: {
					GDMonoMarshal::M_Vector3 from = MARSHALLED_OUT(Vector3, p_var->operator ::Vector3());
					return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Vector3), &from);
				}
				case Variant::TRANSFORM2D: {
					GDMonoMarshal::M_Transform2D from = MARSHALLED_OUT(Transform2D, p_var->operator ::Transform2D());
					return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Transform2D), &from);
				}
				case Variant::PLANE: {
					GDMonoMarshal::M_Plane from = MARSHALLED_OUT(Plane, p_var->operator ::Plane());
					return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Plane), &from);
				}
				case Variant::QUAT: {
					GDMonoMarshal::M_Quat from = MARSHALLED_OUT(Quat, p_var->operator ::Quat());
					return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Quat), &from);
				}
				case Variant::AABB: {
					GDMonoMarshal::M_AABB from = MARSHALLED_OUT(AABB, p_var->operator ::AABB());
					return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(AABB), &from);
				}
				case Variant::BASIS: {
					GDMonoMarshal::M_Basis from = MARSHALLED_OUT(Basis, p_var->operator ::Basis());
					return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Basis), &from);
				}
				case Variant::TRANSFORM: {
					GDMonoMarshal::M_Transform from = MARSHALLED_OUT(Transform, p_var->operator ::Transform());
					return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Transform), &from);
				}
				case Variant::COLOR: {
					GDMonoMarshal::M_Color from = MARSHALLED_OUT(Color, p_var->operator ::Color());
					return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Color), &from);
				}
				case Variant::NODE_PATH:
					return GDMonoUtils::create_managed_from(p_var->operator NodePath());
				case Variant::_RID:
					return GDMonoUtils::create_managed_from(p_var->operator RID());
				case Variant::OBJECT:
					return GDMonoUtils::unmanaged_get_managed(p_var->operator Object *());
				case Variant::DICTIONARY:
					return GDMonoUtils::create_managed_from(p_var->operator Dictionary(), CACHED_CLASS(Dictionary));
				case Variant::ARRAY:
					return GDMonoUtils::create_managed_from(p_var->operator Array(), CACHED_CLASS(Array));
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
				MonoReflectionType *reftype = mono_type_get_object(SCRIPTS_DOMAIN, p_type.type_class->get_mono_type());

				MonoException *exc = NULL;
				GDMonoUtils::IsDictionaryGenericType type_is_dict = CACHED_METHOD_THUNK(MarshalUtils, IsDictionaryGenericType);
				MonoBoolean is_dict = type_is_dict((MonoObject *)reftype, (MonoObject **)&exc);
				UNLIKELY_UNHANDLED_EXCEPTION(exc);

				if (is_dict) {
					return GDMonoUtils::create_managed_from(p_var->operator Dictionary(), p_type.type_class);
				}

				exc = NULL;
				GDMonoUtils::IsArrayGenericType type_is_array = CACHED_METHOD_THUNK(MarshalUtils, IsArrayGenericType);
				MonoBoolean is_array = type_is_array((MonoObject *)reftype, (MonoObject **)&exc);
				UNLIKELY_UNHANDLED_EXCEPTION(exc);

				if (is_array) {
					return GDMonoUtils::create_managed_from(p_var->operator Array(), p_type.type_class);
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

	MonoType *raw_type = tclass->get_mono_type();

	ManagedType type;

	type.type_encoding = mono_type_get_type(raw_type);
	type.type_class = tclass;

	switch (type.type_encoding) {
		case MONO_TYPE_BOOLEAN:
			return (bool)unbox<MonoBoolean>(p_obj);

		case MONO_TYPE_CHAR:
			return unbox<uint16_t>(p_obj);

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
			if (p_obj == NULL)
				return Variant(); // NIL
			return mono_string_to_godot_not_null((MonoString *)p_obj);
		} break;

		case MONO_TYPE_VALUETYPE: {
			GDMonoClass *tclass = type.type_class;

			if (tclass == CACHED_CLASS(Vector2))
				return MARSHALLED_IN(Vector2, (GDMonoMarshal::M_Vector2 *)mono_object_unbox(p_obj));

			if (tclass == CACHED_CLASS(Rect2))
				return MARSHALLED_IN(Rect2, (GDMonoMarshal::M_Rect2 *)mono_object_unbox(p_obj));

			if (tclass == CACHED_CLASS(Transform2D))
				return MARSHALLED_IN(Transform2D, (GDMonoMarshal::M_Transform2D *)mono_object_unbox(p_obj));

			if (tclass == CACHED_CLASS(Vector3))
				return MARSHALLED_IN(Vector3, (GDMonoMarshal::M_Vector3 *)mono_object_unbox(p_obj));

			if (tclass == CACHED_CLASS(Basis))
				return MARSHALLED_IN(Basis, (GDMonoMarshal::M_Basis *)mono_object_unbox(p_obj));

			if (tclass == CACHED_CLASS(Quat))
				return MARSHALLED_IN(Quat, (GDMonoMarshal::M_Quat *)mono_object_unbox(p_obj));

			if (tclass == CACHED_CLASS(Transform))
				return MARSHALLED_IN(Transform, (GDMonoMarshal::M_Transform *)mono_object_unbox(p_obj));

			if (tclass == CACHED_CLASS(AABB))
				return MARSHALLED_IN(AABB, (GDMonoMarshal::M_AABB *)mono_object_unbox(p_obj));

			if (tclass == CACHED_CLASS(Color))
				return MARSHALLED_IN(Color, (GDMonoMarshal::M_Color *)mono_object_unbox(p_obj));

			if (tclass == CACHED_CLASS(Plane))
				return MARSHALLED_IN(Plane, (GDMonoMarshal::M_Plane *)mono_object_unbox(p_obj));

			if (mono_class_is_enum(tclass->get_mono_ptr()))
				return unbox<int32_t>(p_obj);
		} break;

		case MONO_TYPE_ARRAY:
		case MONO_TYPE_SZARRAY: {
			MonoArrayType *array_type = mono_type_get_array_type(type.type_class->get_mono_type());

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
			GDMonoClass *type_class = type.type_class;

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

			if (CACHED_CLASS(Array) == type_class) {
				MonoException *exc = NULL;
				GDMonoUtils::Array_GetPtr get_ptr = CACHED_METHOD_THUNK(Array, GetPtr);
				Array *ptr = get_ptr(p_obj, (MonoObject **)&exc);
				UNLIKELY_UNHANDLED_EXCEPTION(exc);
				return ptr ? Variant(*ptr) : Variant();
			}

			if (CACHED_CLASS(Dictionary) == type_class) {
				MonoException *exc = NULL;
				GDMonoUtils::Dictionary_GetPtr get_ptr = CACHED_METHOD_THUNK(Dictionary, GetPtr);
				Dictionary *ptr = get_ptr(p_obj, (MonoObject **)&exc);
				UNLIKELY_UNHANDLED_EXCEPTION(exc);
				return ptr ? Variant(*ptr) : Variant();
			}
		} break;

		case MONO_TYPE_GENERICINST: {
			MonoReflectionType *reftype = mono_type_get_object(SCRIPTS_DOMAIN, type.type_class->get_mono_type());

			MonoException *exc = NULL;

			GDMonoUtils::IsDictionaryGenericType type_is_dict = CACHED_METHOD_THUNK(MarshalUtils, IsDictionaryGenericType);
			MonoBoolean is_dict = type_is_dict((MonoObject *)reftype, (MonoObject **)&exc);
			UNLIKELY_UNHANDLED_EXCEPTION(exc);

			if (is_dict) {
				MonoException *exc = NULL;
				MonoObject *ret = type.type_class->get_method("GetPtr")->invoke(p_obj, &exc);
				UNLIKELY_UNHANDLED_EXCEPTION(exc);
				return *unbox<Dictionary *>(ret);
			}

			exc = NULL;

			GDMonoUtils::IsArrayGenericType type_is_array = CACHED_METHOD_THUNK(MarshalUtils, IsArrayGenericType);
			MonoBoolean is_array = type_is_array((MonoObject *)reftype, (MonoObject **)&exc);
			UNLIKELY_UNHANDLED_EXCEPTION(exc);

			if (is_array) {
				MonoException *exc = NULL;
				MonoObject *ret = type.type_class->get_method("GetPtr")->invoke(p_obj, &exc);
				UNLIKELY_UNHANDLED_EXCEPTION(exc);
				return *unbox<Array *>(ret);
			}
		} break;
	}

	ERR_EXPLAIN(String() + "Attempted to convert an unmarshallable managed type to Variant. Name: \'" +
				type.type_class->get_name() + "\' Encoding: " + itos(type.type_encoding));
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
	if (!p_array)
		return ret;
	int length = mono_array_length(p_array);
	ret.resize(length);

	for (int i = 0; i < length; i++) {
		MonoObject *elem = mono_array_get(p_array, MonoObject *, i);
		ret[i] = mono_object_to_variant(elem);
	}

	return ret;
}

MonoArray *PoolIntArray_to_mono_array(const PoolIntArray &p_array) {
	PoolIntArray::Read r = p_array.read();

	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(int32_t), p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		mono_array_set(ret, int32_t, i, r[i]);
	}

	return ret;
}

PoolIntArray mono_array_to_PoolIntArray(MonoArray *p_array) {
	PoolIntArray ret;
	if (!p_array)
		return ret;
	int length = mono_array_length(p_array);
	ret.resize(length);
	PoolIntArray::Write w = ret.write();

	for (int i = 0; i < length; i++) {
		w[i] = mono_array_get(p_array, int32_t, i);
	}

	return ret;
}

MonoArray *PoolByteArray_to_mono_array(const PoolByteArray &p_array) {
	PoolByteArray::Read r = p_array.read();

	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(uint8_t), p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		mono_array_set(ret, uint8_t, i, r[i]);
	}

	return ret;
}

PoolByteArray mono_array_to_PoolByteArray(MonoArray *p_array) {
	PoolByteArray ret;
	if (!p_array)
		return ret;
	int length = mono_array_length(p_array);
	ret.resize(length);
	PoolByteArray::Write w = ret.write();

	for (int i = 0; i < length; i++) {
		w[i] = mono_array_get(p_array, uint8_t, i);
	}

	return ret;
}

MonoArray *PoolRealArray_to_mono_array(const PoolRealArray &p_array) {
	PoolRealArray::Read r = p_array.read();

	MonoArray *ret = mono_array_new(mono_domain_get(), REAL_T_MONOCLASS, p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		mono_array_set(ret, real_t, i, r[i]);
	}

	return ret;
}

PoolRealArray mono_array_to_PoolRealArray(MonoArray *p_array) {
	PoolRealArray ret;
	if (!p_array)
		return ret;
	int length = mono_array_length(p_array);
	ret.resize(length);
	PoolRealArray::Write w = ret.write();

	for (int i = 0; i < length; i++) {
		w[i] = mono_array_get(p_array, real_t, i);
	}

	return ret;
}

MonoArray *PoolStringArray_to_mono_array(const PoolStringArray &p_array) {
	PoolStringArray::Read r = p_array.read();

	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(String), p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		MonoString *boxed = mono_string_from_godot(r[i]);
		mono_array_set(ret, MonoString *, i, boxed);
	}

	return ret;
}

PoolStringArray mono_array_to_PoolStringArray(MonoArray *p_array) {
	PoolStringArray ret;
	if (!p_array)
		return ret;
	int length = mono_array_length(p_array);
	ret.resize(length);
	PoolStringArray::Write w = ret.write();

	for (int i = 0; i < length; i++) {
		MonoString *elem = mono_array_get(p_array, MonoString *, i);
		w[i] = mono_string_to_godot(elem);
	}

	return ret;
}

MonoArray *PoolColorArray_to_mono_array(const PoolColorArray &p_array) {
	PoolColorArray::Read r = p_array.read();

	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(Color), p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		M_Color *raw = (M_Color *)mono_array_addr_with_size(ret, sizeof(M_Color), i);
		*raw = MARSHALLED_OUT(Color, r[i]);
	}

	return ret;
}

PoolColorArray mono_array_to_PoolColorArray(MonoArray *p_array) {
	PoolColorArray ret;
	if (!p_array)
		return ret;
	int length = mono_array_length(p_array);
	ret.resize(length);
	PoolColorArray::Write w = ret.write();

	for (int i = 0; i < length; i++) {
		w[i] = MARSHALLED_IN(Color, (M_Color *)mono_array_addr_with_size(p_array, sizeof(M_Color), i));
	}

	return ret;
}

MonoArray *PoolVector2Array_to_mono_array(const PoolVector2Array &p_array) {
	PoolVector2Array::Read r = p_array.read();

	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(Vector2), p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		M_Vector2 *raw = (M_Vector2 *)mono_array_addr_with_size(ret, sizeof(M_Vector2), i);
		*raw = MARSHALLED_OUT(Vector2, r[i]);
	}

	return ret;
}

PoolVector2Array mono_array_to_PoolVector2Array(MonoArray *p_array) {
	PoolVector2Array ret;
	if (!p_array)
		return ret;
	int length = mono_array_length(p_array);
	ret.resize(length);
	PoolVector2Array::Write w = ret.write();

	for (int i = 0; i < length; i++) {
		w[i] = MARSHALLED_IN(Vector2, (M_Vector2 *)mono_array_addr_with_size(p_array, sizeof(M_Vector2), i));
	}

	return ret;
}

MonoArray *PoolVector3Array_to_mono_array(const PoolVector3Array &p_array) {
	PoolVector3Array::Read r = p_array.read();

	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(Vector3), p_array.size());

	for (int i = 0; i < p_array.size(); i++) {
		M_Vector3 *raw = (M_Vector3 *)mono_array_addr_with_size(ret, sizeof(M_Vector3), i);
		*raw = MARSHALLED_OUT(Vector3, r[i]);
	}

	return ret;
}

PoolVector3Array mono_array_to_PoolVector3Array(MonoArray *p_array) {
	PoolVector3Array ret;
	if (!p_array)
		return ret;
	int length = mono_array_length(p_array);
	ret.resize(length);
	PoolVector3Array::Write w = ret.write();

	for (int i = 0; i < length; i++) {
		w[i] = MARSHALLED_IN(Vector3, (M_Vector3 *)mono_array_addr_with_size(p_array, sizeof(M_Vector3), i));
	}

	return ret;
}
} // namespace GDMonoMarshal
