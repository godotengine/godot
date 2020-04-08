/*************************************************************************/
/*  gd_mono_marshal.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "../signal_awaiter_utils.h"
#include "gd_mono.h"
#include "gd_mono_cache.h"
#include "gd_mono_class.h"

namespace GDMonoMarshal {

Variant::Type managed_to_variant_type(const ManagedType &p_type, bool *r_nil_is_variant) {
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
			return Variant::FLOAT;
		case MONO_TYPE_R8:
			return Variant::FLOAT;

		case MONO_TYPE_STRING: {
			return Variant::STRING;
		} break;

		case MONO_TYPE_VALUETYPE: {
			GDMonoClass *vtclass = p_type.type_class;

			if (vtclass == CACHED_CLASS(Vector2))
				return Variant::VECTOR2;

			if (vtclass == CACHED_CLASS(Vector2i))
				return Variant::VECTOR2I;

			if (vtclass == CACHED_CLASS(Rect2))
				return Variant::RECT2;

			if (vtclass == CACHED_CLASS(Rect2i))
				return Variant::RECT2I;

			if (vtclass == CACHED_CLASS(Transform2D))
				return Variant::TRANSFORM2D;

			if (vtclass == CACHED_CLASS(Vector3))
				return Variant::VECTOR3;

			if (vtclass == CACHED_CLASS(Vector3i))
				return Variant::VECTOR3I;

			if (vtclass == CACHED_CLASS(Basis))
				return Variant::BASIS;

			if (vtclass == CACHED_CLASS(Quat))
				return Variant::QUAT;

			if (vtclass == CACHED_CLASS(Transform))
				return Variant::TRANSFORM;

			if (vtclass == CACHED_CLASS(AABB))
				return Variant::AABB;

			if (vtclass == CACHED_CLASS(Color))
				return Variant::COLOR;

			if (vtclass == CACHED_CLASS(Plane))
				return Variant::PLANE;

			if (vtclass == CACHED_CLASS(Callable))
				return Variant::CALLABLE;

			if (vtclass == CACHED_CLASS(SignalInfo))
				return Variant::SIGNAL;

			if (mono_class_is_enum(vtclass->get_mono_ptr()))
				return Variant::INT;
		} break;

		case MONO_TYPE_ARRAY:
		case MONO_TYPE_SZARRAY: {
			MonoArrayType *array_type = mono_type_get_array_type(p_type.type_class->get_mono_type());

			if (array_type->eklass == CACHED_CLASS_RAW(MonoObject))
				return Variant::ARRAY;

			if (array_type->eklass == CACHED_CLASS_RAW(uint8_t))
				return Variant::PACKED_BYTE_ARRAY;

			if (array_type->eklass == CACHED_CLASS_RAW(int32_t))
				return Variant::PACKED_INT32_ARRAY;

			if (array_type->eklass == CACHED_CLASS_RAW(int64_t))
				return Variant::PACKED_INT64_ARRAY;

			if (array_type->eklass == CACHED_CLASS_RAW(float))
				return Variant::PACKED_FLOAT32_ARRAY;

			if (array_type->eklass == CACHED_CLASS_RAW(double))
				return Variant::PACKED_FLOAT64_ARRAY;

			if (array_type->eklass == CACHED_CLASS_RAW(String))
				return Variant::PACKED_STRING_ARRAY;

			if (array_type->eklass == CACHED_CLASS_RAW(Vector2))
				return Variant::PACKED_VECTOR2_ARRAY;

			if (array_type->eklass == CACHED_CLASS_RAW(Vector3))
				return Variant::PACKED_VECTOR3_ARRAY;

			if (array_type->eklass == CACHED_CLASS_RAW(Color))
				return Variant::PACKED_COLOR_ARRAY;
		} break;

		case MONO_TYPE_CLASS: {
			GDMonoClass *type_class = p_type.type_class;

			// GodotObject
			if (CACHED_CLASS(GodotObject)->is_assignable_from(type_class)) {
				return Variant::OBJECT;
			}

			if (CACHED_CLASS(StringName) == type_class) {
				return Variant::STRING_NAME;
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

			// The order in which we check the following interfaces is very important (dictionaries and generics first)

			MonoReflectionType *reftype = mono_type_get_object(mono_domain_get(), type_class->get_mono_type());

			if (GDMonoUtils::Marshal::generic_idictionary_is_assignable_from(reftype)) {
				return Variant::DICTIONARY;
			}

			if (type_class->implements_interface(CACHED_CLASS(System_Collections_IDictionary))) {
				return Variant::DICTIONARY;
			}

			if (GDMonoUtils::Marshal::generic_ienumerable_is_assignable_from(reftype)) {
				return Variant::ARRAY;
			}

			if (type_class->implements_interface(CACHED_CLASS(System_Collections_IEnumerable))) {
				return Variant::ARRAY;
			}
		} break;

		case MONO_TYPE_OBJECT: {
			if (r_nil_is_variant)
				*r_nil_is_variant = true;
			return Variant::NIL;
		} break;

		case MONO_TYPE_GENERICINST: {
			MonoReflectionType *reftype = mono_type_get_object(mono_domain_get(), p_type.type_class->get_mono_type());

			if (GDMonoUtils::Marshal::type_is_generic_dictionary(reftype)) {
				return Variant::DICTIONARY;
			}

			if (GDMonoUtils::Marshal::type_is_generic_array(reftype)) {
				return Variant::ARRAY;
			}

			// The order in which we check the following interfaces is very important (dictionaries and generics first)

			if (GDMonoUtils::Marshal::generic_idictionary_is_assignable_from(reftype))
				return Variant::DICTIONARY;

			if (p_type.type_class->implements_interface(CACHED_CLASS(System_Collections_IDictionary))) {
				return Variant::DICTIONARY;
			}

			if (GDMonoUtils::Marshal::generic_ienumerable_is_assignable_from(reftype))
				return Variant::ARRAY;

			if (p_type.type_class->implements_interface(CACHED_CLASS(System_Collections_IEnumerable))) {
				return Variant::ARRAY;
			}
		} break;

		default: {
		} break;
	}

	if (r_nil_is_variant)
		*r_nil_is_variant = false;

	// Unknown
	return Variant::NIL;
}

bool try_get_array_element_type(const ManagedType &p_array_type, ManagedType &r_elem_type) {
	switch (p_array_type.type_encoding) {
		case MONO_TYPE_GENERICINST: {
			MonoReflectionType *array_reftype = mono_type_get_object(mono_domain_get(), p_array_type.type_class->get_mono_type());

			if (GDMonoUtils::Marshal::type_is_generic_array(array_reftype)) {
				MonoReflectionType *elem_reftype;

				GDMonoUtils::Marshal::array_get_element_type(array_reftype, &elem_reftype);

				r_elem_type = ManagedType::from_reftype(elem_reftype);
				return true;
			}

			MonoReflectionType *elem_reftype;
			if (GDMonoUtils::Marshal::generic_ienumerable_is_assignable_from(array_reftype, &elem_reftype)) {
				r_elem_type = ManagedType::from_reftype(elem_reftype);
				return true;
			}
		} break;
		default: {
		} break;
	}

	return false;
}

bool try_get_dictionary_key_value_types(const ManagedType &p_dictionary_type, ManagedType &r_key_type, ManagedType &r_value_type) {
	switch (p_dictionary_type.type_encoding) {
		case MONO_TYPE_GENERICINST: {
			MonoReflectionType *dict_reftype = mono_type_get_object(mono_domain_get(), p_dictionary_type.type_class->get_mono_type());

			if (GDMonoUtils::Marshal::type_is_generic_dictionary(dict_reftype)) {
				MonoReflectionType *key_reftype;
				MonoReflectionType *value_reftype;

				GDMonoUtils::Marshal::dictionary_get_key_value_types(dict_reftype, &key_reftype, &value_reftype);

				r_key_type = ManagedType::from_reftype(key_reftype);
				r_value_type = ManagedType::from_reftype(value_reftype);
				return true;
			}

			MonoReflectionType *key_reftype, *value_reftype;
			if (GDMonoUtils::Marshal::generic_idictionary_is_assignable_from(dict_reftype, &key_reftype, &value_reftype)) {
				r_key_type = ManagedType::from_reftype(key_reftype);
				r_value_type = ManagedType::from_reftype(value_reftype);
				return true;
			}
		} break;
		default: {
		} break;
	}

	return false;
}

String mono_to_utf8_string(MonoString *p_mono_string) {
	MonoError error;
	char *utf8 = mono_string_to_utf8_checked(p_mono_string, &error);

	if (!mono_error_ok(&error)) {
		ERR_PRINT(String() + "Failed to convert MonoString* to UTF-8: '" + mono_error_get_message(&error) + "'.");
		mono_error_cleanup(&error);
		return String();
	}

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
	CharType *dst = ret.ptrw();

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
			if (p_var->get_type() == Variant::NIL)
				return nullptr; // Otherwise, Variant -> String would return the string "Null"
			return (MonoObject *)mono_string_from_godot(p_var->operator String());
		} break;

		case MONO_TYPE_VALUETYPE: {
			GDMonoClass *vtclass = p_type.type_class;

			if (vtclass == CACHED_CLASS(Vector2)) {
				GDMonoMarshal::M_Vector2 from = MARSHALLED_OUT(Vector2, p_var->operator ::Vector2());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Vector2), &from);
			}

			if (vtclass == CACHED_CLASS(Vector2i)) {
				GDMonoMarshal::M_Vector2i from = MARSHALLED_OUT(Vector2i, p_var->operator ::Vector2i());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Vector2i), &from);
			}

			if (vtclass == CACHED_CLASS(Rect2)) {
				GDMonoMarshal::M_Rect2 from = MARSHALLED_OUT(Rect2, p_var->operator ::Rect2());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Rect2), &from);
			}

			if (vtclass == CACHED_CLASS(Rect2i)) {
				GDMonoMarshal::M_Rect2i from = MARSHALLED_OUT(Rect2i, p_var->operator ::Rect2i());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Rect2i), &from);
			}

			if (vtclass == CACHED_CLASS(Transform2D)) {
				GDMonoMarshal::M_Transform2D from = MARSHALLED_OUT(Transform2D, p_var->operator ::Transform2D());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Transform2D), &from);
			}

			if (vtclass == CACHED_CLASS(Vector3)) {
				GDMonoMarshal::M_Vector3 from = MARSHALLED_OUT(Vector3, p_var->operator ::Vector3());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Vector3), &from);
			}

			if (vtclass == CACHED_CLASS(Vector3i)) {
				GDMonoMarshal::M_Vector3i from = MARSHALLED_OUT(Vector3i, p_var->operator ::Vector3i());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Vector3i), &from);
			}

			if (vtclass == CACHED_CLASS(Basis)) {
				GDMonoMarshal::M_Basis from = MARSHALLED_OUT(Basis, p_var->operator ::Basis());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Basis), &from);
			}

			if (vtclass == CACHED_CLASS(Quat)) {
				GDMonoMarshal::M_Quat from = MARSHALLED_OUT(Quat, p_var->operator ::Quat());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Quat), &from);
			}

			if (vtclass == CACHED_CLASS(Transform)) {
				GDMonoMarshal::M_Transform from = MARSHALLED_OUT(Transform, p_var->operator ::Transform());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Transform), &from);
			}

			if (vtclass == CACHED_CLASS(AABB)) {
				GDMonoMarshal::M_AABB from = MARSHALLED_OUT(AABB, p_var->operator ::AABB());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(AABB), &from);
			}

			if (vtclass == CACHED_CLASS(Color)) {
				GDMonoMarshal::M_Color from = MARSHALLED_OUT(Color, p_var->operator ::Color());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Color), &from);
			}

			if (vtclass == CACHED_CLASS(Plane)) {
				GDMonoMarshal::M_Plane from = MARSHALLED_OUT(Plane, p_var->operator ::Plane());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Plane), &from);
			}

			if (vtclass == CACHED_CLASS(Callable)) {
				GDMonoMarshal::M_Callable from = GDMonoMarshal::callable_to_managed(p_var->operator Callable());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Callable), &from);
			}

			if (vtclass == CACHED_CLASS(SignalInfo)) {
				GDMonoMarshal::M_SignalInfo from = GDMonoMarshal::signal_info_to_managed(p_var->operator Signal());
				return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(SignalInfo), &from);
			}

			if (mono_class_is_enum(vtclass->get_mono_ptr())) {
				MonoType *enum_basetype = mono_class_enum_basetype(vtclass->get_mono_ptr());
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
						ERR_FAIL_V_MSG(nullptr, "Attempted to convert Variant to a managed enum value of unmarshallable base type.");
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
				return (MonoObject *)PackedByteArray_to_mono_array(p_var->operator PackedByteArray());

			if (array_type->eklass == CACHED_CLASS_RAW(int32_t))
				return (MonoObject *)PackedInt32Array_to_mono_array(p_var->operator PackedInt32Array());

			if (array_type->eklass == CACHED_CLASS_RAW(int64_t))
				return (MonoObject *)PackedInt64Array_to_mono_array(p_var->operator PackedInt64Array());

			if (array_type->eklass == CACHED_CLASS_RAW(float))
				return (MonoObject *)PackedFloat32Array_to_mono_array(p_var->operator PackedFloat32Array());

			if (array_type->eklass == CACHED_CLASS_RAW(double))
				return (MonoObject *)PackedFloat64Array_to_mono_array(p_var->operator PackedFloat64Array());

			if (array_type->eklass == CACHED_CLASS_RAW(String))
				return (MonoObject *)PackedStringArray_to_mono_array(p_var->operator PackedStringArray());

			if (array_type->eklass == CACHED_CLASS_RAW(Vector2))
				return (MonoObject *)PackedVector2Array_to_mono_array(p_var->operator PackedVector2Array());

			if (array_type->eklass == CACHED_CLASS_RAW(Vector3))
				return (MonoObject *)PackedVector3Array_to_mono_array(p_var->operator PackedVector3Array());

			if (array_type->eklass == CACHED_CLASS_RAW(Color))
				return (MonoObject *)PackedColorArray_to_mono_array(p_var->operator PackedColorArray());

			ERR_FAIL_V_MSG(nullptr, "Attempted to convert Variant to a managed array of unmarshallable element type.");
		} break;

		case MONO_TYPE_CLASS: {
			GDMonoClass *type_class = p_type.type_class;

			// GodotObject
			if (CACHED_CLASS(GodotObject)->is_assignable_from(type_class)) {
				return GDMonoUtils::unmanaged_get_managed(p_var->operator Object *());
			}

			if (CACHED_CLASS(StringName) == type_class) {
				return GDMonoUtils::create_managed_from(p_var->operator StringName());
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

			// The order in which we check the following interfaces is very important (dictionaries and generics first)

			MonoReflectionType *reftype = mono_type_get_object(mono_domain_get(), type_class->get_mono_type());

			MonoReflectionType *key_reftype, *value_reftype;
			if (GDMonoUtils::Marshal::generic_idictionary_is_assignable_from(reftype, &key_reftype, &value_reftype)) {
				return GDMonoUtils::create_managed_from(p_var->operator Dictionary(),
						GDMonoUtils::Marshal::make_generic_dictionary_type(key_reftype, value_reftype));
			}

			if (type_class->implements_interface(CACHED_CLASS(System_Collections_IDictionary))) {
				return GDMonoUtils::create_managed_from(p_var->operator Dictionary(), CACHED_CLASS(Dictionary));
			}

			MonoReflectionType *elem_reftype;
			if (GDMonoUtils::Marshal::generic_ienumerable_is_assignable_from(reftype, &elem_reftype)) {
				return GDMonoUtils::create_managed_from(p_var->operator Array(),
						GDMonoUtils::Marshal::make_generic_array_type(elem_reftype));
			}

			if (type_class->implements_interface(CACHED_CLASS(System_Collections_IEnumerable))) {
				if (GDMonoCache::tools_godot_api_check()) {
					return GDMonoUtils::create_managed_from(p_var->operator Array(), CACHED_CLASS(Array));
				} else {
					return (MonoObject *)GDMonoMarshal::Array_to_mono_array(p_var->operator Array());
				}
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
				case Variant::FLOAT: {
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
				case Variant::VECTOR2I: {
					GDMonoMarshal::M_Vector2i from = MARSHALLED_OUT(Vector2i, p_var->operator ::Vector2i());
					return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Vector2i), &from);
				}
				case Variant::RECT2: {
					GDMonoMarshal::M_Rect2 from = MARSHALLED_OUT(Rect2, p_var->operator ::Rect2());
					return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Rect2), &from);
				}
				case Variant::RECT2I: {
					GDMonoMarshal::M_Rect2i from = MARSHALLED_OUT(Rect2i, p_var->operator ::Rect2i());
					return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Rect2i), &from);
				}
				case Variant::VECTOR3: {
					GDMonoMarshal::M_Vector3 from = MARSHALLED_OUT(Vector3, p_var->operator ::Vector3());
					return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Vector3), &from);
				}
				case Variant::VECTOR3I: {
					GDMonoMarshal::M_Vector3i from = MARSHALLED_OUT(Vector3i, p_var->operator ::Vector3i());
					return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Vector3i), &from);
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
				case Variant::STRING_NAME:
					return GDMonoUtils::create_managed_from(p_var->operator StringName());
				case Variant::NODE_PATH:
					return GDMonoUtils::create_managed_from(p_var->operator NodePath());
				case Variant::_RID:
					return GDMonoUtils::create_managed_from(p_var->operator RID());
				case Variant::OBJECT:
					return GDMonoUtils::unmanaged_get_managed(p_var->operator Object *());
				case Variant::CALLABLE: {
					GDMonoMarshal::M_Callable from = GDMonoMarshal::callable_to_managed(p_var->operator Callable());
					return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(Callable), &from);
				}
				case Variant::SIGNAL: {
					GDMonoMarshal::M_SignalInfo from = GDMonoMarshal::signal_info_to_managed(p_var->operator Signal());
					return mono_value_box(mono_domain_get(), CACHED_CLASS_RAW(SignalInfo), &from);
				}
				case Variant::DICTIONARY:
					return GDMonoUtils::create_managed_from(p_var->operator Dictionary(), CACHED_CLASS(Dictionary));
				case Variant::ARRAY:
					return GDMonoUtils::create_managed_from(p_var->operator Array(), CACHED_CLASS(Array));
				case Variant::PACKED_BYTE_ARRAY:
					return (MonoObject *)PackedByteArray_to_mono_array(p_var->operator PackedByteArray());
				case Variant::PACKED_INT32_ARRAY:
					return (MonoObject *)PackedInt32Array_to_mono_array(p_var->operator PackedInt32Array());
				case Variant::PACKED_INT64_ARRAY:
					return (MonoObject *)PackedInt64Array_to_mono_array(p_var->operator PackedInt64Array());
				case Variant::PACKED_FLOAT32_ARRAY:
					return (MonoObject *)PackedFloat32Array_to_mono_array(p_var->operator PackedFloat32Array());
				case Variant::PACKED_FLOAT64_ARRAY:
					return (MonoObject *)PackedFloat64Array_to_mono_array(p_var->operator PackedFloat64Array());
				case Variant::PACKED_STRING_ARRAY:
					return (MonoObject *)PackedStringArray_to_mono_array(p_var->operator PackedStringArray());
				case Variant::PACKED_VECTOR2_ARRAY:
					return (MonoObject *)PackedVector2Array_to_mono_array(p_var->operator PackedVector2Array());
				case Variant::PACKED_VECTOR3_ARRAY:
					return (MonoObject *)PackedVector3Array_to_mono_array(p_var->operator PackedVector3Array());
				case Variant::PACKED_COLOR_ARRAY:
					return (MonoObject *)PackedColorArray_to_mono_array(p_var->operator PackedColorArray());
				default:
					return nullptr;
			}
			break;
			case MONO_TYPE_GENERICINST: {
				MonoReflectionType *reftype = mono_type_get_object(mono_domain_get(), p_type.type_class->get_mono_type());

				if (GDMonoUtils::Marshal::type_is_generic_dictionary(reftype)) {
					return GDMonoUtils::create_managed_from(p_var->operator Dictionary(), p_type.type_class);
				}

				if (GDMonoUtils::Marshal::type_is_generic_array(reftype)) {
					return GDMonoUtils::create_managed_from(p_var->operator Array(), p_type.type_class);
				}

				// The order in which we check the following interfaces is very important (dictionaries and generics first)

				MonoReflectionType *key_reftype, *value_reftype;
				if (GDMonoUtils::Marshal::generic_idictionary_is_assignable_from(reftype, &key_reftype, &value_reftype)) {
					return GDMonoUtils::create_managed_from(p_var->operator Dictionary(),
							GDMonoUtils::Marshal::make_generic_dictionary_type(key_reftype, value_reftype));
				}

				if (p_type.type_class->implements_interface(CACHED_CLASS(System_Collections_IDictionary))) {
					return GDMonoUtils::create_managed_from(p_var->operator Dictionary(), CACHED_CLASS(Dictionary));
				}

				MonoReflectionType *elem_reftype;
				if (GDMonoUtils::Marshal::generic_ienumerable_is_assignable_from(reftype, &elem_reftype)) {
					return GDMonoUtils::create_managed_from(p_var->operator Array(),
							GDMonoUtils::Marshal::make_generic_array_type(elem_reftype));
				}

				if (p_type.type_class->implements_interface(CACHED_CLASS(System_Collections_IEnumerable))) {
					if (GDMonoCache::tools_godot_api_check()) {
						return GDMonoUtils::create_managed_from(p_var->operator Array(), CACHED_CLASS(Array));
					} else {
						return (MonoObject *)GDMonoMarshal::Array_to_mono_array(p_var->operator Array());
					}
				}
			} break;
		} break;
	}

	ERR_FAIL_V_MSG(nullptr, "Attempted to convert Variant to an unmarshallable managed type. Name: '" +
									p_type.type_class->get_name() + "' Encoding: " + itos(p_type.type_encoding) + ".");
}

Variant mono_object_to_variant_impl(MonoObject *p_obj, const ManagedType &p_type, bool p_fail_with_err = true) {

	ERR_FAIL_COND_V(!p_type.type_class, Variant());

	switch (p_type.type_encoding) {
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
			if (p_obj == nullptr)
				return Variant(); // NIL
			return mono_string_to_godot_not_null((MonoString *)p_obj);
		} break;

		case MONO_TYPE_VALUETYPE: {
			GDMonoClass *vtclass = p_type.type_class;

			if (vtclass == CACHED_CLASS(Vector2))
				return MARSHALLED_IN(Vector2, unbox_addr<GDMonoMarshal::M_Vector2>(p_obj));

			if (vtclass == CACHED_CLASS(Vector2i))
				return MARSHALLED_IN(Vector2i, unbox_addr<GDMonoMarshal::M_Vector2i>(p_obj));

			if (vtclass == CACHED_CLASS(Rect2))
				return MARSHALLED_IN(Rect2, unbox_addr<GDMonoMarshal::M_Rect2>(p_obj));

			if (vtclass == CACHED_CLASS(Rect2i))
				return MARSHALLED_IN(Rect2i, unbox_addr<GDMonoMarshal::M_Rect2i>(p_obj));

			if (vtclass == CACHED_CLASS(Transform2D))
				return MARSHALLED_IN(Transform2D, unbox_addr<GDMonoMarshal::M_Transform2D>(p_obj));

			if (vtclass == CACHED_CLASS(Vector3))
				return MARSHALLED_IN(Vector3, unbox_addr<GDMonoMarshal::M_Vector3>(p_obj));

			if (vtclass == CACHED_CLASS(Vector3i))
				return MARSHALLED_IN(Vector3i, unbox_addr<GDMonoMarshal::M_Vector3i>(p_obj));

			if (vtclass == CACHED_CLASS(Basis))
				return MARSHALLED_IN(Basis, unbox_addr<GDMonoMarshal::M_Basis>(p_obj));

			if (vtclass == CACHED_CLASS(Quat))
				return MARSHALLED_IN(Quat, unbox_addr<GDMonoMarshal::M_Quat>(p_obj));

			if (vtclass == CACHED_CLASS(Transform))
				return MARSHALLED_IN(Transform, unbox_addr<GDMonoMarshal::M_Transform>(p_obj));

			if (vtclass == CACHED_CLASS(AABB))
				return MARSHALLED_IN(AABB, unbox_addr<GDMonoMarshal::M_AABB>(p_obj));

			if (vtclass == CACHED_CLASS(Color))
				return MARSHALLED_IN(Color, unbox_addr<GDMonoMarshal::M_Color>(p_obj));

			if (vtclass == CACHED_CLASS(Plane))
				return MARSHALLED_IN(Plane, unbox_addr<GDMonoMarshal::M_Plane>(p_obj));

			if (vtclass == CACHED_CLASS(Callable))
				return managed_to_callable(unbox<GDMonoMarshal::M_Callable>(p_obj));

			if (vtclass == CACHED_CLASS(SignalInfo))
				return managed_to_signal_info(unbox<GDMonoMarshal::M_SignalInfo>(p_obj));

			if (mono_class_is_enum(vtclass->get_mono_ptr()))
				return unbox<int32_t>(p_obj);
		} break;

		case MONO_TYPE_ARRAY:
		case MONO_TYPE_SZARRAY: {
			MonoArrayType *array_type = mono_type_get_array_type(p_type.type_class->get_mono_type());

			if (array_type->eklass == CACHED_CLASS_RAW(MonoObject))
				return mono_array_to_Array((MonoArray *)p_obj);

			if (array_type->eklass == CACHED_CLASS_RAW(uint8_t))
				return mono_array_to_PackedByteArray((MonoArray *)p_obj);

			if (array_type->eklass == CACHED_CLASS_RAW(int32_t))
				return mono_array_to_PackedInt32Array((MonoArray *)p_obj);

			if (array_type->eklass == CACHED_CLASS_RAW(int64_t))
				return mono_array_to_PackedInt64Array((MonoArray *)p_obj);

			if (array_type->eklass == CACHED_CLASS_RAW(float))
				return mono_array_to_PackedFloat32Array((MonoArray *)p_obj);

			if (array_type->eklass == CACHED_CLASS_RAW(double))
				return mono_array_to_PackedFloat64Array((MonoArray *)p_obj);

			if (array_type->eklass == CACHED_CLASS_RAW(String))
				return mono_array_to_PackedStringArray((MonoArray *)p_obj);

			if (array_type->eklass == CACHED_CLASS_RAW(Vector2))
				return mono_array_to_PackedVector2Array((MonoArray *)p_obj);

			if (array_type->eklass == CACHED_CLASS_RAW(Vector3))
				return mono_array_to_PackedVector3Array((MonoArray *)p_obj);

			if (array_type->eklass == CACHED_CLASS_RAW(Color))
				return mono_array_to_PackedColorArray((MonoArray *)p_obj);

			if (p_fail_with_err) {
				ERR_FAIL_V_MSG(Variant(), "Attempted to convert a managed array of unmarshallable element type to Variant.");
			} else {
				return Variant();
			}
		} break;

		case MONO_TYPE_CLASS: {
			GDMonoClass *type_class = p_type.type_class;

			// GodotObject
			if (CACHED_CLASS(GodotObject)->is_assignable_from(type_class)) {
				Object *ptr = unbox<Object *>(CACHED_FIELD(GodotObject, ptr)->get_value(p_obj));
				if (ptr != nullptr) {
					Reference *ref = Object::cast_to<Reference>(ptr);
					return ref ? Variant(Ref<Reference>(ref)) : Variant(ptr);
				}
				return Variant();
			}

			if (CACHED_CLASS(StringName) == type_class) {
				StringName *ptr = unbox<StringName *>(CACHED_FIELD(StringName, ptr)->get_value(p_obj));
				return ptr ? Variant(*ptr) : Variant();
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
				MonoException *exc = nullptr;
				Array *ptr = CACHED_METHOD_THUNK(Array, GetPtr).invoke(p_obj, &exc);
				UNHANDLED_EXCEPTION(exc);
				return ptr ? Variant(*ptr) : Variant();
			}

			if (CACHED_CLASS(Dictionary) == type_class) {
				MonoException *exc = nullptr;
				Dictionary *ptr = CACHED_METHOD_THUNK(Dictionary, GetPtr).invoke(p_obj, &exc);
				UNHANDLED_EXCEPTION(exc);
				return ptr ? Variant(*ptr) : Variant();
			}

			// The order in which we check the following interfaces is very important (dictionaries and generics first)

			MonoReflectionType *reftype = mono_type_get_object(mono_domain_get(), type_class->get_mono_type());

			if (GDMonoUtils::Marshal::generic_idictionary_is_assignable_from(reftype)) {
				return GDMonoUtils::Marshal::generic_idictionary_to_dictionary(p_obj);
			}

			if (type_class->implements_interface(CACHED_CLASS(System_Collections_IDictionary))) {
				return GDMonoUtils::Marshal::idictionary_to_dictionary(p_obj);
			}

			if (GDMonoUtils::Marshal::generic_ienumerable_is_assignable_from(reftype)) {
				return GDMonoUtils::Marshal::enumerable_to_array(p_obj);
			}

			if (type_class->implements_interface(CACHED_CLASS(System_Collections_IEnumerable))) {
				return GDMonoUtils::Marshal::enumerable_to_array(p_obj);
			}
		} break;

		case MONO_TYPE_GENERICINST: {
			MonoReflectionType *reftype = mono_type_get_object(mono_domain_get(), p_type.type_class->get_mono_type());

			if (GDMonoUtils::Marshal::type_is_generic_dictionary(reftype)) {
				MonoException *exc = nullptr;
				MonoObject *ret = p_type.type_class->get_method("GetPtr")->invoke(p_obj, &exc);
				UNHANDLED_EXCEPTION(exc);
				return *unbox<Dictionary *>(ret);
			}

			if (GDMonoUtils::Marshal::type_is_generic_array(reftype)) {
				MonoException *exc = nullptr;
				MonoObject *ret = p_type.type_class->get_method("GetPtr")->invoke(p_obj, &exc);
				UNHANDLED_EXCEPTION(exc);
				return *unbox<Array *>(ret);
			}

			// The order in which we check the following interfaces is very important (dictionaries and generics first)

			if (GDMonoUtils::Marshal::generic_idictionary_is_assignable_from(reftype)) {
				return GDMonoUtils::Marshal::generic_idictionary_to_dictionary(p_obj);
			}

			if (p_type.type_class->implements_interface(CACHED_CLASS(System_Collections_IDictionary))) {
				return GDMonoUtils::Marshal::idictionary_to_dictionary(p_obj);
			}

			if (GDMonoUtils::Marshal::generic_ienumerable_is_assignable_from(reftype)) {
				return GDMonoUtils::Marshal::enumerable_to_array(p_obj);
			}

			if (p_type.type_class->implements_interface(CACHED_CLASS(System_Collections_IEnumerable))) {
				return GDMonoUtils::Marshal::enumerable_to_array(p_obj);
			}
		} break;
	}

	if (p_fail_with_err) {
		ERR_FAIL_V_MSG(Variant(), "Attempted to convert an unmarshallable managed type to Variant. Name: '" +
										  p_type.type_class->get_name() + "' Encoding: " + itos(p_type.type_encoding) + ".");
	} else {
		return Variant();
	}
}

Variant mono_object_to_variant(MonoObject *p_obj) {
	if (!p_obj)
		return Variant();

	ManagedType type = ManagedType::from_class(mono_object_get_class(p_obj));

	return mono_object_to_variant_impl(p_obj, type);
}

Variant mono_object_to_variant(MonoObject *p_obj, const ManagedType &p_type) {
	if (!p_obj)
		return Variant();

	return mono_object_to_variant_impl(p_obj, p_type);
}

Variant mono_object_to_variant_no_err(MonoObject *p_obj, const ManagedType &p_type) {
	if (!p_obj)
		return Variant();

	return mono_object_to_variant_impl(p_obj, p_type, /* fail_with_err: */ false);
}

String mono_object_to_variant_string(MonoObject *p_obj, MonoException **r_exc) {
	ManagedType type = ManagedType::from_class(mono_object_get_class(p_obj));
	Variant var = GDMonoMarshal::mono_object_to_variant_no_err(p_obj, type);

	if (var.get_type() == Variant::NIL && p_obj != nullptr) {
		// Cannot convert MonoObject* to Variant; fallback to 'ToString()'.
		MonoException *exc = nullptr;
		MonoString *mono_str = GDMonoUtils::object_to_string(p_obj, &exc);

		if (exc) {
			if (r_exc)
				*r_exc = exc;
			return String();
		}

		return GDMonoMarshal::mono_string_to_godot(mono_str);
	} else {
		return var.operator String();
	}
}

MonoArray *Array_to_mono_array(const Array &p_array) {
	int length = p_array.size();
	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(MonoObject), length);

	for (int i = 0; i < length; i++) {
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

MonoArray *PackedInt32Array_to_mono_array(const PackedInt32Array &p_array) {
	const int32_t *src = p_array.ptr();
	int length = p_array.size();

	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(int32_t), length);

	int32_t *dst = (int32_t *)mono_array_addr(ret, int32_t, 0);
	memcpy(dst, src, length);

	return ret;
}

PackedInt32Array mono_array_to_PackedInt32Array(MonoArray *p_array) {
	PackedInt32Array ret;
	if (!p_array)
		return ret;
	int length = mono_array_length(p_array);
	ret.resize(length);
	int32_t *dst = ret.ptrw();

	const int32_t *src = (const int32_t *)mono_array_addr(p_array, int32_t, 0);
	memcpy(dst, src, length);

	return ret;
}

MonoArray *PackedInt64Array_to_mono_array(const PackedInt64Array &p_array) {
	const int64_t *src = p_array.ptr();
	int length = p_array.size();

	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(int64_t), length);

	int64_t *dst = (int64_t *)mono_array_addr(ret, int64_t, 0);
	memcpy(dst, src, length);

	return ret;
}

PackedInt64Array mono_array_to_PackedInt64Array(MonoArray *p_array) {
	PackedInt64Array ret;
	if (!p_array)
		return ret;
	int length = mono_array_length(p_array);
	ret.resize(length);
	int64_t *dst = ret.ptrw();

	const int64_t *src = (const int64_t *)mono_array_addr(p_array, int64_t, 0);
	memcpy(dst, src, length);

	return ret;
}

MonoArray *PackedByteArray_to_mono_array(const PackedByteArray &p_array) {
	const uint8_t *src = p_array.ptr();
	int length = p_array.size();

	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(uint8_t), length);

	uint8_t *dst = (uint8_t *)mono_array_addr(ret, uint8_t, 0);
	memcpy(dst, src, length);

	return ret;
}

PackedByteArray mono_array_to_PackedByteArray(MonoArray *p_array) {
	PackedByteArray ret;
	if (!p_array)
		return ret;
	int length = mono_array_length(p_array);
	ret.resize(length);
	uint8_t *dst = ret.ptrw();

	const uint8_t *src = (const uint8_t *)mono_array_addr(p_array, uint8_t, 0);
	memcpy(dst, src, length);

	return ret;
}

MonoArray *PackedFloat32Array_to_mono_array(const PackedFloat32Array &p_array) {
	const float *src = p_array.ptr();
	int length = p_array.size();

	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(float), length);

	float *dst = (float *)mono_array_addr(ret, float, 0);
	memcpy(dst, src, length);

	return ret;
}

PackedFloat32Array mono_array_to_PackedFloat32Array(MonoArray *p_array) {
	PackedFloat32Array ret;
	if (!p_array)
		return ret;
	int length = mono_array_length(p_array);
	ret.resize(length);
	float *dst = ret.ptrw();

	const float *src = (const float *)mono_array_addr(p_array, float, 0);
	memcpy(dst, src, length);

	return ret;
}

MonoArray *PackedFloat64Array_to_mono_array(const PackedFloat64Array &p_array) {
	const double *src = p_array.ptr();
	int length = p_array.size();

	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(double), length);

	double *dst = (double *)mono_array_addr(ret, double, 0);
	memcpy(dst, src, length);

	return ret;
}

PackedFloat64Array mono_array_to_PackedFloat64Array(MonoArray *p_array) {
	PackedFloat64Array ret;
	if (!p_array)
		return ret;
	int length = mono_array_length(p_array);
	ret.resize(length);
	double *dst = ret.ptrw();

	const double *src = (const double *)mono_array_addr(p_array, double, 0);
	memcpy(dst, src, length);

	return ret;
}

MonoArray *PackedStringArray_to_mono_array(const PackedStringArray &p_array) {
	const String *r = p_array.ptr();
	int length = p_array.size();

	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(String), length);

	for (int i = 0; i < length; i++) {
		MonoString *boxed = mono_string_from_godot(r[i]);
		mono_array_setref(ret, i, boxed);
	}

	return ret;
}

PackedStringArray mono_array_to_PackedStringArray(MonoArray *p_array) {
	PackedStringArray ret;
	if (!p_array)
		return ret;
	int length = mono_array_length(p_array);
	ret.resize(length);
	String *w = ret.ptrw();

	for (int i = 0; i < length; i++) {
		MonoString *elem = mono_array_get(p_array, MonoString *, i);
		w[i] = mono_string_to_godot(elem);
	}

	return ret;
}

MonoArray *PackedColorArray_to_mono_array(const PackedColorArray &p_array) {
	const Color *src = p_array.ptr();
	int length = p_array.size();

	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(Color), length);

	if constexpr (InteropLayout::MATCHES_Color) {
		Color *dst = (Color *)mono_array_addr(ret, Color, 0);
		memcpy(dst, src, length);
	} else {
		for (int i = 0; i < length; i++) {
			M_Color *raw = (M_Color *)mono_array_addr_with_size(ret, sizeof(M_Color), i);
			*raw = MARSHALLED_OUT(Color, src[i]);
		}
	}

	return ret;
}

PackedColorArray mono_array_to_PackedColorArray(MonoArray *p_array) {
	PackedColorArray ret;
	if (!p_array)
		return ret;
	int length = mono_array_length(p_array);
	ret.resize(length);
	Color *dst = ret.ptrw();

	if constexpr (InteropLayout::MATCHES_Color) {
		const Color *src = (const Color *)mono_array_addr(p_array, Color, 0);
		memcpy(dst, src, length);
	} else {
		for (int i = 0; i < length; i++) {
			dst[i] = MARSHALLED_IN(Color, (M_Color *)mono_array_addr_with_size(p_array, sizeof(M_Color), i));
		}
	}

	return ret;
}

MonoArray *PackedVector2Array_to_mono_array(const PackedVector2Array &p_array) {
	const Vector2 *src = p_array.ptr();
	int length = p_array.size();

	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(Vector2), length);

	if constexpr (InteropLayout::MATCHES_Vector2) {
		Vector2 *dst = (Vector2 *)mono_array_addr(ret, Vector2, 0);
		memcpy(dst, src, length);
	} else {
		for (int i = 0; i < length; i++) {
			M_Vector2 *raw = (M_Vector2 *)mono_array_addr_with_size(ret, sizeof(M_Vector2), i);
			*raw = MARSHALLED_OUT(Vector2, src[i]);
		}
	}

	return ret;
}

PackedVector2Array mono_array_to_PackedVector2Array(MonoArray *p_array) {
	PackedVector2Array ret;
	if (!p_array)
		return ret;
	int length = mono_array_length(p_array);
	ret.resize(length);
	Vector2 *dst = ret.ptrw();

	if constexpr (InteropLayout::MATCHES_Vector2) {
		const Vector2 *src = (const Vector2 *)mono_array_addr(p_array, Vector2, 0);
		memcpy(dst, src, length);
	} else {
		for (int i = 0; i < length; i++) {
			dst[i] = MARSHALLED_IN(Vector2, (M_Vector2 *)mono_array_addr_with_size(p_array, sizeof(M_Vector2), i));
		}
	}

	return ret;
}

MonoArray *PackedVector3Array_to_mono_array(const PackedVector3Array &p_array) {
	const Vector3 *src = p_array.ptr();
	int length = p_array.size();

	MonoArray *ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(Vector3), length);

	if constexpr (InteropLayout::MATCHES_Vector3) {
		Vector3 *dst = (Vector3 *)mono_array_addr(ret, Vector3, 0);
		memcpy(dst, src, length);
	} else {
		for (int i = 0; i < length; i++) {
			M_Vector3 *raw = (M_Vector3 *)mono_array_addr_with_size(ret, sizeof(M_Vector3), i);
			*raw = MARSHALLED_OUT(Vector3, src[i]);
		}
	}

	return ret;
}

PackedVector3Array mono_array_to_PackedVector3Array(MonoArray *p_array) {
	PackedVector3Array ret;
	if (!p_array)
		return ret;
	int length = mono_array_length(p_array);
	ret.resize(length);
	Vector3 *dst = ret.ptrw();

	if constexpr (InteropLayout::MATCHES_Vector3) {
		const Vector3 *src = (const Vector3 *)mono_array_addr(p_array, Vector3, 0);
		memcpy(dst, src, length);
	} else {
		for (int i = 0; i < length; i++) {
			dst[i] = MARSHALLED_IN(Vector3, (M_Vector3 *)mono_array_addr_with_size(p_array, sizeof(M_Vector3), i));
		}
	}

	return ret;
}

Callable managed_to_callable(const M_Callable &p_managed_callable) {
	if (p_managed_callable.delegate) {
		// TODO: Use pooling for ManagedCallable instances.
		CallableCustom *managed_callable = memnew(ManagedCallable(p_managed_callable.delegate));
		return Callable(managed_callable);
	} else {
		Object *target = p_managed_callable.target ?
								 unbox<Object *>(CACHED_FIELD(GodotObject, ptr)->get_value(p_managed_callable.target)) :
								 nullptr;
		StringName *method_ptr = unbox<StringName *>(CACHED_FIELD(StringName, ptr)->get_value(p_managed_callable.method_string_name));
		StringName method = method_ptr ? *method_ptr : StringName();
		return Callable(target, method);
	}
}

M_Callable callable_to_managed(const Callable &p_callable) {
	if (p_callable.is_custom()) {
		CallableCustom *custom = p_callable.get_custom();
		CallableCustom::CompareEqualFunc compare_equal_func = custom->get_compare_equal_func();

		if (compare_equal_func == ManagedCallable::compare_equal_func_ptr) {
			ManagedCallable *managed_callable = static_cast<ManagedCallable *>(custom);
			return {
				nullptr, nullptr,
				managed_callable->get_delegate()
			};
		} else if (compare_equal_func == SignalAwaiterCallable::compare_equal_func_ptr) {
			SignalAwaiterCallable *signal_awaiter_callable = static_cast<SignalAwaiterCallable *>(custom);
			return {
				GDMonoUtils::unmanaged_get_managed(ObjectDB::get_instance(signal_awaiter_callable->get_object())),
				GDMonoUtils::create_managed_from(signal_awaiter_callable->get_signal()),
				nullptr
			};
		} else if (compare_equal_func == EventSignalCallable::compare_equal_func_ptr) {
			EventSignalCallable *event_signal_callable = static_cast<EventSignalCallable *>(custom);
			return {
				GDMonoUtils::unmanaged_get_managed(ObjectDB::get_instance(event_signal_callable->get_object())),
				GDMonoUtils::create_managed_from(event_signal_callable->get_signal()),
				nullptr
			};
		}

		// Some other CallableCustom. We only support ManagedCallable.
		return { nullptr, nullptr, nullptr };
	} else {
		MonoObject *target_managed = GDMonoUtils::unmanaged_get_managed(p_callable.get_object());
		MonoObject *method_string_name_managed = GDMonoUtils::create_managed_from(p_callable.get_method());
		return { target_managed, method_string_name_managed, nullptr };
	}
}

Signal managed_to_signal_info(const M_SignalInfo &p_managed_signal) {
	Object *owner = p_managed_signal.owner ?
							unbox<Object *>(CACHED_FIELD(GodotObject, ptr)->get_value(p_managed_signal.owner)) :
							nullptr;
	StringName *name_ptr = unbox<StringName *>(CACHED_FIELD(StringName, ptr)->get_value(p_managed_signal.name_string_name));
	StringName name = name_ptr ? *name_ptr : StringName();
	return Signal(owner, name);
}

M_SignalInfo signal_info_to_managed(const Signal &p_signal) {
	Object *owner = p_signal.get_object();
	MonoObject *owner_managed = GDMonoUtils::unmanaged_get_managed(owner);
	MonoObject *name_string_name_managed = GDMonoUtils::create_managed_from(p_signal.get_name());
	return { owner_managed, name_string_name_managed };
}

} // namespace GDMonoMarshal
