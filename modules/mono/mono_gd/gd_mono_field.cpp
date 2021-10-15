/*************************************************************************/
/*  gd_mono_field.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "gd_mono_field.h"

#include <mono/metadata/attrdefs.h>

#include "gd_mono_cache.h"
#include "gd_mono_class.h"
#include "gd_mono_marshal.h"
#include "gd_mono_utils.h"

void GDMonoField::set_value_raw(MonoObject *p_object, void *p_ptr) {
	mono_field_set_value(p_object, mono_field, &p_ptr);
}

void GDMonoField::set_value_from_variant(MonoObject *p_object, const Variant &p_value) {
#define SET_FROM_STRUCT(m_type)                                                               \
	{                                                                                         \
		GDMonoMarshal::M_##m_type from = MARSHALLED_OUT(m_type, p_value.operator ::m_type()); \
		mono_field_set_value(p_object, mono_field, &from);                                    \
	}

#define SET_FROM_ARRAY(m_type)                                                                   \
	{                                                                                            \
		MonoArray *managed = GDMonoMarshal::m_type##_to_mono_array(p_value.operator ::m_type()); \
		mono_field_set_value(p_object, mono_field, managed);                                     \
	}

	switch (type.type_encoding) {
		case MONO_TYPE_BOOLEAN: {
			MonoBoolean val = p_value.operator bool();
			mono_field_set_value(p_object, mono_field, &val);
		} break;

		case MONO_TYPE_CHAR: {
			int16_t val = p_value.operator unsigned short();
			mono_field_set_value(p_object, mono_field, &val);
		} break;

		case MONO_TYPE_I1: {
			int8_t val = p_value.operator signed char();
			mono_field_set_value(p_object, mono_field, &val);
		} break;
		case MONO_TYPE_I2: {
			int16_t val = p_value.operator signed short();
			mono_field_set_value(p_object, mono_field, &val);
		} break;
		case MONO_TYPE_I4: {
			int32_t val = p_value.operator signed int();
			mono_field_set_value(p_object, mono_field, &val);
		} break;
		case MONO_TYPE_I8: {
			int64_t val = p_value.operator int64_t();
			mono_field_set_value(p_object, mono_field, &val);
		} break;

		case MONO_TYPE_U1: {
			uint8_t val = p_value.operator unsigned char();
			mono_field_set_value(p_object, mono_field, &val);
		} break;
		case MONO_TYPE_U2: {
			uint16_t val = p_value.operator unsigned short();
			mono_field_set_value(p_object, mono_field, &val);
		} break;
		case MONO_TYPE_U4: {
			uint32_t val = p_value.operator unsigned int();
			mono_field_set_value(p_object, mono_field, &val);
		} break;
		case MONO_TYPE_U8: {
			uint64_t val = p_value.operator uint64_t();
			mono_field_set_value(p_object, mono_field, &val);
		} break;

		case MONO_TYPE_R4: {
			float val = p_value.operator float();
			mono_field_set_value(p_object, mono_field, &val);
		} break;

		case MONO_TYPE_R8: {
			double val = p_value.operator double();
			mono_field_set_value(p_object, mono_field, &val);
		} break;

		case MONO_TYPE_STRING: {
			if (p_value.get_type() == Variant::NIL) {
				// Otherwise, Variant -> String would return the string "Null"
				MonoString *mono_string = NULL;
				mono_field_set_value(p_object, mono_field, mono_string);
			} else {
				MonoString *mono_string = GDMonoMarshal::mono_string_from_godot(p_value);
				mono_field_set_value(p_object, mono_field, mono_string);
			}
		} break;

		case MONO_TYPE_VALUETYPE: {
			GDMonoClass *tclass = type.type_class;

			if (tclass == CACHED_CLASS(Vector2)) {
				SET_FROM_STRUCT(Vector2);
				break;
			}

			if (tclass == CACHED_CLASS(Rect2)) {
				SET_FROM_STRUCT(Rect2);
				break;
			}

			if (tclass == CACHED_CLASS(Transform2D)) {
				SET_FROM_STRUCT(Transform2D);
				break;
			}

			if (tclass == CACHED_CLASS(Vector3)) {
				SET_FROM_STRUCT(Vector3);
				break;
			}

			if (tclass == CACHED_CLASS(Basis)) {
				SET_FROM_STRUCT(Basis);
				break;
			}

			if (tclass == CACHED_CLASS(Quat)) {
				SET_FROM_STRUCT(Quat);
				break;
			}

			if (tclass == CACHED_CLASS(Transform)) {
				SET_FROM_STRUCT(Transform);
				break;
			}

			if (tclass == CACHED_CLASS(AABB)) {
				SET_FROM_STRUCT(AABB);
				break;
			}

			if (tclass == CACHED_CLASS(Color)) {
				SET_FROM_STRUCT(Color);
				break;
			}

			if (tclass == CACHED_CLASS(Plane)) {
				SET_FROM_STRUCT(Plane);
				break;
			}

			if (mono_class_is_enum(tclass->get_mono_ptr())) {
				MonoType *enum_basetype = mono_class_enum_basetype(tclass->get_mono_ptr());
				switch (mono_type_get_type(enum_basetype)) {
					case MONO_TYPE_BOOLEAN: {
						MonoBoolean val = p_value.operator bool();
						mono_field_set_value(p_object, mono_field, &val);
						break;
					}
					case MONO_TYPE_CHAR: {
						uint16_t val = p_value.operator unsigned short();
						mono_field_set_value(p_object, mono_field, &val);
						break;
					}
					case MONO_TYPE_I1: {
						int8_t val = p_value.operator signed char();
						mono_field_set_value(p_object, mono_field, &val);
						break;
					}
					case MONO_TYPE_I2: {
						int16_t val = p_value.operator signed short();
						mono_field_set_value(p_object, mono_field, &val);
						break;
					}
					case MONO_TYPE_I4: {
						int32_t val = p_value.operator signed int();
						mono_field_set_value(p_object, mono_field, &val);
						break;
					}
					case MONO_TYPE_I8: {
						int64_t val = p_value.operator int64_t();
						mono_field_set_value(p_object, mono_field, &val);
						break;
					}
					case MONO_TYPE_U1: {
						uint8_t val = p_value.operator unsigned char();
						mono_field_set_value(p_object, mono_field, &val);
						break;
					}
					case MONO_TYPE_U2: {
						uint16_t val = p_value.operator unsigned short();
						mono_field_set_value(p_object, mono_field, &val);
						break;
					}
					case MONO_TYPE_U4: {
						uint32_t val = p_value.operator unsigned int();
						mono_field_set_value(p_object, mono_field, &val);
						break;
					}
					case MONO_TYPE_U8: {
						uint64_t val = p_value.operator uint64_t();
						mono_field_set_value(p_object, mono_field, &val);
						break;
					}
					default: {
						ERR_FAIL_MSG("Attempted to convert Variant to a managed enum value of unmarshallable base type.");
					}
				}

				break;
			}

			ERR_FAIL_MSG("Attempted to set the value of a field of unmarshallable type: '" + tclass->get_name() + "'.");
		} break;

		case MONO_TYPE_ARRAY:
		case MONO_TYPE_SZARRAY: {
			MonoArrayType *array_type = mono_type_get_array_type(type.type_class->get_mono_type());

			if (array_type->eklass == CACHED_CLASS_RAW(MonoObject)) {
				SET_FROM_ARRAY(Array);
				break;
			}

			if (array_type->eklass == CACHED_CLASS_RAW(uint8_t)) {
				SET_FROM_ARRAY(PoolByteArray);
				break;
			}

			if (array_type->eklass == CACHED_CLASS_RAW(int32_t)) {
				SET_FROM_ARRAY(PoolIntArray);
				break;
			}

			if (array_type->eklass == REAL_T_MONOCLASS) {
				SET_FROM_ARRAY(PoolRealArray);
				break;
			}

			if (array_type->eklass == CACHED_CLASS_RAW(String)) {
				SET_FROM_ARRAY(PoolStringArray);
				break;
			}

			if (array_type->eklass == CACHED_CLASS_RAW(Vector2)) {
				SET_FROM_ARRAY(PoolVector2Array);
				break;
			}

			if (array_type->eklass == CACHED_CLASS_RAW(Vector3)) {
				SET_FROM_ARRAY(PoolVector3Array);
				break;
			}

			if (array_type->eklass == CACHED_CLASS_RAW(Color)) {
				SET_FROM_ARRAY(PoolColorArray);
				break;
			}

			if (array_type->eklass == CACHED_CLASS_RAW(NodePath)) {
				SET_FROM_ARRAY(Array);
				break;
			}

			if (array_type->eklass == CACHED_CLASS_RAW(RID)) {
				SET_FROM_ARRAY(Array);
				break;
			}

			GDMonoClass *array_type_class = GDMono::get_singleton()->get_class(array_type->eklass);
			if (CACHED_CLASS(GodotObject)->is_assignable_from(array_type_class)) {
				MonoArray *managed = GDMonoMarshal::Array_to_mono_array(p_value.operator ::Array(), array_type_class);
				mono_field_set_value(p_object, mono_field, managed);
				break;
			}

			ERR_FAIL_MSG("Attempted to convert Variant to a managed array of unmarshallable element type.");
		} break;

		case MONO_TYPE_CLASS: {
			GDMonoClass *type_class = type.type_class;

			// GodotObject
			if (CACHED_CLASS(GodotObject)->is_assignable_from(type_class)) {
				MonoObject *managed = GDMonoUtils::unmanaged_get_managed(p_value.operator Object *());
				mono_field_set_value(p_object, mono_field, managed);
				break;
			}

			if (CACHED_CLASS(NodePath) == type_class) {
				MonoObject *managed = GDMonoUtils::create_managed_from(p_value.operator NodePath());
				mono_field_set_value(p_object, mono_field, managed);
				break;
			}

			if (CACHED_CLASS(RID) == type_class) {
				MonoObject *managed = GDMonoUtils::create_managed_from(p_value.operator RID());
				mono_field_set_value(p_object, mono_field, managed);
				break;
			}

			// Godot.Collections.Dictionary or IDictionary
			if (CACHED_CLASS(Dictionary) == type_class || type_class == CACHED_CLASS(System_Collections_IDictionary)) {
				MonoObject *managed = GDMonoUtils::create_managed_from(p_value.operator Dictionary(), CACHED_CLASS(Dictionary));
				mono_field_set_value(p_object, mono_field, managed);
				break;
			}

			// Godot.Collections.Array or ICollection or IEnumerable
			if (CACHED_CLASS(Array) == type_class ||
					type_class == CACHED_CLASS(System_Collections_ICollection) ||
					type_class == CACHED_CLASS(System_Collections_IEnumerable)) {
				MonoObject *managed = GDMonoUtils::create_managed_from(p_value.operator Array(), CACHED_CLASS(Array));
				mono_field_set_value(p_object, mono_field, managed);
				break;
			}

			ERR_FAIL_MSG("Attempted to set the value of a field of unmarshallable type: '" + type_class->get_name() + "'.");
		} break;

		case MONO_TYPE_OBJECT: {
			// Variant
			switch (p_value.get_type()) {
				case Variant::BOOL: {
					MonoBoolean val = p_value.operator bool();
					mono_field_set_value(p_object, mono_field, &val);
				} break;
				case Variant::INT: {
					int32_t val = p_value.operator signed int();
					mono_field_set_value(p_object, mono_field, &val);
				} break;
				case Variant::REAL: {
#ifdef REAL_T_IS_DOUBLE
					double val = p_value.operator double();
					mono_field_set_value(p_object, mono_field, &val);
#else
					float val = p_value.operator float();
					mono_field_set_value(p_object, mono_field, &val);
#endif
				} break;
				case Variant::STRING: {
					MonoString *mono_string = GDMonoMarshal::mono_string_from_godot(p_value);
					mono_field_set_value(p_object, mono_field, mono_string);
				} break;
				case Variant::VECTOR2: {
					SET_FROM_STRUCT(Vector2);
				} break;
				case Variant::RECT2: {
					SET_FROM_STRUCT(Rect2);
				} break;
				case Variant::VECTOR3: {
					SET_FROM_STRUCT(Vector3);
				} break;
				case Variant::TRANSFORM2D: {
					SET_FROM_STRUCT(Transform2D);
				} break;
				case Variant::PLANE: {
					SET_FROM_STRUCT(Plane);
				} break;
				case Variant::QUAT: {
					SET_FROM_STRUCT(Quat);
				} break;
				case Variant::AABB: {
					SET_FROM_STRUCT(AABB);
				} break;
				case Variant::BASIS: {
					SET_FROM_STRUCT(Basis);
				} break;
				case Variant::TRANSFORM: {
					SET_FROM_STRUCT(Transform);
				} break;
				case Variant::COLOR: {
					SET_FROM_STRUCT(Color);
				} break;
				case Variant::NODE_PATH: {
					MonoObject *managed = GDMonoUtils::create_managed_from(p_value.operator NodePath());
					mono_field_set_value(p_object, mono_field, managed);
				} break;
				case Variant::_RID: {
					MonoObject *managed = GDMonoUtils::create_managed_from(p_value.operator RID());
					mono_field_set_value(p_object, mono_field, managed);
				} break;
				case Variant::OBJECT: {
					MonoObject *managed = GDMonoUtils::unmanaged_get_managed(p_value.operator Object *());
					mono_field_set_value(p_object, mono_field, managed);
					break;
				}
				case Variant::DICTIONARY: {
					MonoObject *managed = GDMonoUtils::create_managed_from(p_value.operator Dictionary(), CACHED_CLASS(Dictionary));
					mono_field_set_value(p_object, mono_field, managed);
				} break;
				case Variant::ARRAY: {
					MonoObject *managed = GDMonoUtils::create_managed_from(p_value.operator Array(), CACHED_CLASS(Array));
					mono_field_set_value(p_object, mono_field, managed);
				} break;
				case Variant::POOL_BYTE_ARRAY: {
					SET_FROM_ARRAY(PoolByteArray);
				} break;
				case Variant::POOL_INT_ARRAY: {
					SET_FROM_ARRAY(PoolIntArray);
				} break;
				case Variant::POOL_REAL_ARRAY: {
					SET_FROM_ARRAY(PoolRealArray);
				} break;
				case Variant::POOL_STRING_ARRAY: {
					SET_FROM_ARRAY(PoolStringArray);
				} break;
				case Variant::POOL_VECTOR2_ARRAY: {
					SET_FROM_ARRAY(PoolVector2Array);
				} break;
				case Variant::POOL_VECTOR3_ARRAY: {
					SET_FROM_ARRAY(PoolVector3Array);
				} break;
				case Variant::POOL_COLOR_ARRAY: {
					SET_FROM_ARRAY(PoolColorArray);
				} break;
				default:
					break;
			}
		} break;

		case MONO_TYPE_GENERICINST: {
			MonoReflectionType *reftype = mono_type_get_object(mono_domain_get(), type.type_class->get_mono_type());

			// Godot.Collections.Dictionary<TKey, TValue>
			if (GDMonoUtils::Marshal::type_is_generic_dictionary(reftype)) {
				MonoObject *managed = GDMonoUtils::create_managed_from(p_value.operator Dictionary(), type.type_class);
				mono_field_set_value(p_object, mono_field, managed);
				break;
			}

			// Godot.Collections.Array<T>
			if (GDMonoUtils::Marshal::type_is_generic_array(reftype)) {
				MonoObject *managed = GDMonoUtils::create_managed_from(p_value.operator Array(), type.type_class);
				mono_field_set_value(p_object, mono_field, managed);
				break;
			}

			// System.Collections.Generic.Dictionary<TKey, TValue>
			if (GDMonoUtils::Marshal::type_is_system_generic_dictionary(reftype)) {
				MonoReflectionType *key_reftype = nullptr;
				MonoReflectionType *value_reftype = nullptr;
				GDMonoUtils::Marshal::dictionary_get_key_value_types(reftype, &key_reftype, &value_reftype);
				MonoObject *managed = GDMonoMarshal::Dictionary_to_system_generic_dict(p_value.operator Dictionary(),
						type.type_class, key_reftype, value_reftype);
				mono_field_set_value(p_object, mono_field, managed);
				break;
			}

			// System.Collections.Generic.List<T>
			if (GDMonoUtils::Marshal::type_is_system_generic_list(reftype)) {
				MonoReflectionType *elem_reftype = nullptr;
				GDMonoUtils::Marshal::array_get_element_type(reftype, &elem_reftype);
				MonoObject *managed = GDMonoMarshal::Array_to_system_generic_list(p_value.operator Array(),
						type.type_class, elem_reftype);
				mono_field_set_value(p_object, mono_field, managed);
				break;
			}

			// IDictionary<TKey, TValue>
			if (GDMonoUtils::Marshal::type_is_generic_idictionary(reftype)) {
				MonoReflectionType *key_reftype;
				MonoReflectionType *value_reftype;
				GDMonoUtils::Marshal::dictionary_get_key_value_types(reftype, &key_reftype, &value_reftype);
				GDMonoClass *godot_dict_class = GDMonoUtils::Marshal::make_generic_dictionary_type(key_reftype, value_reftype);

				MonoObject *managed = GDMonoUtils::create_managed_from(p_value.operator Dictionary(), godot_dict_class);
				mono_field_set_value(p_object, mono_field, managed);
				break;
			}

			// ICollection<T> or IEnumerable<T>
			if (GDMonoUtils::Marshal::type_is_generic_icollection(reftype) || GDMonoUtils::Marshal::type_is_generic_ienumerable(reftype)) {
				MonoReflectionType *elem_reftype;
				GDMonoUtils::Marshal::array_get_element_type(reftype, &elem_reftype);
				GDMonoClass *godot_array_class = GDMonoUtils::Marshal::make_generic_array_type(elem_reftype);

				MonoObject *managed = GDMonoUtils::create_managed_from(p_value.operator Array(), godot_array_class);
				mono_field_set_value(p_object, mono_field, managed);
				break;
			}

			// GodotObject
			GDMonoClass *type_class = type.type_class;
			if (CACHED_CLASS(GodotObject)->is_assignable_from(type_class)) {
				MonoObject *managed = GDMonoUtils::unmanaged_get_managed(p_value.operator Object *());
				mono_field_set_value(p_object, mono_field, managed);
			}
		} break;

		default: {
			ERR_PRINT("Attempted to set the value of a field of unexpected type encoding: " + itos(type.type_encoding) + ".");
		} break;
	}

#undef SET_FROM_ARRAY_AND_BREAK
#undef SET_FROM_STRUCT_AND_BREAK
}

MonoObject *GDMonoField::get_value(MonoObject *p_object) {
	return mono_field_get_value_object(mono_domain_get(), mono_field, p_object);
}

bool GDMonoField::get_bool_value(MonoObject *p_object) {
	return (bool)GDMonoMarshal::unbox<MonoBoolean>(get_value(p_object));
}

int GDMonoField::get_int_value(MonoObject *p_object) {
	return GDMonoMarshal::unbox<int32_t>(get_value(p_object));
}

String GDMonoField::get_string_value(MonoObject *p_object) {
	MonoObject *val = get_value(p_object);
	return GDMonoMarshal::mono_string_to_godot((MonoString *)val);
}

bool GDMonoField::has_attribute(GDMonoClass *p_attr_class) {
	ERR_FAIL_NULL_V(p_attr_class, false);

	if (!attrs_fetched)
		fetch_attributes();

	if (!attributes)
		return false;

	return mono_custom_attrs_has_attr(attributes, p_attr_class->get_mono_ptr());
}

MonoObject *GDMonoField::get_attribute(GDMonoClass *p_attr_class) {
	ERR_FAIL_NULL_V(p_attr_class, NULL);

	if (!attrs_fetched)
		fetch_attributes();

	if (!attributes)
		return NULL;

	return mono_custom_attrs_get_attr(attributes, p_attr_class->get_mono_ptr());
}

void GDMonoField::fetch_attributes() {
	ERR_FAIL_COND(attributes != NULL);
	attributes = mono_custom_attrs_from_field(owner->get_mono_ptr(), mono_field);
	attrs_fetched = true;
}

bool GDMonoField::is_static() {
	return mono_field_get_flags(mono_field) & MONO_FIELD_ATTR_STATIC;
}

IMonoClassMember::Visibility GDMonoField::get_visibility() {
	switch (mono_field_get_flags(mono_field) & MONO_FIELD_ATTR_FIELD_ACCESS_MASK) {
		case MONO_FIELD_ATTR_PRIVATE:
			return IMonoClassMember::PRIVATE;
		case MONO_FIELD_ATTR_FAM_AND_ASSEM:
			return IMonoClassMember::PROTECTED_AND_INTERNAL;
		case MONO_FIELD_ATTR_ASSEMBLY:
			return IMonoClassMember::INTERNAL;
		case MONO_FIELD_ATTR_FAMILY:
			return IMonoClassMember::PROTECTED;
		case MONO_FIELD_ATTR_PUBLIC:
			return IMonoClassMember::PUBLIC;
		default:
			ERR_FAIL_V(IMonoClassMember::PRIVATE);
	}
}

GDMonoField::GDMonoField(MonoClassField *p_mono_field, GDMonoClass *p_owner) {
	owner = p_owner;
	mono_field = p_mono_field;
	name = String::utf8(mono_field_get_name(mono_field));
	MonoType *field_type = mono_field_get_type(mono_field);
	type.type_encoding = mono_type_get_type(field_type);
	MonoClass *field_type_class = mono_class_from_mono_type(field_type);
	type.type_class = GDMono::get_singleton()->get_class(field_type_class);

	attrs_fetched = false;
	attributes = NULL;
}

GDMonoField::~GDMonoField() {
	if (attributes) {
		mono_custom_attrs_free(attributes);
	}
}
