/**************************************************************************/
/*  gd_mono_field.cpp                                                     */
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
		case MONO_TYPE_VALUETYPE: {
			GDMonoClass *tclass = type.type_class;

			if (tclass == CACHED_CLASS(Vector2)) {
				GDMonoMarshal::M_Vector2 from = MARSHALLED_OUT(Vector2, p_value.operator ::Vector2());
				mono_field_set_value(p_object, mono_field, &from);
				break;
			}

			if (tclass == CACHED_CLASS(Rect2)) {
				GDMonoMarshal::M_Rect2 from = MARSHALLED_OUT(Rect2, p_value.operator ::Rect2());
				mono_field_set_value(p_object, mono_field, &from);
				break;
			}

			if (tclass == CACHED_CLASS(Transform2D)) {
				GDMonoMarshal::M_Transform2D from = MARSHALLED_OUT(Transform2D, p_value.operator ::Transform2D());
				mono_field_set_value(p_object, mono_field, &from);
				break;
			}

			if (tclass == CACHED_CLASS(Vector3)) {
				GDMonoMarshal::M_Vector3 from = MARSHALLED_OUT(Vector3, p_value.operator ::Vector3());
				mono_field_set_value(p_object, mono_field, &from);
				break;
			}

			if (tclass == CACHED_CLASS(Basis)) {
				GDMonoMarshal::M_Basis from = MARSHALLED_OUT(Basis, p_value.operator ::Basis());
				mono_field_set_value(p_object, mono_field, &from);
				break;
			}

			if (tclass == CACHED_CLASS(Quat)) {
				GDMonoMarshal::M_Quat from = MARSHALLED_OUT(Quat, p_value.operator ::Quat());
				mono_field_set_value(p_object, mono_field, &from);
				break;
			}

			if (tclass == CACHED_CLASS(Transform)) {
				GDMonoMarshal::M_Transform from = MARSHALLED_OUT(Transform, p_value.operator ::Transform());
				mono_field_set_value(p_object, mono_field, &from);
				break;
			}

			if (tclass == CACHED_CLASS(AABB)) {
				GDMonoMarshal::M_AABB from = MARSHALLED_OUT(AABB, p_value.operator ::AABB());
				mono_field_set_value(p_object, mono_field, &from);
				break;
			}

			if (tclass == CACHED_CLASS(Color)) {
				GDMonoMarshal::M_Color from = MARSHALLED_OUT(Color, p_value.operator ::Color());
				mono_field_set_value(p_object, mono_field, &from);
				break;
			}

			if (tclass == CACHED_CLASS(Plane)) {
				GDMonoMarshal::M_Plane from = MARSHALLED_OUT(Plane, p_value.operator ::Plane());
				mono_field_set_value(p_object, mono_field, &from);
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
		case MONO_TYPE_ARRAY:
		case MONO_TYPE_SZARRAY: {
			MonoArray *managed = GDMonoMarshal::variant_to_mono_array(p_value, type.type_class);
			if (likely(managed != nullptr)) {
				mono_field_set_value(p_object, mono_field, managed);
			}
		} break;
		case MONO_TYPE_CLASS: {
			MonoObject *managed = GDMonoMarshal::variant_to_mono_object_of_class(p_value, type.type_class);
			if (likely(managed != nullptr)) {
				mono_field_set_value(p_object, mono_field, managed);
			}
		} break;
		case MONO_TYPE_GENERICINST: {
			MonoObject *managed = GDMonoMarshal::variant_to_mono_object_of_genericinst(p_value, type.type_class);
			if (likely(managed != nullptr)) {
				mono_field_set_value(p_object, mono_field, managed);
			}
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
					GDMonoMarshal::M_Vector2 from = MARSHALLED_OUT(Vector2, p_value.operator ::Vector2());
					mono_field_set_value(p_object, mono_field, &from);
				} break;
				case Variant::RECT2: {
					GDMonoMarshal::M_Rect2 from = MARSHALLED_OUT(Rect2, p_value.operator ::Rect2());
					mono_field_set_value(p_object, mono_field, &from);
				} break;
				case Variant::VECTOR3: {
					GDMonoMarshal::M_Vector3 from = MARSHALLED_OUT(Vector3, p_value.operator ::Vector3());
					mono_field_set_value(p_object, mono_field, &from);
				} break;
				case Variant::TRANSFORM2D: {
					GDMonoMarshal::M_Transform2D from = MARSHALLED_OUT(Transform2D, p_value.operator ::Transform2D());
					mono_field_set_value(p_object, mono_field, &from);
				} break;
				case Variant::PLANE: {
					GDMonoMarshal::M_Plane from = MARSHALLED_OUT(Plane, p_value.operator ::Plane());
					mono_field_set_value(p_object, mono_field, &from);
				} break;
				case Variant::QUAT: {
					GDMonoMarshal::M_Quat from = MARSHALLED_OUT(Quat, p_value.operator ::Quat());
					mono_field_set_value(p_object, mono_field, &from);
				} break;
				case Variant::AABB: {
					GDMonoMarshal::M_AABB from = MARSHALLED_OUT(AABB, p_value.operator ::AABB());
					mono_field_set_value(p_object, mono_field, &from);
				} break;
				case Variant::BASIS: {
					GDMonoMarshal::M_Basis from = MARSHALLED_OUT(Basis, p_value.operator ::Basis());
					mono_field_set_value(p_object, mono_field, &from);
				} break;
				case Variant::TRANSFORM: {
					GDMonoMarshal::M_Transform from = MARSHALLED_OUT(Transform, p_value.operator ::Transform());
					mono_field_set_value(p_object, mono_field, &from);
				} break;
				case Variant::COLOR: {
					GDMonoMarshal::M_Color from = MARSHALLED_OUT(Color, p_value.operator ::Color());
					mono_field_set_value(p_object, mono_field, &from);
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
					MonoArray *managed = GDMonoMarshal::PoolByteArray_to_mono_array(p_value.operator ::PoolByteArray());
					mono_field_set_value(p_object, mono_field, managed);
				} break;
				case Variant::POOL_INT_ARRAY: {
					MonoArray *managed = GDMonoMarshal::PoolIntArray_to_mono_array(p_value.operator ::PoolIntArray());
					mono_field_set_value(p_object, mono_field, managed);
				} break;
				case Variant::POOL_REAL_ARRAY: {
					MonoArray *managed = GDMonoMarshal::PoolRealArray_to_mono_array(p_value.operator ::PoolRealArray());
					mono_field_set_value(p_object, mono_field, managed);
				} break;
				case Variant::POOL_STRING_ARRAY: {
					MonoArray *managed = GDMonoMarshal::PoolStringArray_to_mono_array(p_value.operator ::PoolStringArray());
					mono_field_set_value(p_object, mono_field, managed);
				} break;
				case Variant::POOL_VECTOR2_ARRAY: {
					MonoArray *managed = GDMonoMarshal::PoolVector2Array_to_mono_array(p_value.operator ::PoolVector2Array());
					mono_field_set_value(p_object, mono_field, managed);
				} break;
				case Variant::POOL_VECTOR3_ARRAY: {
					MonoArray *managed = GDMonoMarshal::PoolVector3Array_to_mono_array(p_value.operator ::PoolVector3Array());
					mono_field_set_value(p_object, mono_field, managed);
				} break;
				case Variant::POOL_COLOR_ARRAY: {
					MonoArray *managed = GDMonoMarshal::PoolColorArray_to_mono_array(p_value.operator ::PoolColorArray());
					mono_field_set_value(p_object, mono_field, managed);
				} break;
				default:
					break;
			}
		} break;
		default: {
			ERR_PRINT("Attempted to set the value of a field of unexpected type encoding: " + itos(type.type_encoding) + ".");
		} break;
	}
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
