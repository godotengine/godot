/**************************************************************************/
/*  guest_variant.cpp                                                     */
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

#include "guest_datatypes.h"

#include "core/variant/variant.h"
#include "core/variant/variant_utility.h"
#include "sandbox.h"
#include <libriscv/util/crc32.hpp>

namespace riscv {
extern Object *get_object_from_address(const Sandbox &emu, uint64_t addr);
} //namespace riscv

Variant GuestVariant::toVariant(const Sandbox &emu) const {
	switch (type) {
		case Variant::NIL:
			return Variant();
		case Variant::BOOL:
			return v.b;
		case Variant::INT:
			return v.i;
		case Variant::FLOAT:
			return v.f;

		case Variant::VECTOR2:
			return Variant{ Vector2(v.v2f[0], v.v2f[1]) };
		case Variant::VECTOR2I:
			return Variant{ Vector2i(v.v2i[0], v.v2i[1]) };
		case Variant::RECT2:
			return Variant{ Rect2(v.v4f[0], v.v4f[1], v.v4f[2], v.v4f[3]) };
		case Variant::RECT2I:
			return Variant{ Rect2i(v.v4i[0], v.v4i[1], v.v4i[2], v.v4i[3]) };
		case Variant::VECTOR3:
			return Variant{ Vector3(v.v3f[0], v.v3f[1], v.v3f[2]) };
		case Variant::VECTOR3I:
			return Variant{ Vector3i(v.v3i[0], v.v3i[1], v.v3i[2]) };
		case Variant::VECTOR4:
			return Variant{ Vector4(v.v4f[0], v.v4f[1], v.v4f[2], v.v4f[3]) };
		case Variant::VECTOR4I:
			return Variant{ Vector4i(v.v4i[0], v.v4i[1], v.v4i[2], v.v4i[3]) };
		case Variant::COLOR:
			return Variant{ Color(v.v4f[0], v.v4f[1], v.v4f[2], v.v4f[3]) };
		case Variant::PLANE:
			return Variant{ Plane(Vector3(v.v4f[0], v.v4f[1], v.v4f[2]), v.v4f[3]) };

		case Variant::OBJECT: {
			Object *obj = riscv::get_object_from_address(emu, v.i);
			return Variant{ obj };
		}

		default:
			if (std::optional<const Variant *> var_opt = emu.get_scoped_variant(this->v.i)) {
				const Variant *var = *var_opt;
				return *var;
			} else {
				char buffer[128];
				snprintf(buffer, sizeof(buffer), "GuestVariant::toVariant(): %u (%s) is not known/scoped",
						type, GuestVariant::type_name(type));
				throw std::runtime_error(buffer);
			}
	}
}

const Variant *GuestVariant::toVariantPtr(const Sandbox &emu) const {
	if (std::optional<const Variant *> var_opt = emu.get_scoped_variant(this->v.i))
		return var_opt.value();

	char buffer[128];
	snprintf(buffer, sizeof(buffer), "GuestVariant::toVariantPtr(): %u (%s) is not known/scoped",
			type, GuestVariant::type_name(type));
	throw std::runtime_error(buffer);
}

void GuestVariant::set_object(Sandbox &emu, Object *obj) {
	emu.add_scoped_object(obj);
	this->type = Variant::OBJECT;
	this->v.i = (uintptr_t)obj;
}

void GuestVariant::set(Sandbox &emu, const Variant &value, bool implicit_trust) {
	this->type = value.get_type();

	switch (this->type) {
		case Variant::NIL:
			break;
		case Variant::BOOL:
			this->v.b = value;
			break;
		case Variant::INT:
			this->v.i = value;
			break;
		case Variant::FLOAT:
			this->v.f = value;
			break;

		case Variant::VECTOR2: {
			Vector2 vec = value.operator Vector2();
			this->v.v2f[0] = vec.x;
			this->v.v2f[1] = vec.y;
			break;
		}
		case Variant::VECTOR2I: {
			Vector2i vec = value.operator Vector2i();
			this->v.v2i[0] = vec.x;
			this->v.v2i[1] = vec.y;
			break;
		}
		case Variant::RECT2: {
			Rect2 rect = value.operator Rect2();
			this->v.v4f[0] = rect.position[0];
			this->v.v4f[1] = rect.position[1];
			this->v.v4f[2] = rect.size[0];
			this->v.v4f[3] = rect.size[1];
			break;
		}
		case Variant::RECT2I: {
			Rect2i rect = value.operator Rect2i();
			this->v.v4i[0] = rect.position[0];
			this->v.v4i[1] = rect.position[1];
			this->v.v4i[2] = rect.size[0];
			this->v.v4i[3] = rect.size[1];
			break;
		}
		case Variant::VECTOR3: {
			Vector3 vec = value.operator Vector3();
			this->v.v3f[0] = vec.x;
			this->v.v3f[1] = vec.y;
			this->v.v3f[2] = vec.z;
			break;
		}
		case Variant::VECTOR3I: {
			Vector3i vec = value.operator Vector3i();
			this->v.v3i[0] = vec.x;
			this->v.v3i[1] = vec.y;
			this->v.v3i[2] = vec.z;
			break;
		}
		case Variant::VECTOR4: {
			Vector4 vec = value.operator Vector4();
			this->v.v4f[0] = vec.x;
			this->v.v4f[1] = vec.y;
			this->v.v4f[2] = vec.z;
			this->v.v4f[3] = vec.w;
			break;
		}
		case Variant::VECTOR4I: {
			Vector4i vec = value.operator Vector4i();
			this->v.v4i[0] = vec.x;
			this->v.v4i[1] = vec.y;
			this->v.v4i[2] = vec.z;
			this->v.v4i[3] = vec.w;
			break;
		}
		case Variant::COLOR: {
			Color color = value.operator Color();
			this->v.v4f[0] = color.r;
			this->v.v4f[1] = color.g;
			this->v.v4f[2] = color.b;
			this->v.v4f[3] = color.a;
			break;
		}
		case Variant::PLANE: {
			Plane plane = value.operator Plane();
			this->v.v4f[0] = plane.normal.x;
			this->v.v4f[1] = plane.normal.y;
			this->v.v4f[2] = plane.normal.z;
			this->v.v4f[3] = plane.d;
			break;
		}

		case Variant::OBJECT: { // Objects are represented as uintptr_t
			if (!implicit_trust) {
				throw std::runtime_error("GuestVariant::set(): Cannot set OBJECT type without implicit trust");
			}
			// TODO: Check if the object is already scoped?
			Object *obj = value.operator Object *();
			if (!emu.is_allowed_object(obj)) {
				throw std::runtime_error("GuestVariant::set(): Object is not allowed");
			}
			emu.add_scoped_object(obj);
			this->v.i = (uintptr_t)obj;
			break;
		}

		default: {
			if (!implicit_trust)
				throw std::runtime_error("GuestVariant::set(): Cannot set complex type without implicit trust");
			this->v.i = emu.add_scoped_variant(&value);
		}
	}
}

void GuestVariant::create(Sandbox &emu, Variant &&value) {
	this->type = value.get_type();

	switch (this->type) {
		case Variant::NIL:
		case Variant::BOOL:
		case Variant::INT:
		case Variant::FLOAT:

		case Variant::VECTOR2:
		case Variant::VECTOR2I:
		case Variant::RECT2:
		case Variant::RECT2I:
		case Variant::VECTOR3:
		case Variant::VECTOR3I:
		case Variant::VECTOR4:
		case Variant::VECTOR4I:
		case Variant::COLOR:
		case Variant::PLANE:
			this->set(emu, value, true); // Trust the value
			break;

		case Variant::OBJECT: {
			Object *obj = value.operator Object *();
			if (!emu.is_allowed_object(obj))
				throw std::runtime_error("GuestVariant::create(): Object is not allowed");
			emu.add_scoped_object(obj);
			this->v.i = (uintptr_t)obj;
			break;
		}

		default: {
			// Store the variant in the current state
			unsigned int idx = emu.create_scoped_variant(std::move(value));
			this->v.i = idx;
		}
	}
}

void GuestVariant::free(Sandbox &emu) {
	throw std::runtime_error("GuestVariant::free(): Not implemented");
}

static const char *variant_type_names[] = {
	"Nil",

	"Bool", // 1
	"Int",
	"Float",
	"String",

	"Vector2", // 5
	"Vector2i",
	"Rect2",
	"Rect2i",
	"Vector3",
	"Vector3i",
	"Transform2D",
	"Vector4",
	"Vector4i",
	"Plane",
	"Quaternion",
	"AABB",
	"Basis",
	"Transform3D",
	"Projection",

	"Color", // 20
	"StringName",
	"NodePath",
	"RID",
	"Object",
	"Callable",
	"Signal",
	"Dictionary",
	"Array",

	"PackedByteArray", // 29
	"PackedInt32Array",
	"PackedInt64Array",
	"PackedFloat32Array",
	"PackedFloat64Array",
	"PackedStringArray",
	"PackedVector2Array",
	"PackedVector3Array",
	"PackedColorArray",
	"PackedVector4Array",

	"Max",
};

const char *GuestVariant::type_name(int type) {
	if (type < 0 || type >= Variant::VARIANT_MAX) {
		return "Unknown";
	}
	return variant_type_names[type];
}
