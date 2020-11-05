/*************************************************************************/
/*  variant_setget.cpp                                                   */
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

#include "variant.h"

#include "core/class_db.h"
#include "core/core_string_names.h"
#include "core/debugger/engine_debugger.h"

void Variant::set_named(const StringName &p_index, const Variant &p_value, bool *r_valid) {
	bool valid = false;
	switch (type) {
		case VECTOR2: {
			if (p_value.type == Variant::INT) {
				Vector2 *v = reinterpret_cast<Vector2 *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._int;
					valid = true;
				}
			} else if (p_value.type == Variant::FLOAT) {
				Vector2 *v = reinterpret_cast<Vector2 *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._float;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._float;
					valid = true;
				}
			}

		} break;
		case VECTOR2I: {
			if (p_value.type == Variant::INT) {
				Vector2i *v = reinterpret_cast<Vector2i *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._int;
					valid = true;
				}
			} else if (p_value.type == Variant::FLOAT) {
				Vector2i *v = reinterpret_cast<Vector2i *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._float;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._float;
					valid = true;
				}
			}

		} break;
		case RECT2: {
			if (p_value.type == Variant::VECTOR2) {
				Rect2 *v = reinterpret_cast<Rect2 *>(_data._mem);
				//scalar name
				if (p_index == CoreStringNames::singleton->position) {
					v->position = *reinterpret_cast<const Vector2 *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->size) {
					v->size = *reinterpret_cast<const Vector2 *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->end) {
					v->size = *reinterpret_cast<const Vector2 *>(p_value._data._mem) - v->position;
					valid = true;
				}
			}
		} break;
		case RECT2I: {
			if (p_value.type == Variant::VECTOR2I) {
				Rect2i *v = reinterpret_cast<Rect2i *>(_data._mem);
				//scalar name
				if (p_index == CoreStringNames::singleton->position) {
					v->position = *reinterpret_cast<const Vector2i *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->size) {
					v->size = *reinterpret_cast<const Vector2i *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->end) {
					v->size = *reinterpret_cast<const Vector2i *>(p_value._data._mem) - v->position;
					valid = true;
				}
			}
		} break;
		case TRANSFORM2D: {
			if (p_value.type == Variant::VECTOR2) {
				Transform2D *v = _data._transform2d;
				if (p_index == CoreStringNames::singleton->x) {
					v->elements[0] = *reinterpret_cast<const Vector2 *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->elements[1] = *reinterpret_cast<const Vector2 *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->origin) {
					v->elements[2] = *reinterpret_cast<const Vector2 *>(p_value._data._mem);
					valid = true;
				}
			}

		} break;
		case VECTOR3: {
			if (p_value.type == Variant::INT) {
				Vector3 *v = reinterpret_cast<Vector3 *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->z = p_value._data._int;
					valid = true;
				}
			} else if (p_value.type == Variant::FLOAT) {
				Vector3 *v = reinterpret_cast<Vector3 *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._float;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._float;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->z = p_value._data._float;
					valid = true;
				}
			}

		} break;
		case VECTOR3I: {
			if (p_value.type == Variant::INT) {
				Vector3i *v = reinterpret_cast<Vector3i *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->z = p_value._data._int;
					valid = true;
				}
			} else if (p_value.type == Variant::FLOAT) {
				Vector3i *v = reinterpret_cast<Vector3i *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._float;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._float;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->z = p_value._data._float;
					valid = true;
				}
			}

		} break;
		case PLANE: {
			if (p_value.type == Variant::INT) {
				Plane *v = reinterpret_cast<Plane *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->normal.x = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->normal.y = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->normal.z = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->d) {
					v->d = p_value._data._int;
					valid = true;
				}
			} else if (p_value.type == Variant::FLOAT) {
				Plane *v = reinterpret_cast<Plane *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->normal.x = p_value._data._float;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->normal.y = p_value._data._float;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->normal.z = p_value._data._float;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->d) {
					v->d = p_value._data._float;
					valid = true;
				}

			} else if (p_value.type == Variant::VECTOR3) {
				Plane *v = reinterpret_cast<Plane *>(_data._mem);
				if (p_index == CoreStringNames::singleton->normal) {
					v->normal = *reinterpret_cast<const Vector3 *>(p_value._data._mem);
					valid = true;
				}
			}

		} break;
		case QUAT: {
			if (p_value.type == Variant::INT) {
				Quat *v = reinterpret_cast<Quat *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->z = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->w) {
					v->w = p_value._data._int;
					valid = true;
				}
			} else if (p_value.type == Variant::FLOAT) {
				Quat *v = reinterpret_cast<Quat *>(_data._mem);
				if (p_index == CoreStringNames::singleton->x) {
					v->x = p_value._data._float;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->y = p_value._data._float;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->z = p_value._data._float;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->w) {
					v->w = p_value._data._float;
					valid = true;
				}
			}

		} break;
		case AABB: {
			if (p_value.type == Variant::VECTOR3) {
				::AABB *v = _data._aabb;
				//scalar name
				if (p_index == CoreStringNames::singleton->position) {
					v->position = *reinterpret_cast<const Vector3 *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->size) {
					v->size = *reinterpret_cast<const Vector3 *>(p_value._data._mem);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->end) {
					v->size = *reinterpret_cast<const Vector3 *>(p_value._data._mem) - v->position;
					valid = true;
				}
			}
		} break;
		case BASIS: {
			if (p_value.type == Variant::VECTOR3) {
				Basis *v = _data._basis;
				//scalar name
				if (p_index == CoreStringNames::singleton->x) {
					v->set_axis(0, *reinterpret_cast<const Vector3 *>(p_value._data._mem));
					valid = true;
				} else if (p_index == CoreStringNames::singleton->y) {
					v->set_axis(1, *reinterpret_cast<const Vector3 *>(p_value._data._mem));
					valid = true;
				} else if (p_index == CoreStringNames::singleton->z) {
					v->set_axis(2, *reinterpret_cast<const Vector3 *>(p_value._data._mem));
					valid = true;
				}
			}
		} break;
		case TRANSFORM: {
			if (p_value.type == Variant::BASIS && p_index == CoreStringNames::singleton->basis) {
				_data._transform->basis = *p_value._data._basis;
				valid = true;
			} else if (p_value.type == Variant::VECTOR3 && p_index == CoreStringNames::singleton->origin) {
				_data._transform->origin = *reinterpret_cast<const Vector3 *>(p_value._data._mem);
				valid = true;
			}

		} break;
		case COLOR: {
			if (p_value.type == Variant::INT) {
				Color *v = reinterpret_cast<Color *>(_data._mem);
				if (p_index == CoreStringNames::singleton->r) {
					v->r = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->g) {
					v->g = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->b) {
					v->b = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->a) {
					v->a = p_value._data._int;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->r8) {
					v->r = p_value._data._int / 255.0;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->g8) {
					v->g = p_value._data._int / 255.0;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->b8) {
					v->b = p_value._data._int / 255.0;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->a8) {
					v->a = p_value._data._int / 255.0;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->h) {
					v->set_hsv(p_value._data._int, v->get_s(), v->get_v(), v->a);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->s) {
					v->set_hsv(v->get_h(), p_value._data._int, v->get_v(), v->a);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->v) {
					v->set_hsv(v->get_h(), v->get_v(), p_value._data._int, v->a);
					valid = true;
				}
			} else if (p_value.type == Variant::FLOAT) {
				Color *v = reinterpret_cast<Color *>(_data._mem);
				if (p_index == CoreStringNames::singleton->r) {
					v->r = p_value._data._float;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->g) {
					v->g = p_value._data._float;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->b) {
					v->b = p_value._data._float;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->a) {
					v->a = p_value._data._float;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->r8) {
					v->r = p_value._data._float / 255.0;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->g8) {
					v->g = p_value._data._float / 255.0;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->b8) {
					v->b = p_value._data._float / 255.0;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->a8) {
					v->a = p_value._data._float / 255.0;
					valid = true;
				} else if (p_index == CoreStringNames::singleton->h) {
					v->set_hsv(p_value._data._float, v->get_s(), v->get_v(), v->a);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->s) {
					v->set_hsv(v->get_h(), p_value._data._float, v->get_v(), v->a);
					valid = true;
				} else if (p_index == CoreStringNames::singleton->v) {
					v->set_hsv(v->get_h(), v->get_s(), p_value._data._float, v->a);
					valid = true;
				}
			}
		} break;
		case OBJECT: {
#ifdef DEBUG_ENABLED
			if (!_get_obj().obj) {
				break;
			} else if (EngineDebugger::is_active() && ObjectDB::get_instance(_get_obj().id) == nullptr) {
				break;
			}

#endif
			_get_obj().obj->set(p_index, p_value, &valid);

		} break;
		default: {
			set(p_index.operator String(), p_value, &valid);
		} break;
	}

	if (r_valid) {
		*r_valid = valid;
	}
}

Variant Variant::get_named(const StringName &p_index, bool *r_valid) const {
	if (r_valid) {
		*r_valid = true;
	}

	switch (type) {
		case VECTOR2: {
			const Vector2 *v = reinterpret_cast<const Vector2 *>(_data._mem);
			if (p_index == CoreStringNames::singleton->x) {
				return v->x;
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->y;
			}

		} break;
		case VECTOR2I: {
			const Vector2i *v = reinterpret_cast<const Vector2i *>(_data._mem);
			if (p_index == CoreStringNames::singleton->x) {
				return v->x;
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->y;
			}

		} break;
		case RECT2: {
			const Rect2 *v = reinterpret_cast<const Rect2 *>(_data._mem);
			//scalar name
			if (p_index == CoreStringNames::singleton->position) {
				return v->position;
			} else if (p_index == CoreStringNames::singleton->size) {
				return v->size;
			} else if (p_index == CoreStringNames::singleton->end) {
				return v->size + v->position;
			}
		} break;
		case RECT2I: {
			const Rect2i *v = reinterpret_cast<const Rect2i *>(_data._mem);
			//scalar name
			if (p_index == CoreStringNames::singleton->position) {
				return v->position;
			} else if (p_index == CoreStringNames::singleton->size) {
				return v->size;
			} else if (p_index == CoreStringNames::singleton->end) {
				return v->size + v->position;
			}
		} break;
		case TRANSFORM2D: {
			const Transform2D *v = _data._transform2d;
			if (p_index == CoreStringNames::singleton->x) {
				return v->elements[0];
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->elements[1];
			} else if (p_index == CoreStringNames::singleton->origin) {
				return v->elements[2];
			}

		} break;
		case VECTOR3: {
			const Vector3 *v = reinterpret_cast<const Vector3 *>(_data._mem);
			if (p_index == CoreStringNames::singleton->x) {
				return v->x;
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->y;
			} else if (p_index == CoreStringNames::singleton->z) {
				return v->z;
			}

		} break;
		case VECTOR3I: {
			const Vector3i *v = reinterpret_cast<const Vector3i *>(_data._mem);
			if (p_index == CoreStringNames::singleton->x) {
				return v->x;
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->y;
			} else if (p_index == CoreStringNames::singleton->z) {
				return v->z;
			}

		} break;
		case PLANE: {
			const Plane *v = reinterpret_cast<const Plane *>(_data._mem);
			if (p_index == CoreStringNames::singleton->x) {
				return v->normal.x;
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->normal.y;
			} else if (p_index == CoreStringNames::singleton->z) {
				return v->normal.z;
			} else if (p_index == CoreStringNames::singleton->d) {
				return v->d;
			} else if (p_index == CoreStringNames::singleton->normal) {
				return v->normal;
			}

		} break;
		case QUAT: {
			const Quat *v = reinterpret_cast<const Quat *>(_data._mem);
			if (p_index == CoreStringNames::singleton->x) {
				return v->x;
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->y;
			} else if (p_index == CoreStringNames::singleton->z) {
				return v->z;
			} else if (p_index == CoreStringNames::singleton->w) {
				return v->w;
			}

		} break;
		case AABB: {
			const ::AABB *v = _data._aabb;
			//scalar name
			if (p_index == CoreStringNames::singleton->position) {
				return v->position;
			} else if (p_index == CoreStringNames::singleton->size) {
				return v->size;
			} else if (p_index == CoreStringNames::singleton->end) {
				return v->size + v->position;
			}
		} break;
		case BASIS: {
			const Basis *v = _data._basis;
			//scalar name
			if (p_index == CoreStringNames::singleton->x) {
				return v->get_axis(0);
			} else if (p_index == CoreStringNames::singleton->y) {
				return v->get_axis(1);
			} else if (p_index == CoreStringNames::singleton->z) {
				return v->get_axis(2);
			}

		} break;
		case TRANSFORM: {
			if (p_index == CoreStringNames::singleton->basis) {
				return _data._transform->basis;
			} else if (p_index == CoreStringNames::singleton->origin) {
				return _data._transform->origin;
			}

		} break;
		case COLOR: {
			const Color *v = reinterpret_cast<const Color *>(_data._mem);
			if (p_index == CoreStringNames::singleton->r) {
				return v->r;
			} else if (p_index == CoreStringNames::singleton->g) {
				return v->g;
			} else if (p_index == CoreStringNames::singleton->b) {
				return v->b;
			} else if (p_index == CoreStringNames::singleton->a) {
				return v->a;
			} else if (p_index == CoreStringNames::singleton->r8) {
				return int(Math::round(v->r * 255.0));
			} else if (p_index == CoreStringNames::singleton->g8) {
				return int(Math::round(v->g * 255.0));
			} else if (p_index == CoreStringNames::singleton->b8) {
				return int(Math::round(v->b * 255.0));
			} else if (p_index == CoreStringNames::singleton->a8) {
				return int(Math::round(v->a * 255.0));
			} else if (p_index == CoreStringNames::singleton->h) {
				return v->get_h();
			} else if (p_index == CoreStringNames::singleton->s) {
				return v->get_s();
			} else if (p_index == CoreStringNames::singleton->v) {
				return v->get_v();
			}
		} break;
		case OBJECT: {
#ifdef DEBUG_ENABLED
			if (!_get_obj().obj) {
				if (r_valid) {
					*r_valid = false;
				}
				return "Instance base is null.";
			} else {
				if (EngineDebugger::is_active() && !_get_obj().id.is_reference() && ObjectDB::get_instance(_get_obj().id) == nullptr) {
					if (r_valid) {
						*r_valid = false;
					}
					return "Attempted use of stray pointer object.";
				}
			}

#endif

			return _get_obj().obj->get(p_index, r_valid);

		} break;
		default: {
			return get(p_index.operator String(), r_valid);
		}
	}

	if (r_valid) {
		*r_valid = false;
	}
	return Variant();
}

#define DEFAULT_OP_ARRAY_CMD(m_name, m_type, skip_test, cmd)                              \
	case m_name: {                                                                        \
		skip_test;                                                                        \
                                                                                          \
		if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) { \
			int index = p_index;                                                          \
			m_type *arr = reinterpret_cast<m_type *>(_data._mem);                         \
                                                                                          \
			if (index < 0)                                                                \
				index += arr->size();                                                     \
			if (index >= 0 && index < arr->size()) {                                      \
				valid = true;                                                             \
				cmd;                                                                      \
			}                                                                             \
		}                                                                                 \
	} break;

#define DEFAULT_OP_DVECTOR_SET(m_name, m_type, skip_cond)                                    \
	case m_name: {                                                                           \
		if (skip_cond)                                                                       \
			return;                                                                          \
                                                                                             \
		if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {    \
			int index = p_index;                                                             \
			Vector<m_type> *arr = PackedArrayRef<m_type>::get_array_ptr(_data.packed_array); \
                                                                                             \
			if (index < 0)                                                                   \
				index += arr->size();                                                        \
			if (index >= 0 && index < arr->size()) {                                         \
				valid = true;                                                                \
				arr->set(index, p_value);                                                    \
			}                                                                                \
		}                                                                                    \
	} break;

#define DEFAULT_OP_DVECTOR_GET(m_name, m_type)                                                  \
	case m_name: {                                                                              \
		if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {       \
			int index = p_index;                                                                \
			const Vector<m_type> *arr = &PackedArrayRef<m_type>::get_array(_data.packed_array); \
                                                                                                \
			if (index < 0)                                                                      \
				index += arr->size();                                                           \
			if (index >= 0 && index < arr->size()) {                                            \
				valid = true;                                                                   \
				return arr->get(index);                                                         \
			}                                                                                   \
		}                                                                                       \
	} break;

void Variant::set(const Variant &p_index, const Variant &p_value, bool *r_valid) {
	static bool _dummy = false;

	bool &valid = r_valid ? *r_valid : _dummy;
	valid = false;

	switch (type) {
		case NIL: {
			return;
		} break;
		case BOOL: {
			return;
		} break;
		case INT: {
			return;
		} break;
		case FLOAT: {
			return;
		} break;
		case STRING: {
			if (p_index.type != Variant::INT && p_index.type != Variant::FLOAT) {
				return;
			}

			int idx = p_index;
			String *str = reinterpret_cast<String *>(_data._mem);
			int len = str->length();
			if (idx < 0) {
				idx += len;
			}
			if (idx < 0 || idx >= len) {
				return;
			}

			String chr;
			if (p_value.type == Variant::INT || p_value.type == Variant::FLOAT) {
				chr = String::chr(p_value);
			} else if (p_value.type == Variant::STRING) {
				chr = p_value;
			} else {
				return;
			}

			*str = str->substr(0, idx) + chr + str->substr(idx + 1, len);
			valid = true;
			return;

		} break;
		case VECTOR2: {
			if (p_value.type != Variant::INT && p_value.type != Variant::FLOAT) {
				return;
			}

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				// scalar index
				int idx = p_index;

				if (idx < 0) {
					idx += 2;
				}
				if (idx >= 0 && idx < 2) {
					Vector2 *v = reinterpret_cast<Vector2 *>(_data._mem);
					valid = true;
					(*v)[idx] = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Vector2 *v = reinterpret_cast<Vector2 *>(_data._mem);
				if (*str == "x") {
					valid = true;
					v->x = p_value;
					return;
				} else if (*str == "y") {
					valid = true;
					v->y = p_value;
					return;
				}
			}

		} break;
		case VECTOR2I: {
			if (p_value.type != Variant::INT && p_value.type != Variant::FLOAT) {
				return;
			}

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				// scalar index
				int idx = p_index;

				if (idx < 0) {
					idx += 2;
				}
				if (idx >= 0 && idx < 2) {
					Vector2i *v = reinterpret_cast<Vector2i *>(_data._mem);
					valid = true;
					(*v)[idx] = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Vector2i *v = reinterpret_cast<Vector2i *>(_data._mem);
				if (*str == "x") {
					valid = true;
					v->x = p_value;
					return;
				} else if (*str == "y") {
					valid = true;
					v->y = p_value;
					return;
				}
			}

		} break;
		case RECT2: {
			if (p_value.type != Variant::VECTOR2) {
				return;
			}

			if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Rect2 *v = reinterpret_cast<Rect2 *>(_data._mem);
				if (*str == "position") {
					valid = true;
					v->position = p_value;
					return;
				} else if (*str == "size") {
					valid = true;
					v->size = p_value;
					return;
				} else if (*str == "end") {
					valid = true;
					v->size = Vector2(p_value) - v->position;
					return;
				}
			}
		} break;
		case RECT2I: {
			if (p_value.type != Variant::VECTOR2I) {
				return;
			}

			if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Rect2i *v = reinterpret_cast<Rect2i *>(_data._mem);
				if (*str == "position") {
					valid = true;
					v->position = p_value;
					return;
				} else if (*str == "size") {
					valid = true;
					v->size = p_value;
					return;
				} else if (*str == "end") {
					valid = true;
					v->size = Vector2i(p_value) - v->position;
					return;
				}
			}
		} break;
		case TRANSFORM2D: {
			if (p_value.type != Variant::VECTOR2) {
				return;
			}

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				int index = p_index;

				if (index < 0) {
					index += 3;
				}
				if (index >= 0 && index < 3) {
					Transform2D *v = _data._transform2d;

					valid = true;
					v->elements[index] = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING && p_value.get_type() == Variant::VECTOR2) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Transform2D *v = _data._transform2d;
				if (*str == "x") {
					valid = true;
					v->elements[0] = p_value;
					return;
				} else if (*str == "y") {
					valid = true;
					v->elements[1] = p_value;
					return;
				} else if (*str == "origin") {
					valid = true;
					v->elements[2] = p_value;
					return;
				}
			}

		} break;
		case VECTOR3: {
			if (p_value.type != Variant::INT && p_value.type != Variant::FLOAT) {
				return;
			}

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				//scalar index
				int idx = p_index;
				if (idx < 0) {
					idx += 3;
				}
				if (idx >= 0 && idx < 3) {
					Vector3 *v = reinterpret_cast<Vector3 *>(_data._mem);
					valid = true;
					(*v)[idx] = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Vector3 *v = reinterpret_cast<Vector3 *>(_data._mem);
				if (*str == "x") {
					valid = true;
					v->x = p_value;
					return;
				} else if (*str == "y") {
					valid = true;
					v->y = p_value;
					return;
				} else if (*str == "z") {
					valid = true;
					v->z = p_value;
					return;
				}
			}

		} break;
		case VECTOR3I: {
			if (p_value.type != Variant::INT && p_value.type != Variant::FLOAT) {
				return;
			}

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				//scalar index
				int idx = p_index;
				if (idx < 0) {
					idx += 3;
				}
				if (idx >= 0 && idx < 3) {
					Vector3i *v = reinterpret_cast<Vector3i *>(_data._mem);
					valid = true;
					(*v)[idx] = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Vector3i *v = reinterpret_cast<Vector3i *>(_data._mem);
				if (*str == "x") {
					valid = true;
					v->x = p_value;
					return;
				} else if (*str == "y") {
					valid = true;
					v->y = p_value;
					return;
				} else if (*str == "z") {
					valid = true;
					v->z = p_value;
					return;
				}
			}

		} break;
		case PLANE: {
			if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Plane *v = reinterpret_cast<Plane *>(_data._mem);
				if (*str == "x") {
					if (p_value.type != Variant::INT && p_value.type != Variant::FLOAT) {
						return;
					}

					valid = true;
					v->normal.x = p_value;
					return;
				} else if (*str == "y") {
					if (p_value.type != Variant::INT && p_value.type != Variant::FLOAT) {
						return;
					}

					valid = true;
					v->normal.y = p_value;
					return;
				} else if (*str == "z") {
					if (p_value.type != Variant::INT && p_value.type != Variant::FLOAT) {
						return;
					}

					valid = true;
					v->normal.z = p_value;
					return;
				} else if (*str == "normal") {
					if (p_value.type != Variant::VECTOR3) {
						return;
					}

					valid = true;
					v->normal = p_value;
					return;
				} else if (*str == "d") {
					valid = true;
					v->d = p_value;
					return;
				}
			}

		} break;
		case QUAT: {
			if (p_value.type != Variant::INT && p_value.type != Variant::FLOAT) {
				return;
			}

			if (p_index.get_type() == Variant::STRING) {
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Quat *v = reinterpret_cast<Quat *>(_data._mem);
				if (*str == "x") {
					valid = true;
					v->x = p_value;
					return;
				} else if (*str == "y") {
					valid = true;
					v->y = p_value;
					return;
				} else if (*str == "z") {
					valid = true;
					v->z = p_value;
					return;
				} else if (*str == "w") {
					valid = true;
					v->w = p_value;
					return;
				}
			}

		} break;
		case AABB: {
			if (p_value.type != Variant::VECTOR3) {
				return;
			}

			if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				::AABB *v = _data._aabb;
				if (*str == "position") {
					valid = true;
					v->position = p_value;
					return;
				} else if (*str == "size") {
					valid = true;
					v->size = p_value;
					return;
				} else if (*str == "end") {
					valid = true;
					v->size = Vector3(p_value) - v->position;
					return;
				}
			}
		} break;
		case BASIS: {
			if (p_value.type != Variant::VECTOR3) {
				return;
			}

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				int index = p_index;

				if (index < 0) {
					index += 3;
				}
				if (index >= 0 && index < 3) {
					Basis *v = _data._basis;

					valid = true;
					v->set_axis(index, p_value);
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Basis *v = _data._basis;

				if (*str == "x") {
					valid = true;
					v->set_axis(0, p_value);
					return;
				} else if (*str == "y") {
					valid = true;
					v->set_axis(1, p_value);
					return;
				} else if (*str == "z") {
					valid = true;
					v->set_axis(2, p_value);
					return;
				}
			}

		} break;
		case TRANSFORM: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				if (p_value.type != Variant::VECTOR3) {
					return;
				}

				int index = p_index;

				if (index < 0) {
					index += 4;
				}
				if (index >= 0 && index < 4) {
					Transform *v = _data._transform;
					valid = true;
					if (index == 3) {
						v->origin = p_value;
					} else {
						v->basis.set_axis(index, p_value);
					}
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {
				Transform *v = _data._transform;
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);

				if (*str == "basis") {
					if (p_value.type != Variant::BASIS) {
						return;
					}
					valid = true;
					v->basis = p_value;
					return;
				}
				if (*str == "origin") {
					if (p_value.type != Variant::VECTOR3) {
						return;
					}
					valid = true;
					v->origin = p_value;
					return;
				}
			}

		} break;
		case COLOR: {
			if (p_value.type != Variant::INT && p_value.type != Variant::FLOAT) {
				return;
			}

			if (p_index.get_type() == Variant::STRING) {
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Color *v = reinterpret_cast<Color *>(_data._mem);
				if (*str == "r") {
					valid = true;
					v->r = p_value;
					return;
				} else if (*str == "g") {
					valid = true;
					v->g = p_value;
					return;
				} else if (*str == "b") {
					valid = true;
					v->b = p_value;
					return;
				} else if (*str == "a") {
					valid = true;
					v->a = p_value;
					return;
				} else if (*str == "h") {
					valid = true;
					v->set_hsv(p_value, v->get_s(), v->get_v(), v->a);
					return;
				} else if (*str == "s") {
					valid = true;
					v->set_hsv(v->get_h(), p_value, v->get_v(), v->a);
					return;
				} else if (*str == "v") {
					valid = true;
					v->set_hsv(v->get_h(), v->get_s(), p_value, v->a);
					return;
				} else if (*str == "r8") {
					valid = true;
					v->r = float(p_value) / 255.0;
					return;
				} else if (*str == "g8") {
					valid = true;
					v->g = float(p_value) / 255.0;
					return;
				} else if (*str == "b8") {
					valid = true;
					v->b = float(p_value) / 255.0;
					return;
				} else if (*str == "a8") {
					valid = true;
					v->a = float(p_value) / 255.0;
					return;
				}
			} else if (p_index.get_type() == Variant::INT) {
				int idx = p_index;
				if (idx < 0) {
					idx += 4;
				}
				if (idx >= 0 && idx < 4) {
					Color *v = reinterpret_cast<Color *>(_data._mem);
					(*v)[idx] = p_value;
					valid = true;
				}
			}

		} break;
		case STRING_NAME: {
		} break;
		case NODE_PATH: {
		} break;
		case _RID: {
		} break;
		case OBJECT: {
			Object *obj = _get_obj().obj;
			//only if debugging!

			if (obj) {
#ifdef DEBUG_ENABLED
				if (EngineDebugger::is_active() && !_get_obj().id.is_reference() && ObjectDB::get_instance(_get_obj().id) == nullptr) {
					WARN_PRINT("Attempted use of previously freed pointer object.");
					valid = false;
					return;
				}
#endif

				if (p_index.get_type() != Variant::STRING_NAME && p_index.get_type() != Variant::STRING) {
					obj->setvar(p_index, p_value, r_valid);
					return;
				}

				obj->set(p_index, p_value, r_valid);
				return;
			}
		} break;
		case DICTIONARY: {
			Dictionary *dic = reinterpret_cast<Dictionary *>(_data._mem);
			dic->operator[](p_index) = p_value;
			valid = true; //always valid, i guess? should this really be ok?
			return;
		} break;
			DEFAULT_OP_ARRAY_CMD(ARRAY, Array, ;, (*arr)[index] = p_value; return )
			DEFAULT_OP_DVECTOR_SET(PACKED_BYTE_ARRAY, uint8_t, p_value.type != Variant::FLOAT && p_value.type != Variant::INT)
			DEFAULT_OP_DVECTOR_SET(PACKED_INT32_ARRAY, int32_t, p_value.type != Variant::FLOAT && p_value.type != Variant::INT)
			DEFAULT_OP_DVECTOR_SET(PACKED_INT64_ARRAY, int64_t, p_value.type != Variant::FLOAT && p_value.type != Variant::INT)
			DEFAULT_OP_DVECTOR_SET(PACKED_FLOAT32_ARRAY, float, p_value.type != Variant::FLOAT && p_value.type != Variant::INT)
			DEFAULT_OP_DVECTOR_SET(PACKED_FLOAT64_ARRAY, double, p_value.type != Variant::FLOAT && p_value.type != Variant::INT)
			DEFAULT_OP_DVECTOR_SET(PACKED_STRING_ARRAY, String, p_value.type != Variant::STRING)
			DEFAULT_OP_DVECTOR_SET(PACKED_VECTOR2_ARRAY, Vector2, p_value.type != Variant::VECTOR2)
			DEFAULT_OP_DVECTOR_SET(PACKED_VECTOR3_ARRAY, Vector3, p_value.type != Variant::VECTOR3)
			DEFAULT_OP_DVECTOR_SET(PACKED_COLOR_ARRAY, Color, p_value.type != Variant::COLOR)
		default:
			return;
	}
}

Variant Variant::get(const Variant &p_index, bool *r_valid) const {
	static bool _dummy = false;

	bool &valid = r_valid ? *r_valid : _dummy;

	valid = false;

	switch (type) {
		case NIL: {
			return Variant();
		} break;
		case BOOL: {
			return Variant();
		} break;
		case INT: {
			return Variant();
		} break;
		case FLOAT: {
			return Variant();
		} break;
		case STRING: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				//string index

				int idx = p_index;
				const String *str = reinterpret_cast<const String *>(_data._mem);
				if (idx < 0) {
					idx += str->length();
				}
				if (idx >= 0 && idx < str->length()) {
					valid = true;
					return str->substr(idx, 1);
				}
			}

		} break;
		case VECTOR2: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				// scalar index
				int idx = p_index;
				if (idx < 0) {
					idx += 2;
				}
				if (idx >= 0 && idx < 2) {
					const Vector2 *v = reinterpret_cast<const Vector2 *>(_data._mem);
					valid = true;
					return (*v)[idx];
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Vector2 *v = reinterpret_cast<const Vector2 *>(_data._mem);
				if (*str == "x") {
					valid = true;
					return v->x;
				} else if (*str == "y") {
					valid = true;
					return v->y;
				}
			}

		} break;
		case VECTOR2I: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				// scalar index
				int idx = p_index;
				if (idx < 0) {
					idx += 2;
				}
				if (idx >= 0 && idx < 2) {
					const Vector2i *v = reinterpret_cast<const Vector2i *>(_data._mem);
					valid = true;
					return (*v)[idx];
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Vector2i *v = reinterpret_cast<const Vector2i *>(_data._mem);
				if (*str == "x") {
					valid = true;
					return v->x;
				} else if (*str == "y") {
					valid = true;
					return v->y;
				}
			}

		} break;
		case RECT2: {
			if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Rect2 *v = reinterpret_cast<const Rect2 *>(_data._mem);
				if (*str == "position") {
					valid = true;
					return v->position;
				} else if (*str == "size") {
					valid = true;
					return v->size;
				} else if (*str == "end") {
					valid = true;
					return v->size + v->position;
				}
			}
		} break;
		case RECT2I: {
			if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Rect2i *v = reinterpret_cast<const Rect2i *>(_data._mem);
				if (*str == "position") {
					valid = true;
					return v->position;
				} else if (*str == "size") {
					valid = true;
					return v->size;
				} else if (*str == "end") {
					valid = true;
					return v->size + v->position;
				}
			}
		} break;
		case VECTOR3: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				//scalar index
				int idx = p_index;
				if (idx < 0) {
					idx += 3;
				}
				if (idx >= 0 && idx < 3) {
					const Vector3 *v = reinterpret_cast<const Vector3 *>(_data._mem);
					valid = true;
					return (*v)[idx];
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Vector3 *v = reinterpret_cast<const Vector3 *>(_data._mem);
				if (*str == "x") {
					valid = true;
					return v->x;
				} else if (*str == "y") {
					valid = true;
					return v->y;
				} else if (*str == "z") {
					valid = true;
					return v->z;
				}
			}

		} break;
		case VECTOR3I: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				//scalar index
				int idx = p_index;
				if (idx < 0) {
					idx += 3;
				}
				if (idx >= 0 && idx < 3) {
					const Vector3i *v = reinterpret_cast<const Vector3i *>(_data._mem);
					valid = true;
					return (*v)[idx];
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Vector3i *v = reinterpret_cast<const Vector3i *>(_data._mem);
				if (*str == "x") {
					valid = true;
					return v->x;
				} else if (*str == "y") {
					valid = true;
					return v->y;
				} else if (*str == "z") {
					valid = true;
					return v->z;
				}
			}

		} break;
		case TRANSFORM2D: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				int index = p_index;

				if (index < 0) {
					index += 3;
				}
				if (index >= 0 && index < 3) {
					const Transform2D *v = _data._transform2d;

					valid = true;
					return v->elements[index];
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Transform2D *v = _data._transform2d;
				if (*str == "x") {
					valid = true;
					return v->elements[0];
				} else if (*str == "y") {
					valid = true;
					return v->elements[1];
				} else if (*str == "origin") {
					valid = true;
					return v->elements[2];
				}
			}

		} break;
		case PLANE: {
			if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Plane *v = reinterpret_cast<const Plane *>(_data._mem);
				if (*str == "x") {
					valid = true;
					return v->normal.x;
				} else if (*str == "y") {
					valid = true;
					return v->normal.y;
				} else if (*str == "z") {
					valid = true;
					return v->normal.z;
				} else if (*str == "normal") {
					valid = true;
					return v->normal;
				} else if (*str == "d") {
					valid = true;
					return v->d;
				}
			}

		} break;
		case QUAT: {
			if (p_index.get_type() == Variant::STRING) {
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Quat *v = reinterpret_cast<const Quat *>(_data._mem);
				if (*str == "x") {
					valid = true;
					return v->x;
				} else if (*str == "y") {
					valid = true;
					return v->y;
				} else if (*str == "z") {
					valid = true;
					return v->z;
				} else if (*str == "w") {
					valid = true;
					return v->w;
				}
			}

		} break;
		case AABB: {
			if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const ::AABB *v = _data._aabb;
				if (*str == "position") {
					valid = true;
					return v->position;
				} else if (*str == "size") {
					valid = true;
					return v->size;
				} else if (*str == "end") {
					valid = true;
					return v->size + v->position;
				}
			}
		} break;
		case BASIS: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				int index = p_index;
				if (index < 0) {
					index += 3;
				}
				if (index >= 0 && index < 3) {
					const Basis *v = _data._basis;

					valid = true;
					return v->get_axis(index);
				}
			} else if (p_index.get_type() == Variant::STRING) {
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Basis *v = _data._basis;

				if (*str == "x") {
					valid = true;
					return v->get_axis(0);
				} else if (*str == "y") {
					valid = true;
					return v->get_axis(1);
				} else if (*str == "z") {
					valid = true;
					return v->get_axis(2);
				}
			}

		} break;
		case TRANSFORM: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				int index = p_index;
				if (index < 0) {
					index += 4;
				}
				if (index >= 0 && index < 4) {
					const Transform *v = _data._transform;
					valid = true;
					return index == 3 ? v->origin : v->basis.get_axis(index);
				}
			} else if (p_index.get_type() == Variant::STRING) {
				const Transform *v = _data._transform;
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);

				if (*str == "basis") {
					valid = true;
					return v->basis;
				}
				if (*str == "origin") {
					valid = true;
					return v->origin;
				}
			}

		} break;
		case COLOR: {
			if (p_index.get_type() == Variant::STRING) {
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Color *v = reinterpret_cast<const Color *>(_data._mem);
				if (*str == "r") {
					valid = true;
					return v->r;
				} else if (*str == "g") {
					valid = true;
					return v->g;
				} else if (*str == "b") {
					valid = true;
					return v->b;
				} else if (*str == "a") {
					valid = true;
					return v->a;
				} else if (*str == "h") {
					valid = true;
					return v->get_h();
				} else if (*str == "s") {
					valid = true;
					return v->get_s();
				} else if (*str == "v") {
					valid = true;
					return v->get_v();
				} else if (*str == "r8") {
					valid = true;
					return (int)Math::round(v->r * 255.0);
				} else if (*str == "g8") {
					valid = true;
					return (int)Math::round(v->g * 255.0);
				} else if (*str == "b8") {
					valid = true;
					return (int)Math::round(v->b * 255.0);
				} else if (*str == "a8") {
					valid = true;
					return (int)Math::round(v->a * 255.0);
				}
			} else if (p_index.get_type() == Variant::INT) {
				int idx = p_index;
				if (idx < 0) {
					idx += 4;
				}
				if (idx >= 0 && idx < 4) {
					const Color *v = reinterpret_cast<const Color *>(_data._mem);
					valid = true;
					return (*v)[idx];
				}
			}

		} break;
		case STRING_NAME: {
		} break;
		case NODE_PATH: {
		} break;
		case _RID: {
		} break;
		case OBJECT: {
			Object *obj = _get_obj().obj;
			if (obj) {
#ifdef DEBUG_ENABLED

				if (EngineDebugger::is_active() && !_get_obj().id.is_reference() && ObjectDB::get_instance(_get_obj().id) == nullptr) {
					valid = false;
					return "Attempted get on previously freed instance.";
				}
#endif

				if (p_index.get_type() != Variant::STRING) {
					return obj->getvar(p_index, r_valid);
				}

				return obj->get(p_index, r_valid);
			}

		} break;
		case DICTIONARY: {
			const Dictionary *dic = reinterpret_cast<const Dictionary *>(_data._mem);
			const Variant *res = dic->getptr(p_index);
			if (res) {
				valid = true;
				return *res;
			}
		} break;
			DEFAULT_OP_ARRAY_CMD(ARRAY, const Array, ;, return (*arr)[index])
			DEFAULT_OP_DVECTOR_GET(PACKED_BYTE_ARRAY, uint8_t)
			DEFAULT_OP_DVECTOR_GET(PACKED_INT32_ARRAY, int32_t)
			DEFAULT_OP_DVECTOR_GET(PACKED_INT64_ARRAY, int64_t)
			DEFAULT_OP_DVECTOR_GET(PACKED_FLOAT32_ARRAY, float)
			DEFAULT_OP_DVECTOR_GET(PACKED_FLOAT64_ARRAY, double)
			DEFAULT_OP_DVECTOR_GET(PACKED_STRING_ARRAY, String)
			DEFAULT_OP_DVECTOR_GET(PACKED_VECTOR2_ARRAY, Vector2)
			DEFAULT_OP_DVECTOR_GET(PACKED_VECTOR3_ARRAY, Vector3)
			DEFAULT_OP_DVECTOR_GET(PACKED_COLOR_ARRAY, Color)
		default:
			return Variant();
	}

	return Variant();
}

bool Variant::in(const Variant &p_index, bool *r_valid) const {
	if (r_valid) {
		*r_valid = true;
	}

	switch (type) {
		case STRING: {
			if (p_index.get_type() == Variant::STRING) {
				//string index
				String idx = p_index;
				const String *str = reinterpret_cast<const String *>(_data._mem);

				return str->find(idx) != -1;
			}

		} break;
		case OBJECT: {
			Object *obj = _get_obj().obj;
			if (obj) {
				bool valid = false;
#ifdef DEBUG_ENABLED

				if (EngineDebugger::is_active() && !_get_obj().id.is_reference() && ObjectDB::get_instance(_get_obj().id) == nullptr) {
					if (r_valid) {
						*r_valid = false;
					}
					return true; // Attempted get on stray pointer.
				}

#endif

				if (p_index.get_type() != Variant::STRING) {
					obj->getvar(p_index, &valid);
				} else {
					obj->get(p_index, &valid);
				}

				return valid;
			} else {
				if (r_valid) {
					*r_valid = false;
				}
			}
			return false;
		} break;
		case DICTIONARY: {
			const Dictionary *dic = reinterpret_cast<const Dictionary *>(_data._mem);
			return dic->has(p_index);

		} break;
		case ARRAY: {
			const Array *arr = reinterpret_cast<const Array *>(_data._mem);
			int l = arr->size();
			if (l) {
				for (int i = 0; i < l; i++) {
					if (evaluate(OP_EQUAL, (*arr)[i], p_index)) {
						return true;
					}
				}
			}

			return false;

		} break;
		case PACKED_BYTE_ARRAY: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				int index = p_index;
				const Vector<uint8_t> *arr = &PackedArrayRef<uint8_t>::get_array(_data.packed_array);
				int l = arr->size();
				if (l) {
					const uint8_t *r = arr->ptr();
					for (int i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}

		} break;
		case PACKED_INT32_ARRAY: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				int32_t index = p_index;
				const Vector<int32_t> *arr = &PackedArrayRef<int32_t>::get_array(_data.packed_array);
				int32_t l = arr->size();
				if (l) {
					const int32_t *r = arr->ptr();
					for (int32_t i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}
		} break;
		case PACKED_INT64_ARRAY: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				int64_t index = p_index;
				const Vector<int64_t> *arr = &PackedArrayRef<int64_t>::get_array(_data.packed_array);
				int64_t l = arr->size();
				if (l) {
					const int64_t *r = arr->ptr();
					for (int64_t i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}
		} break;
		case PACKED_FLOAT32_ARRAY: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				real_t index = p_index;
				const Vector<float> *arr = &PackedArrayRef<float>::get_array(_data.packed_array);
				int l = arr->size();
				if (l) {
					const float *r = arr->ptr();
					for (int i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}

		} break;
		case PACKED_FLOAT64_ARRAY: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::FLOAT) {
				real_t index = p_index;
				const Vector<double> *arr = &PackedArrayRef<double>::get_array(_data.packed_array);
				int l = arr->size();
				if (l) {
					const double *r = arr->ptr();
					for (int i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}

		} break;
		case PACKED_STRING_ARRAY: {
			if (p_index.get_type() == Variant::STRING) {
				String index = p_index;
				const Vector<String> *arr = &PackedArrayRef<String>::get_array(_data.packed_array);

				int l = arr->size();
				if (l) {
					const String *r = arr->ptr();
					for (int i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}

		} break; //25
		case PACKED_VECTOR2_ARRAY: {
			if (p_index.get_type() == Variant::VECTOR2) {
				Vector2 index = p_index;
				const Vector<Vector2> *arr = &PackedArrayRef<Vector2>::get_array(_data.packed_array);

				int l = arr->size();
				if (l) {
					const Vector2 *r = arr->ptr();
					for (int i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}

		} break;
		case PACKED_VECTOR3_ARRAY: {
			if (p_index.get_type() == Variant::VECTOR3) {
				Vector3 index = p_index;
				const Vector<Vector3> *arr = &PackedArrayRef<Vector3>::get_array(_data.packed_array);

				int l = arr->size();
				if (l) {
					const Vector3 *r = arr->ptr();
					for (int i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}

		} break;
		case PACKED_COLOR_ARRAY: {
			if (p_index.get_type() == Variant::COLOR) {
				Color index = p_index;
				const Vector<Color> *arr = &PackedArrayRef<Color>::get_array(_data.packed_array);

				int l = arr->size();
				if (l) {
					const Color *r = arr->ptr();
					for (int i = 0; i < l; i++) {
						if (r[i] == index) {
							return true;
						}
					}
				}

				return false;
			}
		} break;
		default: {
		}
	}

	if (r_valid) {
		*r_valid = false;
	}
	return false;
}

void Variant::get_property_list(List<PropertyInfo> *p_list) const {
	switch (type) {
		case VECTOR2: {
			p_list->push_back(PropertyInfo(Variant::FLOAT, "x"));
			p_list->push_back(PropertyInfo(Variant::FLOAT, "y"));

		} break;
		case VECTOR2I: {
			p_list->push_back(PropertyInfo(Variant::INT, "x"));
			p_list->push_back(PropertyInfo(Variant::INT, "y"));

		} break;
		case RECT2: {
			p_list->push_back(PropertyInfo(Variant::VECTOR2, "position"));
			p_list->push_back(PropertyInfo(Variant::VECTOR2, "size"));
			p_list->push_back(PropertyInfo(Variant::VECTOR2, "end"));

		} break;
		case RECT2I: {
			p_list->push_back(PropertyInfo(Variant::VECTOR2I, "position"));
			p_list->push_back(PropertyInfo(Variant::VECTOR2I, "size"));
			p_list->push_back(PropertyInfo(Variant::VECTOR2I, "end"));

		} break;
		case VECTOR3: {
			p_list->push_back(PropertyInfo(Variant::FLOAT, "x"));
			p_list->push_back(PropertyInfo(Variant::FLOAT, "y"));
			p_list->push_back(PropertyInfo(Variant::FLOAT, "z"));

		} break;
		case VECTOR3I: {
			p_list->push_back(PropertyInfo(Variant::INT, "x"));
			p_list->push_back(PropertyInfo(Variant::INT, "y"));
			p_list->push_back(PropertyInfo(Variant::INT, "z"));

		} break;
		case TRANSFORM2D: {
			p_list->push_back(PropertyInfo(Variant::VECTOR2, "x"));
			p_list->push_back(PropertyInfo(Variant::VECTOR2, "y"));
			p_list->push_back(PropertyInfo(Variant::VECTOR2, "origin"));

		} break;
		case PLANE: {
			p_list->push_back(PropertyInfo(Variant::VECTOR3, "normal"));
			p_list->push_back(PropertyInfo(Variant::FLOAT, "x"));
			p_list->push_back(PropertyInfo(Variant::FLOAT, "y"));
			p_list->push_back(PropertyInfo(Variant::FLOAT, "z"));
			p_list->push_back(PropertyInfo(Variant::FLOAT, "d"));

		} break;
		case QUAT: {
			p_list->push_back(PropertyInfo(Variant::FLOAT, "x"));
			p_list->push_back(PropertyInfo(Variant::FLOAT, "y"));
			p_list->push_back(PropertyInfo(Variant::FLOAT, "z"));
			p_list->push_back(PropertyInfo(Variant::FLOAT, "w"));

		} break;
		case AABB: {
			p_list->push_back(PropertyInfo(Variant::VECTOR3, "position"));
			p_list->push_back(PropertyInfo(Variant::VECTOR3, "size"));
			p_list->push_back(PropertyInfo(Variant::VECTOR3, "end"));
		} break;
		case BASIS: {
			p_list->push_back(PropertyInfo(Variant::VECTOR3, "x"));
			p_list->push_back(PropertyInfo(Variant::VECTOR3, "y"));
			p_list->push_back(PropertyInfo(Variant::VECTOR3, "z"));

		} break;
		case TRANSFORM: {
			p_list->push_back(PropertyInfo(Variant::BASIS, "basis"));
			p_list->push_back(PropertyInfo(Variant::VECTOR3, "origin"));

		} break;
		case COLOR: {
			p_list->push_back(PropertyInfo(Variant::FLOAT, "r"));
			p_list->push_back(PropertyInfo(Variant::FLOAT, "g"));
			p_list->push_back(PropertyInfo(Variant::FLOAT, "b"));
			p_list->push_back(PropertyInfo(Variant::FLOAT, "a"));
			p_list->push_back(PropertyInfo(Variant::FLOAT, "h"));
			p_list->push_back(PropertyInfo(Variant::FLOAT, "s"));
			p_list->push_back(PropertyInfo(Variant::FLOAT, "v"));
			p_list->push_back(PropertyInfo(Variant::INT, "r8"));
			p_list->push_back(PropertyInfo(Variant::INT, "g8"));
			p_list->push_back(PropertyInfo(Variant::INT, "b8"));
			p_list->push_back(PropertyInfo(Variant::INT, "a8"));

		} break;
		case STRING_NAME: {
		} break;
		case NODE_PATH: {
		} break;
		case _RID: {
		} break;
		case OBJECT: {
			Object *obj = _get_obj().obj;
			if (obj) {
#ifdef DEBUG_ENABLED

				if (EngineDebugger::is_active() && !_get_obj().id.is_reference() && ObjectDB::get_instance(_get_obj().id) == nullptr) {
					WARN_PRINT("Attempted get_property list on previously freed instance.");
					return;
				}

#endif

				obj->get_property_list(p_list);
			}

		} break;
		case DICTIONARY: {
			const Dictionary *dic = reinterpret_cast<const Dictionary *>(_data._mem);
			List<Variant> keys;
			dic->get_key_list(&keys);
			for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {
				if (E->get().get_type() == Variant::STRING) {
					p_list->push_back(PropertyInfo(Variant::STRING, E->get()));
				}
			}
		} break;
		case ARRAY:
		case PACKED_BYTE_ARRAY:
		case PACKED_INT32_ARRAY:
		case PACKED_INT64_ARRAY:
		case PACKED_FLOAT32_ARRAY:
		case PACKED_FLOAT64_ARRAY:
		case PACKED_STRING_ARRAY:
		case PACKED_VECTOR2_ARRAY:
		case PACKED_VECTOR3_ARRAY:
		case PACKED_COLOR_ARRAY: {
			//nothing
		} break;
		default: {
		}
	}
}

bool Variant::iter_init(Variant &r_iter, bool &valid) const {
	valid = true;
	switch (type) {
		case INT: {
			r_iter = 0;
			return _data._int > 0;
		} break;
		case FLOAT: {
			r_iter = 0;
			return _data._float > 0.0;
		} break;
		case VECTOR2: {
			double from = reinterpret_cast<const Vector2 *>(_data._mem)->x;
			double to = reinterpret_cast<const Vector2 *>(_data._mem)->y;

			r_iter = from;

			return from < to;
		} break;
		case VECTOR2I: {
			int64_t from = reinterpret_cast<const Vector2i *>(_data._mem)->x;
			int64_t to = reinterpret_cast<const Vector2i *>(_data._mem)->y;

			r_iter = from;

			return from < to;
		} break;
		case VECTOR3: {
			double from = reinterpret_cast<const Vector3 *>(_data._mem)->x;
			double to = reinterpret_cast<const Vector3 *>(_data._mem)->y;
			double step = reinterpret_cast<const Vector3 *>(_data._mem)->z;

			r_iter = from;

			if (from == to) {
				return false;
			} else if (from < to) {
				return step > 0;
			}
			return step < 0;
		} break;
		case VECTOR3I: {
			int64_t from = reinterpret_cast<const Vector3i *>(_data._mem)->x;
			int64_t to = reinterpret_cast<const Vector3i *>(_data._mem)->y;
			int64_t step = reinterpret_cast<const Vector3i *>(_data._mem)->z;

			r_iter = from;

			if (from == to) {
				return false;
			} else if (from < to) {
				return step > 0;
			}
			return step < 0;
		} break;
		case OBJECT: {
			if (!_get_obj().obj) {
				valid = false;
				return false;
			}

#ifdef DEBUG_ENABLED

			if (EngineDebugger::is_active() && !_get_obj().id.is_reference() && ObjectDB::get_instance(_get_obj().id) == nullptr) {
				valid = false;
				return false;
			}

#endif
			Callable::CallError ce;
			ce.error = Callable::CallError::CALL_OK;
			Array ref;
			ref.push_back(r_iter);
			Variant vref = ref;
			const Variant *refp[] = { &vref };
			Variant ret = _get_obj().obj->call(CoreStringNames::get_singleton()->_iter_init, refp, 1, ce);

			if (ref.size() != 1 || ce.error != Callable::CallError::CALL_OK) {
				valid = false;
				return false;
			}

			r_iter = ref[0];
			return ret;
		} break;

		case STRING: {
			const String *str = reinterpret_cast<const String *>(_data._mem);
			if (str->empty()) {
				return false;
			}
			r_iter = 0;
			return true;
		} break;
		case DICTIONARY: {
			const Dictionary *dic = reinterpret_cast<const Dictionary *>(_data._mem);
			if (dic->empty()) {
				return false;
			}

			const Variant *next = dic->next(nullptr);
			r_iter = *next;
			return true;

		} break;
		case ARRAY: {
			const Array *arr = reinterpret_cast<const Array *>(_data._mem);
			if (arr->empty()) {
				return false;
			}
			r_iter = 0;
			return true;
		} break;
		case PACKED_BYTE_ARRAY: {
			const Vector<uint8_t> *arr = &PackedArrayRef<uint8_t>::get_array(_data.packed_array);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;

		} break;
		case PACKED_INT32_ARRAY: {
			const Vector<int32_t> *arr = &PackedArrayRef<int32_t>::get_array(_data.packed_array);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;

		} break;
		case PACKED_INT64_ARRAY: {
			const Vector<int64_t> *arr = &PackedArrayRef<int64_t>::get_array(_data.packed_array);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;

		} break;
		case PACKED_FLOAT32_ARRAY: {
			const Vector<float> *arr = &PackedArrayRef<float>::get_array(_data.packed_array);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;

		} break;
		case PACKED_FLOAT64_ARRAY: {
			const Vector<double> *arr = &PackedArrayRef<double>::get_array(_data.packed_array);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;

		} break;
		case PACKED_STRING_ARRAY: {
			const Vector<String> *arr = &PackedArrayRef<String>::get_array(_data.packed_array);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;
		} break;
		case PACKED_VECTOR2_ARRAY: {
			const Vector<Vector2> *arr = &PackedArrayRef<Vector2>::get_array(_data.packed_array);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;
		} break;
		case PACKED_VECTOR3_ARRAY: {
			const Vector<Vector3> *arr = &PackedArrayRef<Vector3>::get_array(_data.packed_array);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;
		} break;
		case PACKED_COLOR_ARRAY: {
			const Vector<Color> *arr = &PackedArrayRef<Color>::get_array(_data.packed_array);
			if (arr->size() == 0) {
				return false;
			}
			r_iter = 0;
			return true;

		} break;
		default: {
		}
	}

	valid = false;
	return false;
}

bool Variant::iter_next(Variant &r_iter, bool &valid) const {
	valid = true;
	switch (type) {
		case INT: {
			int64_t idx = r_iter;
			idx++;
			if (idx >= _data._int) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		case FLOAT: {
			int64_t idx = r_iter;
			idx++;
			if (idx >= _data._float) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		case VECTOR2: {
			double to = reinterpret_cast<const Vector2 *>(_data._mem)->y;

			double idx = r_iter;
			idx++;

			if (idx >= to) {
				return false;
			}

			r_iter = idx;
			return true;
		} break;
		case VECTOR2I: {
			int64_t to = reinterpret_cast<const Vector2i *>(_data._mem)->y;

			int64_t idx = r_iter;
			idx++;

			if (idx >= to) {
				return false;
			}

			r_iter = idx;
			return true;
		} break;
		case VECTOR3: {
			double to = reinterpret_cast<const Vector3 *>(_data._mem)->y;
			double step = reinterpret_cast<const Vector3 *>(_data._mem)->z;

			double idx = r_iter;
			idx += step;

			if (step < 0 && idx <= to) {
				return false;
			}

			if (step > 0 && idx >= to) {
				return false;
			}

			r_iter = idx;
			return true;
		} break;
		case VECTOR3I: {
			int64_t to = reinterpret_cast<const Vector3i *>(_data._mem)->y;
			int64_t step = reinterpret_cast<const Vector3i *>(_data._mem)->z;

			int64_t idx = r_iter;
			idx += step;

			if (step < 0 && idx <= to) {
				return false;
			}

			if (step > 0 && idx >= to) {
				return false;
			}

			r_iter = idx;
			return true;
		} break;
		case OBJECT: {
			if (!_get_obj().obj) {
				valid = false;
				return false;
			}

#ifdef DEBUG_ENABLED

			if (EngineDebugger::is_active() && !_get_obj().id.is_reference() && ObjectDB::get_instance(_get_obj().id) == nullptr) {
				valid = false;
				return false;
			}

#endif
			Callable::CallError ce;
			ce.error = Callable::CallError::CALL_OK;
			Array ref;
			ref.push_back(r_iter);
			Variant vref = ref;
			const Variant *refp[] = { &vref };
			Variant ret = _get_obj().obj->call(CoreStringNames::get_singleton()->_iter_next, refp, 1, ce);

			if (ref.size() != 1 || ce.error != Callable::CallError::CALL_OK) {
				valid = false;
				return false;
			}

			r_iter = ref[0];

			return ret;
		} break;

		case STRING: {
			const String *str = reinterpret_cast<const String *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= str->length()) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		case DICTIONARY: {
			const Dictionary *dic = reinterpret_cast<const Dictionary *>(_data._mem);
			const Variant *next = dic->next(&r_iter);
			if (!next) {
				return false;
			}

			r_iter = *next;
			return true;

		} break;
		case ARRAY: {
			const Array *arr = reinterpret_cast<const Array *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		case PACKED_BYTE_ARRAY: {
			const Vector<uint8_t> *arr = &PackedArrayRef<uint8_t>::get_array(_data.packed_array);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;

		} break;
		case PACKED_INT32_ARRAY: {
			const Vector<int32_t> *arr = &PackedArrayRef<int32_t>::get_array(_data.packed_array);
			int32_t idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;

		} break;
		case PACKED_INT64_ARRAY: {
			const Vector<int64_t> *arr = &PackedArrayRef<int64_t>::get_array(_data.packed_array);
			int64_t idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;

		} break;
		case PACKED_FLOAT32_ARRAY: {
			const Vector<float> *arr = &PackedArrayRef<float>::get_array(_data.packed_array);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;

		} break;
		case PACKED_FLOAT64_ARRAY: {
			const Vector<double> *arr = &PackedArrayRef<double>::get_array(_data.packed_array);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;

		} break;
		case PACKED_STRING_ARRAY: {
			const Vector<String> *arr = &PackedArrayRef<String>::get_array(_data.packed_array);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		case PACKED_VECTOR2_ARRAY: {
			const Vector<Vector2> *arr = &PackedArrayRef<Vector2>::get_array(_data.packed_array);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		case PACKED_VECTOR3_ARRAY: {
			const Vector<Vector3> *arr = &PackedArrayRef<Vector3>::get_array(_data.packed_array);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		case PACKED_COLOR_ARRAY: {
			const Vector<Color> *arr = &PackedArrayRef<Color>::get_array(_data.packed_array);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size()) {
				return false;
			}
			r_iter = idx;
			return true;
		} break;
		default: {
		}
	}

	valid = false;
	return false;
}

Variant Variant::iter_get(const Variant &r_iter, bool &r_valid) const {
	r_valid = true;
	switch (type) {
		case INT: {
			return r_iter;
		} break;
		case FLOAT: {
			return r_iter;
		} break;
		case VECTOR2: {
			return r_iter;
		} break;
		case VECTOR2I: {
			return r_iter;
		} break;
		case VECTOR3: {
			return r_iter;
		} break;
		case VECTOR3I: {
			return r_iter;
		} break;
		case OBJECT: {
			if (!_get_obj().obj) {
				r_valid = false;
				return Variant();
			}
#ifdef DEBUG_ENABLED
			if (EngineDebugger::is_active() && !_get_obj().id.is_reference() && ObjectDB::get_instance(_get_obj().id) == nullptr) {
				r_valid = false;
				return Variant();
			}

#endif
			Callable::CallError ce;
			ce.error = Callable::CallError::CALL_OK;
			const Variant *refp[] = { &r_iter };
			Variant ret = _get_obj().obj->call(CoreStringNames::get_singleton()->_iter_get, refp, 1, ce);

			if (ce.error != Callable::CallError::CALL_OK) {
				r_valid = false;
				return Variant();
			}

			//r_iter=ref[0];

			return ret;
		} break;

		case STRING: {
			const String *str = reinterpret_cast<const String *>(_data._mem);
			return str->substr(r_iter, 1);
		} break;
		case DICTIONARY: {
			return r_iter; //iterator is the same as the key

		} break;
		case ARRAY: {
			const Array *arr = reinterpret_cast<const Array *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case PACKED_BYTE_ARRAY: {
			const Vector<uint8_t> *arr = &PackedArrayRef<uint8_t>::get_array(_data.packed_array);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case PACKED_INT32_ARRAY: {
			const Vector<int32_t> *arr = &PackedArrayRef<int32_t>::get_array(_data.packed_array);
			int32_t idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case PACKED_INT64_ARRAY: {
			const Vector<int64_t> *arr = &PackedArrayRef<int64_t>::get_array(_data.packed_array);
			int64_t idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case PACKED_FLOAT32_ARRAY: {
			const Vector<float> *arr = &PackedArrayRef<float>::get_array(_data.packed_array);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case PACKED_FLOAT64_ARRAY: {
			const Vector<double> *arr = &PackedArrayRef<double>::get_array(_data.packed_array);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case PACKED_STRING_ARRAY: {
			const Vector<String> *arr = &PackedArrayRef<String>::get_array(_data.packed_array);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case PACKED_VECTOR2_ARRAY: {
			const Vector<Vector2> *arr = &PackedArrayRef<Vector2>::get_array(_data.packed_array);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case PACKED_VECTOR3_ARRAY: {
			const Vector<Vector3> *arr = &PackedArrayRef<Vector3>::get_array(_data.packed_array);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case PACKED_COLOR_ARRAY: {
			const Vector<Color> *arr = &PackedArrayRef<Color>::get_array(_data.packed_array);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		default: {
		}
	}

	r_valid = false;
	return Variant();
}

Variant Variant::duplicate(bool deep) const {
	switch (type) {
		case OBJECT: {
			/*  breaks stuff :(
			if (deep && !_get_obj().ref.is_null()) {
				Ref<Resource> resource = _get_obj().ref;
				if (resource.is_valid()) {
					return resource->duplicate(true);
				}
			}
			*/
			return *this;
		} break;
		case DICTIONARY:
			return operator Dictionary().duplicate(deep);
		case ARRAY:
			return operator Array().duplicate(deep);
		default:
			return *this;
	}
}

void Variant::blend(const Variant &a, const Variant &b, float c, Variant &r_dst) {
	if (a.type != b.type) {
		if (a.is_num() && b.is_num()) {
			real_t va = a;
			real_t vb = b;
			r_dst = va + vb * c;
		} else {
			r_dst = a;
		}
		return;
	}

	switch (a.type) {
		case NIL: {
			r_dst = Variant();
		}
			return;
		case INT: {
			int64_t va = a._data._int;
			int64_t vb = b._data._int;
			r_dst = int(va + vb * c + 0.5);
		}
			return;
		case FLOAT: {
			double ra = a._data._float;
			double rb = b._data._float;
			r_dst = ra + rb * c;
		}
			return;
		case VECTOR2: {
			r_dst = *reinterpret_cast<const Vector2 *>(a._data._mem) + *reinterpret_cast<const Vector2 *>(b._data._mem) * c;
		}
			return;
		case VECTOR2I: {
			int32_t vax = reinterpret_cast<const Vector2i *>(a._data._mem)->x;
			int32_t vbx = reinterpret_cast<const Vector2i *>(b._data._mem)->x;
			int32_t vay = reinterpret_cast<const Vector2i *>(a._data._mem)->y;
			int32_t vby = reinterpret_cast<const Vector2i *>(b._data._mem)->y;
			r_dst = Vector2i(int32_t(vax + vbx * c + 0.5), int32_t(vay + vby * c + 0.5));
		}
			return;
		case RECT2: {
			const Rect2 *ra = reinterpret_cast<const Rect2 *>(a._data._mem);
			const Rect2 *rb = reinterpret_cast<const Rect2 *>(b._data._mem);
			r_dst = Rect2(ra->position + rb->position * c, ra->size + rb->size * c);
		}
			return;
		case RECT2I: {
			const Rect2i *ra = reinterpret_cast<const Rect2i *>(a._data._mem);
			const Rect2i *rb = reinterpret_cast<const Rect2i *>(b._data._mem);

			int32_t vax = ra->position.x;
			int32_t vay = ra->position.y;
			int32_t vbx = ra->size.x;
			int32_t vby = ra->size.y;
			int32_t vcx = rb->position.x;
			int32_t vcy = rb->position.y;
			int32_t vdx = rb->size.x;
			int32_t vdy = rb->size.y;

			r_dst = Rect2i(int32_t(vax + vbx * c + 0.5), int32_t(vay + vby * c + 0.5), int32_t(vcx + vdx * c + 0.5), int32_t(vcy + vdy * c + 0.5));
		}
			return;
		case VECTOR3: {
			r_dst = *reinterpret_cast<const Vector3 *>(a._data._mem) + *reinterpret_cast<const Vector3 *>(b._data._mem) * c;
		}
			return;
		case VECTOR3I: {
			int32_t vax = reinterpret_cast<const Vector3i *>(a._data._mem)->x;
			int32_t vbx = reinterpret_cast<const Vector3i *>(b._data._mem)->x;
			int32_t vay = reinterpret_cast<const Vector3i *>(a._data._mem)->y;
			int32_t vby = reinterpret_cast<const Vector3i *>(b._data._mem)->y;
			int32_t vaz = reinterpret_cast<const Vector3i *>(a._data._mem)->z;
			int32_t vbz = reinterpret_cast<const Vector3i *>(b._data._mem)->z;
			r_dst = Vector3i(int32_t(vax + vbx * c + 0.5), int32_t(vay + vby * c + 0.5), int32_t(vaz + vbz * c + 0.5));
		}
			return;
		case AABB: {
			const ::AABB *ra = reinterpret_cast<const ::AABB *>(a._data._mem);
			const ::AABB *rb = reinterpret_cast<const ::AABB *>(b._data._mem);
			r_dst = ::AABB(ra->position + rb->position * c, ra->size + rb->size * c);
		}
			return;
		case QUAT: {
			Quat empty_rot;
			const Quat *qa = reinterpret_cast<const Quat *>(a._data._mem);
			const Quat *qb = reinterpret_cast<const Quat *>(b._data._mem);
			r_dst = *qa * empty_rot.slerp(*qb, c);
		}
			return;
		case COLOR: {
			const Color *ca = reinterpret_cast<const Color *>(a._data._mem);
			const Color *cb = reinterpret_cast<const Color *>(b._data._mem);
			float new_r = ca->r + cb->r * c;
			float new_g = ca->g + cb->g * c;
			float new_b = ca->b + cb->b * c;
			float new_a = ca->a + cb->a * c;
			new_r = new_r > 1.0 ? 1.0 : new_r;
			new_g = new_g > 1.0 ? 1.0 : new_g;
			new_b = new_b > 1.0 ? 1.0 : new_b;
			new_a = new_a > 1.0 ? 1.0 : new_a;
			r_dst = Color(new_r, new_g, new_b, new_a);
		}
			return;
		default: {
			r_dst = c < 0.5 ? a : b;
		}
			return;
	}
}

void Variant::interpolate(const Variant &a, const Variant &b, float c, Variant &r_dst) {
	if (a.type != b.type) {
		if (a.is_num() && b.is_num()) {
			//not as efficient but..
			real_t va = a;
			real_t vb = b;
			r_dst = va + (vb - va) * c;

		} else {
			r_dst = a;
		}
		return;
	}

	switch (a.type) {
		case NIL: {
			r_dst = Variant();
		}
			return;
		case BOOL: {
			r_dst = a;
		}
			return;
		case INT: {
			int64_t va = a._data._int;
			int64_t vb = b._data._int;
			r_dst = int(va + (vb - va) * c);
		}
			return;
		case FLOAT: {
			real_t va = a._data._float;
			real_t vb = b._data._float;
			r_dst = va + (vb - va) * c;
		}
			return;
		case STRING: {
			//this is pretty funny and bizarre, but artists like to use it for typewritter effects
			String sa = *reinterpret_cast<const String *>(a._data._mem);
			String sb = *reinterpret_cast<const String *>(b._data._mem);
			String dst;
			int sa_len = sa.length();
			int sb_len = sb.length();
			int csize = sa_len + (sb_len - sa_len) * c;
			if (csize == 0) {
				r_dst = "";
				return;
			}
			dst.resize(csize + 1);
			dst[csize] = 0;
			int split = csize / 2;

			for (int i = 0; i < csize; i++) {
				char32_t chr = ' ';

				if (i < split) {
					if (i < sa.length()) {
						chr = sa[i];
					} else if (i < sb.length()) {
						chr = sb[i];
					}

				} else {
					if (i < sb.length()) {
						chr = sb[i];
					} else if (i < sa.length()) {
						chr = sa[i];
					}
				}

				dst[i] = chr;
			}

			r_dst = dst;
		}
			return;
		case VECTOR2: {
			r_dst = reinterpret_cast<const Vector2 *>(a._data._mem)->lerp(*reinterpret_cast<const Vector2 *>(b._data._mem), c);
		}
			return;
		case VECTOR2I: {
			int32_t vax = reinterpret_cast<const Vector2i *>(a._data._mem)->x;
			int32_t vbx = reinterpret_cast<const Vector2i *>(b._data._mem)->x;
			int32_t vay = reinterpret_cast<const Vector2i *>(a._data._mem)->y;
			int32_t vby = reinterpret_cast<const Vector2i *>(b._data._mem)->y;
			r_dst = Vector2i(int32_t(vax + vbx * c + 0.5), int32_t(vay + vby * c + 0.5));
		}
			return;

		case RECT2: {
			r_dst = Rect2(reinterpret_cast<const Rect2 *>(a._data._mem)->position.lerp(reinterpret_cast<const Rect2 *>(b._data._mem)->position, c), reinterpret_cast<const Rect2 *>(a._data._mem)->size.lerp(reinterpret_cast<const Rect2 *>(b._data._mem)->size, c));
		}
			return;
		case RECT2I: {
			const Rect2i *ra = reinterpret_cast<const Rect2i *>(a._data._mem);
			const Rect2i *rb = reinterpret_cast<const Rect2i *>(b._data._mem);

			int32_t vax = ra->position.x;
			int32_t vay = ra->position.y;
			int32_t vbx = ra->size.x;
			int32_t vby = ra->size.y;
			int32_t vcx = rb->position.x;
			int32_t vcy = rb->position.y;
			int32_t vdx = rb->size.x;
			int32_t vdy = rb->size.y;

			r_dst = Rect2i(int32_t(vax + vbx * c + 0.5), int32_t(vay + vby * c + 0.5), int32_t(vcx + vdx * c + 0.5), int32_t(vcy + vdy * c + 0.5));
		}
			return;

		case VECTOR3: {
			r_dst = reinterpret_cast<const Vector3 *>(a._data._mem)->lerp(*reinterpret_cast<const Vector3 *>(b._data._mem), c);
		}
			return;
		case VECTOR3I: {
			int32_t vax = reinterpret_cast<const Vector3i *>(a._data._mem)->x;
			int32_t vbx = reinterpret_cast<const Vector3i *>(b._data._mem)->x;
			int32_t vay = reinterpret_cast<const Vector3i *>(a._data._mem)->y;
			int32_t vby = reinterpret_cast<const Vector3i *>(b._data._mem)->y;
			int32_t vaz = reinterpret_cast<const Vector3i *>(a._data._mem)->z;
			int32_t vbz = reinterpret_cast<const Vector3i *>(b._data._mem)->z;
			r_dst = Vector3i(int32_t(vax + vbx * c + 0.5), int32_t(vay + vby * c + 0.5), int32_t(vaz + vbz * c + 0.5));
		}
			return;

		case TRANSFORM2D: {
			r_dst = a._data._transform2d->interpolate_with(*b._data._transform2d, c);
		}
			return;
		case PLANE: {
			r_dst = a;
		}
			return;
		case QUAT: {
			r_dst = reinterpret_cast<const Quat *>(a._data._mem)->slerp(*reinterpret_cast<const Quat *>(b._data._mem), c);
		}
			return;
		case AABB: {
			r_dst = ::AABB(a._data._aabb->position.lerp(b._data._aabb->position, c), a._data._aabb->size.lerp(b._data._aabb->size, c));
		}
			return;
		case BASIS: {
			r_dst = Transform(*a._data._basis).interpolate_with(Transform(*b._data._basis), c).basis;
		}
			return;
		case TRANSFORM: {
			r_dst = a._data._transform->interpolate_with(*b._data._transform, c);
		}
			return;
		case COLOR: {
			r_dst = reinterpret_cast<const Color *>(a._data._mem)->lerp(*reinterpret_cast<const Color *>(b._data._mem), c);
		}
			return;
		case STRING_NAME: {
			r_dst = a;
		}
			return;
		case NODE_PATH: {
			r_dst = a;
		}
			return;
		case _RID: {
			r_dst = a;
		}
			return;
		case OBJECT: {
			r_dst = a;
		}
			return;
		case DICTIONARY: {
		}
			return;
		case ARRAY: {
			r_dst = a;
		}
			return;
		case PACKED_BYTE_ARRAY: {
			r_dst = a;
		}
			return;
		case PACKED_INT32_ARRAY: {
			const Vector<int32_t> *arr_a = &PackedArrayRef<int32_t>::get_array(a._data.packed_array);
			const Vector<int32_t> *arr_b = &PackedArrayRef<int32_t>::get_array(b._data.packed_array);
			int32_t sz = arr_a->size();
			if (sz == 0 || arr_b->size() != sz) {
				r_dst = a;
			} else {
				Vector<int32_t> v;
				v.resize(sz);
				{
					int32_t *vw = v.ptrw();
					const int32_t *ar = arr_a->ptr();
					const int32_t *br = arr_b->ptr();

					Variant va;
					for (int32_t i = 0; i < sz; i++) {
						Variant::interpolate(ar[i], br[i], c, va);
						vw[i] = va;
					}
				}
				r_dst = v;
			}
		}
			return;
		case PACKED_INT64_ARRAY: {
			const Vector<int64_t> *arr_a = &PackedArrayRef<int64_t>::get_array(a._data.packed_array);
			const Vector<int64_t> *arr_b = &PackedArrayRef<int64_t>::get_array(b._data.packed_array);
			int64_t sz = arr_a->size();
			if (sz == 0 || arr_b->size() != sz) {
				r_dst = a;
			} else {
				Vector<int64_t> v;
				v.resize(sz);
				{
					int64_t *vw = v.ptrw();
					const int64_t *ar = arr_a->ptr();
					const int64_t *br = arr_b->ptr();

					Variant va;
					for (int64_t i = 0; i < sz; i++) {
						Variant::interpolate(ar[i], br[i], c, va);
						vw[i] = va;
					}
				}
				r_dst = v;
			}
		}
			return;
		case PACKED_FLOAT32_ARRAY: {
			const Vector<float> *arr_a = &PackedArrayRef<float>::get_array(a._data.packed_array);
			const Vector<float> *arr_b = &PackedArrayRef<float>::get_array(b._data.packed_array);
			int sz = arr_a->size();
			if (sz == 0 || arr_b->size() != sz) {
				r_dst = a;
			} else {
				Vector<float> v;
				v.resize(sz);
				{
					float *vw = v.ptrw();
					const float *ar = arr_a->ptr();
					const float *br = arr_b->ptr();

					Variant va;
					for (int i = 0; i < sz; i++) {
						Variant::interpolate(ar[i], br[i], c, va);
						vw[i] = va;
					}
				}
				r_dst = v;
			}
		}
			return;
		case PACKED_FLOAT64_ARRAY: {
			const Vector<double> *arr_a = &PackedArrayRef<double>::get_array(a._data.packed_array);
			const Vector<double> *arr_b = &PackedArrayRef<double>::get_array(b._data.packed_array);
			int sz = arr_a->size();
			if (sz == 0 || arr_b->size() != sz) {
				r_dst = a;
			} else {
				Vector<double> v;
				v.resize(sz);
				{
					double *vw = v.ptrw();
					const double *ar = arr_a->ptr();
					const double *br = arr_b->ptr();

					Variant va;
					for (int i = 0; i < sz; i++) {
						Variant::interpolate(ar[i], br[i], c, va);
						vw[i] = va;
					}
				}
				r_dst = v;
			}
		}
			return;
		case PACKED_STRING_ARRAY: {
			r_dst = a;
		}
			return;
		case PACKED_VECTOR2_ARRAY: {
			const Vector<Vector2> *arr_a = &PackedArrayRef<Vector2>::get_array(a._data.packed_array);
			const Vector<Vector2> *arr_b = &PackedArrayRef<Vector2>::get_array(b._data.packed_array);
			int sz = arr_a->size();
			if (sz == 0 || arr_b->size() != sz) {
				r_dst = a;
			} else {
				Vector<Vector2> v;
				v.resize(sz);
				{
					Vector2 *vw = v.ptrw();
					const Vector2 *ar = arr_a->ptr();
					const Vector2 *br = arr_b->ptr();

					for (int i = 0; i < sz; i++) {
						vw[i] = ar[i].lerp(br[i], c);
					}
				}
				r_dst = v;
			}
		}
			return;
		case PACKED_VECTOR3_ARRAY: {
			const Vector<Vector3> *arr_a = &PackedArrayRef<Vector3>::get_array(a._data.packed_array);
			const Vector<Vector3> *arr_b = &PackedArrayRef<Vector3>::get_array(b._data.packed_array);
			int sz = arr_a->size();
			if (sz == 0 || arr_b->size() != sz) {
				r_dst = a;
			} else {
				Vector<Vector3> v;
				v.resize(sz);
				{
					Vector3 *vw = v.ptrw();
					const Vector3 *ar = arr_a->ptr();
					const Vector3 *br = arr_b->ptr();

					for (int i = 0; i < sz; i++) {
						vw[i] = ar[i].lerp(br[i], c);
					}
				}
				r_dst = v;
			}
		}
			return;
		case PACKED_COLOR_ARRAY: {
			const Vector<Color> *arr_a = &PackedArrayRef<Color>::get_array(a._data.packed_array);
			const Vector<Color> *arr_b = &PackedArrayRef<Color>::get_array(b._data.packed_array);
			int sz = arr_a->size();
			if (sz == 0 || arr_b->size() != sz) {
				r_dst = a;
			} else {
				Vector<Color> v;
				v.resize(sz);
				{
					Color *vw = v.ptrw();
					const Color *ar = arr_a->ptr();
					const Color *br = arr_b->ptr();

					for (int i = 0; i < sz; i++) {
						vw[i] = ar[i].lerp(br[i], c);
					}
				}
				r_dst = v;
			}
		}
			return;
		default: {
			r_dst = a;
		}
	}
}
