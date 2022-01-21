/*************************************************************************/
/*  marshalls.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "marshalls.h"

#include "core/object/ref_counted.h"
#include "core/os/keyboard.h"
#include "core/string/print_string.h"

#include <limits.h>
#include <stdio.h>

void EncodedObjectAsID::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_object_id", "id"), &EncodedObjectAsID::set_object_id);
	ClassDB::bind_method(D_METHOD("get_object_id"), &EncodedObjectAsID::get_object_id);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "object_id"), "set_object_id", "get_object_id");
}

void EncodedObjectAsID::set_object_id(ObjectID p_id) {
	id = p_id;
}

ObjectID EncodedObjectAsID::get_object_id() const {
	return id;
}

#define ERR_FAIL_ADD_OF(a, b, err) ERR_FAIL_COND_V(((int32_t)(b)) < 0 || ((int32_t)(a)) < 0 || ((int32_t)(a)) > INT_MAX - ((int32_t)(b)), err)
#define ERR_FAIL_MUL_OF(a, b, err) ERR_FAIL_COND_V(((int32_t)(a)) < 0 || ((int32_t)(b)) <= 0 || ((int32_t)(a)) > INT_MAX / ((int32_t)(b)), err)

#define ENCODE_MASK 0xFF
#define ENCODE_FLAG_64 1 << 16
#define ENCODE_FLAG_OBJECT_AS_ID 1 << 16

static Error _decode_string(const uint8_t *&buf, int &len, int *r_len, String &r_string) {
	ERR_FAIL_COND_V(len < 4, ERR_INVALID_DATA);

	int32_t strlen = decode_uint32(buf);
	int32_t pad = 0;

	// Handle padding
	if (strlen % 4) {
		pad = 4 - strlen % 4;
	}

	buf += 4;
	len -= 4;

	// Ensure buffer is big enough
	ERR_FAIL_ADD_OF(strlen, pad, ERR_FILE_EOF);
	ERR_FAIL_COND_V(strlen < 0 || strlen + pad > len, ERR_FILE_EOF);

	String str;
	ERR_FAIL_COND_V(str.parse_utf8((const char *)buf, strlen), ERR_INVALID_DATA);
	r_string = str;

	// Add padding
	strlen += pad;

	// Update buffer pos, left data count, and return size
	buf += strlen;
	len -= strlen;
	if (r_len) {
		(*r_len) += 4 + strlen;
	}

	return OK;
}

Error decode_variant(Variant &r_variant, const uint8_t *p_buffer, int p_len, int *r_len, bool p_allow_objects) {
	const uint8_t *buf = p_buffer;
	int len = p_len;

	ERR_FAIL_COND_V(len < 4, ERR_INVALID_DATA);

	uint32_t type = decode_uint32(buf);

	ERR_FAIL_COND_V((type & ENCODE_MASK) >= Variant::VARIANT_MAX, ERR_INVALID_DATA);

	buf += 4;
	len -= 4;
	if (r_len) {
		*r_len = 4;
	}

	// Note: We cannot use sizeof(real_t) for decoding, in case a different size is encoded.
	// Decoding math types always checks for the encoded size, while encoding always uses compilation setting.
	// This does lead to some code duplication for decoding, but compatibility is the priority.
	switch (type & ENCODE_MASK) {
		case Variant::NIL: {
			r_variant = Variant();
		} break;
		case Variant::BOOL: {
			ERR_FAIL_COND_V(len < 4, ERR_INVALID_DATA);
			bool val = decode_uint32(buf);
			r_variant = val;
			if (r_len) {
				(*r_len) += 4;
			}
		} break;
		case Variant::INT: {
			if (type & ENCODE_FLAG_64) {
				ERR_FAIL_COND_V(len < 8, ERR_INVALID_DATA);
				int64_t val = decode_uint64(buf);
				r_variant = val;
				if (r_len) {
					(*r_len) += 8;
				}

			} else {
				ERR_FAIL_COND_V(len < 4, ERR_INVALID_DATA);
				int32_t val = decode_uint32(buf);
				r_variant = val;
				if (r_len) {
					(*r_len) += 4;
				}
			}

		} break;
		case Variant::FLOAT: {
			if (type & ENCODE_FLAG_64) {
				ERR_FAIL_COND_V((size_t)len < sizeof(double), ERR_INVALID_DATA);
				double val = decode_double(buf);
				r_variant = val;
				if (r_len) {
					(*r_len) += sizeof(double);
				}
			} else {
				ERR_FAIL_COND_V((size_t)len < sizeof(float), ERR_INVALID_DATA);
				float val = decode_float(buf);
				r_variant = val;
				if (r_len) {
					(*r_len) += sizeof(float);
				}
			}

		} break;
		case Variant::STRING: {
			String str;
			Error err = _decode_string(buf, len, r_len, str);
			if (err) {
				return err;
			}
			r_variant = str;

		} break;

		// math types
		case Variant::VECTOR2: {
			Vector2 val;
			if (type & ENCODE_FLAG_64) {
				ERR_FAIL_COND_V((size_t)len < sizeof(double) * 2, ERR_INVALID_DATA);
				val.x = decode_double(&buf[0]);
				val.y = decode_double(&buf[sizeof(double)]);

				if (r_len) {
					(*r_len) += sizeof(double) * 2;
				}
			} else {
				ERR_FAIL_COND_V((size_t)len < sizeof(float) * 2, ERR_INVALID_DATA);
				val.x = decode_float(&buf[0]);
				val.y = decode_float(&buf[sizeof(float)]);

				if (r_len) {
					(*r_len) += sizeof(float) * 2;
				}
			}
			r_variant = val;

		} break;
		case Variant::VECTOR2I: {
			ERR_FAIL_COND_V(len < 4 * 2, ERR_INVALID_DATA);
			Vector2i val;
			val.x = decode_uint32(&buf[0]);
			val.y = decode_uint32(&buf[4]);
			r_variant = val;

			if (r_len) {
				(*r_len) += 4 * 2;
			}

		} break;
		case Variant::RECT2: {
			Rect2 val;
			if (type & ENCODE_FLAG_64) {
				ERR_FAIL_COND_V((size_t)len < sizeof(double) * 4, ERR_INVALID_DATA);
				val.position.x = decode_double(&buf[0]);
				val.position.y = decode_double(&buf[sizeof(double)]);
				val.size.x = decode_double(&buf[sizeof(double) * 2]);
				val.size.y = decode_double(&buf[sizeof(double) * 3]);

				if (r_len) {
					(*r_len) += sizeof(double) * 4;
				}
			} else {
				ERR_FAIL_COND_V((size_t)len < sizeof(float) * 4, ERR_INVALID_DATA);
				val.position.x = decode_float(&buf[0]);
				val.position.y = decode_float(&buf[sizeof(float)]);
				val.size.x = decode_float(&buf[sizeof(float) * 2]);
				val.size.y = decode_float(&buf[sizeof(float) * 3]);

				if (r_len) {
					(*r_len) += sizeof(float) * 4;
				}
			}
			r_variant = val;

		} break;
		case Variant::RECT2I: {
			ERR_FAIL_COND_V(len < 4 * 4, ERR_INVALID_DATA);
			Rect2i val;
			val.position.x = decode_uint32(&buf[0]);
			val.position.y = decode_uint32(&buf[4]);
			val.size.x = decode_uint32(&buf[8]);
			val.size.y = decode_uint32(&buf[12]);
			r_variant = val;

			if (r_len) {
				(*r_len) += 4 * 4;
			}

		} break;
		case Variant::VECTOR3: {
			Vector3 val;
			if (type & ENCODE_FLAG_64) {
				ERR_FAIL_COND_V((size_t)len < sizeof(double) * 3, ERR_INVALID_DATA);
				val.x = decode_double(&buf[0]);
				val.y = decode_double(&buf[sizeof(double)]);
				val.z = decode_double(&buf[sizeof(double) * 2]);

				if (r_len) {
					(*r_len) += sizeof(double) * 3;
				}
			} else {
				ERR_FAIL_COND_V((size_t)len < sizeof(float) * 3, ERR_INVALID_DATA);
				val.x = decode_float(&buf[0]);
				val.y = decode_float(&buf[sizeof(float)]);
				val.z = decode_float(&buf[sizeof(float) * 2]);

				if (r_len) {
					(*r_len) += sizeof(float) * 3;
				}
			}
			r_variant = val;

		} break;
		case Variant::VECTOR3I: {
			ERR_FAIL_COND_V(len < 4 * 3, ERR_INVALID_DATA);
			Vector3i val;
			val.x = decode_uint32(&buf[0]);
			val.y = decode_uint32(&buf[4]);
			val.z = decode_uint32(&buf[8]);
			r_variant = val;

			if (r_len) {
				(*r_len) += 4 * 3;
			}

		} break;
		case Variant::TRANSFORM2D: {
			Transform2D val;
			if (type & ENCODE_FLAG_64) {
				ERR_FAIL_COND_V((size_t)len < sizeof(double) * 6, ERR_INVALID_DATA);
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 2; j++) {
						val.elements[i][j] = decode_double(&buf[(i * 2 + j) * sizeof(double)]);
					}
				}

				if (r_len) {
					(*r_len) += sizeof(double) * 6;
				}
			} else {
				ERR_FAIL_COND_V((size_t)len < sizeof(float) * 6, ERR_INVALID_DATA);
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 2; j++) {
						val.elements[i][j] = decode_float(&buf[(i * 2 + j) * sizeof(float)]);
					}
				}

				if (r_len) {
					(*r_len) += sizeof(float) * 6;
				}
			}
			r_variant = val;

		} break;
		case Variant::PLANE: {
			Plane val;
			if (type & ENCODE_FLAG_64) {
				ERR_FAIL_COND_V((size_t)len < sizeof(double) * 4, ERR_INVALID_DATA);
				val.normal.x = decode_double(&buf[0]);
				val.normal.y = decode_double(&buf[sizeof(double)]);
				val.normal.z = decode_double(&buf[sizeof(double) * 2]);
				val.d = decode_double(&buf[sizeof(double) * 3]);

				if (r_len) {
					(*r_len) += sizeof(double) * 4;
				}
			} else {
				ERR_FAIL_COND_V((size_t)len < sizeof(float) * 4, ERR_INVALID_DATA);
				val.normal.x = decode_float(&buf[0]);
				val.normal.y = decode_float(&buf[sizeof(float)]);
				val.normal.z = decode_float(&buf[sizeof(float) * 2]);
				val.d = decode_float(&buf[sizeof(float) * 3]);

				if (r_len) {
					(*r_len) += sizeof(float) * 4;
				}
			}
			r_variant = val;

		} break;
		case Variant::QUATERNION: {
			Quaternion val;
			if (type & ENCODE_FLAG_64) {
				ERR_FAIL_COND_V((size_t)len < sizeof(double) * 4, ERR_INVALID_DATA);
				val.x = decode_double(&buf[0]);
				val.y = decode_double(&buf[sizeof(double)]);
				val.z = decode_double(&buf[sizeof(double) * 2]);
				val.w = decode_double(&buf[sizeof(double) * 3]);

				if (r_len) {
					(*r_len) += sizeof(double) * 4;
				}
			} else {
				ERR_FAIL_COND_V((size_t)len < sizeof(float) * 4, ERR_INVALID_DATA);
				val.x = decode_float(&buf[0]);
				val.y = decode_float(&buf[sizeof(float)]);
				val.z = decode_float(&buf[sizeof(float) * 2]);
				val.w = decode_float(&buf[sizeof(float) * 3]);

				if (r_len) {
					(*r_len) += sizeof(float) * 4;
				}
			}
			r_variant = val;

		} break;
		case Variant::AABB: {
			AABB val;
			if (type & ENCODE_FLAG_64) {
				ERR_FAIL_COND_V((size_t)len < sizeof(double) * 6, ERR_INVALID_DATA);
				val.position.x = decode_double(&buf[0]);
				val.position.y = decode_double(&buf[sizeof(double)]);
				val.position.z = decode_double(&buf[sizeof(double) * 2]);
				val.size.x = decode_double(&buf[sizeof(double) * 3]);
				val.size.y = decode_double(&buf[sizeof(double) * 4]);
				val.size.z = decode_double(&buf[sizeof(double) * 5]);

				if (r_len) {
					(*r_len) += sizeof(double) * 6;
				}
			} else {
				ERR_FAIL_COND_V((size_t)len < sizeof(float) * 6, ERR_INVALID_DATA);
				val.position.x = decode_float(&buf[0]);
				val.position.y = decode_float(&buf[sizeof(float)]);
				val.position.z = decode_float(&buf[sizeof(float) * 2]);
				val.size.x = decode_float(&buf[sizeof(float) * 3]);
				val.size.y = decode_float(&buf[sizeof(float) * 4]);
				val.size.z = decode_float(&buf[sizeof(float) * 5]);

				if (r_len) {
					(*r_len) += sizeof(float) * 6;
				}
			}
			r_variant = val;

		} break;
		case Variant::BASIS: {
			Basis val;
			if (type & ENCODE_FLAG_64) {
				ERR_FAIL_COND_V((size_t)len < sizeof(double) * 9, ERR_INVALID_DATA);
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {
						val.elements[i][j] = decode_double(&buf[(i * 3 + j) * sizeof(double)]);
					}
				}

				if (r_len) {
					(*r_len) += sizeof(double) * 9;
				}
			} else {
				ERR_FAIL_COND_V((size_t)len < sizeof(float) * 9, ERR_INVALID_DATA);
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {
						val.elements[i][j] = decode_float(&buf[(i * 3 + j) * sizeof(float)]);
					}
				}

				if (r_len) {
					(*r_len) += sizeof(float) * 9;
				}
			}
			r_variant = val;

		} break;
		case Variant::TRANSFORM3D: {
			Transform3D val;
			if (type & ENCODE_FLAG_64) {
				ERR_FAIL_COND_V((size_t)len < sizeof(double) * 12, ERR_INVALID_DATA);
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {
						val.basis.elements[i][j] = decode_double(&buf[(i * 3 + j) * sizeof(double)]);
					}
				}
				val.origin[0] = decode_double(&buf[sizeof(double) * 9]);
				val.origin[1] = decode_double(&buf[sizeof(double) * 10]);
				val.origin[2] = decode_double(&buf[sizeof(double) * 11]);

				if (r_len) {
					(*r_len) += sizeof(double) * 12;
				}
			} else {
				ERR_FAIL_COND_V((size_t)len < sizeof(float) * 12, ERR_INVALID_DATA);
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {
						val.basis.elements[i][j] = decode_float(&buf[(i * 3 + j) * sizeof(float)]);
					}
				}
				val.origin[0] = decode_float(&buf[sizeof(float) * 9]);
				val.origin[1] = decode_float(&buf[sizeof(float) * 10]);
				val.origin[2] = decode_float(&buf[sizeof(float) * 11]);

				if (r_len) {
					(*r_len) += sizeof(float) * 12;
				}
			}
			r_variant = val;

		} break;
		// misc types
		case Variant::COLOR: {
			ERR_FAIL_COND_V(len < 4 * 4, ERR_INVALID_DATA);
			Color val;
			val.r = decode_float(&buf[0]);
			val.g = decode_float(&buf[4]);
			val.b = decode_float(&buf[8]);
			val.a = decode_float(&buf[12]);
			r_variant = val;

			if (r_len) {
				(*r_len) += 4 * 4; // Colors should always be in single-precision.
			}
		} break;
		case Variant::STRING_NAME: {
			String str;
			Error err = _decode_string(buf, len, r_len, str);
			if (err) {
				return err;
			}
			r_variant = StringName(str);

		} break;

		case Variant::NODE_PATH: {
			ERR_FAIL_COND_V(len < 4, ERR_INVALID_DATA);
			int32_t strlen = decode_uint32(buf);

			if (strlen & 0x80000000) {
				//new format
				ERR_FAIL_COND_V(len < 12, ERR_INVALID_DATA);
				Vector<StringName> names;
				Vector<StringName> subnames;

				uint32_t namecount = strlen &= 0x7FFFFFFF;
				uint32_t subnamecount = decode_uint32(buf + 4);
				uint32_t flags = decode_uint32(buf + 8);

				len -= 12;
				buf += 12;

				if (flags & 2) { // Obsolete format with property separate from subpath
					subnamecount++;
				}

				uint32_t total = namecount + subnamecount;

				if (r_len) {
					(*r_len) += 12;
				}

				for (uint32_t i = 0; i < total; i++) {
					String str;
					Error err = _decode_string(buf, len, r_len, str);
					if (err) {
						return err;
					}

					if (i < namecount) {
						names.push_back(str);
					} else {
						subnames.push_back(str);
					}
				}

				r_variant = NodePath(names, subnames, flags & 1);

			} else {
				//old format, just a string

				ERR_FAIL_V(ERR_INVALID_DATA);
			}

		} break;
		case Variant::RID: {
			r_variant = RID();
		} break;
		case Variant::OBJECT: {
			if (type & ENCODE_FLAG_OBJECT_AS_ID) {
				//this _is_ allowed
				ERR_FAIL_COND_V(len < 8, ERR_INVALID_DATA);
				ObjectID val = ObjectID(decode_uint64(buf));
				if (r_len) {
					(*r_len) += 8;
				}

				if (val.is_null()) {
					r_variant = (Object *)nullptr;
				} else {
					Ref<EncodedObjectAsID> obj_as_id;
					obj_as_id.instantiate();
					obj_as_id->set_object_id(val);

					r_variant = obj_as_id;
				}

			} else {
				ERR_FAIL_COND_V(!p_allow_objects, ERR_UNAUTHORIZED);

				String str;
				Error err = _decode_string(buf, len, r_len, str);
				if (err) {
					return err;
				}

				if (str.is_empty()) {
					r_variant = (Object *)nullptr;
				} else {
					Object *obj = ClassDB::instantiate(str);

					ERR_FAIL_COND_V(!obj, ERR_UNAVAILABLE);
					ERR_FAIL_COND_V(len < 4, ERR_INVALID_DATA);

					int32_t count = decode_uint32(buf);
					buf += 4;
					len -= 4;
					if (r_len) {
						(*r_len) += 4; // Size of count number.
					}

					for (int i = 0; i < count; i++) {
						str = String();
						err = _decode_string(buf, len, r_len, str);
						if (err) {
							return err;
						}

						Variant value;
						int used;
						err = decode_variant(value, buf, len, &used, p_allow_objects);
						if (err) {
							return err;
						}

						buf += used;
						len -= used;
						if (r_len) {
							(*r_len) += used;
						}

						obj->set(str, value);
					}

					if (Object::cast_to<RefCounted>(obj)) {
						REF ref = REF(Object::cast_to<RefCounted>(obj));
						r_variant = ref;
					} else {
						r_variant = obj;
					}
				}
			}

		} break;
		case Variant::CALLABLE: {
			r_variant = Callable();
		} break;
		case Variant::SIGNAL: {
			r_variant = Signal();
		} break;

		case Variant::DICTIONARY: {
			ERR_FAIL_COND_V(len < 4, ERR_INVALID_DATA);
			int32_t count = decode_uint32(buf);
			//  bool shared = count&0x80000000;
			count &= 0x7FFFFFFF;

			buf += 4;
			len -= 4;

			if (r_len) {
				(*r_len) += 4; // Size of count number.
			}

			Dictionary d;

			for (int i = 0; i < count; i++) {
				Variant key, value;

				int used;
				Error err = decode_variant(key, buf, len, &used, p_allow_objects);
				ERR_FAIL_COND_V_MSG(err != OK, err, "Error when trying to decode Variant.");

				buf += used;
				len -= used;
				if (r_len) {
					(*r_len) += used;
				}

				err = decode_variant(value, buf, len, &used, p_allow_objects);
				ERR_FAIL_COND_V_MSG(err != OK, err, "Error when trying to decode Variant.");

				buf += used;
				len -= used;
				if (r_len) {
					(*r_len) += used;
				}

				d[key] = value;
			}

			r_variant = d;

		} break;
		case Variant::ARRAY: {
			ERR_FAIL_COND_V(len < 4, ERR_INVALID_DATA);
			int32_t count = decode_uint32(buf);
			//  bool shared = count&0x80000000;
			count &= 0x7FFFFFFF;

			buf += 4;
			len -= 4;

			if (r_len) {
				(*r_len) += 4; // Size of count number.
			}

			Array varr;

			for (int i = 0; i < count; i++) {
				int used = 0;
				Variant v;
				Error err = decode_variant(v, buf, len, &used, p_allow_objects);
				ERR_FAIL_COND_V_MSG(err != OK, err, "Error when trying to decode Variant.");
				buf += used;
				len -= used;
				varr.push_back(v);
				if (r_len) {
					(*r_len) += used;
				}
			}

			r_variant = varr;

		} break;

		// arrays
		case Variant::PACKED_BYTE_ARRAY: {
			ERR_FAIL_COND_V(len < 4, ERR_INVALID_DATA);
			int32_t count = decode_uint32(buf);
			buf += 4;
			len -= 4;
			ERR_FAIL_COND_V(count < 0 || count > len, ERR_INVALID_DATA);

			Vector<uint8_t> data;

			if (count) {
				data.resize(count);
				uint8_t *w = data.ptrw();
				for (int32_t i = 0; i < count; i++) {
					w[i] = buf[i];
				}
			}

			r_variant = data;

			if (r_len) {
				if (count % 4) {
					(*r_len) += 4 - count % 4;
				}
				(*r_len) += 4 + count;
			}

		} break;
		case Variant::PACKED_INT32_ARRAY: {
			ERR_FAIL_COND_V(len < 4, ERR_INVALID_DATA);
			int32_t count = decode_uint32(buf);
			buf += 4;
			len -= 4;
			ERR_FAIL_MUL_OF(count, 4, ERR_INVALID_DATA);
			ERR_FAIL_COND_V(count < 0 || count * 4 > len, ERR_INVALID_DATA);

			Vector<int32_t> data;

			if (count) {
				//const int*rbuf=(const int*)buf;
				data.resize(count);
				int32_t *w = data.ptrw();
				for (int32_t i = 0; i < count; i++) {
					w[i] = decode_uint32(&buf[i * 4]);
				}
			}
			r_variant = Variant(data);
			if (r_len) {
				(*r_len) += 4 + count * sizeof(int32_t);
			}

		} break;
		case Variant::PACKED_INT64_ARRAY: {
			ERR_FAIL_COND_V(len < 4, ERR_INVALID_DATA);
			int32_t count = decode_uint32(buf);
			buf += 4;
			len -= 4;
			ERR_FAIL_MUL_OF(count, 8, ERR_INVALID_DATA);
			ERR_FAIL_COND_V(count < 0 || count * 8 > len, ERR_INVALID_DATA);

			Vector<int64_t> data;

			if (count) {
				//const int*rbuf=(const int*)buf;
				data.resize(count);
				int64_t *w = data.ptrw();
				for (int64_t i = 0; i < count; i++) {
					w[i] = decode_uint64(&buf[i * 8]);
				}
			}
			r_variant = Variant(data);
			if (r_len) {
				(*r_len) += 4 + count * sizeof(int64_t);
			}

		} break;
		case Variant::PACKED_FLOAT32_ARRAY: {
			ERR_FAIL_COND_V(len < 4, ERR_INVALID_DATA);
			int32_t count = decode_uint32(buf);
			buf += 4;
			len -= 4;
			ERR_FAIL_MUL_OF(count, 4, ERR_INVALID_DATA);
			ERR_FAIL_COND_V(count < 0 || count * 4 > len, ERR_INVALID_DATA);

			Vector<float> data;

			if (count) {
				//const float*rbuf=(const float*)buf;
				data.resize(count);
				float *w = data.ptrw();
				for (int32_t i = 0; i < count; i++) {
					w[i] = decode_float(&buf[i * 4]);
				}
			}
			r_variant = data;

			if (r_len) {
				(*r_len) += 4 + count * sizeof(float);
			}

		} break;
		case Variant::PACKED_FLOAT64_ARRAY: {
			ERR_FAIL_COND_V(len < 4, ERR_INVALID_DATA);
			int32_t count = decode_uint32(buf);
			buf += 4;
			len -= 4;
			ERR_FAIL_MUL_OF(count, 8, ERR_INVALID_DATA);
			ERR_FAIL_COND_V(count < 0 || count * 8 > len, ERR_INVALID_DATA);

			Vector<double> data;

			if (count) {
				data.resize(count);
				double *w = data.ptrw();
				for (int64_t i = 0; i < count; i++) {
					w[i] = decode_double(&buf[i * 8]);
				}
			}
			r_variant = data;

			if (r_len) {
				(*r_len) += 4 + count * sizeof(double);
			}

		} break;
		case Variant::PACKED_STRING_ARRAY: {
			ERR_FAIL_COND_V(len < 4, ERR_INVALID_DATA);
			int32_t count = decode_uint32(buf);

			Vector<String> strings;
			buf += 4;
			len -= 4;

			if (r_len) {
				(*r_len) += 4; // Size of count number.
			}

			for (int32_t i = 0; i < count; i++) {
				String str;
				Error err = _decode_string(buf, len, r_len, str);
				if (err) {
					return err;
				}

				strings.push_back(str);
			}

			r_variant = strings;

		} break;
		case Variant::PACKED_VECTOR2_ARRAY: {
			ERR_FAIL_COND_V(len < 4, ERR_INVALID_DATA);
			int32_t count = decode_uint32(buf);
			buf += 4;
			len -= 4;

			Vector<Vector2> varray;

			if (type & ENCODE_FLAG_64) {
				ERR_FAIL_MUL_OF(count, sizeof(double) * 2, ERR_INVALID_DATA);
				ERR_FAIL_COND_V(count < 0 || count * sizeof(double) * 2 > (size_t)len, ERR_INVALID_DATA);

				if (r_len) {
					(*r_len) += 4; // Size of count number.
				}

				if (count) {
					varray.resize(count);
					Vector2 *w = varray.ptrw();

					for (int32_t i = 0; i < count; i++) {
						w[i].x = decode_double(buf + i * sizeof(double) * 2 + sizeof(double) * 0);
						w[i].y = decode_double(buf + i * sizeof(double) * 2 + sizeof(double) * 1);
					}

					int adv = sizeof(double) * 2 * count;

					if (r_len) {
						(*r_len) += adv;
					}
					len -= adv;
					buf += adv;
				}
			} else {
				ERR_FAIL_MUL_OF(count, sizeof(float) * 2, ERR_INVALID_DATA);
				ERR_FAIL_COND_V(count < 0 || count * sizeof(float) * 2 > (size_t)len, ERR_INVALID_DATA);

				if (r_len) {
					(*r_len) += 4; // Size of count number.
				}

				if (count) {
					varray.resize(count);
					Vector2 *w = varray.ptrw();

					for (int32_t i = 0; i < count; i++) {
						w[i].x = decode_float(buf + i * sizeof(float) * 2 + sizeof(float) * 0);
						w[i].y = decode_float(buf + i * sizeof(float) * 2 + sizeof(float) * 1);
					}

					int adv = sizeof(float) * 2 * count;

					if (r_len) {
						(*r_len) += adv;
					}
				}
			}
			r_variant = varray;

		} break;
		case Variant::PACKED_VECTOR3_ARRAY: {
			ERR_FAIL_COND_V(len < 4, ERR_INVALID_DATA);
			int32_t count = decode_uint32(buf);
			buf += 4;
			len -= 4;

			Vector<Vector3> varray;

			if (type & ENCODE_FLAG_64) {
				ERR_FAIL_MUL_OF(count, sizeof(double) * 3, ERR_INVALID_DATA);
				ERR_FAIL_COND_V(count < 0 || count * sizeof(double) * 3 > (size_t)len, ERR_INVALID_DATA);

				if (r_len) {
					(*r_len) += 4; // Size of count number.
				}

				if (count) {
					varray.resize(count);
					Vector3 *w = varray.ptrw();

					for (int32_t i = 0; i < count; i++) {
						w[i].x = decode_double(buf + i * sizeof(double) * 3 + sizeof(double) * 0);
						w[i].y = decode_double(buf + i * sizeof(double) * 3 + sizeof(double) * 1);
						w[i].z = decode_double(buf + i * sizeof(double) * 3 + sizeof(double) * 2);
					}

					int adv = sizeof(double) * 3 * count;

					if (r_len) {
						(*r_len) += adv;
					}
					len -= adv;
					buf += adv;
				}
			} else {
				ERR_FAIL_MUL_OF(count, sizeof(float) * 3, ERR_INVALID_DATA);
				ERR_FAIL_COND_V(count < 0 || count * sizeof(float) * 3 > (size_t)len, ERR_INVALID_DATA);

				if (r_len) {
					(*r_len) += 4; // Size of count number.
				}

				if (count) {
					varray.resize(count);
					Vector3 *w = varray.ptrw();

					for (int32_t i = 0; i < count; i++) {
						w[i].x = decode_float(buf + i * sizeof(float) * 3 + sizeof(float) * 0);
						w[i].y = decode_float(buf + i * sizeof(float) * 3 + sizeof(float) * 1);
						w[i].z = decode_float(buf + i * sizeof(float) * 3 + sizeof(float) * 2);
					}

					int adv = sizeof(float) * 3 * count;

					if (r_len) {
						(*r_len) += adv;
					}
					len -= adv;
					buf += adv;
				}
			}
			r_variant = varray;

		} break;
		case Variant::PACKED_COLOR_ARRAY: {
			ERR_FAIL_COND_V(len < 4, ERR_INVALID_DATA);
			int32_t count = decode_uint32(buf);
			buf += 4;
			len -= 4;

			ERR_FAIL_MUL_OF(count, 4 * 4, ERR_INVALID_DATA);
			ERR_FAIL_COND_V(count < 0 || count * 4 * 4 > len, ERR_INVALID_DATA);

			Vector<Color> carray;

			if (r_len) {
				(*r_len) += 4; // Size of count number.
			}

			if (count) {
				carray.resize(count);
				Color *w = carray.ptrw();

				for (int32_t i = 0; i < count; i++) {
					// Colors should always be in single-precision.
					w[i].r = decode_float(buf + i * 4 * 4 + 4 * 0);
					w[i].g = decode_float(buf + i * 4 * 4 + 4 * 1);
					w[i].b = decode_float(buf + i * 4 * 4 + 4 * 2);
					w[i].a = decode_float(buf + i * 4 * 4 + 4 * 3);
				}

				int adv = 4 * 4 * count;

				if (r_len) {
					(*r_len) += adv;
				}
			}

			r_variant = carray;

		} break;
		default: {
			ERR_FAIL_V(ERR_BUG);
		}
	}

	return OK;
}

static void _encode_string(const String &p_string, uint8_t *&buf, int &r_len) {
	CharString utf8 = p_string.utf8();

	if (buf) {
		encode_uint32(utf8.length(), buf);
		buf += 4;
		memcpy(buf, utf8.get_data(), utf8.length());
		buf += utf8.length();
	}

	r_len += 4 + utf8.length();
	while (r_len % 4) {
		r_len++; //pad
		if (buf) {
			*(buf++) = 0;
		}
	}
}

Error encode_variant(const Variant &p_variant, uint8_t *r_buffer, int &r_len, bool p_full_objects, int p_depth) {
	ERR_FAIL_COND_V_MSG(p_depth > Variant::MAX_RECURSION_DEPTH, ERR_OUT_OF_MEMORY, "Potential inifite recursion detected. Bailing.");
	uint8_t *buf = r_buffer;

	r_len = 0;

	uint32_t flags = 0;

	switch (p_variant.get_type()) {
		case Variant::INT: {
			int64_t val = p_variant;
			if (val > (int64_t)INT_MAX || val < (int64_t)INT_MIN) {
				flags |= ENCODE_FLAG_64;
			}
		} break;
		case Variant::FLOAT: {
			double d = p_variant;
			float f = d;
			if (double(f) != d) {
				flags |= ENCODE_FLAG_64;
			}
		} break;
		case Variant::OBJECT: {
			// Test for potential wrong values sent by the debugger when it breaks.
			Object *obj = p_variant.get_validated_object();
			if (!obj) {
				// Object is invalid, send a nullptr instead.
				if (buf) {
					encode_uint32(Variant::NIL, buf);
				}
				r_len += 4;
				return OK;
			}

			if (!p_full_objects) {
				flags |= ENCODE_FLAG_OBJECT_AS_ID;
			}
		} break;
		default: {
		} // nothing to do at this stage
	}

	if (buf) {
		encode_uint32(p_variant.get_type() | flags, buf);
		buf += 4;
	}
	r_len += 4;

	switch (p_variant.get_type()) {
		case Variant::NIL: {
			//nothing to do
		} break;
		case Variant::BOOL: {
			if (buf) {
				encode_uint32(p_variant.operator bool(), buf);
			}

			r_len += 4;

		} break;
		case Variant::INT: {
			if (flags & ENCODE_FLAG_64) {
				//64 bits
				if (buf) {
					encode_uint64(p_variant.operator int64_t(), buf);
				}

				r_len += 8;
			} else {
				if (buf) {
					encode_uint32(p_variant.operator int32_t(), buf);
				}

				r_len += 4;
			}
		} break;
		case Variant::FLOAT: {
			if (flags & ENCODE_FLAG_64) {
				if (buf) {
					encode_double(p_variant.operator double(), buf);
				}

				r_len += 8;

			} else {
				if (buf) {
					encode_float(p_variant.operator float(), buf);
				}

				r_len += 4;
			}

		} break;
		case Variant::NODE_PATH: {
			NodePath np = p_variant;
			if (buf) {
				encode_uint32(uint32_t(np.get_name_count()) | 0x80000000, buf); //for compatibility with the old format
				encode_uint32(np.get_subname_count(), buf + 4);
				uint32_t np_flags = 0;
				if (np.is_absolute()) {
					np_flags |= 1;
				}

				encode_uint32(np_flags, buf + 8);

				buf += 12;
			}

			r_len += 12;

			int total = np.get_name_count() + np.get_subname_count();

			for (int i = 0; i < total; i++) {
				String str;

				if (i < np.get_name_count()) {
					str = np.get_name(i);
				} else {
					str = np.get_subname(i - np.get_name_count());
				}

				CharString utf8 = str.utf8();

				int pad = 0;

				if (utf8.length() % 4) {
					pad = 4 - utf8.length() % 4;
				}

				if (buf) {
					encode_uint32(utf8.length(), buf);
					buf += 4;
					memcpy(buf, utf8.get_data(), utf8.length());
					buf += pad + utf8.length();
				}

				r_len += 4 + utf8.length() + pad;
			}

		} break;
		case Variant::STRING:
		case Variant::STRING_NAME: {
			_encode_string(p_variant, buf, r_len);

		} break;

		// math types
		case Variant::VECTOR2: {
			if (buf) {
				Vector2 v2 = p_variant;
				encode_real(v2.x, &buf[0]);
				encode_real(v2.y, &buf[sizeof(real_t)]);
			}

			r_len += 2 * sizeof(real_t);

		} break;
		case Variant::VECTOR2I: {
			if (buf) {
				Vector2i v2 = p_variant;
				encode_uint32(v2.x, &buf[0]);
				encode_uint32(v2.y, &buf[4]);
			}

			r_len += 2 * 4;

		} break;
		case Variant::RECT2: {
			if (buf) {
				Rect2 r2 = p_variant;
				encode_real(r2.position.x, &buf[0]);
				encode_real(r2.position.y, &buf[sizeof(real_t)]);
				encode_real(r2.size.x, &buf[sizeof(real_t) * 2]);
				encode_real(r2.size.y, &buf[sizeof(real_t) * 3]);
			}
			r_len += 4 * sizeof(real_t);

		} break;
		case Variant::RECT2I: {
			if (buf) {
				Rect2i r2 = p_variant;
				encode_uint32(r2.position.x, &buf[0]);
				encode_uint32(r2.position.y, &buf[4]);
				encode_uint32(r2.size.x, &buf[8]);
				encode_uint32(r2.size.y, &buf[12]);
			}
			r_len += 4 * 4;

		} break;
		case Variant::VECTOR3: {
			if (buf) {
				Vector3 v3 = p_variant;
				encode_real(v3.x, &buf[0]);
				encode_real(v3.y, &buf[sizeof(real_t)]);
				encode_real(v3.z, &buf[sizeof(real_t) * 2]);
			}

			r_len += 3 * sizeof(real_t);

		} break;
		case Variant::VECTOR3I: {
			if (buf) {
				Vector3i v3 = p_variant;
				encode_uint32(v3.x, &buf[0]);
				encode_uint32(v3.y, &buf[4]);
				encode_uint32(v3.z, &buf[8]);
			}

			r_len += 3 * 4;

		} break;
		case Variant::TRANSFORM2D: {
			if (buf) {
				Transform2D val = p_variant;
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 2; j++) {
						memcpy(&buf[(i * 2 + j) * sizeof(real_t)], &val.elements[i][j], sizeof(real_t));
					}
				}
			}

			r_len += 6 * sizeof(real_t);

		} break;
		case Variant::PLANE: {
			if (buf) {
				Plane p = p_variant;
				encode_real(p.normal.x, &buf[0]);
				encode_real(p.normal.y, &buf[sizeof(real_t)]);
				encode_real(p.normal.z, &buf[sizeof(real_t) * 2]);
				encode_real(p.d, &buf[sizeof(real_t) * 3]);
			}

			r_len += 4 * sizeof(real_t);

		} break;
		case Variant::QUATERNION: {
			if (buf) {
				Quaternion q = p_variant;
				encode_real(q.x, &buf[0]);
				encode_real(q.y, &buf[sizeof(real_t)]);
				encode_real(q.z, &buf[sizeof(real_t) * 2]);
				encode_real(q.w, &buf[sizeof(real_t) * 3]);
			}

			r_len += 4 * sizeof(real_t);

		} break;
		case Variant::AABB: {
			if (buf) {
				AABB aabb = p_variant;
				encode_real(aabb.position.x, &buf[0]);
				encode_real(aabb.position.y, &buf[sizeof(real_t)]);
				encode_real(aabb.position.z, &buf[sizeof(real_t) * 2]);
				encode_real(aabb.size.x, &buf[sizeof(real_t) * 3]);
				encode_real(aabb.size.y, &buf[sizeof(real_t) * 4]);
				encode_real(aabb.size.z, &buf[sizeof(real_t) * 5]);
			}

			r_len += 6 * sizeof(real_t);

		} break;
		case Variant::BASIS: {
			if (buf) {
				Basis val = p_variant;
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {
						memcpy(&buf[(i * 3 + j) * sizeof(real_t)], &val.elements[i][j], sizeof(real_t));
					}
				}
			}

			r_len += 9 * sizeof(real_t);

		} break;
		case Variant::TRANSFORM3D: {
			if (buf) {
				Transform3D val = p_variant;
				for (int i = 0; i < 3; i++) {
					for (int j = 0; j < 3; j++) {
						memcpy(&buf[(i * 3 + j) * sizeof(real_t)], &val.basis.elements[i][j], sizeof(real_t));
					}
				}

				encode_real(val.origin.x, &buf[sizeof(real_t) * 9]);
				encode_real(val.origin.y, &buf[sizeof(real_t) * 10]);
				encode_real(val.origin.z, &buf[sizeof(real_t) * 11]);
			}

			r_len += 12 * sizeof(real_t);

		} break;

		// misc types
		case Variant::COLOR: {
			if (buf) {
				Color c = p_variant;
				encode_float(c.r, &buf[0]);
				encode_float(c.g, &buf[4]);
				encode_float(c.b, &buf[8]);
				encode_float(c.a, &buf[12]);
			}

			r_len += 4 * 4; // Colors should always be in single-precision.

		} break;
		case Variant::RID: {
		} break;
		case Variant::CALLABLE: {
		} break;
		case Variant::SIGNAL: {
		} break;
		case Variant::OBJECT: {
			if (p_full_objects) {
				Object *obj = p_variant;
				if (!obj) {
					if (buf) {
						encode_uint32(0, buf);
					}
					r_len += 4;

				} else {
					_encode_string(obj->get_class(), buf, r_len);

					List<PropertyInfo> props;
					obj->get_property_list(&props);

					int pc = 0;
					for (const PropertyInfo &E : props) {
						if (!(E.usage & PROPERTY_USAGE_STORAGE)) {
							continue;
						}
						pc++;
					}

					if (buf) {
						encode_uint32(pc, buf);
						buf += 4;
					}

					r_len += 4;

					for (const PropertyInfo &E : props) {
						if (!(E.usage & PROPERTY_USAGE_STORAGE)) {
							continue;
						}

						_encode_string(E.name, buf, r_len);

						int len;
						Error err = encode_variant(obj->get(E.name), buf, len, p_full_objects, p_depth + 1);
						ERR_FAIL_COND_V(err, err);
						ERR_FAIL_COND_V(len % 4, ERR_BUG);
						r_len += len;
						if (buf) {
							buf += len;
						}
					}
				}
			} else {
				if (buf) {
					Object *obj = p_variant.get_validated_object();
					ObjectID id;
					if (obj) {
						id = obj->get_instance_id();
					}

					encode_uint64(id, buf);
				}

				r_len += 8;
			}

		} break;
		case Variant::DICTIONARY: {
			Dictionary d = p_variant;

			if (buf) {
				encode_uint32(uint32_t(d.size()), buf);
				buf += 4;
			}
			r_len += 4;

			List<Variant> keys;
			d.get_key_list(&keys);

			for (const Variant &E : keys) {
				/*
				CharString utf8 = E->->utf8();

				if (buf) {
					encode_uint32(utf8.length()+1,buf);
					buf+=4;
					memcpy(buf,utf8.get_data(),utf8.length()+1);
				}

				r_len+=4+utf8.length()+1;
				while (r_len%4)
					r_len++; //pad
				*/
				int len;
				Error err = encode_variant(E, buf, len, p_full_objects, p_depth + 1);
				ERR_FAIL_COND_V(err, err);
				ERR_FAIL_COND_V(len % 4, ERR_BUG);
				r_len += len;
				if (buf) {
					buf += len;
				}
				Variant *v = d.getptr(E);
				ERR_FAIL_COND_V(!v, ERR_BUG);
				err = encode_variant(*v, buf, len, p_full_objects, p_depth + 1);
				ERR_FAIL_COND_V(err, err);
				ERR_FAIL_COND_V(len % 4, ERR_BUG);
				r_len += len;
				if (buf) {
					buf += len;
				}
			}

		} break;
		case Variant::ARRAY: {
			Array v = p_variant;

			if (buf) {
				encode_uint32(uint32_t(v.size()), buf);
				buf += 4;
			}

			r_len += 4;

			for (int i = 0; i < v.size(); i++) {
				int len;
				Error err = encode_variant(v.get(i), buf, len, p_full_objects, p_depth + 1);
				ERR_FAIL_COND_V(err, err);
				ERR_FAIL_COND_V(len % 4, ERR_BUG);
				r_len += len;
				if (buf) {
					buf += len;
				}
			}

		} break;
		// arrays
		case Variant::PACKED_BYTE_ARRAY: {
			Vector<uint8_t> data = p_variant;
			int datalen = data.size();
			int datasize = sizeof(uint8_t);

			if (buf) {
				encode_uint32(datalen, buf);
				buf += 4;
				const uint8_t *r = data.ptr();
				memcpy(buf, &r[0], datalen * datasize);
				buf += datalen * datasize;
			}

			r_len += 4 + datalen * datasize;
			while (r_len % 4) {
				r_len++;
				if (buf) {
					*(buf++) = 0;
				}
			}

		} break;
		case Variant::PACKED_INT32_ARRAY: {
			Vector<int32_t> data = p_variant;
			int datalen = data.size();
			int datasize = sizeof(int32_t);

			if (buf) {
				encode_uint32(datalen, buf);
				buf += 4;
				const int32_t *r = data.ptr();
				for (int32_t i = 0; i < datalen; i++) {
					encode_uint32(r[i], &buf[i * datasize]);
				}
			}

			r_len += 4 + datalen * datasize;

		} break;
		case Variant::PACKED_INT64_ARRAY: {
			Vector<int64_t> data = p_variant;
			int datalen = data.size();
			int datasize = sizeof(int64_t);

			if (buf) {
				encode_uint32(datalen, buf);
				buf += 4;
				const int64_t *r = data.ptr();
				for (int64_t i = 0; i < datalen; i++) {
					encode_uint64(r[i], &buf[i * datasize]);
				}
			}

			r_len += 4 + datalen * datasize;

		} break;
		case Variant::PACKED_FLOAT32_ARRAY: {
			Vector<float> data = p_variant;
			int datalen = data.size();
			int datasize = sizeof(float);

			if (buf) {
				encode_uint32(datalen, buf);
				buf += 4;
				const float *r = data.ptr();
				for (int i = 0; i < datalen; i++) {
					encode_float(r[i], &buf[i * datasize]);
				}
			}

			r_len += 4 + datalen * datasize;

		} break;
		case Variant::PACKED_FLOAT64_ARRAY: {
			Vector<double> data = p_variant;
			int datalen = data.size();
			int datasize = sizeof(double);

			if (buf) {
				encode_uint32(datalen, buf);
				buf += 4;
				const double *r = data.ptr();
				for (int i = 0; i < datalen; i++) {
					encode_double(r[i], &buf[i * datasize]);
				}
			}

			r_len += 4 + datalen * datasize;

		} break;
		case Variant::PACKED_STRING_ARRAY: {
			Vector<String> data = p_variant;
			int len = data.size();

			if (buf) {
				encode_uint32(len, buf);
				buf += 4;
			}

			r_len += 4;

			for (int i = 0; i < len; i++) {
				CharString utf8 = data.get(i).utf8();

				if (buf) {
					encode_uint32(utf8.length() + 1, buf);
					buf += 4;
					memcpy(buf, utf8.get_data(), utf8.length() + 1);
					buf += utf8.length() + 1;
				}

				r_len += 4 + utf8.length() + 1;
				while (r_len % 4) {
					r_len++; //pad
					if (buf) {
						*(buf++) = 0;
					}
				}
			}

		} break;
		case Variant::PACKED_VECTOR2_ARRAY: {
			Vector<Vector2> data = p_variant;
			int len = data.size();

			if (buf) {
				encode_uint32(len, buf);
				buf += 4;
			}

			r_len += 4;

			if (buf) {
				for (int i = 0; i < len; i++) {
					Vector2 v = data.get(i);

					encode_real(v.x, &buf[0]);
					encode_real(v.y, &buf[sizeof(real_t)]);
					buf += sizeof(real_t) * 2;
				}
			}

			r_len += sizeof(real_t) * 2 * len;

		} break;
		case Variant::PACKED_VECTOR3_ARRAY: {
			Vector<Vector3> data = p_variant;
			int len = data.size();

			if (buf) {
				encode_uint32(len, buf);
				buf += 4;
			}

			r_len += 4;

			if (buf) {
				for (int i = 0; i < len; i++) {
					Vector3 v = data.get(i);

					encode_real(v.x, &buf[0]);
					encode_real(v.y, &buf[sizeof(real_t)]);
					encode_real(v.z, &buf[sizeof(real_t) * 2]);
					buf += sizeof(real_t) * 3;
				}
			}

			r_len += sizeof(real_t) * 3 * len;

		} break;
		case Variant::PACKED_COLOR_ARRAY: {
			Vector<Color> data = p_variant;
			int len = data.size();

			if (buf) {
				encode_uint32(len, buf);
				buf += 4;
			}

			r_len += 4;

			if (buf) {
				for (int i = 0; i < len; i++) {
					Color c = data.get(i);

					encode_float(c.r, &buf[0]);
					encode_float(c.g, &buf[4]);
					encode_float(c.b, &buf[8]);
					encode_float(c.a, &buf[12]);
					buf += 4 * 4; // Colors should always be in single-precision.
				}
			}

			r_len += 4 * 4 * len;

		} break;
		default: {
			ERR_FAIL_V(ERR_BUG);
		}
	}

	return OK;
}
