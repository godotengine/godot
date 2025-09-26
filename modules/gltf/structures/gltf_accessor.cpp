/**************************************************************************/
/*  gltf_accessor.cpp                                                     */
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

#include "gltf_accessor.h"
#include "gltf_accessor.compat.inc"

#include "../gltf_state.h"

void GLTFAccessor::_bind_methods() {
	BIND_ENUM_CONSTANT(TYPE_SCALAR);
	BIND_ENUM_CONSTANT(TYPE_VEC2);
	BIND_ENUM_CONSTANT(TYPE_VEC3);
	BIND_ENUM_CONSTANT(TYPE_VEC4);
	BIND_ENUM_CONSTANT(TYPE_MAT2);
	BIND_ENUM_CONSTANT(TYPE_MAT3);
	BIND_ENUM_CONSTANT(TYPE_MAT4);

	BIND_ENUM_CONSTANT(COMPONENT_TYPE_NONE);
	BIND_ENUM_CONSTANT(COMPONENT_TYPE_SIGNED_BYTE);
	BIND_ENUM_CONSTANT(COMPONENT_TYPE_UNSIGNED_BYTE);
	BIND_ENUM_CONSTANT(COMPONENT_TYPE_SIGNED_SHORT);
	BIND_ENUM_CONSTANT(COMPONENT_TYPE_UNSIGNED_SHORT);
	BIND_ENUM_CONSTANT(COMPONENT_TYPE_SIGNED_INT);
	BIND_ENUM_CONSTANT(COMPONENT_TYPE_UNSIGNED_INT);
	BIND_ENUM_CONSTANT(COMPONENT_TYPE_SINGLE_FLOAT);
	BIND_ENUM_CONSTANT(COMPONENT_TYPE_DOUBLE_FLOAT);
	BIND_ENUM_CONSTANT(COMPONENT_TYPE_HALF_FLOAT);
	BIND_ENUM_CONSTANT(COMPONENT_TYPE_SIGNED_LONG);
	BIND_ENUM_CONSTANT(COMPONENT_TYPE_UNSIGNED_LONG);

	ClassDB::bind_static_method("GLTFAccessor", D_METHOD("from_dictionary", "dictionary"), &GLTFAccessor::from_dictionary);
	ClassDB::bind_method(D_METHOD("to_dictionary"), &GLTFAccessor::to_dictionary);

	ClassDB::bind_method(D_METHOD("get_buffer_view"), &GLTFAccessor::get_buffer_view);
	ClassDB::bind_method(D_METHOD("set_buffer_view", "buffer_view"), &GLTFAccessor::set_buffer_view);
	ClassDB::bind_method(D_METHOD("get_byte_offset"), &GLTFAccessor::get_byte_offset);
	ClassDB::bind_method(D_METHOD("set_byte_offset", "byte_offset"), &GLTFAccessor::set_byte_offset);
	ClassDB::bind_method(D_METHOD("get_component_type"), &GLTFAccessor::get_component_type);
	ClassDB::bind_method(D_METHOD("set_component_type", "component_type"), &GLTFAccessor::set_component_type);
	ClassDB::bind_method(D_METHOD("get_normalized"), &GLTFAccessor::get_normalized);
	ClassDB::bind_method(D_METHOD("set_normalized", "normalized"), &GLTFAccessor::set_normalized);
	ClassDB::bind_method(D_METHOD("get_count"), &GLTFAccessor::get_count);
	ClassDB::bind_method(D_METHOD("set_count", "count"), &GLTFAccessor::set_count);
	ClassDB::bind_method(D_METHOD("get_accessor_type"), &GLTFAccessor::get_accessor_type);
	ClassDB::bind_method(D_METHOD("set_accessor_type", "accessor_type"), &GLTFAccessor::set_accessor_type);
	ClassDB::bind_method(D_METHOD("get_type"), &GLTFAccessor::get_type);
	ClassDB::bind_method(D_METHOD("set_type", "type"), &GLTFAccessor::set_type);
	ClassDB::bind_method(D_METHOD("get_min"), &GLTFAccessor::get_min);
	ClassDB::bind_method(D_METHOD("set_min", "min"), &GLTFAccessor::set_min);
	ClassDB::bind_method(D_METHOD("get_max"), &GLTFAccessor::get_max);
	ClassDB::bind_method(D_METHOD("set_max", "max"), &GLTFAccessor::set_max);
	ClassDB::bind_method(D_METHOD("get_sparse_count"), &GLTFAccessor::get_sparse_count);
	ClassDB::bind_method(D_METHOD("set_sparse_count", "sparse_count"), &GLTFAccessor::set_sparse_count);
	ClassDB::bind_method(D_METHOD("get_sparse_indices_buffer_view"), &GLTFAccessor::get_sparse_indices_buffer_view);
	ClassDB::bind_method(D_METHOD("set_sparse_indices_buffer_view", "sparse_indices_buffer_view"), &GLTFAccessor::set_sparse_indices_buffer_view);
	ClassDB::bind_method(D_METHOD("get_sparse_indices_byte_offset"), &GLTFAccessor::get_sparse_indices_byte_offset);
	ClassDB::bind_method(D_METHOD("set_sparse_indices_byte_offset", "sparse_indices_byte_offset"), &GLTFAccessor::set_sparse_indices_byte_offset);
	ClassDB::bind_method(D_METHOD("get_sparse_indices_component_type"), &GLTFAccessor::get_sparse_indices_component_type);
	ClassDB::bind_method(D_METHOD("set_sparse_indices_component_type", "sparse_indices_component_type"), &GLTFAccessor::set_sparse_indices_component_type);
	ClassDB::bind_method(D_METHOD("get_sparse_values_buffer_view"), &GLTFAccessor::get_sparse_values_buffer_view);
	ClassDB::bind_method(D_METHOD("set_sparse_values_buffer_view", "sparse_values_buffer_view"), &GLTFAccessor::set_sparse_values_buffer_view);
	ClassDB::bind_method(D_METHOD("get_sparse_values_byte_offset"), &GLTFAccessor::get_sparse_values_byte_offset);
	ClassDB::bind_method(D_METHOD("set_sparse_values_byte_offset", "sparse_values_byte_offset"), &GLTFAccessor::set_sparse_values_byte_offset);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "buffer_view"), "set_buffer_view", "get_buffer_view"); // GLTFBufferViewIndex
	ADD_PROPERTY(PropertyInfo(Variant::INT, "byte_offset"), "set_byte_offset", "get_byte_offset"); // int
	ADD_PROPERTY(PropertyInfo(Variant::INT, "component_type"), "set_component_type", "get_component_type"); // int
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "normalized"), "set_normalized", "get_normalized"); // bool
	ADD_PROPERTY(PropertyInfo(Variant::INT, "count"), "set_count", "get_count"); // int
	ADD_PROPERTY(PropertyInfo(Variant::INT, "accessor_type"), "set_accessor_type", "get_accessor_type"); // GLTFAccessor::GLTFAccessorType
	ADD_PROPERTY(PropertyInfo(Variant::INT, "type", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_type", "get_type"); // Deprecated, int for GLTFAccessor::GLTFAccessorType
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT64_ARRAY, "min"), "set_min", "get_min"); // Vector<real_t>
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT64_ARRAY, "max"), "set_max", "get_max"); // Vector<real_t>
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sparse_count"), "set_sparse_count", "get_sparse_count"); // int
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sparse_indices_buffer_view"), "set_sparse_indices_buffer_view", "get_sparse_indices_buffer_view"); // int
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sparse_indices_byte_offset"), "set_sparse_indices_byte_offset", "get_sparse_indices_byte_offset"); // int
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sparse_indices_component_type"), "set_sparse_indices_component_type", "get_sparse_indices_component_type"); // int
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sparse_values_buffer_view"), "set_sparse_values_buffer_view", "get_sparse_values_buffer_view"); // int
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sparse_values_byte_offset"), "set_sparse_values_byte_offset", "get_sparse_values_byte_offset"); // int
}

// Property getters and setters.

GLTFBufferViewIndex GLTFAccessor::get_buffer_view() const {
	return buffer_view;
}

void GLTFAccessor::set_buffer_view(GLTFBufferViewIndex p_buffer_view) {
	buffer_view = p_buffer_view;
}

int64_t GLTFAccessor::get_byte_offset() const {
	return byte_offset;
}

void GLTFAccessor::set_byte_offset(int64_t p_byte_offset) {
	byte_offset = p_byte_offset;
}

GLTFAccessor::GLTFComponentType GLTFAccessor::get_component_type() const {
	return component_type;
}

void GLTFAccessor::set_component_type(GLTFComponentType p_component_type) {
	component_type = (GLTFComponentType)p_component_type;
}

bool GLTFAccessor::get_normalized() const {
	return normalized;
}

void GLTFAccessor::set_normalized(bool p_normalized) {
	normalized = p_normalized;
}

int64_t GLTFAccessor::get_count() const {
	return count;
}

void GLTFAccessor::set_count(int64_t p_count) {
	count = p_count;
}

GLTFAccessor::GLTFAccessorType GLTFAccessor::get_accessor_type() const {
	return accessor_type;
}

void GLTFAccessor::set_accessor_type(GLTFAccessorType p_accessor_type) {
	accessor_type = p_accessor_type;
}

int GLTFAccessor::get_type() const {
	return (int)accessor_type;
}

void GLTFAccessor::set_type(int p_accessor_type) {
	accessor_type = (GLTFAccessorType)p_accessor_type; // TODO: Register enum
}

Vector<double> GLTFAccessor::get_min() const {
	return Vector<double>(min);
}

void GLTFAccessor::set_min(const Vector<double> &p_min) {
	min = Vector<double>(p_min);
}

Vector<double> GLTFAccessor::get_max() const {
	return Vector<double>(max);
}

void GLTFAccessor::set_max(const Vector<double> &p_max) {
	max = Vector<double>(p_max);
}

int64_t GLTFAccessor::get_sparse_count() const {
	return sparse_count;
}

void GLTFAccessor::set_sparse_count(int64_t p_sparse_count) {
	sparse_count = p_sparse_count;
}

GLTFBufferViewIndex GLTFAccessor::get_sparse_indices_buffer_view() const {
	return sparse_indices_buffer_view;
}

void GLTFAccessor::set_sparse_indices_buffer_view(GLTFBufferViewIndex p_sparse_indices_buffer_view) {
	sparse_indices_buffer_view = p_sparse_indices_buffer_view;
}

int64_t GLTFAccessor::get_sparse_indices_byte_offset() const {
	return sparse_indices_byte_offset;
}

void GLTFAccessor::set_sparse_indices_byte_offset(int64_t p_sparse_indices_byte_offset) {
	sparse_indices_byte_offset = p_sparse_indices_byte_offset;
}

GLTFAccessor::GLTFComponentType GLTFAccessor::get_sparse_indices_component_type() const {
	return sparse_indices_component_type;
}

void GLTFAccessor::set_sparse_indices_component_type(GLTFComponentType p_sparse_indices_component_type) {
	sparse_indices_component_type = (GLTFComponentType)p_sparse_indices_component_type;
}

GLTFBufferViewIndex GLTFAccessor::get_sparse_values_buffer_view() const {
	return sparse_values_buffer_view;
}

void GLTFAccessor::set_sparse_values_buffer_view(GLTFBufferViewIndex p_sparse_values_buffer_view) {
	sparse_values_buffer_view = p_sparse_values_buffer_view;
}

int64_t GLTFAccessor::get_sparse_values_byte_offset() const {
	return sparse_values_byte_offset;
}

void GLTFAccessor::set_sparse_values_byte_offset(int64_t p_sparse_values_byte_offset) {
	sparse_values_byte_offset = p_sparse_values_byte_offset;
}

// Trivial helper functions.

void GLTFAccessor::_calculate_min_and_max(const PackedFloat64Array &p_numbers) {
	const int64_t vector_size = _get_vector_size();
	ERR_FAIL_COND(vector_size <= 0 || p_numbers.size() % vector_size != 0);
	min.resize(vector_size);
	max.resize(vector_size);
	// Initialize min and max with the first vector element values.
	for (int64_t in_vec = 0; in_vec < vector_size; in_vec++) {
		min.write[in_vec] = p_numbers[in_vec];
		max.write[in_vec] = p_numbers[in_vec];
	}
	// Iterate over the rest of the vectors.
	for (int64_t which_vec = vector_size; which_vec < p_numbers.size(); which_vec += vector_size) {
		for (int64_t in_vec = 0; in_vec < vector_size; in_vec++) {
			min.write[in_vec] = MIN(p_numbers[which_vec + in_vec], min[in_vec]);
			max.write[in_vec] = MAX(p_numbers[which_vec + in_vec], max[in_vec]);
		}
	}
	// 3.6.2.5: For floating-point components, JSON-stored minimum and maximum values represent single precision
	// floats and SHOULD be rounded to single precision before usage to avoid any potential boundary mismatches.
	// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#accessors-bounds
	if (component_type == GLTFAccessor::COMPONENT_TYPE_SINGLE_FLOAT) {
		for (int64_t i = 0; i < min.size(); i++) {
			min.write[i] = (double)(float)min[i];
			max.write[i] = (double)(float)max[i];
		}
	}
}

void GLTFAccessor::_determine_pad_skip(int64_t &r_skip_every, int64_t &r_skip_bytes) const {
	// 3.6.2.4. Accessors of matrix type have data stored in column-major order. The start of each column MUST be aligned to 4-byte boundaries.
	// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#data-alignment
	switch (component_type) {
		case GLTFAccessor::COMPONENT_TYPE_SIGNED_BYTE:
		case GLTFAccessor::COMPONENT_TYPE_UNSIGNED_BYTE: {
			if (accessor_type == GLTFAccessor::TYPE_MAT2) {
				r_skip_every = 2;
				r_skip_bytes = 2;
			}
			if (accessor_type == GLTFAccessor::TYPE_MAT3) {
				r_skip_every = 3;
				r_skip_bytes = 1;
			}
		} break;
		case GLTFAccessor::COMPONENT_TYPE_SIGNED_SHORT:
		case GLTFAccessor::COMPONENT_TYPE_UNSIGNED_SHORT: {
			if (accessor_type == GLTFAccessor::TYPE_MAT3) {
				r_skip_every = 6;
				r_skip_bytes = 2;
			}
		} break;
		default: {
		} break;
	}
}

int64_t GLTFAccessor::_determine_padded_byte_count(int64_t p_raw_byte_size) const {
	// 3.6.2.4. Accessors of matrix type have data stored in column-major order. The start of each column MUST be aligned to 4-byte boundaries.
	// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#data-alignment
	switch (component_type) {
		case GLTFAccessor::COMPONENT_TYPE_SIGNED_BYTE:
		case GLTFAccessor::COMPONENT_TYPE_UNSIGNED_BYTE: {
			if (accessor_type == GLTFAccessor::TYPE_MAT2) {
				return p_raw_byte_size * 2;
			}
			if (accessor_type == GLTFAccessor::TYPE_MAT3) {
				return p_raw_byte_size * 4 / 3;
			}
		} break;
		case GLTFAccessor::COMPONENT_TYPE_SIGNED_SHORT:
		case GLTFAccessor::COMPONENT_TYPE_UNSIGNED_SHORT: {
			if (accessor_type == GLTFAccessor::TYPE_MAT3) {
				return p_raw_byte_size * 4 / 3;
			}
		} break;
		default: {
		} break;
	}
	return p_raw_byte_size;
}

PackedFloat64Array GLTFAccessor::_filter_numbers(const PackedFloat64Array &p_numbers) const {
	PackedFloat64Array filtered_numbers = p_numbers;
	for (int64_t i = 0; i < p_numbers.size(); i++) {
		const double num = p_numbers[i];
		if (!Math::is_finite(num)) {
			// 3.6.2.2. "Values of NaN, +Infinity, and -Infinity MUST NOT be present."
			// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#accessor-data-types
			filtered_numbers.set(i, 0.0);
		} else if (component_type == GLTFAccessor::COMPONENT_TYPE_SINGLE_FLOAT) {
			filtered_numbers.set(i, (double)(float)num);
		}
	}
	return filtered_numbers;
}

String GLTFAccessor::_get_component_type_name(const GLTFComponentType p_component) {
	// These names are only for debugging and printing error messages, glTF uses the numeric values.
	switch (p_component) {
		case GLTFAccessor::COMPONENT_TYPE_NONE:
			return "None";
		case GLTFAccessor::COMPONENT_TYPE_SIGNED_BYTE:
			return "Byte";
		case GLTFAccessor::COMPONENT_TYPE_UNSIGNED_BYTE:
			return "UByte";
		case GLTFAccessor::COMPONENT_TYPE_SIGNED_SHORT:
			return "Short";
		case GLTFAccessor::COMPONENT_TYPE_UNSIGNED_SHORT:
			return "UShort";
		case GLTFAccessor::COMPONENT_TYPE_SIGNED_INT:
			return "Int";
		case GLTFAccessor::COMPONENT_TYPE_UNSIGNED_INT:
			return "UInt";
		case GLTFAccessor::COMPONENT_TYPE_SINGLE_FLOAT:
			return "Float";
		case GLTFAccessor::COMPONENT_TYPE_DOUBLE_FLOAT:
			return "Double";
		case GLTFAccessor::COMPONENT_TYPE_HALF_FLOAT:
			return "Half";
		case GLTFAccessor::COMPONENT_TYPE_SIGNED_LONG:
			return "Long";
		case GLTFAccessor::COMPONENT_TYPE_UNSIGNED_LONG:
			return "ULong";
	}

	return "<Error>";
}

GLTFAccessor::GLTFComponentType GLTFAccessor::_get_indices_component_type_for_size(const int64_t p_size) {
	ERR_FAIL_COND_V(p_size < 0, GLTFAccessor::COMPONENT_TYPE_NONE);
	// 3.7.2.1. indices accessor MUST NOT contain the maximum possible value for the component type used
	// (i.e., 255 for unsigned bytes, 65535 for unsigned shorts, 4294967295 for unsigned ints).
	// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes-overview
	if (unlikely(p_size > 4294967294LL)) {
		return GLTFAccessor::COMPONENT_TYPE_UNSIGNED_LONG;
	}
	if (p_size > 65534LL) {
		return GLTFAccessor::COMPONENT_TYPE_UNSIGNED_INT;
	}
	if (p_size > 254LL) {
		return GLTFAccessor::COMPONENT_TYPE_UNSIGNED_SHORT;
	}
	return GLTFAccessor::COMPONENT_TYPE_UNSIGNED_BYTE;
}

GLTFAccessor::GLTFAccessorType GLTFAccessor::_get_accessor_type_from_str(const String &p_string) {
	if (p_string == "SCALAR") {
		return GLTFAccessor::TYPE_SCALAR;
	}
	if (p_string == "VEC2") {
		return GLTFAccessor::TYPE_VEC2;
	}
	if (p_string == "VEC3") {
		return GLTFAccessor::TYPE_VEC3;
	}
	if (p_string == "VEC4") {
		return GLTFAccessor::TYPE_VEC4;
	}
	if (p_string == "MAT2") {
		return GLTFAccessor::TYPE_MAT2;
	}
	if (p_string == "MAT3") {
		return GLTFAccessor::TYPE_MAT3;
	}
	if (p_string == "MAT4") {
		return GLTFAccessor::TYPE_MAT4;
	}
	ERR_FAIL_V(GLTFAccessor::TYPE_SCALAR);
}

String GLTFAccessor::_get_accessor_type_name() const {
	switch (accessor_type) {
		case GLTFAccessor::TYPE_SCALAR:
			return "SCALAR";
		case GLTFAccessor::TYPE_VEC2:
			return "VEC2";
		case GLTFAccessor::TYPE_VEC3:
			return "VEC3";
		case GLTFAccessor::TYPE_VEC4:
			return "VEC4";
		case GLTFAccessor::TYPE_MAT2:
			return "MAT2";
		case GLTFAccessor::TYPE_MAT3:
			return "MAT3";
		case GLTFAccessor::TYPE_MAT4:
			return "MAT4";
		default:
			break;
	}
	ERR_FAIL_V("SCALAR");
}

int64_t GLTFAccessor::_get_vector_size() const {
	switch (accessor_type) {
		case GLTFAccessor::TYPE_SCALAR:
			return 1;
		case GLTFAccessor::TYPE_VEC2:
			return 2;
		case GLTFAccessor::TYPE_VEC3:
			return 3;
		case GLTFAccessor::TYPE_VEC4:
			return 4;
		case GLTFAccessor::TYPE_MAT2:
			return 4;
		case GLTFAccessor::TYPE_MAT3:
			return 9;
		case GLTFAccessor::TYPE_MAT4:
			return 16;
		default:
			break;
	}
	ERR_FAIL_V(0);
}

int64_t GLTFAccessor::_get_bytes_per_component(const GLTFComponentType p_component_type) {
	switch (p_component_type) {
		case GLTFAccessor::COMPONENT_TYPE_NONE:
			ERR_FAIL_V(0);
		case GLTFAccessor::COMPONENT_TYPE_SIGNED_BYTE:
		case GLTFAccessor::COMPONENT_TYPE_UNSIGNED_BYTE:
			return 1;
		case GLTFAccessor::COMPONENT_TYPE_SIGNED_SHORT:
		case GLTFAccessor::COMPONENT_TYPE_UNSIGNED_SHORT:
		case GLTFAccessor::COMPONENT_TYPE_HALF_FLOAT:
			return 2;
		case GLTFAccessor::COMPONENT_TYPE_SIGNED_INT:
		case GLTFAccessor::COMPONENT_TYPE_UNSIGNED_INT:
		case GLTFAccessor::COMPONENT_TYPE_SINGLE_FLOAT:
			return 4;
		case GLTFAccessor::COMPONENT_TYPE_DOUBLE_FLOAT:
		case GLTFAccessor::COMPONENT_TYPE_SIGNED_LONG:
		case GLTFAccessor::COMPONENT_TYPE_UNSIGNED_LONG:
			return 8;
	}
	ERR_FAIL_V(0);
}

int64_t GLTFAccessor::_get_bytes_per_vector() const {
	const int64_t raw_byte_size = _get_bytes_per_component(component_type) * _get_vector_size();
	return _determine_padded_byte_count(raw_byte_size);
}

bool GLTFAccessor::is_equal_exact(const Ref<GLTFAccessor> &p_other) const {
	if (p_other.is_null()) {
		return false;
	}
	return (buffer_view == p_other->buffer_view &&
			byte_offset == p_other->byte_offset &&
			component_type == p_other->component_type &&
			normalized == p_other->normalized &&
			count == p_other->count &&
			accessor_type == p_other->accessor_type &&
			min == p_other->min &&
			max == p_other->max &&
			sparse_count == p_other->sparse_count &&
			sparse_indices_buffer_view == p_other->sparse_indices_buffer_view &&
			sparse_indices_byte_offset == p_other->sparse_indices_byte_offset &&
			sparse_indices_component_type == p_other->sparse_indices_component_type &&
			sparse_values_buffer_view == p_other->sparse_values_buffer_view &&
			sparse_values_byte_offset == p_other->sparse_values_byte_offset);
}

// Private encode functions.

PackedFloat64Array GLTFAccessor::_encode_variants_as_floats(const Array &p_input_data, Variant::Type p_variant_type) const {
	const int64_t vector_size = _get_vector_size();
	const int64_t input_size = p_input_data.size();
	PackedFloat64Array numbers;
	numbers.resize(input_size * vector_size);
	for (int64_t input_index = 0; input_index < input_size; input_index++) {
		Variant variant = p_input_data[input_index];
		const int64_t vector_offset = input_index * vector_size;
		switch (p_variant_type) {
			case Variant::NIL:
			case Variant::BOOL:
			case Variant::INT:
			case Variant::FLOAT: {
				// For scalar values, just append them. Variant can convert all of these to double. Some padding may also be needed.
				numbers.set(vector_offset, variant);
				if (unlikely(vector_size > 1)) {
					for (int64_t i = 1; i < vector_size; i++) {
						numbers.set(vector_offset + i, 0.0);
					}
				}
			} break;
			case Variant::PLANE:
			case Variant::QUATERNION:
			case Variant::RECT2: {
				// Evil hack that relies on the structure of Variant, but it's the
				// only way to accomplish this without a ton of code duplication.
				*(Variant::Type *)&variant = Variant::VECTOR4;
			}
				[[fallthrough]];
			case Variant::VECTOR2:
			case Variant::VECTOR3:
			case Variant::VECTOR4: {
				// Variant can handle converting Vector2/3/4 to Vector4 for us.
				Vector4 vec = variant;
				for (int64_t i = 0; i < vector_size; i++) {
					numbers.set(vector_offset + i, vec[i]);
				}
				if (unlikely(vector_size > 4)) {
					for (int64_t i = 4; i < vector_size; i++) {
						numbers.set(vector_offset + i, 0.0);
					}
				}
			} break;
			case Variant::RECT2I: {
				*(Variant::Type *)&variant = Variant::VECTOR4I;
			}
				[[fallthrough]];
			case Variant::VECTOR2I:
			case Variant::VECTOR3I:
			case Variant::VECTOR4I: {
				// Variant can handle converting Vector2i/3i/4i to Vector4i for us.
				Vector4i vec = variant;
				for (int64_t i = 0; i < vector_size; i++) {
					numbers.set(vector_offset + i, vec[i]);
				}
				if (unlikely(vector_size > 4)) {
					for (int64_t i = 4; i < vector_size; i++) {
						numbers.set(vector_offset + i, 0.0);
					}
				}
			} break;
			case Variant::COLOR: {
				Color c = variant;
				for (int64_t i = 0; i < vector_size; i++) {
					numbers.set(vector_offset + i, c[i]);
				}
				if (unlikely(vector_size > 4)) {
					for (int64_t i = 4; i < vector_size; i++) {
						numbers.set(vector_offset + i, 0.0);
					}
				}
			} break;
			case Variant::TRANSFORM2D:
			case Variant::BASIS:
			case Variant::TRANSFORM3D:
			case Variant::PROJECTION: {
				// Variant can handle converting Transform2D/Transform3D/Basis to Projection for us.
				Projection p = variant;
				if (vector_size == 16) {
					for (int64_t i = 0; i < 4; i++) {
						numbers.set(vector_offset + 4 * i, p.columns[i][0]);
						numbers.set(vector_offset + 4 * i + 1, p.columns[i][1]);
						numbers.set(vector_offset + 4 * i + 2, p.columns[i][2]);
						numbers.set(vector_offset + 4 * i + 3, p.columns[i][3]);
					}
				} else if (vector_size == 9) {
					for (int64_t i = 0; i < 3; i++) {
						numbers.set(vector_offset + 3 * i, p.columns[i][0]);
						numbers.set(vector_offset + 3 * i + 1, p.columns[i][1]);
						numbers.set(vector_offset + 3 * i + 2, p.columns[i][2]);
					}
				} else if (vector_size == 4) {
					numbers.set(vector_offset, p.columns[0][0]);
					numbers.set(vector_offset + 1, p.columns[0][1]);
					numbers.set(vector_offset + 2, p.columns[1][0]);
					numbers.set(vector_offset + 3, p.columns[1][1]);
				}
			} break;
			default: {
				ERR_FAIL_V_MSG(PackedFloat64Array(), "glTF export: Cannot encode accessor from Variant of type " + Variant::get_type_name(p_variant_type) + ".");
			}
		}
	}
	return numbers;
}

void GLTFAccessor::_store_sparse_indices_into_state(const Ref<GLTFState> &p_gltf_state, const PackedInt64Array &p_sparse_indices, const bool p_deduplicate) {
	// The byte offset of a sparse accessor's indices buffer view MUST be a multiple of the indices primitive componentType.
	// https://github.com/KhronosGroup/glTF/blob/main/specification/2.0/schema/accessor.sparse.indices.schema.json
	const int64_t bytes_per_index = _get_bytes_per_component(sparse_indices_component_type);
	PackedByteArray indices_bytes;
	indices_bytes.resize(bytes_per_index * p_sparse_indices.size());
	uint8_t *ret_write = indices_bytes.ptrw();
	int64_t ret_byte_offset = 0;
	for (int64_t i = 0; i < p_sparse_indices.size(); i++) {
		switch (sparse_indices_component_type) {
			case GLTFAccessor::COMPONENT_TYPE_UNSIGNED_BYTE: {
				*(uint8_t *)&ret_write[ret_byte_offset] = p_sparse_indices[i];
			} break;
			case GLTFAccessor::COMPONENT_TYPE_UNSIGNED_SHORT: {
				*(uint16_t *)&ret_write[ret_byte_offset] = p_sparse_indices[i];
			} break;
			case GLTFAccessor::COMPONENT_TYPE_UNSIGNED_INT: {
				*(uint32_t *)&ret_write[ret_byte_offset] = p_sparse_indices[i];
			} break;
			case GLTFAccessor::COMPONENT_TYPE_UNSIGNED_LONG: {
				*(uint64_t *)&ret_write[ret_byte_offset] = p_sparse_indices[i];
			} break;
			default: {
				ERR_FAIL_MSG("glTF export: Invalid sparse indices component type '" + _get_component_type_name(sparse_indices_component_type) + "' for sparse accessor indices.");
			} break;
		}
		ret_byte_offset += bytes_per_index;
	}
	const GLTFBufferViewIndex buffer_view_index = GLTFBufferView::write_new_buffer_view_into_state(p_gltf_state, indices_bytes, bytes_per_index, GLTFBufferView::TARGET_NONE, -1, 0, p_deduplicate);
	ERR_FAIL_COND_MSG(buffer_view_index == -1, "glTF export: Failed to write sparse indices into glTF state.");
	set_sparse_indices_buffer_view(buffer_view_index);
}

// Low-level encode functions.

GLTFAccessor::GLTFComponentType GLTFAccessor::get_minimal_integer_component_type_from_ints(const PackedInt64Array &p_numbers) {
	bool has_negative = false;
	for (int64_t i = 0; i < p_numbers.size(); i++) {
		if (p_numbers[i] < 0) {
			has_negative = true;
			break;
		}
	}
	if (has_negative) {
		GLTFComponentType ret = GLTFAccessor::COMPONENT_TYPE_SIGNED_BYTE;
		for (int64_t i = 0; i < p_numbers.size(); i++) {
			const int64_t num = p_numbers[i];
			if (ret == GLTFAccessor::COMPONENT_TYPE_SIGNED_BYTE && (num < -128LL || num > 127LL)) {
				ret = GLTFAccessor::COMPONENT_TYPE_SIGNED_SHORT;
			}
			if (ret == GLTFAccessor::COMPONENT_TYPE_SIGNED_SHORT && (num < -32768LL || num > 32767LL)) {
				ret = GLTFAccessor::COMPONENT_TYPE_SIGNED_INT;
			}
			if (ret == GLTFAccessor::COMPONENT_TYPE_SIGNED_INT && (num < -2147483648LL || num > 2147483647LL)) {
				return GLTFAccessor::COMPONENT_TYPE_SIGNED_LONG;
			}
		}
		return ret;
	}
	GLTFComponentType ret = GLTFAccessor::COMPONENT_TYPE_UNSIGNED_BYTE;
	for (int64_t i = 0; i < p_numbers.size(); i++) {
		const int64_t num = p_numbers[i];
		// 3.7.2.1. indices accessor MUST NOT contain the maximum possible value for the component type used
		// (i.e., 255 for unsigned bytes, 65535 for unsigned shorts, 4294967295 for unsigned ints).
		// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes-overview
		if (ret == GLTFAccessor::COMPONENT_TYPE_UNSIGNED_BYTE && num > 254LL) {
			ret = GLTFAccessor::COMPONENT_TYPE_UNSIGNED_SHORT;
		}
		if (ret == GLTFAccessor::COMPONENT_TYPE_UNSIGNED_SHORT && num > 65534LL) {
			ret = GLTFAccessor::COMPONENT_TYPE_UNSIGNED_INT;
		}
		if (ret == GLTFAccessor::COMPONENT_TYPE_UNSIGNED_INT && num > 4294967294LL) {
			return GLTFAccessor::COMPONENT_TYPE_UNSIGNED_LONG;
		}
	}
	return ret;
}

PackedByteArray GLTFAccessor::encode_floats_as_bytes(const PackedFloat64Array &p_input_numbers) {
	// Filter and update `count`, `min`, and `max` based on the given data.
	PackedFloat64Array filtered_numbers = _filter_numbers(p_input_numbers);
	count = filtered_numbers.size() / _get_vector_size();
	_calculate_min_and_max(filtered_numbers);
	// Actually encode the data.
	const int64_t input_size = filtered_numbers.size();
	const int64_t bytes_per_component = _get_bytes_per_component(component_type);
	int64_t raw_byte_size = _determine_padded_byte_count(bytes_per_component * input_size);
	int64_t skip_every = 0;
	int64_t skip_bytes = 0;
	_determine_pad_skip(skip_every, skip_bytes);
	PackedByteArray ret;
	ret.resize(raw_byte_size);
	uint8_t *ret_write = ret.ptrw();
	int64_t ret_byte_offset = 0;
	for (int64_t i = 0; i < input_size; i++) {
		switch (component_type) {
			case COMPONENT_TYPE_NONE: {
				ERR_FAIL_V_MSG(ret, "glTF export: Invalid component type 'NONE' for glTF accessor.");
			} break;
			case COMPONENT_TYPE_SIGNED_BYTE: {
				*(int8_t *)&ret_write[ret_byte_offset] = filtered_numbers[i];
			} break;
			case COMPONENT_TYPE_UNSIGNED_BYTE: {
				*(uint8_t *)&ret_write[ret_byte_offset] = filtered_numbers[i];
			} break;
			case COMPONENT_TYPE_SIGNED_SHORT: {
				*(int16_t *)&ret_write[ret_byte_offset] = filtered_numbers[i];
			} break;
			case COMPONENT_TYPE_UNSIGNED_SHORT: {
				*(uint16_t *)&ret_write[ret_byte_offset] = filtered_numbers[i];
			} break;
			case COMPONENT_TYPE_SIGNED_INT: {
				*(int32_t *)&ret_write[ret_byte_offset] = filtered_numbers[i];
			} break;
			case COMPONENT_TYPE_UNSIGNED_INT: {
				*(uint32_t *)&ret_write[ret_byte_offset] = filtered_numbers[i];
			} break;
			case COMPONENT_TYPE_SINGLE_FLOAT: {
				*(float *)&ret_write[ret_byte_offset] = filtered_numbers[i];
			} break;
			case COMPONENT_TYPE_DOUBLE_FLOAT: {
				*(double *)&ret_write[ret_byte_offset] = filtered_numbers[i];
			} break;
			case COMPONENT_TYPE_HALF_FLOAT: {
				*(uint16_t *)&ret_write[ret_byte_offset] = Math::make_half_float(filtered_numbers[i]);
			} break;
			case COMPONENT_TYPE_SIGNED_LONG: {
				// Note: This can potentially result in precision loss because int64_t can store some values that double can't.
				*(int64_t *)&ret_write[ret_byte_offset] = filtered_numbers[i];
			} break;
			case COMPONENT_TYPE_UNSIGNED_LONG: {
				// Note: This can potentially result in precision loss because uint64_t can store some values that double can't.
				*(uint64_t *)&ret_write[ret_byte_offset] = filtered_numbers[i];
			} break;
			default: {
				ERR_FAIL_V_MSG(ret, "glTF export: Godot does not support writing glTF accessor components of type '" + itos(component_type) + "'.");
			} break;
		}
		ret_byte_offset += bytes_per_component;
		if (unlikely(skip_every > 0)) {
			if ((i + 1) % skip_every == 0) {
				ret_byte_offset += skip_bytes;
			}
		}
	}
	ERR_FAIL_COND_V_MSG(ret_byte_offset != raw_byte_size, ret, "glTF export: Accessor encoded data did not write exactly the expected number of bytes.");
	return ret;
}

PackedByteArray GLTFAccessor::encode_ints_as_bytes(const PackedInt64Array &p_input_numbers) {
	// Filter and update `count`, `min`, and `max` based on the given data.
	count = p_input_numbers.size() / _get_vector_size();
	_calculate_min_and_max(Variant(p_input_numbers));
	// Actually encode the data.
	const int64_t input_size = p_input_numbers.size();
	const int64_t bytes_per_component = _get_bytes_per_component(component_type);
	int64_t raw_byte_size = _determine_padded_byte_count(bytes_per_component * input_size);
	int64_t skip_every = 0;
	int64_t skip_bytes = 0;
	_determine_pad_skip(skip_every, skip_bytes);
	PackedByteArray ret;
	ret.resize(raw_byte_size);
	uint8_t *ret_write = ret.ptrw();
	int64_t ret_byte_offset = 0;
	for (int64_t i = 0; i < input_size; i++) {
		switch (component_type) {
			case COMPONENT_TYPE_NONE: {
				ERR_FAIL_V_MSG(ret, "glTF export: Invalid component type 'NONE' for glTF accessor.");
			} break;
			case COMPONENT_TYPE_SIGNED_BYTE: {
				*(int8_t *)&ret_write[ret_byte_offset] = p_input_numbers[i];
			} break;
			case COMPONENT_TYPE_UNSIGNED_BYTE: {
				*(uint8_t *)&ret_write[ret_byte_offset] = p_input_numbers[i];
			} break;
			case COMPONENT_TYPE_SIGNED_SHORT: {
				*(int16_t *)&ret_write[ret_byte_offset] = p_input_numbers[i];
			} break;
			case COMPONENT_TYPE_UNSIGNED_SHORT: {
				*(uint16_t *)&ret_write[ret_byte_offset] = p_input_numbers[i];
			} break;
			case COMPONENT_TYPE_SIGNED_INT: {
				*(int32_t *)&ret_write[ret_byte_offset] = p_input_numbers[i];
			} break;
			case COMPONENT_TYPE_UNSIGNED_INT: {
				*(uint32_t *)&ret_write[ret_byte_offset] = p_input_numbers[i];
			} break;
			case COMPONENT_TYPE_SINGLE_FLOAT: {
				*(float *)&ret_write[ret_byte_offset] = p_input_numbers[i];
			} break;
			case COMPONENT_TYPE_DOUBLE_FLOAT: {
				*(double *)&ret_write[ret_byte_offset] = p_input_numbers[i];
			} break;
			case COMPONENT_TYPE_HALF_FLOAT: {
				*(uint16_t *)&ret_write[ret_byte_offset] = Math::make_half_float(p_input_numbers[i]);
			} break;
			case COMPONENT_TYPE_SIGNED_LONG: {
				*(int64_t *)&ret_write[ret_byte_offset] = p_input_numbers[i];
			} break;
			case COMPONENT_TYPE_UNSIGNED_LONG: {
				*(uint64_t *)&ret_write[ret_byte_offset] = p_input_numbers[i];
			} break;
			default: {
				ERR_FAIL_V_MSG(ret, "glTF export: Godot does not support writing glTF accessor components of type '" + itos(component_type) + "'.");
			} break;
		}
		ret_byte_offset += bytes_per_component;
		if (unlikely(skip_every > 0)) {
			if ((i + 1) % skip_every == 0) {
				ret_byte_offset += skip_bytes;
			}
		}
	}
	ERR_FAIL_COND_V_MSG(ret_byte_offset != raw_byte_size, ret, "glTF export: Accessor encoded data did not write exactly the expected number of bytes.");
	return ret;
}

PackedByteArray GLTFAccessor::encode_variants_as_bytes(const Array &p_input_data, Variant::Type p_variant_type) {
	const int64_t bytes_per_vec = _get_bytes_per_vector();
	ERR_FAIL_COND_V_MSG(bytes_per_vec == 0, PackedByteArray(), "glTF export: Cannot encode an accessor of this type.");
	PackedFloat64Array numbers = _encode_variants_as_floats(p_input_data, p_variant_type);
	return encode_floats_as_bytes(numbers);
}

GLTFAccessorIndex GLTFAccessor::store_accessor_data_into_state(const Ref<GLTFState> &p_gltf_state, const PackedByteArray &p_data_bytes, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target, const GLTFBufferIndex p_buffer_index, const bool p_deduplicate) {
	ERR_FAIL_COND_V_MSG(p_data_bytes.is_empty(), -1, "glTF export: Cannot store nothing.");
	// Update `count` based on the size of the data. It's possible that `count` may already be correct, but this function is public, so this prevents footguns.
	const int64_t bytes_per_vec = _get_bytes_per_vector();
	ERR_FAIL_COND_V_MSG(bytes_per_vec == 0 || p_data_bytes.size() % bytes_per_vec != 0, -1, "glTF export: Tried to store an accessor with data that is not a multiple of the accessor's bytes per vector.");
	count = p_data_bytes.size() / bytes_per_vec;
	// 3.6.2.4. The byte offset of an accessor's buffer view MUST be a multiple of the accessor's primitive size.
	// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#data-alignment
	const int64_t alignment = _get_bytes_per_component(component_type);
	// 3.6.2.4. Each element of a vertex attribute MUST be aligned to 4-byte boundaries inside a bufferView.
	int64_t byte_stride = -1;
	if (p_buffer_view_target == GLTFBufferView::TARGET_ARRAY_BUFFER) {
		byte_stride = bytes_per_vec;
		ERR_FAIL_COND_V_MSG(byte_stride < 4 || byte_stride % 4 != 0, -1, "glTF export: Vertex attributes using TARGET_ARRAY_BUFFER must have a byte stride that is a multiple of 4 as required by section 3.6.2.4 of the glTF specification.");
	}
	// Write the data into a new buffer view.
	const GLTFBufferViewIndex buffer_view_index = GLTFBufferView::write_new_buffer_view_into_state(p_gltf_state, p_data_bytes, alignment, p_buffer_view_target, byte_stride, 0, p_deduplicate);
	ERR_FAIL_COND_V_MSG(buffer_view_index == -1, -1, "glTF export: Accessor failed to write new buffer view into glTF state.");
	set_buffer_view(buffer_view_index);
	// Add the new accessor to the state, but check for duplicates first.
	TypedArray<GLTFAccessor> state_accessors = p_gltf_state->get_accessors();
	const GLTFAccessorIndex accessor_count = state_accessors.size();
	for (GLTFAccessorIndex i = 0; i < accessor_count; i++) {
		Ref<GLTFAccessor> existing_accessor = state_accessors[i];
		if (is_equal_exact(existing_accessor)) {
			// An identical accessor already exists in the state, so just return the index.
			return i;
		}
	}
	Ref<GLTFAccessor> self = this;
	state_accessors.append(self);
	p_gltf_state->set_accessors(state_accessors);
	return accessor_count;
}

Ref<GLTFAccessor> GLTFAccessor::make_new_accessor_without_data(GLTFAccessorType p_accessor_type, GLTFComponentType p_component_type) {
	Ref<GLTFAccessor> accessor;
	accessor.instantiate();
	accessor->set_accessor_type(p_accessor_type);
	accessor->set_component_type(p_component_type);
	return accessor;
}

// High-level encode functions.

GLTFAccessorIndex GLTFAccessor::encode_new_accessor_from_colors(const Ref<GLTFState> &p_gltf_state, const PackedColorArray &p_input_data, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target, const bool p_deduplicate) {
	ERR_FAIL_COND_V_MSG(p_input_data.is_empty(), -1, "glTF export: Cannot encode an accessor from an empty array.");
	PackedFloat64Array numbers;
	numbers.resize(p_input_data.size() * 4);
	for (int64_t i = 0; i < p_input_data.size(); i++) {
		const Color &color = p_input_data[i];
		numbers.set(i * 4, color.r);
		numbers.set(i * 4 + 1, color.g);
		numbers.set(i * 4 + 2, color.b);
		numbers.set(i * 4 + 3, color.a);
	}
	Ref<GLTFAccessor> accessor = make_new_accessor_without_data(TYPE_VEC4, COMPONENT_TYPE_SINGLE_FLOAT);
	PackedByteArray encoded_bytes = accessor->encode_floats_as_bytes(numbers);
	ERR_FAIL_COND_V_MSG(encoded_bytes.is_empty(), -1, "glTF export: Accessor failed to encode data as bytes (was the input data empty?).");
	return accessor->store_accessor_data_into_state(p_gltf_state, encoded_bytes, p_buffer_view_target, 0, p_deduplicate);
}

GLTFAccessorIndex GLTFAccessor::encode_new_accessor_from_float64s(const Ref<GLTFState> &p_gltf_state, const PackedFloat64Array &p_input_data, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target, const bool p_deduplicate) {
	ERR_FAIL_COND_V_MSG(p_input_data.is_empty(), -1, "glTF export: Cannot encode an accessor from an empty array.");
	Ref<GLTFAccessor> accessor = make_new_accessor_without_data(TYPE_SCALAR, COMPONENT_TYPE_SINGLE_FLOAT);
	PackedByteArray encoded_bytes = accessor->encode_floats_as_bytes(p_input_data);
	ERR_FAIL_COND_V_MSG(encoded_bytes.is_empty(), -1, "glTF export: Accessor failed to encode data as bytes (was the input data empty?).");
	return accessor->store_accessor_data_into_state(p_gltf_state, encoded_bytes, p_buffer_view_target, 0, p_deduplicate);
}

GLTFAccessorIndex GLTFAccessor::encode_new_accessor_from_int32s(const Ref<GLTFState> &p_gltf_state, const PackedInt32Array &p_input_data, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target, const bool p_deduplicate) {
	ERR_FAIL_COND_V_MSG(p_input_data.is_empty(), -1, "glTF export: Cannot encode an accessor from an empty array.");
	PackedInt64Array numbers;
	numbers.resize(p_input_data.size());
	for (int64_t i = 0; i < p_input_data.size(); i++) {
		numbers.set(i, p_input_data[i]);
	}
	const GLTFComponentType component_type = get_minimal_integer_component_type_from_ints(numbers);
	Ref<GLTFAccessor> accessor = make_new_accessor_without_data(TYPE_SCALAR, component_type);
	PackedByteArray encoded_bytes = accessor->encode_ints_as_bytes(numbers);
	ERR_FAIL_COND_V_MSG(encoded_bytes.is_empty(), -1, "glTF export: Accessor failed to encode data as bytes (was the input data empty?).");
	return accessor->store_accessor_data_into_state(p_gltf_state, encoded_bytes, p_buffer_view_target, 0, p_deduplicate);
}

GLTFAccessorIndex GLTFAccessor::encode_new_accessor_from_int64s(const Ref<GLTFState> &p_gltf_state, const PackedInt64Array &p_input_data, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target, const bool p_deduplicate) {
	ERR_FAIL_COND_V_MSG(p_input_data.is_empty(), -1, "glTF export: Cannot encode an accessor from an empty array.");
	const GLTFComponentType component_type = get_minimal_integer_component_type_from_ints(p_input_data);
	Ref<GLTFAccessor> accessor = make_new_accessor_without_data(TYPE_SCALAR, component_type);
	PackedByteArray encoded_bytes = accessor->encode_ints_as_bytes(p_input_data);
	ERR_FAIL_COND_V_MSG(encoded_bytes.is_empty(), -1, "glTF export: Accessor failed to encode data as bytes (was the input data empty?).");
	return accessor->store_accessor_data_into_state(p_gltf_state, encoded_bytes, p_buffer_view_target, 0, p_deduplicate);
}

GLTFAccessorIndex GLTFAccessor::encode_new_accessor_from_quaternions(const Ref<GLTFState> &p_gltf_state, const Vector<Quaternion> &p_input_data, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target, const bool p_deduplicate) {
	ERR_FAIL_COND_V_MSG(p_input_data.is_empty(), -1, "glTF export: Cannot encode an accessor from an empty array.");
	PackedFloat64Array numbers;
	numbers.resize(p_input_data.size() * 4);
	for (int64_t i = 0; i < p_input_data.size(); i++) {
		const Quaternion &quat = p_input_data[i];
		numbers.set(i * 4, quat.x);
		numbers.set(i * 4 + 1, quat.y);
		numbers.set(i * 4 + 2, quat.z);
		numbers.set(i * 4 + 3, quat.w);
	}
	Ref<GLTFAccessor> accessor = make_new_accessor_without_data(TYPE_VEC4, COMPONENT_TYPE_SINGLE_FLOAT);
	PackedByteArray encoded_bytes = accessor->encode_floats_as_bytes(numbers);
	ERR_FAIL_COND_V_MSG(encoded_bytes.is_empty(), -1, "glTF export: Accessor failed to encode data as bytes (was the input data empty?).");
	return accessor->store_accessor_data_into_state(p_gltf_state, encoded_bytes, p_buffer_view_target, 0, p_deduplicate);
}

GLTFAccessorIndex GLTFAccessor::encode_new_accessor_from_variants(const Ref<GLTFState> &p_gltf_state, const Array &p_input_data, Variant::Type p_variant_type, GLTFAccessorType p_accessor_type, GLTFComponentType p_component_type, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target, const bool p_deduplicate) {
	ERR_FAIL_COND_V_MSG(p_input_data.is_empty(), -1, "glTF export: Cannot encode an accessor from an empty array.");
	Ref<GLTFAccessor> accessor = make_new_accessor_without_data(p_accessor_type, p_component_type);
	// Write the data into a new buffer view.
	PackedByteArray encoded_bytes = accessor->encode_variants_as_bytes(p_input_data, p_variant_type);
	ERR_FAIL_COND_V_MSG(encoded_bytes.is_empty(), -1, "glTF export: Accessor failed to encode data as bytes (was the input data empty?).");
	return accessor->store_accessor_data_into_state(p_gltf_state, encoded_bytes, p_buffer_view_target, 0, p_deduplicate);
}

GLTFAccessorIndex GLTFAccessor::encode_new_accessor_from_vector2s(const Ref<GLTFState> &p_gltf_state, const PackedVector2Array &p_input_data, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target, const bool p_deduplicate) {
	ERR_FAIL_COND_V_MSG(p_input_data.is_empty(), -1, "glTF export: Cannot encode an accessor from an empty array.");
	PackedFloat64Array numbers;
	numbers.resize(p_input_data.size() * 2);
	for (int64_t i = 0; i < p_input_data.size(); i++) {
		const Vector2 &vec = p_input_data[i];
		numbers.set(i * 2, vec.x);
		numbers.set(i * 2 + 1, vec.y);
	}
	Ref<GLTFAccessor> accessor = make_new_accessor_without_data(TYPE_VEC2, COMPONENT_TYPE_SINGLE_FLOAT);
	PackedByteArray encoded_bytes = accessor->encode_floats_as_bytes(numbers);
	ERR_FAIL_COND_V_MSG(encoded_bytes.is_empty(), -1, "glTF export: Accessor failed to encode data as bytes (was the input data empty?).");
	return accessor->store_accessor_data_into_state(p_gltf_state, encoded_bytes, p_buffer_view_target, 0, p_deduplicate);
}

GLTFAccessorIndex GLTFAccessor::encode_new_accessor_from_vector3s(const Ref<GLTFState> &p_gltf_state, const PackedVector3Array &p_input_data, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target, const bool p_deduplicate) {
	ERR_FAIL_COND_V_MSG(p_input_data.is_empty(), -1, "glTF export: Cannot encode an accessor from an empty array.");
	PackedFloat64Array numbers;
	numbers.resize(p_input_data.size() * 3);
	for (int64_t i = 0; i < p_input_data.size(); i++) {
		const Vector3 &vec = p_input_data[i];
		numbers.set(i * 3, vec.x);
		numbers.set(i * 3 + 1, vec.y);
		numbers.set(i * 3 + 2, vec.z);
	}
	Ref<GLTFAccessor> accessor = make_new_accessor_without_data(TYPE_VEC3, COMPONENT_TYPE_SINGLE_FLOAT);
	PackedByteArray encoded_bytes = accessor->encode_floats_as_bytes(numbers);
	ERR_FAIL_COND_V_MSG(encoded_bytes.is_empty(), -1, "glTF export: Accessor failed to encode data as bytes (was the input data empty?).");
	return accessor->store_accessor_data_into_state(p_gltf_state, encoded_bytes, p_buffer_view_target, 0, p_deduplicate);
}

GLTFAccessorIndex GLTFAccessor::encode_new_accessor_from_vector4s(const Ref<GLTFState> &p_gltf_state, const PackedVector4Array &p_input_data, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target, const bool p_deduplicate) {
	ERR_FAIL_COND_V_MSG(p_input_data.is_empty(), -1, "glTF export: Cannot encode an accessor from an empty array.");
	PackedFloat64Array numbers;
	numbers.resize(p_input_data.size() * 4);
	for (int64_t i = 0; i < p_input_data.size(); i++) {
		const Vector4 &vec = p_input_data[i];
		numbers.set(i * 4, vec.x);
		numbers.set(i * 4 + 1, vec.y);
		numbers.set(i * 4 + 2, vec.z);
		numbers.set(i * 4 + 3, vec.w);
	}
	Ref<GLTFAccessor> accessor = make_new_accessor_without_data(TYPE_VEC4, COMPONENT_TYPE_SINGLE_FLOAT);
	PackedByteArray encoded_bytes = accessor->encode_floats_as_bytes(numbers);
	ERR_FAIL_COND_V_MSG(encoded_bytes.is_empty(), -1, "glTF export: Accessor failed to encode data as bytes (was the input data empty?).");
	return accessor->store_accessor_data_into_state(p_gltf_state, encoded_bytes, p_buffer_view_target, 0, p_deduplicate);
}

GLTFAccessorIndex GLTFAccessor::encode_new_accessor_from_vector4is(const Ref<GLTFState> &p_gltf_state, const Vector<Vector4i> &p_input_data, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target, const bool p_deduplicate) {
	ERR_FAIL_COND_V_MSG(p_input_data.is_empty(), -1, "glTF export: Cannot encode an accessor from an empty array.");
	PackedInt64Array numbers;
	numbers.resize(p_input_data.size() * 4);
	for (int64_t i = 0; i < p_input_data.size(); i++) {
		const Vector4i &vec = p_input_data[i];
		numbers.set(i * 4, vec.x);
		numbers.set(i * 4 + 1, vec.y);
		numbers.set(i * 4 + 2, vec.z);
		numbers.set(i * 4 + 3, vec.w);
	}
	const GLTFComponentType component_type = get_minimal_integer_component_type_from_ints(numbers);
	Ref<GLTFAccessor> accessor = make_new_accessor_without_data(TYPE_VEC4, component_type);
	PackedByteArray encoded_bytes = accessor->encode_ints_as_bytes(numbers);
	ERR_FAIL_COND_V_MSG(encoded_bytes.is_empty(), -1, "glTF export: Accessor failed to encode data as bytes (was the input data empty?).");
	return accessor->store_accessor_data_into_state(p_gltf_state, encoded_bytes, p_buffer_view_target, 0, p_deduplicate);
}

GLTFAccessorIndex GLTFAccessor::encode_new_sparse_accessor_from_vec3s(const Ref<GLTFState> &p_gltf_state, const PackedVector3Array &p_input_data, const PackedVector3Array &p_base_reference_data, const double p_tolerance_multiplier, const GLTFBufferView::ArrayBufferTarget p_main_buffer_view_target, const bool p_deduplicate) {
	const int64_t input_size = p_input_data.size();
	ERR_FAIL_COND_V_MSG(input_size == 0, -1, "glTF export: Cannot encode an accessor from an empty array.");
	const bool is_base_empty = p_base_reference_data.is_empty();
	ERR_FAIL_COND_V_MSG(!is_base_empty && p_base_reference_data.size() != input_size, -1, "glTF export: Base reference data must either be empty, or have the same size as the main input data.");
	PackedInt64Array sparse_indices;
	PackedFloat64Array sparse_values;
	PackedFloat64Array dense_values;
	int64_t highest_index = 0;
	dense_values.resize(input_size * 3);
	for (int64_t i = 0; i < input_size; i++) {
		Vector3 vec = p_input_data[i];
		Vector3 base_ref_vec;
		Vector3 displacement;
		if (is_base_empty) {
			base_ref_vec = Vector3();
			displacement = vec;
		} else {
			base_ref_vec = p_base_reference_data[i];
			displacement = vec - base_ref_vec;
		}
		if ((displacement * p_tolerance_multiplier).is_zero_approx()) {
			vec = base_ref_vec;
		} else {
			highest_index = i;
			sparse_indices.append(i);
			sparse_values.append(vec.x);
			sparse_values.append(vec.y);
			sparse_values.append(vec.z);
		}
		dense_values.set(i * 3, vec.x);
		dense_values.set(i * 3 + 1, vec.y);
		dense_values.set(i * 3 + 2, vec.z);
	}
	// Check if the sparse accessor actually saves space, or if it's better to just use a normal accessor.
	const int64_t sparse_count = sparse_indices.size();
	const int64_t bytes_per_value_component = _get_bytes_per_component(COMPONENT_TYPE_SINGLE_FLOAT);
	const GLTFComponentType indices_component_type = _get_indices_component_type_for_size(highest_index);
	const int64_t sparse_data_bytes = _get_bytes_per_component(indices_component_type) * sparse_count + bytes_per_value_component * sparse_values.size();
	const int64_t dense_data_bytes = bytes_per_value_component * 3 * input_size;
	// Sparse accessors require more JSON, a bit under 200 characters when minified, so factor that in.
	constexpr int64_t sparse_json_fluff = 200;
	Ref<GLTFAccessor> accessor = make_new_accessor_without_data(TYPE_VEC3, COMPONENT_TYPE_SINGLE_FLOAT);
	if (sparse_data_bytes + sparse_json_fluff >= dense_data_bytes) {
		// Sparse accessor is not worth it, just use a normal accessor instead.
		// However, note that we use the calculated dense values instead of the original input data.
		// This way, regardless of the underlying storage layout, the data is the same in both cases.
		PackedByteArray encoded_bytes = accessor->encode_floats_as_bytes(dense_values);
		ERR_FAIL_COND_V_MSG(encoded_bytes.is_empty(), -1, "glTF export: Accessor failed to encode data as bytes (was the input data empty?).");
		return accessor->store_accessor_data_into_state(p_gltf_state, encoded_bytes, p_main_buffer_view_target, 0, p_deduplicate);
	}
	// Encode as a sparse accessor.
	if (sparse_count > 0) {
		accessor->set_sparse_count(sparse_count);
		accessor->set_sparse_indices_component_type(indices_component_type);
		accessor->_store_sparse_indices_into_state(p_gltf_state, sparse_indices, p_deduplicate);
		const PackedByteArray sparse_values_encoded_bytes = accessor->encode_floats_as_bytes(sparse_values);
		ERR_FAIL_COND_V_MSG(sparse_values_encoded_bytes.is_empty(), -1, "glTF export: Accessor failed to encode sparse values as bytes.");
		// Note: Sparse values always use TARGET_NONE, it does NOT match the target of the main buffer view.
		const GLTFBufferViewIndex sparse_values_buffer_view_index = GLTFBufferView::write_new_buffer_view_into_state(p_gltf_state, sparse_values_encoded_bytes, bytes_per_value_component, GLTFBufferView::TARGET_NONE, -1, 0, p_deduplicate);
		accessor->set_sparse_values_buffer_view(sparse_values_buffer_view_index);
	}
	// If the base reference data is empty, just directly add the accessor with only sparse data.
	if (is_base_empty) {
		// This is similar to `encode_floats_as_bytes` + `store_accessor_data_into_state` but we don't write a buffer view.
		// Filter and update `count`, `min`, and `max` based on the given data.
		accessor->set_count(input_size);
		const PackedFloat64Array filtered_numbers = accessor->_filter_numbers(dense_values);
		accessor->_calculate_min_and_max(filtered_numbers);
		// Add the new accessor to the state, but check for duplicates first.
		TypedArray<GLTFAccessor> state_accessors = p_gltf_state->get_accessors();
		const GLTFAccessorIndex accessor_count = state_accessors.size();
		for (GLTFAccessorIndex i = 0; i < accessor_count; i++) {
			Ref<GLTFAccessor> existing_accessor = state_accessors[i];
			if (accessor->is_equal_exact(existing_accessor)) {
				// An identical accessor already exists in the state, so just return the index.
				return i;
			}
		}
		state_accessors.append(accessor);
		p_gltf_state->set_accessors(state_accessors);
		return accessor_count;
	}
	// Encode the base reference alongside the sparse data.
	PackedFloat64Array base_reference_values;
	base_reference_values.resize(input_size * 3);
	for (int64_t i = 0; i < input_size; i++) {
		const Vector3 &base_ref_vec = p_base_reference_data[i];
		base_reference_values.set(i * 3, base_ref_vec.x);
		base_reference_values.set(i * 3 + 1, base_ref_vec.y);
		base_reference_values.set(i * 3 + 2, base_ref_vec.z);
	}
	const PackedByteArray base_reference_encoded_bytes = accessor->encode_floats_as_bytes(base_reference_values);
	ERR_FAIL_COND_V_MSG(base_reference_encoded_bytes.is_empty(), -1, "glTF export: Accessor failed to encode data as bytes (was the input data empty?).");
	return accessor->store_accessor_data_into_state(p_gltf_state, base_reference_encoded_bytes, p_main_buffer_view_target, 0, p_deduplicate);
}

// Dictionary conversion.

Ref<GLTFAccessor> GLTFAccessor::from_dictionary(const Dictionary &p_dict) {
	// See https://github.com/KhronosGroup/glTF/blob/main/specification/2.0/schema/accessor.schema.json
	Ref<GLTFAccessor> accessor;
	accessor.instantiate();
	if (p_dict.has("bufferView")) {
		// bufferView is optional. If not present, the accessor is considered to be zero-initialized.
		accessor->buffer_view = p_dict["bufferView"];
	}
	if (p_dict.has("byteOffset")) {
		accessor->byte_offset = p_dict["byteOffset"];
	}
	if (p_dict.has("componentType")) {
		accessor->component_type = (GLTFAccessor::GLTFComponentType)(int32_t)p_dict["componentType"];
	}
	if (p_dict.has("count")) {
		accessor->count = p_dict["count"];
	}
	if (accessor->count <= 0) {
		ERR_PRINT("glTF import: Invalid accessor count " + itos(accessor->count) + " for accessor. Accessor count must be greater than 0.");
	}
	if (p_dict.has("max")) {
		accessor->max = p_dict["max"];
	}
	if (p_dict.has("min")) {
		accessor->min = p_dict["min"];
	}
	if (p_dict.has("normalized")) {
		accessor->normalized = p_dict["normalized"];
	}
	if (p_dict.has("sparse")) {
		// See https://github.com/KhronosGroup/glTF/blob/main/specification/2.0/schema/accessor.sparse.schema.json
		const Dictionary &sparse_dict = p_dict["sparse"];
		ERR_FAIL_COND_V(!sparse_dict.has("count"), accessor);
		accessor->sparse_count = sparse_dict["count"];
		ERR_FAIL_COND_V(!sparse_dict.has("indices"), accessor);
		const Dictionary &sparse_indices_dict = sparse_dict["indices"];
		ERR_FAIL_COND_V(!sparse_indices_dict.has("bufferView"), accessor);
		accessor->sparse_indices_buffer_view = sparse_indices_dict["bufferView"];
		ERR_FAIL_COND_V(!sparse_indices_dict.has("componentType"), accessor);
		accessor->sparse_indices_component_type = (GLTFAccessor::GLTFComponentType)(int32_t)sparse_indices_dict["componentType"];
		if (sparse_indices_dict.has("byteOffset")) {
			accessor->sparse_indices_byte_offset = sparse_indices_dict["byteOffset"];
		}
		ERR_FAIL_COND_V(!sparse_dict.has("values"), accessor);
		const Dictionary &sparse_values_dict = sparse_dict["values"];
		ERR_FAIL_COND_V(!sparse_values_dict.has("bufferView"), accessor);
		accessor->sparse_values_buffer_view = sparse_values_dict["bufferView"];
		if (sparse_values_dict.has("byteOffset")) {
			accessor->sparse_values_byte_offset = sparse_values_dict["byteOffset"];
		}
	}
	accessor->accessor_type = _get_accessor_type_from_str(p_dict["type"]);
	return accessor;
}

Dictionary GLTFAccessor::to_dictionary() const {
	Dictionary dict;
	if (buffer_view != -1) {
		// bufferView may be omitted to zero-initialize the buffer. When this happens, byteOffset MUST also be omitted.
		if (byte_offset > 0) {
			dict["byteOffset"] = byte_offset;
		}
		dict["bufferView"] = buffer_view;
	}
	dict["componentType"] = component_type;
	dict["count"] = count;
	dict["max"] = max;
	dict["min"] = min;
	dict["normalized"] = normalized;
	dict["type"] = _get_accessor_type_name();

	if (sparse_count > 0) {
		Dictionary sparse_indices_dict;
		sparse_indices_dict["bufferView"] = sparse_indices_buffer_view;
		sparse_indices_dict["componentType"] = sparse_indices_component_type;
		if (sparse_indices_byte_offset > 0) {
			sparse_indices_dict["byteOffset"] = sparse_indices_byte_offset;
		}
		Dictionary sparse_values_dict;
		sparse_values_dict["bufferView"] = sparse_values_buffer_view;
		if (sparse_values_byte_offset > 0) {
			sparse_values_dict["byteOffset"] = sparse_values_byte_offset;
		}
		Dictionary sparse_dict;
		sparse_dict["count"] = sparse_count;
		sparse_dict["indices"] = sparse_indices_dict;
		sparse_dict["values"] = sparse_values_dict;
		dict["sparse"] = sparse_dict;
	}
	return dict;
}
