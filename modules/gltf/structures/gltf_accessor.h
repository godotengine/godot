/**************************************************************************/
/*  gltf_accessor.h                                                       */
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

#pragma once

#include "../gltf_defines.h"

#include "gltf_buffer_view.h"

class GLTFAccessor : public Resource {
	GDCLASS(GLTFAccessor, Resource);
	friend class GLTFDocument;

public:
	enum GLTFAccessorType {
		TYPE_SCALAR,
		TYPE_VEC2,
		TYPE_VEC3,
		TYPE_VEC4,
		TYPE_MAT2,
		TYPE_MAT3,
		TYPE_MAT4,
	};

	enum GLTFComponentType {
		COMPONENT_TYPE_NONE = 0,
		COMPONENT_TYPE_SIGNED_BYTE = 5120,
		COMPONENT_TYPE_UNSIGNED_BYTE = 5121,
		COMPONENT_TYPE_SIGNED_SHORT = 5122,
		COMPONENT_TYPE_UNSIGNED_SHORT = 5123,
		COMPONENT_TYPE_SIGNED_INT = 5124,
		COMPONENT_TYPE_UNSIGNED_INT = 5125,
		COMPONENT_TYPE_SINGLE_FLOAT = 5126,
		COMPONENT_TYPE_DOUBLE_FLOAT = 5130,
		COMPONENT_TYPE_HALF_FLOAT = 5131,
		COMPONENT_TYPE_SIGNED_LONG = 5134,
		COMPONENT_TYPE_UNSIGNED_LONG = 5135,
	};

private:
	GLTFBufferViewIndex buffer_view = -1;
	int64_t byte_offset = 0;
	GLTFComponentType component_type = COMPONENT_TYPE_NONE;
	bool normalized = false;
	int64_t count = 0;
	GLTFAccessorType accessor_type = GLTFAccessorType::TYPE_SCALAR;
	Vector<double> min;
	Vector<double> max;
	int64_t sparse_count = 0;
	GLTFBufferViewIndex sparse_indices_buffer_view = 0;
	int64_t sparse_indices_byte_offset = 0;
	GLTFComponentType sparse_indices_component_type = COMPONENT_TYPE_NONE;
	GLTFBufferViewIndex sparse_values_buffer_view = 0;
	int64_t sparse_values_byte_offset = 0;

	// Trivial helper functions.
	void _calculate_min_and_max(const PackedFloat64Array &p_numbers);
	void _determine_pad_skip(int64_t &r_skip_every, int64_t &r_skip_bytes) const;
	int64_t _determine_padded_byte_count(int64_t p_raw_byte_size) const;
	PackedFloat64Array _filter_numbers(const PackedFloat64Array &p_numbers) const;
	static String _get_component_type_name(const GLTFComponentType p_component);
	static GLTFComponentType _get_indices_component_type_for_size(const int64_t p_size);
	static GLTFAccessorType _get_accessor_type_from_str(const String &p_string);
	String _get_accessor_type_name() const;
	int64_t _get_vector_size() const;
	static int64_t _get_numbers_per_variant_for_gltf(Variant::Type p_variant_type);
	static int64_t _get_bytes_per_component(const GLTFComponentType p_component_type);
	int64_t _get_bytes_per_vector() const;

	// Private decode functions.
	PackedInt64Array _decode_sparse_indices(const Ref<GLTFState> &p_gltf_state, const Vector<Ref<GLTFBufferView>> &p_buffer_views) const;
	template <typename T>
	Vector<T> _decode_raw_numbers(const Ref<GLTFState> &p_gltf_state, const Vector<Ref<GLTFBufferView>> &p_buffer_views, bool p_sparse_values) const;
	template <typename T>
	Vector<T> _decode_as_numbers(const Ref<GLTFState> &p_gltf_state) const;

	// Private encode functions.
	PackedFloat64Array _encode_variants_as_floats(const Array &p_input_data, Variant::Type p_variant_type) const;
	void _store_sparse_indices_into_state(const Ref<GLTFState> &p_gltf_state, const PackedInt64Array &p_sparse_indices, const bool p_deduplicate = true);

protected:
	static void _bind_methods();

#ifndef DISABLE_DEPRECATED
	// 32-bit and non-const versions for compatibility.
	GLTFBufferViewIndex _get_buffer_view_bind_compat_106220();
	int _get_byte_offset_bind_compat_106220();
	int _get_component_type_bind_compat_106220();
	void _set_component_type_bind_compat_106220(int p_component_type);
	bool _get_normalized_bind_compat_106220();
	int _get_count_bind_compat_106220();
	GLTFAccessorType _get_accessor_type_bind_compat_106220();
	int _get_type_bind_compat_106220();
	Vector<double> _get_min_bind_compat_106220();
	Vector<double> _get_max_bind_compat_106220();
	int _get_sparse_count_bind_compat_106220();
	int _get_sparse_indices_buffer_view_bind_compat_106220();
	int _get_sparse_indices_byte_offset_bind_compat_106220();
	int _get_sparse_indices_component_type_bind_compat_106220();
	void _set_sparse_indices_component_type_bind_compat_106220(int p_sparse_indices_component_type);
	int _get_sparse_values_buffer_view_bind_compat_106220();
	int _get_sparse_values_byte_offset_bind_compat_106220();
	static void _bind_compatibility_methods();
#endif // DISABLE_DEPRECATED

public:
	// Property getters and setters.
	GLTFBufferViewIndex get_buffer_view() const;
	void set_buffer_view(GLTFBufferViewIndex p_buffer_view);

	int64_t get_byte_offset() const;
	void set_byte_offset(int64_t p_byte_offset);

	GLTFComponentType get_component_type() const;
	void set_component_type(GLTFComponentType p_component_type);

	bool get_normalized() const;
	void set_normalized(bool p_normalized);

	int64_t get_count() const;
	void set_count(int64_t p_count);

	GLTFAccessorType get_accessor_type() const;
	void set_accessor_type(GLTFAccessorType p_accessor_type);

	int get_type() const;
	void set_type(int p_accessor_type);

	Vector<double> get_min() const;
	void set_min(const Vector<double> &p_min);

	Vector<double> get_max() const;
	void set_max(const Vector<double> &p_max);

	int64_t get_sparse_count() const;
	void set_sparse_count(int64_t p_sparse_count);

	GLTFBufferViewIndex get_sparse_indices_buffer_view() const;
	void set_sparse_indices_buffer_view(GLTFBufferViewIndex p_sparse_indices_buffer_view);

	int64_t get_sparse_indices_byte_offset() const;
	void set_sparse_indices_byte_offset(int64_t p_sparse_indices_byte_offset);

	GLTFComponentType get_sparse_indices_component_type() const;
	void set_sparse_indices_component_type(GLTFComponentType p_sparse_indices_component_type);

	GLTFBufferViewIndex get_sparse_values_buffer_view() const;
	void set_sparse_values_buffer_view(GLTFBufferViewIndex p_sparse_values_buffer_view);

	int64_t get_sparse_values_byte_offset() const;
	void set_sparse_values_byte_offset(int64_t p_sparse_values_byte_offset);

	bool is_equal_exact(const Ref<GLTFAccessor> &p_other) const;

	// High-level decode functions.
	PackedColorArray decode_as_colors(const Ref<GLTFState> &p_gltf_state) const;
	PackedFloat32Array decode_as_float32s(const Ref<GLTFState> &p_gltf_state) const;
	PackedFloat64Array decode_as_float64s(const Ref<GLTFState> &p_gltf_state) const;
	PackedInt32Array decode_as_int32s(const Ref<GLTFState> &p_gltf_state) const;
	PackedInt64Array decode_as_int64s(const Ref<GLTFState> &p_gltf_state) const;
	Vector<Quaternion> decode_as_quaternions(const Ref<GLTFState> &p_gltf_state) const;
	PackedVector2Array decode_as_vector2s(const Ref<GLTFState> &p_gltf_state) const;
	PackedVector3Array decode_as_vector3s(const Ref<GLTFState> &p_gltf_state) const;
	PackedVector4Array decode_as_vector4s(const Ref<GLTFState> &p_gltf_state) const;
	Array decode_as_variants(const Ref<GLTFState> &p_gltf_state, Variant::Type p_variant_type) const;

	// Low-level encode functions.
	static GLTFComponentType get_minimal_integer_component_type_from_ints(const PackedInt64Array &p_numbers);
	PackedByteArray encode_floats_as_bytes(const PackedFloat64Array &p_input_numbers);
	PackedByteArray encode_ints_as_bytes(const PackedInt64Array &p_input_numbers);
	PackedByteArray encode_variants_as_bytes(const Array &p_input_data, Variant::Type p_variant_type);

	GLTFAccessorIndex store_accessor_data_into_state(const Ref<GLTFState> &p_gltf_state, const PackedByteArray &p_data_bytes, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target = GLTFBufferView::TARGET_NONE, const GLTFBufferIndex p_buffer_index = 0, const bool p_deduplicate = true);
	static Ref<GLTFAccessor> make_new_accessor_without_data(GLTFAccessorType p_accessor_type = TYPE_SCALAR, GLTFComponentType p_component_type = COMPONENT_TYPE_SINGLE_FLOAT);

	// High-level encode functions.
	static GLTFAccessorIndex encode_new_accessor_from_colors(const Ref<GLTFState> &p_gltf_state, const PackedColorArray &p_input_data, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target = GLTFBufferView::TARGET_NONE, const bool p_deduplicate = true);
	static GLTFAccessorIndex encode_new_accessor_from_float64s(const Ref<GLTFState> &p_gltf_state, const PackedFloat64Array &p_input_data, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target = GLTFBufferView::TARGET_NONE, const bool p_deduplicate = true);
	static GLTFAccessorIndex encode_new_accessor_from_int32s(const Ref<GLTFState> &p_gltf_state, const PackedInt32Array &p_input_data, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target = GLTFBufferView::TARGET_NONE, const bool p_deduplicate = true);
	static GLTFAccessorIndex encode_new_accessor_from_int64s(const Ref<GLTFState> &p_gltf_state, const PackedInt64Array &p_input_data, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target = GLTFBufferView::TARGET_NONE, const bool p_deduplicate = true);
	static GLTFAccessorIndex encode_new_accessor_from_quaternions(const Ref<GLTFState> &p_gltf_state, const Vector<Quaternion> &p_input_data, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target = GLTFBufferView::TARGET_NONE, const bool p_deduplicate = true);
	static GLTFAccessorIndex encode_new_accessor_from_variants(const Ref<GLTFState> &p_gltf_state, const Array &p_input_data, Variant::Type p_variant_type, GLTFAccessorType p_accessor_type = TYPE_SCALAR, GLTFComponentType p_component_type = COMPONENT_TYPE_SINGLE_FLOAT, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target = GLTFBufferView::TARGET_NONE, const bool p_deduplicate = true);
	static GLTFAccessorIndex encode_new_accessor_from_vector2s(const Ref<GLTFState> &p_gltf_state, const PackedVector2Array &p_input_data, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target = GLTFBufferView::TARGET_NONE, const bool p_deduplicate = true);
	static GLTFAccessorIndex encode_new_accessor_from_vector3s(const Ref<GLTFState> &p_gltf_state, const PackedVector3Array &p_input_data, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target = GLTFBufferView::TARGET_NONE, const bool p_deduplicate = true);
	static GLTFAccessorIndex encode_new_accessor_from_vector4s(const Ref<GLTFState> &p_gltf_state, const PackedVector4Array &p_input_data, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target = GLTFBufferView::TARGET_NONE, const bool p_deduplicate = true);
	static GLTFAccessorIndex encode_new_accessor_from_vector4is(const Ref<GLTFState> &p_gltf_state, const Vector<Vector4i> &p_input_data, const GLTFBufferView::ArrayBufferTarget p_buffer_view_target = GLTFBufferView::TARGET_NONE, const bool p_deduplicate = true);
	static GLTFAccessorIndex encode_new_sparse_accessor_from_vec3s(const Ref<GLTFState> &p_gltf_state, const PackedVector3Array &p_input_data, const PackedVector3Array &p_base_reference_data, const double p_tolerance_multiplier = 1.0, const GLTFBufferView::ArrayBufferTarget p_main_buffer_view_target = GLTFBufferView::TARGET_NONE, const bool p_deduplicate = true);

	// Dictionary conversion.
	static Ref<GLTFAccessor> from_dictionary(const Dictionary &p_dict);
	Dictionary to_dictionary() const;
};

VARIANT_ENUM_CAST(GLTFAccessor::GLTFAccessorType);
VARIANT_ENUM_CAST(GLTFAccessor::GLTFComponentType);
