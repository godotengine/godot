/**************************************************************************/
/*  gltf_accessor.compat.inc                                              */
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

#ifndef DISABLE_DEPRECATED

// 32-bit versions for compatibility.

GLTFBufferViewIndex GLTFAccessor::_get_buffer_view_bind_compat_106220() {
	return get_buffer_view();
}

int GLTFAccessor::_get_byte_offset_bind_compat_106220() {
	return get_byte_offset();
}

int GLTFAccessor::_get_component_type_bind_compat_106220() {
	return (int)get_component_type();
}

void GLTFAccessor::_set_component_type_bind_compat_106220(int p_component_type) {
	set_component_type((GLTFComponentType)p_component_type);
}

bool GLTFAccessor::_get_normalized_bind_compat_106220() {
	return get_normalized();
}

int GLTFAccessor::_get_count_bind_compat_106220() {
	return get_count();
}

GLTFAccessor::GLTFAccessorType GLTFAccessor::_get_accessor_type_bind_compat_106220() {
	return get_accessor_type();
}

int GLTFAccessor::_get_type_bind_compat_106220() {
	return (int)get_accessor_type();
}

Vector<double> GLTFAccessor::_get_min_bind_compat_106220() {
	return get_min();
}

Vector<double> GLTFAccessor::_get_max_bind_compat_106220() {
	return get_max();
}

int GLTFAccessor::_get_sparse_count_bind_compat_106220() {
	return get_sparse_count();
}

int GLTFAccessor::_get_sparse_indices_buffer_view_bind_compat_106220() {
	return get_sparse_indices_buffer_view();
}

int GLTFAccessor::_get_sparse_indices_byte_offset_bind_compat_106220() {
	return get_sparse_indices_byte_offset();
}

int GLTFAccessor::_get_sparse_indices_component_type_bind_compat_106220() {
	return (int)get_sparse_indices_component_type();
}

void GLTFAccessor::_set_sparse_indices_component_type_bind_compat_106220(int p_sparse_indices_component_type) {
	set_sparse_indices_component_type((GLTFComponentType)p_sparse_indices_component_type);
}

int GLTFAccessor::_get_sparse_values_buffer_view_bind_compat_106220() {
	return get_sparse_values_buffer_view();
}

int GLTFAccessor::_get_sparse_values_byte_offset_bind_compat_106220() {
	return get_sparse_values_byte_offset();
}

void GLTFAccessor::_bind_compatibility_methods() {
	// 32-bit versions for compatibility.
	ClassDB::bind_compatibility_method(D_METHOD("get_buffer_view"), &GLTFAccessor::_get_buffer_view_bind_compat_106220);
	ClassDB::bind_compatibility_method(D_METHOD("get_byte_offset"), &GLTFAccessor::_get_byte_offset_bind_compat_106220);
	ClassDB::bind_compatibility_method(D_METHOD("get_component_type"), &GLTFAccessor::_get_component_type_bind_compat_106220);
	ClassDB::bind_compatibility_method(D_METHOD("set_component_type", "component_type"), &GLTFAccessor::_set_component_type_bind_compat_106220);
	ClassDB::bind_compatibility_method(D_METHOD("get_normalized"), &GLTFAccessor::_get_normalized_bind_compat_106220);
	ClassDB::bind_compatibility_method(D_METHOD("get_count"), &GLTFAccessor::_get_count_bind_compat_106220);
	ClassDB::bind_compatibility_method(D_METHOD("get_accessor_type"), &GLTFAccessor::_get_accessor_type_bind_compat_106220);
	ClassDB::bind_compatibility_method(D_METHOD("get_type"), &GLTFAccessor::_get_type_bind_compat_106220);
	ClassDB::bind_compatibility_method(D_METHOD("get_min"), &GLTFAccessor::_get_min_bind_compat_106220);
	ClassDB::bind_compatibility_method(D_METHOD("get_max"), &GLTFAccessor::_get_max_bind_compat_106220);
	ClassDB::bind_compatibility_method(D_METHOD("get_sparse_count"), &GLTFAccessor::_get_sparse_count_bind_compat_106220);
	ClassDB::bind_compatibility_method(D_METHOD("get_sparse_indices_buffer_view"), &GLTFAccessor::_get_sparse_indices_buffer_view_bind_compat_106220);
	ClassDB::bind_compatibility_method(D_METHOD("get_sparse_indices_byte_offset"), &GLTFAccessor::_get_sparse_indices_byte_offset_bind_compat_106220);
	ClassDB::bind_compatibility_method(D_METHOD("get_sparse_indices_component_type"), &GLTFAccessor::_get_sparse_indices_component_type_bind_compat_106220);
	ClassDB::bind_compatibility_method(D_METHOD("set_sparse_indices_component_type", "sparse_indices_component_type"), &GLTFAccessor::_set_sparse_indices_component_type_bind_compat_106220);
	ClassDB::bind_compatibility_method(D_METHOD("get_sparse_values_buffer_view"), &GLTFAccessor::_get_sparse_values_buffer_view_bind_compat_106220);
	ClassDB::bind_compatibility_method(D_METHOD("get_sparse_values_byte_offset"), &GLTFAccessor::_get_sparse_values_byte_offset_bind_compat_106220);
}

#endif // DISABLE_DEPRECATED
