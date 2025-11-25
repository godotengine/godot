/**************************************************************************/
/*  gltf_buffer_view.cpp                                                  */
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

#include "gltf_buffer_view.h"
#include "gltf_buffer_view.compat.inc"

#include "../gltf_state.h"

void GLTFBufferView::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load_buffer_view_data", "state"), &GLTFBufferView::load_buffer_view_data);

	ClassDB::bind_static_method("GLTFBufferView", D_METHOD("from_dictionary", "dictionary"), &GLTFBufferView::from_dictionary);
	ClassDB::bind_method(D_METHOD("to_dictionary"), &GLTFBufferView::to_dictionary);

	ClassDB::bind_method(D_METHOD("get_buffer"), &GLTFBufferView::get_buffer);
	ClassDB::bind_method(D_METHOD("set_buffer", "buffer"), &GLTFBufferView::set_buffer);
	ClassDB::bind_method(D_METHOD("get_byte_offset"), &GLTFBufferView::get_byte_offset);
	ClassDB::bind_method(D_METHOD("set_byte_offset", "byte_offset"), &GLTFBufferView::set_byte_offset);
	ClassDB::bind_method(D_METHOD("get_byte_length"), &GLTFBufferView::get_byte_length);
	ClassDB::bind_method(D_METHOD("set_byte_length", "byte_length"), &GLTFBufferView::set_byte_length);
	ClassDB::bind_method(D_METHOD("get_byte_stride"), &GLTFBufferView::get_byte_stride);
	ClassDB::bind_method(D_METHOD("set_byte_stride", "byte_stride"), &GLTFBufferView::set_byte_stride);
	ClassDB::bind_method(D_METHOD("get_indices"), &GLTFBufferView::get_indices);
	ClassDB::bind_method(D_METHOD("set_indices", "indices"), &GLTFBufferView::set_indices);
	ClassDB::bind_method(D_METHOD("get_vertex_attributes"), &GLTFBufferView::get_vertex_attributes);
	ClassDB::bind_method(D_METHOD("set_vertex_attributes", "is_attributes"), &GLTFBufferView::set_vertex_attributes);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "buffer"), "set_buffer", "get_buffer"); // GLTFBufferIndex
	ADD_PROPERTY(PropertyInfo(Variant::INT, "byte_offset"), "set_byte_offset", "get_byte_offset"); // int
	ADD_PROPERTY(PropertyInfo(Variant::INT, "byte_length"), "set_byte_length", "get_byte_length"); // int
	ADD_PROPERTY(PropertyInfo(Variant::INT, "byte_stride"), "set_byte_stride", "get_byte_stride"); // int
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "indices"), "set_indices", "get_indices"); // bool
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "vertex_attributes"), "set_vertex_attributes", "get_vertex_attributes"); // bool
}

GLTFBufferIndex GLTFBufferView::get_buffer() const {
	return buffer;
}

void GLTFBufferView::set_buffer(GLTFBufferIndex p_buffer) {
	buffer = p_buffer;
}

int64_t GLTFBufferView::get_byte_offset() const {
	return byte_offset;
}

void GLTFBufferView::set_byte_offset(int64_t p_byte_offset) {
	byte_offset = p_byte_offset;
}

int64_t GLTFBufferView::get_byte_length() const {
	return byte_length;
}

void GLTFBufferView::set_byte_length(int64_t p_byte_length) {
	byte_length = p_byte_length;
}

int64_t GLTFBufferView::get_byte_stride() const {
	return byte_stride;
}

void GLTFBufferView::set_byte_stride(int64_t p_byte_stride) {
	byte_stride = p_byte_stride;
}

bool GLTFBufferView::get_indices() const {
	return indices;
}

void GLTFBufferView::set_indices(bool p_indices) {
	indices = p_indices;
}

bool GLTFBufferView::get_vertex_attributes() const {
	return vertex_attributes;
}

void GLTFBufferView::set_vertex_attributes(bool p_attributes) {
	vertex_attributes = p_attributes;
}

Vector<uint8_t> GLTFBufferView::load_buffer_view_data(const Ref<GLTFState> p_gltf_state) const {
	ERR_FAIL_COND_V(p_gltf_state.is_null(), Vector<uint8_t>());
	const Vector<PackedByteArray> &buffers = p_gltf_state->get_buffers();
	ERR_FAIL_INDEX_V(buffer, buffers.size(), Vector<uint8_t>());
	const PackedByteArray &buffer_data = buffers[buffer];
	const int64_t byte_end = byte_offset + byte_length;
	// Note that for buffer views with a byte stride, the parts of this data which get used may
	// only be determined in combination with the accessors that reference this buffer view.
	return buffer_data.slice(byte_offset, byte_end);
}

GLTFBufferViewIndex GLTFBufferView::write_new_buffer_view_into_state(const Ref<GLTFState> &p_gltf_state, const PackedByteArray &p_input_data, const int64_t p_alignment, const ArrayBufferTarget p_target, const int64_t p_byte_stride, const GLTFBufferIndex p_buffer_index, const bool p_deduplicate) {
	ERR_FAIL_COND_V_MSG(p_buffer_index < 0, -1, "Buffer index must be greater than or equal to zero.");
	const bool target_is_indices = p_target == ArrayBufferTarget::TARGET_ELEMENT_ARRAY_BUFFER;
	const bool target_is_vertex_attributes = p_target == ArrayBufferTarget::TARGET_ARRAY_BUFFER;
	if (target_is_vertex_attributes) {
		ERR_FAIL_COND_V_MSG(p_byte_stride < 4 || p_byte_stride % 4 != 0, -1, "glTF export: Vertex attributes using TARGET_ARRAY_BUFFER must have a byte stride that is a multiple of 4 as required by section 3.6.2.4 of the glTF specification.");
	}
	// Check for duplicate buffer views before adding a new one.
	Vector<Ref<GLTFBufferView>> state_buffer_views = p_gltf_state->get_buffer_views();
	const int buffer_view_index = state_buffer_views.size();
	if (p_deduplicate) {
		for (int i = 0; i < buffer_view_index; i++) {
			const Ref<GLTFBufferView> &existing_buffer_view = state_buffer_views[i];
			if (existing_buffer_view->get_byte_offset() % p_alignment == 0 &&
					existing_buffer_view->get_byte_length() == p_input_data.size() &&
					existing_buffer_view->get_byte_stride() == p_byte_stride &&
					existing_buffer_view->get_indices() == target_is_indices &&
					existing_buffer_view->get_vertex_attributes() == target_is_vertex_attributes) {
				if (existing_buffer_view->load_buffer_view_data(p_gltf_state) == p_input_data) {
					// Duplicate found, return the index of the existing buffer view.
					return i;
				}
			}
		}
	}
	// Write the data into the buffer at the specified index.
	Vector<PackedByteArray> state_buffers = p_gltf_state->get_buffers();
	if (state_buffers.size() <= p_buffer_index) {
		state_buffers.resize(p_buffer_index + 1);
	}
	PackedByteArray state_buffer = state_buffers[p_buffer_index];
	const int64_t input_data_size = p_input_data.size();
	// This is used by accessors. The byte offset of an accessor MUST be a multiple of the accessor's component size.
	// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#data-alignment
	int64_t byte_offset = state_buffer.size();
	if (byte_offset % p_alignment != 0) {
		byte_offset += p_alignment - (byte_offset % p_alignment);
	}
	state_buffer.resize(byte_offset + input_data_size);
	uint8_t *buffer_ptr = state_buffer.ptrw();
	memcpy(buffer_ptr + byte_offset, p_input_data.ptr(), input_data_size);
	state_buffers.set(p_buffer_index, state_buffer);
	p_gltf_state->set_buffers(state_buffers);
	// Create a new GLTFBufferView that references the new buffer.
	Ref<GLTFBufferView> buffer_view;
	buffer_view.instantiate();
	buffer_view->set_buffer(p_buffer_index);
	buffer_view->set_byte_offset(byte_offset);
	buffer_view->set_byte_length(input_data_size);
	buffer_view->set_byte_stride(p_byte_stride);
	buffer_view->set_indices(target_is_indices);
	buffer_view->set_vertex_attributes(target_is_vertex_attributes);
	// Add the new buffer view to the state.
	state_buffer_views.append(buffer_view);
	p_gltf_state->set_buffer_views(state_buffer_views);
	return buffer_view_index;
}

Ref<GLTFBufferView> GLTFBufferView::from_dictionary(const Dictionary &p_dict) {
	// See https://github.com/KhronosGroup/glTF/blob/main/specification/2.0/schema/bufferView.schema.json
	Ref<GLTFBufferView> buffer_view;
	buffer_view.instantiate();
	if (p_dict.has("buffer")) {
		buffer_view->set_buffer(p_dict["buffer"]);
	}
	if (p_dict.has("byteLength")) {
		buffer_view->set_byte_length(p_dict["byteLength"]);
	}
	if (p_dict.has("byteOffset")) {
		buffer_view->set_byte_offset(p_dict["byteOffset"]);
	}
	if (p_dict.has("byteStride")) {
		buffer_view->byte_stride = p_dict["byteStride"];
		if (buffer_view->byte_stride < 4 || buffer_view->byte_stride > 252 || buffer_view->byte_stride % 4 != 0) {
			ERR_PRINT("glTF import: Invalid byte stride " + itos(buffer_view->byte_stride) + " for buffer view. If defined, byte stride must be a multiple of 4 and between 4 and 252.");
		}
	}
	if (p_dict.has("target")) {
		const int target = p_dict["target"];
		buffer_view->indices = target == ArrayBufferTarget::TARGET_ELEMENT_ARRAY_BUFFER;
		buffer_view->vertex_attributes = target == ArrayBufferTarget::TARGET_ARRAY_BUFFER;
	}
	return buffer_view;
}

Dictionary GLTFBufferView::to_dictionary() const {
	Dictionary dict;
	ERR_FAIL_COND_V_MSG(buffer == -1, dict, "Buffer index must be set to a valid buffer before converting to Dictionary.");
	dict["buffer"] = buffer;
	dict["byteLength"] = byte_length;
	if (byte_offset != 0) {
		dict["byteOffset"] = byte_offset;
	}
	if (byte_stride != -1) {
		dict["byteStride"] = byte_stride;
	}
	if (indices) {
		dict["target"] = ArrayBufferTarget::TARGET_ELEMENT_ARRAY_BUFFER;
	} else if (vertex_attributes) {
		dict["target"] = ArrayBufferTarget::TARGET_ARRAY_BUFFER;
	}
	return dict;
}
