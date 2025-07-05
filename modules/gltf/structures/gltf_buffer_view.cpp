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
	ERR_FAIL_COND_V_MSG(byte_stride > 0, Vector<uint8_t>(), "Buffer views with byte stride are not yet supported by this method.");
	const TypedArray<Vector<uint8_t>> &buffers = p_gltf_state->get_buffers();
	ERR_FAIL_INDEX_V(buffer, buffers.size(), Vector<uint8_t>());
	const PackedByteArray &buffer_data = buffers[buffer];
	const int64_t byte_end = byte_offset + byte_length;
	return buffer_data.slice(byte_offset, byte_end);
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
