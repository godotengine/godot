/**************************************************************************/
/*  surface_data_rd.h                                                     */
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

#include "core/object/ref_counted.h"
#include "core/variant/typed_array.h"
#include "servers/rendering/rendering_server_enums.h"

class SurfaceDataRD : public RefCounted {
	GDCLASS(SurfaceDataRD, RefCounted);

protected:
	static void _bind_methods();

public:
	RenderingServerEnums::PrimitiveType primitive = RenderingServerEnums::PRIMITIVE_MAX;
	uint64_t format = 0;
	int vertex_count = 0;
	int index_count = 0;
	int max_vertex_count = 0;
	int max_index_count = 0;
	RID vertex_buffer;
	RID attribute_buffer;
	RID skin_buffer;
	RID index_buffer;
	RID indirect_buffer;
	int indirect_buffer_offset = 0;
	RID material;
	uint64_t input_mask = 0;
	TypedArray<RID> lod_index_buffers;
	Array vertex_attributes;
	TypedArray<RID> vertex_buffers;
	PackedInt64Array vertex_buffer_offsets;

	RenderingServerEnums::PrimitiveType get_primitive_type() const { return primitive; }
	uint64_t get_format() const { return format; }
	int get_vertex_count() const { return vertex_count; }
	int get_index_count() const { return index_count; }
	int get_max_vertex_count() const { return max_vertex_count; }
	int get_max_index_count() const { return max_index_count; }
	RID get_vertex_buffer() const { return vertex_buffer; }
	RID get_attribute_buffer() const { return attribute_buffer; }
	RID get_skin_buffer() const { return skin_buffer; }
	RID get_index_buffer() const { return index_buffer; }
	RID get_indirect_buffer() const { return indirect_buffer; }
	int get_indirect_buffer_offset() const { return indirect_buffer_offset; }
	RID get_material() const { return material; }
	uint64_t get_input_mask() const { return input_mask; }
	TypedArray<RID> get_lod_index_buffers() const { return lod_index_buffers; }
	Array get_vertex_attributes() const { return vertex_attributes; }
	TypedArray<RID> get_vertex_buffers() const { return vertex_buffers; }
	PackedInt64Array get_vertex_buffer_offsets() const { return vertex_buffer_offsets; }
};
