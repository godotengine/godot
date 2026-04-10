/**************************************************************************/
/*  surface_data_rd.cpp                                                   */
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

#include "surface_data_rd.h"

#include "core/object/class_db.h"
#include "servers/rendering/rendering_server.h" // IWYU pragma: keep. For VARIANT_ENUM_CAST of RSE::PrimitiveType.

void SurfaceDataRD::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_primitive_type"), &SurfaceDataRD::get_primitive_type);
	ClassDB::bind_method(D_METHOD("get_format"), &SurfaceDataRD::get_format);
	ClassDB::bind_method(D_METHOD("get_vertex_count"), &SurfaceDataRD::get_vertex_count);
	ClassDB::bind_method(D_METHOD("get_index_count"), &SurfaceDataRD::get_index_count);
	ClassDB::bind_method(D_METHOD("get_max_vertex_count"), &SurfaceDataRD::get_max_vertex_count);
	ClassDB::bind_method(D_METHOD("get_max_index_count"), &SurfaceDataRD::get_max_index_count);
	ClassDB::bind_method(D_METHOD("get_vertex_buffer"), &SurfaceDataRD::get_vertex_buffer);
	ClassDB::bind_method(D_METHOD("get_attribute_buffer"), &SurfaceDataRD::get_attribute_buffer);
	ClassDB::bind_method(D_METHOD("get_skin_buffer"), &SurfaceDataRD::get_skin_buffer);
	ClassDB::bind_method(D_METHOD("get_index_buffer"), &SurfaceDataRD::get_index_buffer);
	ClassDB::bind_method(D_METHOD("get_indirect_buffer"), &SurfaceDataRD::get_indirect_buffer);
	ClassDB::bind_method(D_METHOD("get_indirect_buffer_offset"), &SurfaceDataRD::get_indirect_buffer_offset);
	ClassDB::bind_method(D_METHOD("get_material"), &SurfaceDataRD::get_material);
	ClassDB::bind_method(D_METHOD("get_input_mask"), &SurfaceDataRD::get_input_mask);
	ClassDB::bind_method(D_METHOD("get_lod_index_buffers"), &SurfaceDataRD::get_lod_index_buffers);
	ClassDB::bind_method(D_METHOD("get_vertex_attributes"), &SurfaceDataRD::get_vertex_attributes);
	ClassDB::bind_method(D_METHOD("get_vertex_buffers"), &SurfaceDataRD::get_vertex_buffers);
	ClassDB::bind_method(D_METHOD("get_vertex_buffer_offsets"), &SurfaceDataRD::get_vertex_buffer_offsets);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "primitive_type", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_primitive_type");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "format", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_format");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "vertex_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_vertex_count");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "index_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_index_count");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_vertex_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_max_vertex_count");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_index_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_max_index_count");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "vertex_buffer", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_vertex_buffer");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "attribute_buffer", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_attribute_buffer");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "skin_buffer", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_skin_buffer");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "index_buffer", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_index_buffer");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "indirect_buffer", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_indirect_buffer");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "indirect_buffer_offset", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_indirect_buffer_offset");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "material", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_material");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "input_mask", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_input_mask");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "lod_index_buffers", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_lod_index_buffers");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "vertex_attributes", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_vertex_attributes");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "vertex_buffers", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_vertex_buffers");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT64_ARRAY, "vertex_buffer_offsets", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "", "get_vertex_buffer_offsets");
}
