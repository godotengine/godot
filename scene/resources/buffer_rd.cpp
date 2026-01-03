/**************************************************************************/
/*  texture_rd.cpp                                                        */
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

#include "buffer_rd.h"

////////////////////////////////////////////////////////////////////////////
// Texture2DRD

void BufferRD::_bind_methods() {
	// ClassDB::bind_method(D_METHOD("set_texture_rd_rid", "texture_rd_rid"), &Texture2DRD::set_texture_rd_rid);
	// ClassDB::bind_method(D_METHOD("get_texture_rd_rid"), &Texture2DRD::get_texture_rd_rid);

	// ADD_PROPERTY(PropertyInfo(Variant::RID, "texture_rd_rid", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_texture_rd_rid", "get_texture_rd_rid");
}


RID BufferRD::get_rid() const {
	if (buffer_rid.is_null()) {
		// We are in trouble, create something temporary.
		buffer_rid = RenderingServer::get_singleton()->texture_2d_placeholder_create();
	}

	return buffer_rid;
}

void BufferRD::set_buffer_rd_rid(RID p_buffer_rd_rid) {
	ERR_FAIL_NULL(RS::get_singleton());

	if (p_buffer_rd_rid.is_valid()) {
		RS::get_singleton()->call_on_render_thread(callable_mp(this, &BufferRD::_set_buffer_rd_rid).bind(p_buffer_rd_rid));
	} else if (buffer_rid.is_valid()) {
		RS::get_singleton()->free(buffer_rid);
		buffer_rid = RID();
		size = Size2i();

		notify_property_list_changed();
		emit_changed();
	}
}

void BufferRD::_set_buffer_rd_rid(RID p_buffer_rd_rid) {
	ERR_FAIL_NULL(RD::get_singleton());
	//ERR_FAIL_COND(!RD::get_singleton()->buffer_is_valid(p_buffer_rd_rid));

	buffer_rd_rid = p_buffer_rd_rid;

	if (buffer_rid.is_valid()) {
		// RS::get_singleton()->buffer_replace(buffer_rid, RS::get_singleton()->buffer_rd_create(p_buffer_rd_rid));
	} else {
		// buffer_rid = RS::get_singleton()->buffer_rd_create(p_buffer_rd_rid);
	}

	notify_property_list_changed();
	emit_changed();
}

RID BufferRD::get_buffer_rd_rid() const {
	return buffer_rd_rid;
}

BufferRD::BufferRD() {
	size = Size2i();
}

BufferRD::~BufferRD() {
	if (buffer_rid.is_valid()) {
		ERR_FAIL_NULL(RS::get_singleton());
		RS::get_singleton()->free(buffer_rid);
		buffer_rid = RID();
	}
}