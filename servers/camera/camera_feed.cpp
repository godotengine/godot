/**************************************************************************/
/*  camera_feed.cpp                                                       */
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

#include "camera_feed.h"

#include "servers/rendering_server.h"

void CameraFeed::_bind_methods() {
	// The setters prefixed with _ are only exposed so we can have feeds through GDExtension!
	// They should not be called by the end user.

	ClassDB::bind_method(D_METHOD("get_id"), &CameraFeed::get_id);

	ClassDB::bind_method(D_METHOD("is_active"), &CameraFeed::is_active);
	ClassDB::bind_method(D_METHOD("set_active", "active"), &CameraFeed::set_active);

	ClassDB::bind_method(D_METHOD("get_name"), &CameraFeed::get_name);
	ClassDB::bind_method(D_METHOD("_set_name", "name"), &CameraFeed::set_name);

	ClassDB::bind_method(D_METHOD("get_position"), &CameraFeed::get_position);
	ClassDB::bind_method(D_METHOD("_set_position", "position"), &CameraFeed::set_position);

	// Note, for transform some feeds may override what the user sets (such as ARKit)
	ClassDB::bind_method(D_METHOD("get_transform"), &CameraFeed::get_transform);
	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &CameraFeed::set_transform);

	ClassDB::bind_method(D_METHOD("get_format"), &CameraFeed::get_format);
	ClassDB::bind_method(D_METHOD("set_format"), &CameraFeed::set_format);

	ADD_GROUP("Feed", "feed_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "feed_is_active"), "set_active", "is_active");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "feed_transform"), "set_transform", "get_transform");

	BIND_ENUM_CONSTANT(FEED_UNSPECIFIED);
	BIND_ENUM_CONSTANT(FEED_FRONT);
	BIND_ENUM_CONSTANT(FEED_BACK);
}

int CameraFeed::get_id() const {
	return id;
}

bool CameraFeed::is_active() const {
	return active;
}

void CameraFeed::set_active(bool p_is_active) {
	if (p_is_active == active) {
		// all good
	} else if (p_is_active) {
		// attempt to activate this feed
		if (activate_feed()) {
			print_line("Activate " + name);
			active = true;
		}
	} else {
		// just deactivate it
		deactivate_feed();
		print_line("Deactivate " + name);
		active = false;
	}
}

String CameraFeed::get_name() const {
	return name;
}

void CameraFeed::set_name(String p_name) {
	name = p_name;
}

int CameraFeed::get_base_width() const {
	return base_width;
}

int CameraFeed::get_base_height() const {
	return base_height;
}

int CameraFeed::get_format() const {
	return format;
}

void CameraFeed::set_format(int p_format) {
	format = p_format;
}

CameraFeed::FeedPosition CameraFeed::get_position() const {
	return position;
}

void CameraFeed::set_position(CameraFeed::FeedPosition p_position) {
	position = p_position;
}

Transform2D CameraFeed::get_transform() const {
	return transform;
}

void CameraFeed::set_transform(const Transform2D &p_transform) {
	transform = p_transform;
}

RID CameraFeed::get_texture() {
	return texture;
}

void CameraFeed::set_texture(Ref<Image> &diffuse, Ref<Image> &normal) {
	if (diffuse != NULL) {
		if (diffuse_texture.is_null()) {
			diffuse_texture = RenderingServer::get_singleton()->texture_2d_create(diffuse);
			RenderingServer::get_singleton()->canvas_texture_set_channel(texture, RenderingServer::CANVAS_TEXTURE_CHANNEL_DIFFUSE, diffuse_texture);
		}
	}

	if (normal != NULL) {
		if (normal_texture.is_null()) {
			normal_texture = RenderingServer::get_singleton()->texture_2d_create(normal);
			RenderingServer::get_singleton()->canvas_texture_set_channel(texture, RenderingServer::CANVAS_TEXTURE_CHANNEL_NORMAL, normal_texture);
		}
	}
}

CameraFeed::CameraFeed() {
	// initialize our feed
	id = CameraServer::get_singleton()->get_free_id();
	name = "???";
	base_width = 0;
	base_height = 0;
	format = 0;
	active = false;
	position = CameraFeed::FEED_UNSPECIFIED;
	transform = Transform2D(1.0, 0.0, 0.0, -1.0, 0.0, 1.0);

	// Set textures
	texture = RenderingServer::get_singleton()->canvas_texture_create();
}

CameraFeed::~CameraFeed() {
	// Free our textures
	ERR_FAIL_NULL(RenderingServer::get_singleton());
	RenderingServer::get_singleton()->free(texture);
}

bool CameraFeed::activate_feed() {
	// nothing to do here
	return true;
}

void CameraFeed::deactivate_feed() {
	// nothing to do here
}
