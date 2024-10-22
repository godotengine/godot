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
	ClassDB::bind_method(D_METHOD("get_id"), &CameraFeed::get_id);
	ClassDB::bind_method(D_METHOD("get_name"), &CameraFeed::get_name);
	ClassDB::bind_method(D_METHOD("get_position"), &CameraFeed::get_position);
	ClassDB::bind_method(D_METHOD("get_width"), &CameraFeed::get_width);
	ClassDB::bind_method(D_METHOD("get_heigth"), &CameraFeed::get_height);
	ClassDB::bind_method(D_METHOD("get_datatype"), &CameraFeed::get_datatype);

	ClassDB::bind_method(D_METHOD("is_active"), &CameraFeed::is_active);
	ClassDB::bind_method(D_METHOD("set_active", "active"), &CameraFeed::set_active);

	ClassDB::bind_method(D_METHOD("get_name"), &CameraFeed::get_name);
	ClassDB::bind_method(D_METHOD("get_position"), &CameraFeed::get_position);

	// Note, for transform some feeds may override what the user sets (such as ARKit)
	ClassDB::bind_method(D_METHOD("get_transform"), &CameraFeed::get_transform);
	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &CameraFeed::set_transform);

	ClassDB::bind_method(D_METHOD("get_datatype"), &CameraFeed::get_datatype);

	ADD_SIGNAL(MethodInfo("frame_changed"));
	ADD_SIGNAL(MethodInfo("format_changed"));

	ADD_GROUP("Feed", "feed_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "feed_is_active"), "set_active", "is_active");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "feed_transform"), "set_transform", "get_transform");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "formats"), "", "get_formats");

	BIND_ENUM_CONSTANT(FEED_UNSPECIFIED);
	BIND_ENUM_CONSTANT(FEED_FRONT);
	BIND_ENUM_CONSTANT(FEED_BACK);

	BIND_ENUM_CONSTANT(FEED_UNSUPPORTED);
	BIND_ENUM_CONSTANT(FEED_RGB);
	BIND_ENUM_CONSTANT(FEED_RGBA);
	BIND_ENUM_CONSTANT(FEED_YCBCR);
	BIND_ENUM_CONSTANT(FEED_YCBCR_SEP);
	BIND_ENUM_CONSTANT(FEED_NV12);
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
			active = true;
		}
	} else {
		// just deactivate it
		deactivate_feed();
		active = false;
	}
}

String CameraFeed::get_name() const {
	return name;
}

int CameraFeed::get_width() const {
	return width;
}

int CameraFeed::get_height() const {
	return height;
}

CameraFeed::FeedPosition CameraFeed::get_position() const {
	return position;
}

CameraFeed::FeedDataType CameraFeed::get_datatype() const {
	return datatype;
}

Transform2D CameraFeed::get_transform() const {
	return transform;
}

void CameraFeed::set_transform(const Transform2D &p_transform) {
	transform = p_transform;
}

RID CameraFeed::get_texture() const {
	return texture;
}

Ref<Image> CameraFeed::get_image(RenderingServer::CanvasTextureChannel channel) {
	return channel_image[channel];
}

void CameraFeed::set_image(RenderingServer::CanvasTextureChannel channel, const Ref<Image> &image) {
	if (channel_image[channel] != image) {
		channel_image[channel] = image;
		RenderingServer::get_singleton()->free(channel_texture[channel]);
		channel_texture[channel] = RenderingServer::get_singleton()->texture_2d_create(image);
		RenderingServer::get_singleton()->canvas_texture_set_channel(texture, channel, channel_texture[channel]);
	} else {
		RenderingServer::get_singleton()->texture_2d_update(channel_texture[channel], image);
	}
}

void CameraFeed::set_image(RenderingServer::CanvasTextureChannel channel, uint8_t *data, size_t offset, size_t len) {
	Ref<Image> image = channel_image[channel];
	ERR_FAIL_COND_MSG(image.is_null(), "Channel not initialized");
	Vector<uint8_t> image_data = image->get_data();
	uint8_t *dest = image_data.ptrw();
	memcpy(dest, data + offset, len);
	image->set_data(image->get_width(), image->get_height(), false, image->get_format(), image_data);
	RenderingServer::get_singleton()->texture_2d_update(channel_texture[channel], image);
}

CameraFeed::CameraFeed() {
	// initialize our feed
	id = CameraServer::get_singleton()->get_free_id();
	name = "?";
	width = 0;
	height = 0;
	active = false;
	position = CameraFeed::FEED_UNSPECIFIED;
	datatype = CameraFeed::FEED_UNSUPPORTED;
	transform = Transform2D(1.0, 0.0, 0.0, -1.0, 0.0, 1.0);
	texture = RenderingServer::get_singleton()->canvas_texture_create();
}

CameraFeed::~CameraFeed() {
	// Free our textures
	ERR_FAIL_NULL(RenderingServer::get_singleton());
	RenderingServer::get_singleton()->free(texture);
	for (size_t i = 0; i < 3; i++) {
		RenderingServer::get_singleton()->free(channel_texture[i]);
	}
}

bool CameraFeed::activate_feed() {
	// nothing to do here
	return true;
}

void CameraFeed::deactivate_feed() {
	// nothing to do here
}