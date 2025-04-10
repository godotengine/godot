/**************************************************************************/
/*  camera_texture.cpp                                                    */
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

#include "camera_texture.h"

#include "servers/camera/camera_feed.h"

void CameraTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_camera_feed_id", "feed_id"), &CameraTexture::set_camera_feed_id);
	ClassDB::bind_method(D_METHOD("get_camera_feed_id"), &CameraTexture::get_camera_feed_id);

	ClassDB::bind_method(D_METHOD("set_which_feed", "which_feed"), &CameraTexture::set_which_feed);
	ClassDB::bind_method(D_METHOD("get_which_feed"), &CameraTexture::get_which_feed);

	ClassDB::bind_method(D_METHOD("set_camera_active", "active"), &CameraTexture::set_camera_active);
	ClassDB::bind_method(D_METHOD("get_camera_active"), &CameraTexture::get_camera_active);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "camera_feed_id"), "set_camera_feed_id", "get_camera_feed_id");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "which_feed"), "set_which_feed", "get_which_feed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "camera_is_active"), "set_camera_active", "get_camera_active");
	ADD_PROPERTY_DEFAULT("camera_is_active", false);
}

void CameraTexture::_on_format_changed() {
	// FIXME: `emit_changed` is more appropriate, but causes errors for some reason.
	callable_mp((Resource *)this, &Resource::emit_changed).call_deferred();
}

int CameraTexture::get_width() const {
	Ref<CameraFeed> feed = CameraServer::get_singleton()->get_feed_by_id(camera_feed_id);
	if (feed.is_valid()) {
		return feed->get_base_width();
	} else {
		return 0;
	}
}

int CameraTexture::get_height() const {
	Ref<CameraFeed> feed = CameraServer::get_singleton()->get_feed_by_id(camera_feed_id);
	if (feed.is_valid()) {
		return feed->get_base_height();
	} else {
		return 0;
	}
}

bool CameraTexture::has_alpha() const {
	return false;
}

RID CameraTexture::get_rid() const {
	Ref<CameraFeed> feed = CameraServer::get_singleton()->get_feed_by_id(camera_feed_id);
	if (feed.is_valid()) {
		return feed->get_texture(which_feed);
	} else {
		if (_texture.is_null()) {
			_texture = RenderingServer::get_singleton()->texture_2d_placeholder_create();
		}
		return _texture;
	}
}

Ref<Image> CameraTexture::get_image() const {
	return RenderingServer::get_singleton()->texture_2d_get(get_rid());
}

void CameraTexture::set_camera_feed_id(int p_new_id) {
	Ref<CameraFeed> feed = CameraServer::get_singleton()->get_feed_by_id(camera_feed_id);
	if (feed.is_valid()) {
		if (feed->is_connected("format_changed", callable_mp(this, &CameraTexture::_on_format_changed))) {
			feed->disconnect("format_changed", callable_mp(this, &CameraTexture::_on_format_changed));
		}
	}

	camera_feed_id = p_new_id;

	feed = CameraServer::get_singleton()->get_feed_by_id(camera_feed_id);
	if (feed.is_valid()) {
		feed->connect("format_changed", callable_mp(this, &CameraTexture::_on_format_changed));
	}

	notify_property_list_changed();
	callable_mp((Resource *)this, &Resource::emit_changed).call_deferred();
}

int CameraTexture::get_camera_feed_id() const {
	return camera_feed_id;
}

void CameraTexture::set_which_feed(CameraServer::FeedImage p_which) {
	which_feed = p_which;
	notify_property_list_changed();
	callable_mp((Resource *)this, &Resource::emit_changed).call_deferred();
}

CameraServer::FeedImage CameraTexture::get_which_feed() const {
	return which_feed;
}

void CameraTexture::set_camera_active(bool p_active) {
	Ref<CameraFeed> feed = CameraServer::get_singleton()->get_feed_by_id(camera_feed_id);
	if (feed.is_valid()) {
		feed->set_active(p_active);
		notify_property_list_changed();
		callable_mp((Resource *)this, &Resource::emit_changed).call_deferred();
	}
}

bool CameraTexture::get_camera_active() const {
	Ref<CameraFeed> feed = CameraServer::get_singleton()->get_feed_by_id(camera_feed_id);
	if (feed.is_valid()) {
		return feed->is_active();
	} else {
		return false;
	}
}

CameraTexture::CameraTexture() {}

CameraTexture::~CameraTexture() {
	if (_texture.is_valid()) {
		ERR_FAIL_NULL(RenderingServer::get_singleton());
		RenderingServer::get_singleton()->free(_texture);
	}
}
