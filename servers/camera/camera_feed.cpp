/*************************************************************************/
/*  camera_feed.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "camera_feed.h"

#include "servers/rendering_server.h"

void CameraFeed::_bind_methods() {
// FIXME: Disabled during Vulkan refactoring, should be ported.
#if 0
	// The setters prefixed with _ are only exposed so we can have feeds through GDNative!
	// They should not be called by the end user.

	ClassDB::bind_method(D_METHOD("get_id"), &CameraFeed::get_id);
	ClassDB::bind_method(D_METHOD("get_name"), &CameraFeed::get_name);
	ClassDB::bind_method(D_METHOD("_set_name", "name"), &CameraFeed::set_name);

	ClassDB::bind_method(D_METHOD("is_active"), &CameraFeed::is_active);
	ClassDB::bind_method(D_METHOD("set_active", "active"), &CameraFeed::set_active);

	ClassDB::bind_method(D_METHOD("get_position"), &CameraFeed::get_position);
	ClassDB::bind_method(D_METHOD("_set_position", "position"), &CameraFeed::set_position);

	// Note, for transform some feeds may override what the user sets (such as ARKit)
	ClassDB::bind_method(D_METHOD("get_transform"), &CameraFeed::get_transform);
	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &CameraFeed::set_transform);

	ClassDB::bind_method(D_METHOD("_set_RGB_img", "rgb_img"), &CameraFeed::set_RGB_img);
	ClassDB::bind_method(D_METHOD("_set_YCbCr_img", "ycbcr_img"), &CameraFeed::set_YCbCr_img);
	ClassDB::bind_method(D_METHOD("_allocate_texture", "width", "height", "format", "texture_type", "data_type"), &CameraFeed::allocate_texture);

	ADD_GROUP("Feed", "feed_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "feed_is_active"), "set_active", "is_active");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "feed_transform"), "set_transform", "get_transform");

	BIND_ENUM_CONSTANT(FEED_NOIMAGE);
	BIND_ENUM_CONSTANT(FEED_RGB);
	BIND_ENUM_CONSTANT(FEED_YCBCR);
	BIND_ENUM_CONSTANT(FEED_YCBCR_SEP);

	BIND_ENUM_CONSTANT(FEED_UNSPECIFIED);
	BIND_ENUM_CONSTANT(FEED_FRONT);
	BIND_ENUM_CONSTANT(FEED_BACK);
#endif
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

CameraFeed::FeedDataType CameraFeed::get_datatype() const {
	return datatype;
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

RID CameraFeed::get_texture(CameraServer::FeedImage p_which) {
	return texture[p_which];
}

CameraFeed::CameraFeed() {
	// initialize our feed
	id = CameraServer::get_singleton()->get_free_id();
	name = "???";
	active = false;
	datatype = CameraFeed::FEED_RGB;
	position = CameraFeed::FEED_UNSPECIFIED;
	transform = Transform2D(1.0, 0.0, 0.0, -1.0, 0.0, 1.0);

// FIXME: Disabled during Vulkan refactoring, should be ported.
#if 0
	// create a texture object
	RenderingServer *vs = RenderingServer::get_singleton();
	texture[CameraServer::FEED_Y_IMAGE] = vs->texture_create(); // also used for RGBA
	texture[CameraServer::FEED_CBCR_IMAGE] = vs->texture_create();
#endif
}

CameraFeed::CameraFeed(String p_name, FeedPosition p_position) {
	// initialize our feed
	id = CameraServer::get_singleton()->get_free_id();
	base_width = 0;
	base_height = 0;
	name = p_name;
	active = false;
	datatype = CameraFeed::FEED_NOIMAGE;
	position = p_position;
	transform = Transform2D(1.0, 0.0, 0.0, -1.0, 0.0, 1.0);

// FIXME: Disabled during Vulkan refactoring, should be ported.
#if 0
	// create a texture object
	RenderingServer *vs = RenderingServer::get_singleton();
	texture[CameraServer::FEED_Y_IMAGE] = vs->texture_create(); // also used for RGBA
	texture[CameraServer::FEED_CBCR_IMAGE] = vs->texture_create();
#endif
}

CameraFeed::~CameraFeed() {
// FIXME: Disabled during Vulkan refactoring, should be ported.
#if 0
	// Free our textures
	RenderingServer *vs = RenderingServer::get_singleton();
	vs->free(texture[CameraServer::FEED_Y_IMAGE]);
	vs->free(texture[CameraServer::FEED_CBCR_IMAGE]);
#endif
}

void CameraFeed::set_RGB_img(const Ref<Image> &p_rgb_img) {
// FIXME: Disabled during Vulkan refactoring, should be ported.
#if 0
	ERR_FAIL_COND(p_rgb_img.is_null());
	if (active) {
		RenderingServer *vs = RenderingServer::get_singleton();

		int new_width = p_rgb_img->get_width();
		int new_height = p_rgb_img->get_height();

		if ((base_width != new_width) || (base_height != new_height)) {
			// We're assuming here that our camera image doesn't change around formats etc, allocate the whole lot...
			base_width = new_width;
			base_height = new_height;

			vs->texture_allocate(texture[CameraServer::FEED_RGBA_IMAGE], new_width, new_height, 0, Image::FORMAT_RGB8, RS::TEXTURE_TYPE_2D, RS::TEXTURE_FLAGS_DEFAULT);
		}

		vs->texture_set_data(texture[CameraServer::FEED_RGBA_IMAGE], p_rgb_img);
		datatype = CameraFeed::FEED_RGB;
	}
#endif
}

void CameraFeed::set_YCbCr_img(const Ref<Image> &p_ycbcr_img) {
// FIXME: Disabled during Vulkan refactoring, should be ported.
#if 0
	ERR_FAIL_COND(p_ycbcr_img.is_null());
	if (active) {
		RenderingServer *vs = RenderingServer::get_singleton();

		int new_width = p_ycbcr_img->get_width();
		int new_height = p_ycbcr_img->get_height();

		if ((base_width != new_width) || (base_height != new_height)) {
			// We're assuming here that our camera image doesn't change around formats etc, allocate the whole lot...
			base_width = new_width;
			base_height = new_height;

			vs->texture_allocate(texture[CameraServer::FEED_RGBA_IMAGE], new_width, new_height, 0, Image::FORMAT_RGB8, RS::TEXTURE_TYPE_2D, RS::TEXTURE_FLAGS_DEFAULT);
		}

		vs->texture_set_data(texture[CameraServer::FEED_RGBA_IMAGE], p_ycbcr_img);
		datatype = CameraFeed::FEED_YCBCR;
	}
#endif
}

void CameraFeed::set_YCbCr_imgs(const Ref<Image> &p_y_img, const Ref<Image> &p_cbcr_img) {
// FIXME: Disabled during Vulkan refactoring, should be ported.
#if 0
	ERR_FAIL_COND(p_y_img.is_null());
	ERR_FAIL_COND(p_cbcr_img.is_null());
	if (active) {
		RenderingServer *vs = RenderingServer::get_singleton();

		///@TODO investigate whether we can use thirdparty/misc/yuv2rgb.h here to convert our YUV data to RGB, our shader approach is potentially faster though..
		// Wondering about including that into multiple projects, may cause issues.
		// That said, if we convert to RGB, we could enable using texture resources again...

		int new_y_width = p_y_img->get_width();
		int new_y_height = p_y_img->get_height();
		int new_cbcr_width = p_cbcr_img->get_width();
		int new_cbcr_height = p_cbcr_img->get_height();

		if ((base_width != new_y_width) || (base_height != new_y_height)) {
			// We're assuming here that our camera image doesn't change around formats etc, allocate the whole lot...
			base_width = new_y_width;
			base_height = new_y_height;

			vs->texture_allocate(texture[CameraServer::FEED_Y_IMAGE], new_y_width, new_y_height, 0, Image::FORMAT_R8, RS::TEXTURE_TYPE_2D, RS::TEXTURE_FLAG_USED_FOR_STREAMING);

			///@TODO GLES2 doesn't support FORMAT_RG8, need to do some form of conversion
			vs->texture_allocate(texture[CameraServer::FEED_CBCR_IMAGE], new_cbcr_width, new_cbcr_height, 0, Image::FORMAT_RG8, RS::TEXTURE_TYPE_2D, RS::TEXTURE_FLAG_USED_FOR_STREAMING);
		}

		vs->texture_set_data(texture[CameraServer::FEED_Y_IMAGE], p_y_img);
		vs->texture_set_data(texture[CameraServer::FEED_CBCR_IMAGE], p_cbcr_img);
		datatype = CameraFeed::FEED_YCBCR_SEP;
	}
#endif
}

// FIXME: Disabled during Vulkan refactoring, should be ported.
#if 0
void CameraFeed::allocate_texture(int p_width, int p_height, Image::Format p_format, RenderingServer::TextureType p_texture_type, FeedDataType p_data_type) {
	RenderingServer *vs = RenderingServer::get_singleton();

	if ((base_width != p_width) || (base_height != p_height)) {
		// We're assuming here that our camera image doesn't change around formats etc, allocate the whole lot...
		base_width = p_width;
		base_height = p_height;

		vs->texture_allocate(texture[0], p_width, p_height, 0, p_format, p_texture_type, RS::TEXTURE_FLAGS_DEFAULT);
	}

	datatype = p_data_type;
}
#endif

bool CameraFeed::activate_feed() {
	// nothing to do here
	return true;
}

void CameraFeed::deactivate_feed() {
	// nothing to do here
}
