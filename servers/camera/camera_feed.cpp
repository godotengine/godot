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
	ClassDB::bind_method(D_METHOD("set_name", "name"), &CameraFeed::set_name);

	ClassDB::bind_method(D_METHOD("get_position"), &CameraFeed::get_position);
	ClassDB::bind_method(D_METHOD("_set_position", "position"), &CameraFeed::set_position);

	// Note, for transform some feeds may override what the user sets (such as ARKit)
	ClassDB::bind_method(D_METHOD("get_transform"), &CameraFeed::get_transform);
	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &CameraFeed::set_transform);

	ClassDB::bind_method(D_METHOD("_set_RGB_img", "rgb_img"), &CameraFeed::set_RGB_img);
	ClassDB::bind_method(D_METHOD("_set_YCbCr_img", "ycbcr_img"), &CameraFeed::set_YCbCr_img);
	ClassDB::bind_method(D_METHOD("set_external", "width", "height"), &CameraFeed::set_external);
	ClassDB::bind_method(D_METHOD("set_external_depthmap", "depthbuffer", "width", "height"), &CameraFeed::set_external_depthmap);
	ClassDB::bind_method(D_METHOD("is_depthmap_available"), &CameraFeed::is_depthmap_available);
	ClassDB::bind_method(D_METHOD("set_should_display_depthmap", "enabled"), &CameraFeed::set_should_display_depthmap);
	ClassDB::bind_method(D_METHOD("should_display_depthmap"), &CameraFeed::should_display_depthmap);
	ClassDB::bind_method(D_METHOD("set_max_depth_meters"), &CameraFeed::set_max_depth_meters);

	ClassDB::bind_method(D_METHOD("get_texture", "feed_image_type"), &CameraFeed::get_texture);
	ClassDB::bind_method(D_METHOD("get_texture_tex_id", "feed_image_type"), &CameraFeed::get_texture_tex_id);

	ClassDB::bind_method(D_METHOD("get_datatype"), &CameraFeed::get_datatype);

	ADD_GROUP("Feed", "feed_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "feed_is_active"), "set_active", "is_active");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "feed_transform"), "set_transform", "get_transform");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "display_ar_depthmap"), "set_should_display_depthmap", "should_display_depthmap");

	BIND_ENUM_CONSTANT(FEED_NOIMAGE);
	BIND_ENUM_CONSTANT(FEED_RGB);
	BIND_ENUM_CONSTANT(FEED_YCBCR);
	BIND_ENUM_CONSTANT(FEED_YCBCR_SEP);
	BIND_ENUM_CONSTANT(FEED_EXTERNAL);

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

uint64_t CameraFeed::get_texture_tex_id(CameraServer::FeedImage p_which) {
	return RenderingServer::get_singleton()->texture_get_native_handle(texture[p_which]);
}

CameraFeed::CameraFeed() {
	// initialize our feed
	id = CameraServer::get_singleton()->get_free_id();
	base_width = 0;
	base_height = 0;
	name = "???";
	active = false;
	datatype = CameraFeed::FEED_RGB;
	depth_map_datatype = CameraFeed::FEED_NOIMAGE;
	position = CameraFeed::FEED_UNSPECIFIED;
	transform = Transform2D(1.0, 0.0, 0.0, -1.0, 0.0, 1.0);
	texture[CameraServer::FEED_Y_IMAGE] = RenderingServer::get_singleton()->texture_2d_placeholder_create();
	texture[CameraServer::FEED_CBCR_IMAGE] = RenderingServer::get_singleton()->texture_2d_placeholder_create();
	texture[CameraServer::FEED_DEPTHMAP] = RenderingServer::get_singleton()->texture_2d_placeholder_create();
	depthmap_is_available = false;
	display_depthmap = false;
	maxDepthMeters = 30.f;
}

CameraFeed::CameraFeed(String p_name, FeedPosition p_position) {
	// initialize our feed
	id = CameraServer::get_singleton()->get_free_id();
	base_width = 0;
	base_height = 0;
	name = p_name;
	active = false;
	datatype = CameraFeed::FEED_NOIMAGE;
	depth_map_datatype = CameraFeed::FEED_NOIMAGE;
	position = p_position;
	transform = Transform2D(1.0, 0.0, 0.0, -1.0, 0.0, 1.0);
	texture[CameraServer::FEED_Y_IMAGE] = RenderingServer::get_singleton()->texture_2d_placeholder_create();
	texture[CameraServer::FEED_CBCR_IMAGE] = RenderingServer::get_singleton()->texture_2d_placeholder_create();
	texture[CameraServer::FEED_DEPTHMAP] = RenderingServer::get_singleton()->texture_2d_placeholder_create();
	depthmap_is_available = false;
	display_depthmap = false;
	maxDepthMeters = 30.f;
}

CameraFeed::~CameraFeed() {
	// Free our textures
	ERR_FAIL_NULL(RenderingServer::get_singleton());
	RenderingServer::get_singleton()->free(texture[CameraServer::FEED_Y_IMAGE]);
	RenderingServer::get_singleton()->free(texture[CameraServer::FEED_CBCR_IMAGE]);
	RenderingServer::get_singleton()->free(texture[CameraServer::FEED_DEPTHMAP]);
}

void CameraFeed::set_RGB_img(const Ref<Image> &p_rgb_img) {
	ERR_FAIL_COND(p_rgb_img.is_null());
	if (active) {
		int new_width = p_rgb_img->get_width();
		int new_height = p_rgb_img->get_height();

		if ((base_width != new_width) || (base_height != new_height)) {
			// We're assuming here that our camera image doesn't change around formats etc, allocate the whole lot...
			base_width = new_width;
			base_height = new_height;

			RID new_texture = RenderingServer::get_singleton()->texture_2d_create(p_rgb_img);
			RenderingServer::get_singleton()->texture_replace(texture[CameraServer::FEED_RGBA_IMAGE], new_texture);
		} else {
			RenderingServer::get_singleton()->texture_2d_update(texture[CameraServer::FEED_RGBA_IMAGE], p_rgb_img);
		}

		datatype = CameraFeed::FEED_RGB;
	}
}

void CameraFeed::set_YCbCr_img(const Ref<Image> &p_ycbcr_img) {
	ERR_FAIL_COND(p_ycbcr_img.is_null());
	if (active) {
		int new_width = p_ycbcr_img->get_width();
		int new_height = p_ycbcr_img->get_height();

		if ((base_width != new_width) || (base_height != new_height)) {
			// We're assuming here that our camera image doesn't change around formats etc, allocate the whole lot...
			base_width = new_width;
			base_height = new_height;

			RID new_texture = RenderingServer::get_singleton()->texture_2d_create(p_ycbcr_img);
			RenderingServer::get_singleton()->texture_replace(texture[CameraServer::FEED_RGBA_IMAGE], new_texture);
		} else {
			RenderingServer::get_singleton()->texture_2d_update(texture[CameraServer::FEED_RGBA_IMAGE], p_ycbcr_img);
		}

		datatype = CameraFeed::FEED_YCBCR;
	}
}

void CameraFeed::set_YCbCr_imgs(const Ref<Image> &p_y_img, const Ref<Image> &p_cbcr_img) {
	ERR_FAIL_COND(p_y_img.is_null());
	ERR_FAIL_COND(p_cbcr_img.is_null());
	if (active) {
		///@TODO investigate whether we can use thirdparty/misc/yuv2rgb.h here to convert our YUV data to RGB, our shader approach is potentially faster though..
		// Wondering about including that into multiple projects, may cause issues.
		// That said, if we convert to RGB, we could enable using texture resources again...

		int new_y_width = p_y_img->get_width();
		int new_y_height = p_y_img->get_height();

		if ((base_width != new_y_width) || (base_height != new_y_height)) {
			// We're assuming here that our camera image doesn't change around formats etc, allocate the whole lot...
			base_width = new_y_width;
			base_height = new_y_height;
			{
				RID new_texture = RenderingServer::get_singleton()->texture_2d_create(p_y_img);
				RenderingServer::get_singleton()->texture_replace(texture[CameraServer::FEED_Y_IMAGE], new_texture);
			}
			{
				RID new_texture = RenderingServer::get_singleton()->texture_2d_create(p_cbcr_img);
				RenderingServer::get_singleton()->texture_replace(texture[CameraServer::FEED_CBCR_IMAGE], new_texture);
			}
		} else {
			RenderingServer::get_singleton()->texture_2d_update(texture[CameraServer::FEED_Y_IMAGE], p_y_img);
			RenderingServer::get_singleton()->texture_2d_update(texture[CameraServer::FEED_CBCR_IMAGE], p_cbcr_img);
		}

		datatype = CameraFeed::FEED_YCBCR_SEP;
	}
}

void CameraFeed::set_external(int p_width, int p_height) {
	if ((base_width != p_width) || (base_height != p_height)) 
	{
		// We're assuming here that our camera image doesn't change around formats etc, allocate the whole lot...
		base_width = p_width;
		base_height = p_height;

		RID new_texture = RenderingServer::get_singleton()->texture_set_external(texture[CameraServer::FEED_YCBCR_IMAGE], p_width, p_height);
		RenderingServer::get_singleton()->texture_replace(texture[CameraServer::FEED_YCBCR_IMAGE], new_texture);
	}

	datatype = CameraFeed::FEED_EXTERNAL;
}

void CameraFeed::set_external_depthmap(const PackedByteArray& p_depthbuffer, int p_width, int p_height) {
	// We're assuming here that our camera image doesn't change around formats etc, allocate the whole lot...
	depthmap_is_available = true;
	depthmap_base_width = p_width;
	depthmap_base_height = p_height;

    // Create the image
    Ref<Image> image = Image::create_from_data(p_width, p_height, false, Image::FORMAT_RG8, p_depthbuffer); // Using L8 format for 8-bit per channel

    // Creating a texture from this image and replacing the placeholder
    RID new_texture = RS::get_singleton()->texture_2d_create(image);
	RenderingServer::get_singleton()->texture_replace(texture[CameraServer::FEED_DEPTHMAP], new_texture);

	depth_map_datatype = CameraFeed::FEED_EXTERNAL;
}

bool CameraFeed::is_depthmap_available() {
	return depthmap_is_available;
}

void CameraFeed::set_should_display_depthmap(bool p_enabled) {
	display_depthmap = p_enabled;
}

bool CameraFeed::should_display_depthmap() {
	return display_depthmap;
}

void CameraFeed::set_max_depth_meters(float p_maxDepthMeters) {
	maxDepthMeters = p_maxDepthMeters;
}

unsigned int CameraFeed::get_external_depthmap() {
	return depthmap_handle;
}

float CameraFeed::get_maxDepthMeters() {
	return maxDepthMeters;
}

int CameraFeed::get_depthmap_base_width() const {
	return depthmap_base_width;
}

int CameraFeed::get_depthmap_base_height() const {
	return depthmap_base_height;
}


bool CameraFeed::activate_feed() {
	// nothing to do here
	return true;
}

void CameraFeed::deactivate_feed() {
	// nothing to do here
}
