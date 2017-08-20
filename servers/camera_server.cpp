/*************************************************************************/
/*  camera_server.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "camera_server.h"
#include "visual_server.h"

// Include this for now until we find the proper way to load our textures
#include "platform_config.h"
#ifndef GLES3_INCLUDE_H
#include <GLES3/gl3.h>
#else
#include GLES3_INCLUDE_H
#endif

////////////////////////////////////////////////////////
// CameraFeed

int CameraFeed::get_id() const {
	return id;
};

bool CameraFeed::get_is_active() const {
	return active;
};

void CameraFeed::set_is_active(bool p_is_active) {
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
	};
};

String CameraFeed::get_name() const {
	return name;
};

CameraFeed::FeedDataType CameraFeed::get_datatype() const {
	return datatype;
};

CameraFeed::FeedPosition CameraFeed::get_position() const {
	return position;
};

RID CameraFeed::get_texture(int p_which) {
	int set;

	if (state == FEED_WAITING_ON_0) {
		// still waiting on set 0 to be updated, so return 1
		set = 1;
	} else if (state == FEED_UPDATING_0) {
		// set 0 is still being updated, so return 1
		set = 1;
	} else if (state == FEED_0_IS_AVAILABLE) {
		// set 0 is available, we'll start using it and tell our camera to update set 1
		set = 0;
		if ((datatype == FEED_RGB) || (p_which == 1)) {
			state = FEED_WAITING_ON_1;
		}
	} else if (state == FEED_WAITING_ON_1) {
		// still waiting on set 1 to be updated, so return 0
		set = 0;
	} else if (state == FEED_UPDATING_1) {
		// set 1 is still being updated, so return 0
		set = 0;
	} else if (state == FEED_1_IS_AVAILABLE) {
		// set 1 is available, we'll start using it and tell our camera to update set 1
		set = 1;
		if ((datatype == FEED_RGB) || (p_which == 1)) {
			state = FEED_WAITING_ON_0;
		}
	};

	return texture[set][p_which];
};

CameraFeed::CameraFeed() {
	// initialize our feed
	id = CameraServer::get_singleton()->get_free_id();
	name = "???";
	active = false;
	datatype = CameraFeed::FEED_RGB;
	position = CameraFeed::FEED_UNSPECIFIED;

	// create a texture object
	///@TODO rewrite this so we only need one texture but can support two planes, or that we always convert to RGB
	VisualServer *vs = VisualServer::get_singleton();
	texture[0][0] = vs->texture_create();
	texture[0][1] = vs->texture_create();
	texture[1][0] = vs->texture_create();
	texture[1][1] = vs->texture_create();

	state = FEED_WAITING_ON_0;
};

CameraFeed::CameraFeed(String p_name, FeedPosition p_position) {
	// initialize our feed
	id = CameraServer::get_singleton()->get_free_id();
	base_width = 0;
	base_height = 0;
	name = p_name;
	active = false;
	datatype = CameraFeed::FEED_NOIMAGE;
	position = p_position;

	// create a texture object
	///@TODO rewrite this so we only need one texture but can support two planes, or that we always convert to RGB
	VisualServer *vs = VisualServer::get_singleton();
	texture[0][0] = vs->texture_create();
	texture[0][1] = vs->texture_create();
	texture[1][0] = vs->texture_create();
	texture[1][1] = vs->texture_create();
	state = FEED_WAITING_ON_0;
};

CameraFeed::~CameraFeed() {
	// Free our textures
	VisualServer *vs = VisualServer::get_singleton();
	vs->free(texture[0][0]);
	vs->free(texture[0][1]);
	vs->free(texture[1][0]);
	vs->free(texture[1][1]);
};

int CameraFeed::write_to_set() {
	if (state == FEED_WAITING_ON_0) {
		state = FEED_UPDATING_0;
		return 0;
	} else if (state == FEED_WAITING_ON_1) {
		state = FEED_UPDATING_1;
		return 1;
	} else {
		return -1;
	};
};

bool CameraFeed::is_waiting() {
	if (state == FEED_WAITING_ON_0) {
		return true;
	} else if (state == FEED_WAITING_ON_1) {
		return true;
	} else {
		return false;
	};
};

void CameraFeed::set_texture_data_RGB(unsigned char *p_data, int p_width, int p_height) {
	int set = write_to_set();
	if (set >= 0) {
		VisualServer *vs = VisualServer::get_singleton();

		if ((base_width != p_width) || (base_height != p_height)) {
			// We're assuming here that our camera image doesn't change around formats etc, allocate the whole lot...
			base_width = p_width;
			base_height = p_height;

			vs->texture_allocate(texture[0][0], p_width, p_height, Image::FORMAT_RGB8, VS::TEXTURE_FLAGS_DEFAULT);
			img_data[0][0].resize(p_width * p_height * 3);
			vs->texture_allocate(texture[1][0], p_width, p_height, Image::FORMAT_RGB8, VS::TEXTURE_FLAGS_DEFAULT);
			img_data[1][0].resize(p_width * p_height * 3);
		};

		PoolVector<uint8_t>::Write w = img_data[set][0].write();
		memcpy(w.ptr(), p_data, p_width * p_height * 3);

		Ref<Image> img;
		img.instance();
		img->create(p_width, p_height, 0, Image::FORMAT_RGB8, img_data[set][0]);
		vs->texture_set_data(texture[set][0], img);

		datatype = CameraFeed::FEED_RGB;

		state = set == 0 ? FEED_0_IS_AVAILABLE : FEED_1_IS_AVAILABLE;
	};
};

void CameraFeed::set_texture_data_YCbCr(unsigned char *p_y_data, int p_y_width, int p_y_height, unsigned char *p_cbcr_data, int p_cbcr_width, int p_cbcr_height) {
	int set = write_to_set();
	if (set >= 0) {
		VisualServer *vs = VisualServer::get_singleton();

		///@TODO investigate whether we can use thirdparty/misc/yuv2rgb.h here to convert our YUV data to RGB, our shader approach is potentially faster though..
		// Wondering about including that into multiple projects, may cause issues.
		// That said, if we convert to RGB, we could enable using texture resources again...

		if ((base_width != p_y_width) || (base_height != p_y_height)) {
			// We're assuming here that our camera image doesn't change around formats etc, allocate the whole lot...
			base_width = p_y_width;
			base_height = p_y_height;

			vs->texture_allocate(texture[0][0], p_y_width, p_y_height, Image::FORMAT_R8, VS::TEXTURE_FLAG_USED_FOR_STREAMING);
			img_data[0][0].resize(p_y_width * p_y_height);
			vs->texture_allocate(texture[1][0], p_y_width, p_y_height, Image::FORMAT_R8, VS::TEXTURE_FLAG_USED_FOR_STREAMING);
			img_data[1][0].resize(p_y_width * p_y_height);

			vs->texture_allocate(texture[0][1], p_cbcr_width, p_cbcr_height, Image::FORMAT_RG8, VS::TEXTURE_FLAGS_DEFAULT);
			img_data[0][1].resize(p_cbcr_width * p_cbcr_height * 2);
			vs->texture_allocate(texture[1][1], p_cbcr_width, p_cbcr_height, Image::FORMAT_RG8, VS::TEXTURE_FLAGS_DEFAULT);
			img_data[1][1].resize(p_cbcr_width * p_cbcr_height * 2);
		};

		{
			PoolVector<uint8_t>::Write w = img_data[set][0].write();
			memcpy(w.ptr(), p_y_data, p_y_width * p_y_height);

			Ref<Image> img;
			img.instance();
			img->create(p_y_width, p_y_height, 0, Image::FORMAT_R8, img_data[set][0]);
			vs->texture_set_data(texture[set][0], img);
		}

		{
			PoolVector<uint8_t>::Write w = img_data[set][1].write();
			memcpy(w.ptr(), p_cbcr_data, p_cbcr_width * p_cbcr_height * 2);

			Ref<Image> img;
			img.instance();
			img->create(p_cbcr_width, p_cbcr_height, 0, Image::FORMAT_RG8, img_data[set][1]);
			vs->texture_set_data(texture[set][1], img);
		}

		datatype = CameraFeed::FEED_YCbCr;

		state = set == 0 ? FEED_0_IS_AVAILABLE : FEED_1_IS_AVAILABLE;
	};
};

bool CameraFeed::activate_feed() {
	// nothing to do here
	return true;
};

void CameraFeed::deactivate_feed(){
	// nothing to do here
};

////////////////////////////////////////////////////////
// CameraServer

void CameraServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("feeds"), &CameraServer::get_feeds);
	ClassDB::bind_method(D_METHOD("feed_is_active", "id"), &CameraServer::feed_is_active);
	ClassDB::bind_method(D_METHOD("feed_set_active", "id", "active"), &CameraServer::feed_set_active);

	ADD_SIGNAL(MethodInfo("camera_feed_added", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("camera_feed_removed", PropertyInfo(Variant::INT, "id")));
};

CameraServer *CameraServer::singleton = NULL;

CameraServer *CameraServer::get_singleton() {
	return singleton;
};

int CameraServer::get_free_id() {
	bool id_exists = true;
	int newid = 0;

	// find a free id
	while (id_exists) {
		newid++;
		id_exists = false;
		for (int i = 0; i < feeds.size() && !id_exists; i++) {
			if (feeds[i]->get_id() == newid) {
				id_exists = true;
			};
		};
	};

	return newid;
};

int CameraServer::get_feed_index(int p_id) const {
	for (int i = 0; i < feeds.size(); i++) {
		if (feeds[i]->get_id() == p_id) {
			return i;
		};
	};

	return -1;
};

CameraFeed *CameraServer::get_feed(int p_id) {
	int index = get_feed_index(p_id);

	if (index == -1) {
		return NULL;
	} else {
		return feeds[index];
	}
};

void CameraServer::add_feed(CameraFeed *p_feed) {
	// add our feed
	feeds.push_back(p_feed);

// record for debugging
#ifdef DEBUG_ENABLED
	print_line("Registered camera " + p_feed->get_name() + " with id " + itos(p_feed->get_id()) + " position " + itos(p_feed->get_position()) + " at index " + itos(feeds.size() - 1));
#endif

	// let whomever is interested know
	emit_signal("camera_feed_added", p_feed->get_id());
};

void CameraServer::remove_feed(int p_id) {
	for (int i = 0; i < feeds.size(); i++) {
		if (feeds[i]->get_id() == p_id) {
			CameraFeed *remove_feed = feeds[i];

// record for debugging
#ifdef DEBUG_ENABLED
			print_line("Removed camera " + remove_feed->get_name() + " with id " + itos(remove_feed->get_id()) + " position " + itos(remove_feed->get_position()));
#endif

			// remove it from our array
			feeds.remove(i);

			// and delete our object
			delete remove_feed;

			// let whomever is interested know
			emit_signal("camera_feed_removed", p_id);
			return;
		};
	};
};

Array CameraServer::get_feeds() const {
	Array return_feeds;

	for (int i = 0; i < feeds.size(); i++) {
		Dictionary feed_info;

		feed_info["id"] = feeds[i]->get_id();
		feed_info["name"] = feeds[i]->get_name();
		feed_info["position"] = feeds[i]->get_position();

		return_feeds.push_back(feed_info);
	};

	return return_feeds;
};

bool CameraServer::feed_is_active(int p_id) const {
	int index = get_feed_index(p_id);
	ERR_FAIL_COND_V(index == -1, false);

	return feeds[index]->get_is_active();
};

void CameraServer::feed_set_active(int p_id, bool p_set_active) {
	int index = get_feed_index(p_id);
	ERR_FAIL_COND(index == -1);

	feeds[index]->set_is_active(p_set_active);
};

RID CameraServer::feed_texture(int p_id, int p_texture) const {
	int index = get_feed_index(p_id);
	ERR_FAIL_COND_V(index == -1, RID());

	return feeds[index]->get_texture(p_texture);
};

CameraServer::CameraServer() {
	singleton = this;
};

CameraServer::~CameraServer() {
	singleton = NULL;
};
