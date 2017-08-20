/*************************************************************************/
/*  camera_server.h                                                       */
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

#ifndef CAMERA_SERVER_H
#define CAMERA_SERVER_H

#include "object.h"
#include "os/thread_safe.h"
#include "rid.h"
#include "variant.h"

/**
	@author Bastiaan Olij <mux213@gmail.com>

	The camera server is a singleton object that gives access to the various
	camera feeds that can be used as the background for our environment.
**/

class CameraFeed {
public:
	enum FeedState {
		FEED_WAITING_ON_0,
		FEED_UPDATING_0,
		FEED_0_IS_AVAILABLE,
		FEED_WAITING_ON_1,
		FEED_UPDATING_1,
		FEED_1_IS_AVAILABLE,
	};

	enum FeedDataType {
		FEED_NOIMAGE, // we don't have an image yet
		FEED_RGB, // our texture will contain a normal RGB texture that can be used directly
		FEED_YCbCr // our texture is a texture atlas with two planes, first plane contains Y data, second plane contains CbCr data
	};

	enum FeedPosition {
		FEED_UNSPECIFIED, // we have no idea
		FEED_FRONT, // this is a camera on the front of the device
		FEED_BACK // this is a camera on the back of the device
	};

private:
	int id; // unique id for this, for internal use in case feeds are removed
	int base_width;
	int base_height;
	PoolVector<uint8_t> img_data[2][2];

protected:
	String name; // name of our camera feed
	FeedDataType datatype; // type of texture data stored
	FeedPosition position;

	bool active; // only when active do we actually update the camera texture each frame
	FeedState state; // our process state
	RID texture[2][2]; // two pairs of texture objects we are updating, we need 1 for RGB, but 2 for YCbCr
	int write_to_set();

public:
	int get_id() const;
	bool get_is_active() const;
	void set_is_active(bool p_is_active);
	String get_name() const;
	FeedDataType get_datatype() const;
	FeedPosition get_position() const;
	RID get_texture(int p_which);

	CameraFeed();
	CameraFeed(String p_name, FeedPosition p_position = CameraFeed::FEED_UNSPECIFIED);
	virtual ~CameraFeed();

	bool is_waiting();
	void set_texture_data_RGB(unsigned char *p_data, int p_width, int p_height);
	void set_texture_data_YCbCr(unsigned char *p_y_data, int p_y_width, int p_y_height, unsigned char *p_cbcr_data, int p_cbcr_width, int p_cbcr_height);

	virtual bool activate_feed();
	virtual void deactivate_feed();
};

class CameraServer : public Object {
	GDCLASS(CameraServer, Object);
	_THREAD_SAFE_CLASS_

private:
protected:
	Vector<CameraFeed *> feeds;

	static CameraServer *singleton;

	static void _bind_methods();

public:
	static CameraServer *get_singleton();

	// Because our position in our vector may change, we need to be able to convert our unique id to our current index
	int get_free_id();
	int get_feed_index(int p_id) const;
	CameraFeed *get_feed(int p_id);
	void add_feed(CameraFeed *p_feed);
	void remove_feed(int p_id); // note this will also destruct our instance!

	Array get_feeds() const;

	bool feed_is_active(int p_id) const;
	void feed_set_active(int p_id, bool p_set_active);
	RID feed_texture(int p_id, int p_texture) const;

	CameraServer();
	~CameraServer();
};

VARIANT_ENUM_CAST(CameraFeed::FeedPosition);

#endif /* CAMERA_SERVER_H */
