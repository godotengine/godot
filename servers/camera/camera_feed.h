/**************************************************************************/
/*  camera_feed.h                                                         */
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

#ifndef CAMERA_FEED_H
#define CAMERA_FEED_H

#include "core/io/image.h"
#include "core/math/transform_2d.h"
#include "servers/camera_server.h"
#include "servers/rendering_server.h"

/**
	The camera server is a singleton object that gives access to the various
	camera feeds that can be used as the background for our environment.
**/

class CameraFeed : public RefCounted {
	GDCLASS(CameraFeed, RefCounted);

public:
	enum FeedPosition {
		FEED_UNSPECIFIED, // we have no idea
		FEED_FRONT, // this is a camera on the front of the device
		FEED_BACK // this is a camera on the back of the device
	};

private:
	int id; // unique id for this, for internal use in case feeds are removed

protected:
	String name; // name of our camera feed
	FeedPosition position; // position of camera on the device
	int format = 0; // format id
	Transform2D transform; // display transform

	bool active; // only when active do we actually update the camera texture each frame

	uint32_t base_width; // Base width of camera frames
	uint32_t base_height; // Base height of camera frames

	RID texture, diffuse_texture, normal_texture; // Canvas textures

	static void _bind_methods();

public:
	int get_id() const;
	
	int get_format() const;
	virtual void set_format(int type);

	bool is_active() const;
	void set_active(bool p_is_active);

	String get_name() const;
	void set_name(String p_name);

	int get_base_width() const;
	int get_base_height() const;

	FeedPosition get_position() const;
	void set_position(FeedPosition p_position);

	Transform2D get_transform() const;
	void set_transform(const Transform2D &p_transform);

	RID get_texture();
	void set_texture(Ref<Image> &diffuse, Ref<Image> &normal);

	CameraFeed();
	virtual ~CameraFeed();

	virtual bool activate_feed();
	virtual void deactivate_feed();
};

VARIANT_ENUM_CAST(CameraFeed::FeedPosition);

#endif // CAMERA_FEED_H
