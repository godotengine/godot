/**************************************************************************/
/*  camera_web.h                                                          */
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

#pragma once

#include "servers/camera/camera_feed.h"
#include "servers/camera/camera_server.h"

#include "modules/camera/buffer_decoder.h"

// Shared camera data types, kept in a platform/web-level header so this
// module does not need to include the full CameraDriverWeb driver header.
#include "platform/web/camera_types_web.h"

class CameraDriverWeb;

class CameraFeedWeb : public CameraFeed {
	GDSOFTCLASS(CameraFeedWeb, CameraFeed);

	String device_id;
	BufferDecoder *buffer_decoder = nullptr;
	int current_pixel_format = -1;
	int current_frame_width = 0;
	int current_frame_height = 0;
	String detected_format_name;
	bool synthesized_format = false;

	static BufferDecoder *_create_buffer_decoder(CameraFeedWeb *p_feed, int p_pixel_format);
	static String _get_format_name(int p_pixel_format);
	Size2i _get_requested_size() const;
	static void _on_get_pixel_data(void *p_context, const uint8_t *p_data, const int p_length, const int p_width, const int p_height, const int p_pixel_format, const int p_facing_mode, const char *p_error);
	static void _on_denied_callback(void *p_context);
	static void _on_formats_callback(void *p_context, const char *p_result);

public:
	virtual bool activate_feed() override;
	virtual void deactivate_feed() override;
	virtual bool set_format(int p_index, const Dictionary &p_parameters) override;
	virtual Array get_formats() const override;
	virtual FeedFormat get_format() const override;
	String get_device_id() const { return device_id; }

	CameraFeedWeb(const CameraInfo &info);
	~CameraFeedWeb();
};

class CameraWeb : public CameraServer {
	GDSOFTCLASS(CameraWeb, CameraServer);

	CameraDriverWeb *driver = nullptr;
	uint64_t request_id = 0;
	void _cleanup();
	void _update_feeds();
	static void _on_get_cameras_callback(void *p_context, const Vector<CameraInfo> &p_camera_info);

public:
	virtual void set_monitoring_feeds(bool p_monitoring_feeds) override;

	~CameraWeb();
};
