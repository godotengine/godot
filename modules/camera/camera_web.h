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

#include "platform/web/camera_driver_web.h"
#include "servers/camera/camera_feed.h"
#include "servers/camera_server.h"

#include <atomic>

class CameraFeedWeb : public CameraFeed {
	GDSOFTCLASS(CameraFeedWeb, CameraFeed);

private:
	String device_id;
	Ref<Image> image;
	Vector<uint8_t> data;
	static void _on_get_pixeldata(void *context, const uint8_t *rawdata, const int length, const int p_width, const int p_height, const char *error);
	static void _on_denied_callback(void *context);

protected:
public:
	bool activate_feed() override;
	void deactivate_feed() override;
	bool set_format(int p_index, const Dictionary &p_parameters) override;
	Array get_formats() const override;
	FeedFormat get_format() const override;

	CameraFeedWeb(const CameraInfo &info);
	~CameraFeedWeb();
};

class CameraWeb : public CameraServer {
	GDCLASS(CameraWeb, CameraServer);

private:
	CameraDriverWeb *camera_driver_web = nullptr;
	std::atomic<bool> activating;
	void _cleanup();
	void _update_feeds();
	static void _on_get_cameras_callback(void *context, const Vector<CameraInfo> &camera_info);

protected:
public:
	void set_monitoring_feeds(bool p_monitoring_feeds) override;

	~CameraWeb();
};
