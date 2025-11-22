/**************************************************************************/
/*  camera_feed_pipewire.h                                                */
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

#ifdef SOWRAP_ENABLED
#include "drivers/pipewire/pipewire-so_wrap.h"
#else
#include <pipewire/pipewire.h>
#endif

class CameraFeedPipeWire : public CameraFeed {
	GDSOFTCLASS(CameraFeedPipeWire, CameraFeed);

	struct FeedFormat {
		uint32_t media_subtype;
		uint32_t format;
		spa_rectangle resolution;
		spa_fraction framerate;
	};

	static void on_node_info(void *data, const struct pw_node_info *info);
	static void on_node_param(void *data, int seq, uint32_t id, uint32_t index, uint32_t next, const struct spa_pod *param);
	static void on_proxy_destroy(void *data);
	static void on_stream_destroy(void *data);
	static void on_stream_process(void *data);

	static const struct pw_node_events node_events;
	static const struct pw_proxy_events proxy_events;
	static const struct pw_stream_events stream_events;

	uint32_t object_id = 0;
	pw_stream *stream = nullptr;
	pw_proxy *proxy = nullptr;
	Vector<FeedFormat> formats;
	spa_hook node_listener = {};
	spa_hook proxy_listener = {};
	spa_hook stream_listener = {};

	void add_format(const uint32_t media_subtype, const uint32_t format, const spa_rectangle resolution, const spa_fraction framerate);
	int set_stream_format(FeedFormat p_format);

public:
	CameraFeedPipeWire(int p_id, pw_stream *p_stream, pw_proxy *p_proxy, const char *p_name);
	virtual ~CameraFeedPipeWire() override;

	uint32_t get_object_id() const;

	virtual bool set_format(int p_index, const Dictionary &p_parameters) override;
	virtual Array get_formats() const override;

	virtual bool activate_feed() override;
	virtual void deactivate_feed() override;
};
