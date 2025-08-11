/**************************************************************************/
/*  camera_feed_pipewire.cpp                                              */
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

#include "camera_feed_pipewire.h"

#include "camera_pipewire.h"

GODOT_GCC_WARNING_PUSH
GODOT_GCC_WARNING_IGNORE("-Wmissing-field-initializers")
GODOT_CLANG_WARNING_PUSH
GODOT_CLANG_WARNING_IGNORE("-Wmissing-field-initializers")

#include <spa/debug/types.h>
#include <spa/param/video/format-utils.h>

void CameraFeedPipeWire::on_node_info(void *data, const struct pw_node_info *info) {
	CameraFeedPipeWire *feed = (CameraFeedPipeWire *)data;
	CameraPipeWire *server = (CameraPipeWire *)CameraServer::get_singleton();

	if (info == nullptr || !(info->change_mask & PW_NODE_CHANGE_MASK_PARAMS)) {
		return;
	}

	for (uint32_t i = 0; i < info->n_params; i++) {
		spa_param_info *param = &info->params[i];
		if (!(param->flags & SPA_PARAM_INFO_READ)) {
			continue;
		}
		pw_node_enum_params((pw_node *)feed->proxy, ++param->seq, param->id, 0, -1, nullptr);
	}
	server->sync_wait(feed->proxy);
}

void CameraFeedPipeWire::on_node_param(void *data, int seq, uint32_t id, uint32_t index, uint32_t next, const struct spa_pod *param) {
	uint32_t media_type, media_subtype, format;
	spa_rectangle resolution = {};
	const spa_pod_prop *framerate_prop;
	const spa_pod *framerate_pod;
	uint32_t n_framerates, framerate_choice;
	const spa_fraction *framerate_values;

	CameraFeedPipeWire *feed = (CameraFeedPipeWire *)data;

	if (id != SPA_PARAM_EnumFormat || param == nullptr) {
		return;
	}
	if (spa_format_parse(param, &media_type, &media_subtype) < 0) {
		return;
	}
	if (media_type != SPA_MEDIA_TYPE_video) {
		return;
	}

	if (media_subtype == SPA_MEDIA_SUBTYPE_raw) {
		spa_video_info_raw info = {};
		spa_format_video_raw_parse(param, &info);
		switch (info.format) {
			case SPA_VIDEO_FORMAT_YUY2:
			case SPA_VIDEO_FORMAT_RGB:
			case SPA_VIDEO_FORMAT_NV12:
				format = info.format;
				resolution = info.size;
				break;
			default:
				return;
		}
	} else {
		return;
	}

	framerate_prop = spa_pod_find_prop(param, nullptr, SPA_FORMAT_VIDEO_framerate);
	if (!framerate_prop) {
		return;
	}
	framerate_pod = spa_pod_get_values(&framerate_prop->value, &n_framerates, &framerate_choice);
	if (framerate_pod->type != SPA_TYPE_Fraction) {
		return;
	}
	framerate_values = (spa_fraction *)SPA_POD_BODY(framerate_pod);
	if (framerate_choice == SPA_CHOICE_None) {
		feed->add_format(media_subtype, format, resolution, framerate_values[0]);
	} else if (framerate_choice == SPA_CHOICE_Enum) {
		// Index 0 is the default.
		for (uint32_t i = 1; i < n_framerates; i++) {
			feed->add_format(media_subtype, format, resolution, framerate_values[i]);
		}
	}
}

void CameraFeedPipeWire::on_proxy_destroy(void *data) {
	CameraFeedPipeWire *feed = (CameraFeedPipeWire *)data;
	feed->proxy = nullptr;
}

void CameraFeedPipeWire::on_stream_destroy(void *data) {
	CameraFeedPipeWire *feed = (CameraFeedPipeWire *)data;
	feed->stream = nullptr;
}

void CameraFeedPipeWire::on_stream_process(void *data) {
	pw_buffer *b = nullptr;
	spa_buffer *buf = nullptr;
	CameraFeedPipeWire *feed = (CameraFeedPipeWire *)data;
	FeedFormat format = feed->formats[feed->selected_format];
	pw_stream *stream = feed->stream;

	if (stream == nullptr) {
		return;
	}

	while (true) {
		pw_buffer *t;
		if ((t = pw_stream_dequeue_buffer(stream)) == nullptr) {
			break;
		}
		if (b) {
			pw_stream_queue_buffer(stream, b);
		}
		b = t;
	}
	ERR_FAIL_NULL_MSG(b, "Out of buffer.");

	buf = b->buffer;
	// This is running on PipeWire's thread loop, make sure 'frame_changed' signal is emitted on the main thread.
	if (format.media_subtype == SPA_MEDIA_SUBTYPE_raw) {
		if (format.format == SPA_VIDEO_FORMAT_YUY2) {
			Vector<uint8_t> bytes;
			bytes.resize(buf->datas[0].maxsize); // codespell:ignore datas
			memcpy(bytes.ptrw(), buf->datas[0].data, bytes.size()); // codespell:ignore datas
			Ref<Image> image = Image::create_from_data(format.resolution.width, format.resolution.height, false, Image::FORMAT_RG8, bytes);
			feed->set_ycbcr_image(image);
		}
		if (format.format == SPA_VIDEO_FORMAT_RGB) {
			Vector<uint8_t> bytes;
			bytes.resize(buf->datas[0].maxsize); // codespell:ignore datas
			memcpy(bytes.ptrw(), buf->datas[0].data, bytes.size()); // codespell:ignore datas
			Ref<Image> image = Image::create_from_data(format.resolution.width, format.resolution.height, false, Image::FORMAT_RGB8, bytes);
			feed->set_rgb_image(image);
		}
		if (format.format == SPA_VIDEO_FORMAT_NV12) {
			Vector<uint8_t> bytes_y;
			Vector<uint8_t> bytes_uv;
			bytes_y.resize(format.resolution.width * format.resolution.height);
			bytes_uv.resize(bytes_y.size() / 2);
			ERR_FAIL_COND(buf->datas[0].maxsize - bytes_y.size() - bytes_uv.size() < 0); // codespell:ignore datas
			memcpy(bytes_y.ptrw(), buf->datas[0].data, bytes_y.size()); // codespell:ignore datas
			memcpy(bytes_uv.ptrw(), (uint8_t *)buf->datas[0].data + bytes_y.size(), bytes_uv.size()); // codespell:ignore datas
			Ref<Image> image_y = Image::create_from_data(format.resolution.width, format.resolution.height, false, Image::FORMAT_R8, bytes_y);
			Ref<Image> image_uv = Image::create_from_data(format.resolution.width / 2, format.resolution.height / 2, false, Image::FORMAT_RG8, bytes_uv);
			feed->set_ycbcr_images(image_y, image_uv);
		}
	} else {
		ERR_FAIL_MSG("Unsupported format.");
	}
}

const struct pw_node_events CameraFeedPipeWire::node_events = {
	.version = PW_VERSION_NODE_EVENTS,
	.info = on_node_info,
	.param = on_node_param,
};

const struct pw_proxy_events CameraFeedPipeWire::proxy_events = {
	.version = PW_VERSION_PROXY_EVENTS,
	.destroy = on_proxy_destroy,
	.bound = nullptr,
	.removed = nullptr,
	.done = nullptr,
	.error = nullptr,
	.bound_props = nullptr,
};

const struct pw_stream_events CameraFeedPipeWire::stream_events = {
	.version = PW_VERSION_STREAM_EVENTS,
	.destroy = on_stream_destroy,
	.state_changed = nullptr,
	.control_info = nullptr,
	.io_changed = nullptr,
	.param_changed = nullptr,
	.add_buffer = nullptr,
	.remove_buffer = nullptr,
	.process = on_stream_process,
	.drained = nullptr,
	.command = nullptr,
	.trigger_done = nullptr,
};

void CameraFeedPipeWire::add_format(const uint32_t media_subtype, const uint32_t format, const spa_rectangle resolution, const spa_fraction framerate) {
	FeedFormat feed_format = {};
	feed_format.media_subtype = media_subtype;
	feed_format.format = format;
	feed_format.resolution = resolution;
	feed_format.framerate = framerate;
	formats.push_back(feed_format);
}

int CameraFeedPipeWire::set_stream_format(FeedFormat p_format) {
	const struct spa_pod *param[1];
	uint8_t buffer[1024];
	struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));
	param[0] = (spa_pod *)spa_pod_builder_add_object(&b,
			SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat,
			SPA_FORMAT_mediaType, SPA_POD_Id(SPA_MEDIA_TYPE_video),
			SPA_FORMAT_mediaSubtype, SPA_POD_Id(p_format.media_subtype),
			SPA_FORMAT_VIDEO_format, SPA_POD_Id(p_format.format),
			SPA_FORMAT_VIDEO_size, SPA_POD_Rectangle(&p_format.resolution),
			SPA_FORMAT_VIDEO_framerate, SPA_POD_Fraction(&p_format.framerate));
	int result = pw_stream_update_params(stream, param, 1);
	return result;
}

CameraFeedPipeWire::CameraFeedPipeWire(int p_id, pw_stream *p_stream, pw_proxy *p_proxy, const char *p_name) :
		object_id(p_id), stream(p_stream), proxy(p_proxy) {
	set_name(p_name);
	pw_node_add_listener((pw_node *)proxy, &node_listener, &node_events, this);
	pw_proxy_add_listener(proxy, &proxy_listener, &proxy_events, this);
	pw_stream_add_listener(stream, &stream_listener, &stream_events, this);
}

CameraFeedPipeWire::~CameraFeedPipeWire() {
	CameraPipeWire *server = (CameraPipeWire *)CameraServer::get_singleton();
	if (server == nullptr) {
		return;
	}
	if (is_active()) {
		deactivate_feed();
	}
	server->thread_lock();
	if (stream) {
		pw_stream_destroy(stream);
		spa_hook_remove(&stream_listener);
	}
	if (proxy) {
		pw_proxy_destroy(proxy);
		spa_hook_remove(&proxy_listener);
		spa_hook_remove(&node_listener);
	}
	server->thread_unlock();
}

uint32_t CameraFeedPipeWire::get_object_id() const {
	return object_id;
}

bool CameraFeedPipeWire::set_format(int p_index, const Dictionary &p_parameters) {
	CameraPipeWire *server = (CameraPipeWire *)CameraServer::get_singleton();
	server->thread_lock();
	server->sync_wait(proxy);
	server->thread_unlock();

	ERR_FAIL_INDEX_V_MSG(p_index, formats.size(), false, "Invalid format index.");

	selected_format = p_index;
	if (is_active()) {
		server->thread_lock();
		int result = set_stream_format(formats[selected_format]);
		server->sync_wait(proxy);
		server->thread_unlock();
		return result == 0;
	}
	return true;
}

Array CameraFeedPipeWire::get_formats() const {
	CameraPipeWire *server = (CameraPipeWire *)CameraServer::get_singleton();
	server->thread_lock();
	server->sync_wait(proxy);
	server->thread_unlock();

	Array result;
	for (const FeedFormat &format : formats) {
		Dictionary dictionary;
		if (format.media_subtype == SPA_MEDIA_SUBTYPE_raw) {
			dictionary["format"] = spa_debug_type_find_short_name(spa_type_video_format, format.format);
		} else {
			dictionary["format"] = spa_debug_type_find_short_name(spa_type_media_subtype, format.media_subtype);
		}
		dictionary["width"] = format.resolution.width;
		dictionary["height"] = format.resolution.height;
		dictionary["framerate_denominator"] = format.framerate.denom;
		dictionary["framerate_numerator"] = format.framerate.num;
		result.push_back(dictionary);
	}
	return result;
}

bool CameraFeedPipeWire::activate_feed() {
	int result;
	ERR_FAIL_NULL_V(stream, false);
	ERR_FAIL_NULL_V(proxy, false);
	ERR_FAIL_COND_V(formats.is_empty(), false);
	CameraPipeWire *server = (CameraPipeWire *)CameraServer::get_singleton();
	FeedFormat feed_format = {};
	server->thread_lock();
	pw_stream_flags stream_flags = pw_stream_flags(PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS);
	result = pw_stream_connect(stream, PW_DIRECTION_INPUT, PW_ID_ANY, stream_flags, nullptr, 0);
	if (result == 0) {
		if (selected_format == -1) {
			feed_format = formats[0];
		} else {
			feed_format = formats[selected_format];
		}
		result = set_stream_format(feed_format);
	}
	server->sync_wait(proxy);
	server->thread_unlock();
	return result == 0;
}

void CameraFeedPipeWire::deactivate_feed() {
	ERR_FAIL_NULL(stream);
	ERR_FAIL_NULL(proxy);
	CameraPipeWire *server = (CameraPipeWire *)CameraServer::get_singleton();
	server->thread_lock();
	pw_stream_disconnect(stream);
	server->sync_wait(proxy);
	server->thread_unlock();
}
