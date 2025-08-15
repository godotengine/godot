/**************************************************************************/
/*  camera_web.cpp                                                        */
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

#include "camera_web.h"

#include "core/io/json.h"

void CameraFeedWeb::_on_get_pixeldata(void *context, const uint8_t *rawdata, const int length, const int p_width, const int p_height, const char *error) {
	CameraFeedWeb *feed = reinterpret_cast<CameraFeedWeb *>(context);
	if (error) {
		if (feed->is_active()) {
			feed->deactivate_feed();
		};
		String error_str = String::utf8(error);
		ERR_PRINT(vformat("Camera feed error from JS: %s", error_str));
		return;
	}

	if (context == nullptr || rawdata == nullptr || length < 0 || p_width <= 0 || p_height <= 0) {
		if (feed->is_active()) {
			feed->deactivate_feed();
		};
		ERR_PRINT("Camera feed error: Invalid pixel data received.");
		return;
	}

	Vector<uint8_t> data = feed->data;
	Ref<Image> image = feed->image;

	if (length != data.size()) {
		int64_t size = Image::get_image_data_size(p_width, p_height, Image::FORMAT_RGBA8, false);
		data.resize(length > size ? length : size);
	}
	memcpy(data.ptrw(), rawdata, length);

	image->initialize_data(p_width, p_height, false, Image::FORMAT_RGBA8, data);
	feed->set_rgb_image(image);
}

void CameraFeedWeb::_on_denied_callback(void *context) {
	CameraFeedWeb *feed = reinterpret_cast<CameraFeedWeb *>(context);
	feed->deactivate_feed();
}

bool CameraFeedWeb::activate_feed() {
	ERR_FAIL_COND_V_MSG(selected_format == -1, false, "CameraFeed format needs to be set before activating.");

	// Initialize image when activating the feed
	if (image.is_null()) {
		image.instantiate();
	}

	int width = parameters.get(KEY_WIDTH, 0);
	int height = parameters.get(KEY_HEIGHT, 0);
	// Firefox ESR (128.11.0esr) does not implement MediaStreamTrack.getCapabilities(), so 'formats' will be empty.
	if (formats.size() > selected_format) {
		CameraFeed::FeedFormat f = formats[selected_format];
		width = width > 0 ? width : f.width;
		height = height > 0 ? height : f.height;
	}
	CameraDriverWeb::get_singleton()->get_pixel_data(this, device_id, width, height, &_on_get_pixeldata, &_on_denied_callback);
	return true;
}

void CameraFeedWeb::deactivate_feed() {
	CameraDriverWeb::get_singleton()->stop_stream(device_id);
	// Release the image when deactivating the feed
	image.unref();
	data.clear();
}

bool CameraFeedWeb::set_format(int p_index, const Dictionary &p_parameters) {
	ERR_FAIL_COND_V_MSG(active, false, "Feed is active.");
	ERR_FAIL_INDEX_V_MSG(p_index, formats.size(), false, "Invalid format index.");

	selected_format = p_index;
	parameters = p_parameters;
	return true;
}

Array CameraFeedWeb::get_formats() const {
	Array result;
	for (const FeedFormat &feed_format : formats) {
		Dictionary dictionary;
		dictionary["width"] = feed_format.width;
		dictionary["height"] = feed_format.height;
		dictionary["format"] = feed_format.format;
		result.push_back(dictionary);
	}
	return result;
}

CameraFeed::FeedFormat CameraFeedWeb::get_format() const {
	CameraFeed::FeedFormat feed_format = {};
	return selected_format == -1 ? feed_format : formats[selected_format];
}

CameraFeedWeb::CameraFeedWeb(const CameraInfo &info) {
	name = info.label;
	device_id = info.device_id;

	FeedFormat feed_format;
	feed_format.width = info.capability.width;
	feed_format.height = info.capability.height;
	feed_format.format = String("RGBA");
	formats.append(feed_format);
}

CameraFeedWeb::~CameraFeedWeb() {
	if (is_active()) {
		deactivate_feed();
	}
}

void CameraWeb::_on_get_cameras_callback(void *context, const Vector<CameraInfo> &camera_info) {
	CameraWeb *server = static_cast<CameraWeb *>(context);
	for (int i = server->feeds.size() - 1; i >= 0; i--) {
		server->remove_feed(server->feeds[i]);
	}
	for (int i = 0; i < camera_info.size(); i++) {
		CameraInfo info = camera_info[i];
		Ref<CameraFeedWeb> feed = memnew(CameraFeedWeb(info));
		server->add_feed(feed);
	}
	server->activating.clear();
	server->call_deferred("emit_signal", SNAME(CameraServer::feeds_updated_signal_name));
}

void CameraWeb::_update_feeds() {
	activating.set();
	camera_driver_web->get_cameras((void *)this, &_on_get_cameras_callback);
}

void CameraWeb::_cleanup() {
	if (camera_driver_web != nullptr) {
		camera_driver_web->stop_stream();
		memdelete(camera_driver_web);
		camera_driver_web = nullptr;
	}
}

void CameraWeb::set_monitoring_feeds(bool p_monitoring_feeds) {
	if (p_monitoring_feeds == monitoring_feeds || activating.is_set()) {
		return;
	}

	CameraServer::set_monitoring_feeds(p_monitoring_feeds);
	if (p_monitoring_feeds) {
		if (camera_driver_web == nullptr) {
			camera_driver_web = new CameraDriverWeb();
		}
		_update_feeds();
	} else {
		_cleanup();
	}
}

CameraWeb::~CameraWeb() {
	_cleanup();
}
