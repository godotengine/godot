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

namespace {
const String KEY_HEIGHT("height");
const String KEY_WIDTH("width");
} //namespace

void CameraFeedWeb::_on_get_pixel_data(void *p_context, const uint8_t *p_data, const int p_length, const int p_width, const int p_height, const int p_orientation, const int p_facing_mode, const char *p_error) {
	// Validate context first to avoid dereferencing null on error paths.
	ERR_FAIL_NULL_MSG(p_context, "Camera feed error: Null context received.");

	CameraFeedWeb *feed = reinterpret_cast<CameraFeedWeb *>(p_context);

	if (p_error) {
		if (feed->is_active()) {
			feed->deactivate_feed();
		}
		String error_str = String::utf8(p_error);
		ERR_PRINT(vformat("Camera feed error from JS: %s", error_str));
		return;
	}

	if (p_data == nullptr || p_length < 0 || p_width <= 0 || p_height <= 0) {
		if (feed->is_active()) {
			feed->deactivate_feed();
		}
		ERR_PRINT("Camera feed error: Invalid pixel data received.");
		return;
	}

	// Update feed position based on facing mode.
	// p_facing_mode: 0=unknown, 1=user/front, 2=environment/back
	if (p_facing_mode == 1) {
		feed->position = CameraFeed::FEED_FRONT;
	} else if (p_facing_mode == 2) {
		feed->position = CameraFeed::FEED_BACK;
	}

	// Apply rotation based on screen orientation (convert degrees to radians).
	// Also apply vertical flip for all cameras.
	feed->transform = Transform2D();
	feed->transform = feed->transform.rotated(Math::deg_to_rad(static_cast<float>(p_orientation)));
	feed->transform = feed->transform.scaled(Vector2(1, -1));

	Vector<uint8_t> &data = feed->data;
	Ref<Image> image = feed->image;

	const int64_t expected_size = Image::get_image_data_size(p_width, p_height, Image::FORMAT_RGBA8, false);
	if (p_length < expected_size) {
		if (feed->is_active()) {
			feed->deactivate_feed();
		}
		ERR_PRINT("Camera feed error: Received pixel data smaller than expected.");
		return;
	}

	if (data.size() != expected_size) {
		data.resize(expected_size);
	}
	// Copy exactly the expected size (ignore any trailing bytes in 'p_data').
	memcpy(data.ptrw(), p_data, expected_size);

	image->initialize_data(p_width, p_height, false, Image::FORMAT_RGBA8, data);
	feed->set_rgb_image(image);
}

void CameraFeedWeb::_on_denied_callback(void *p_context) {
	ERR_FAIL_NULL_MSG(p_context, "Camera feed error: Null context received in denied callback.");
	CameraFeedWeb *feed = reinterpret_cast<CameraFeedWeb *>(p_context);
	feed->deactivate_feed();
}

bool CameraFeedWeb::activate_feed() {
	if (is_active()) {
		WARN_PRINT("Camera feed is already active.");
		return true;
	}
	ERR_FAIL_COND_V_MSG(selected_format == -1, false, "CameraFeed format needs to be set before activating.");

	// Initialize image when activating the feed.
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
	CameraDriverWeb *driver = CameraDriverWeb::get_singleton();
	ERR_FAIL_NULL_V_MSG(driver, false, "CameraDriverWeb singleton is not initialized.");
	driver->get_pixel_data(this, device_id, width, height, &_on_get_pixel_data, &_on_denied_callback);
	return true;
}

void CameraFeedWeb::deactivate_feed() {
	CameraDriverWeb *driver = CameraDriverWeb::get_singleton();
	ERR_FAIL_NULL_MSG(driver, "CameraDriverWeb singleton is not initialized.");
	driver->stop_stream(device_id);
	// Release the image when deactivating the feed.
	image.unref();
	data.clear();
}

bool CameraFeedWeb::set_format(int p_index, const Dictionary &p_parameters) {
	ERR_FAIL_COND_V_MSG(active, false, "Feed is active.");
	ERR_FAIL_INDEX_V_MSG(p_index, formats.size(), false, "Invalid format index.");

	selected_format = p_index;
	parameters = p_parameters.duplicate();
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
	if (selected_format < 0 || selected_format >= formats.size()) {
		return feed_format;
	}
	return formats[selected_format];
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

void CameraWeb::_on_get_cameras_callback(void *p_context, const Vector<CameraInfo> &p_camera_info) {
	CameraWeb *server = static_cast<CameraWeb *>(p_context);

	// Deactivate all feeds before removing them.
	for (int i = 0; i < server->feeds.size(); i++) {
		Ref<CameraFeedWeb> feed = server->feeds[i];
		if (feed.is_valid() && feed->is_active()) {
			feed->deactivate_feed();
		}
	}

	for (int i = server->feeds.size() - 1; i >= 0; i--) {
		server->remove_feed(server->feeds[i]);
	}
	for (int i = 0; i < p_camera_info.size(); i++) {
		CameraInfo info = p_camera_info[i];
		Ref<CameraFeedWeb> feed = memnew(CameraFeedWeb(info));
		server->add_feed(feed);
	}
	server->activating.clear();
	server->call_deferred("emit_signal", SNAME(CameraServer::feeds_updated_signal_name));
}

void CameraWeb::_update_feeds() {
	activating.set();
	driver->get_cameras((void *)this, &_on_get_cameras_callback);
}

void CameraWeb::_cleanup() {
	if (driver != nullptr) {
		driver->stop_stream();
		memdelete(driver);
		driver = nullptr;
	}
}

void CameraWeb::set_monitoring_feeds(bool p_monitoring_feeds) {
	if (p_monitoring_feeds == monitoring_feeds || activating.is_set()) {
		return;
	}

	CameraServer::set_monitoring_feeds(p_monitoring_feeds);
	if (p_monitoring_feeds) {
		if (driver == nullptr) {
			driver = new CameraDriverWeb();
		}
		_update_feeds();
	} else {
		_cleanup();
	}
}

CameraWeb::~CameraWeb() {
	_cleanup();
}
