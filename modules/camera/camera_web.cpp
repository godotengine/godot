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

#include "core/templates/hash_set.h"

#include "platform/web/camera_driver_web.h"

namespace {
// Pixel format codes - must match FORMAT_CODE_* in library_godot_camera.js.
constexpr int FORMAT_CODE_RGBA = 0;
constexpr int FORMAT_CODE_NV12 = 1;

const String KEY_HEIGHT("height");
const String KEY_WIDTH("width");
} // namespace

BufferDecoder *CameraFeedWeb::_create_buffer_decoder(CameraFeedWeb *p_feed, int p_pixel_format) {
	switch (p_pixel_format) {
		case FORMAT_CODE_NV12:
			return memnew(Nv12BufferDecoder(p_feed));
		case FORMAT_CODE_RGBA:
			return memnew(CopyBufferDecoder(p_feed, CopyBufferDecoder::rgba));
		default:
			// Unknown format - drop frames to avoid mis-sizing the destination image.
			ERR_PRINT(vformat("Camera feed error: Unknown pixel format code %d.", p_pixel_format));
			return memnew(NullBufferDecoder(p_feed));
	}
}

void CameraFeedWeb::_on_get_pixel_data(void *p_context, const uint8_t *p_data, const int p_length, const int p_width, const int p_height, const int p_pixel_format, const int p_facing_mode, const char *p_error) {
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

	if (p_data == nullptr || p_length <= 0 || p_width <= 0 || p_height <= 0) {
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

	// (Re)create the decoder when the format or frame size changes. The browser
	// may pick a resolution different from what was requested, so the stored
	// format is updated to reflect the actually delivered dimensions before
	// BufferDecoder reads them in its constructor.
	const CameraFeed::FeedFormat current_format = feed->get_format();
	const bool size_changed = current_format.width != p_width || current_format.height != p_height;
	if (feed->buffer_decoder == nullptr || feed->current_pixel_format != p_pixel_format || size_changed) {
		if (feed->selected_format >= 0 && feed->selected_format < feed->formats.size()) {
			FeedFormat &f = feed->formats.write[feed->selected_format];
			f.width = p_width;
			f.height = p_height;
		}
		if (feed->buffer_decoder) {
			memdelete(feed->buffer_decoder);
			feed->buffer_decoder = nullptr;
		}
		feed->buffer_decoder = _create_buffer_decoder(feed, p_pixel_format);
		feed->current_pixel_format = p_pixel_format;
	}

	StreamingBuffer buffer;
	buffer.start = const_cast<uint8_t *>(p_data);
	buffer.length = p_length;
	buffer.pitch = 0; // Tightly packed; decoders will derive the stride from length.
	feed->buffer_decoder->decode(buffer);
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

	int width = parameters.get(KEY_WIDTH, 0);
	int height = parameters.get(KEY_HEIGHT, 0);
	// Defensive check in case formats list is shorter than expected.
	if (formats.size() > selected_format) {
		CameraFeed::FeedFormat f = formats[selected_format];
		width = width > 0 ? width : f.width;
		height = height > 0 ? height : f.height;
	}

	CameraDriverWeb *driver = CameraDriverWeb::get_singleton();
	ERR_FAIL_NULL_V_MSG(driver, false, "CameraDriverWeb singleton is not initialized.");

	// 'this' is passed as a raw context pointer to JS. This is safe because
	// Web builds are single-threaded - stop_stream() synchronously cancels all
	// pending callbacks, so no callback can fire after deactivate_feed() returns.
	driver->get_pixel_data(this, device_id, width, height,
			&_on_get_pixel_data, &_on_denied_callback);
	return true;
}

void CameraFeedWeb::deactivate_feed() {
	CameraDriverWeb *driver = CameraDriverWeb::get_singleton();
	ERR_FAIL_NULL_MSG(driver, "CameraDriverWeb singleton is not initialized.");
	driver->stop_stream(device_id);

	if (buffer_decoder) {
		memdelete(buffer_decoder);
		buffer_decoder = nullptr;
	}
	current_pixel_format = -1;
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
		dictionary["frame_numerator"] = feed_format.frame_numerator;
		dictionary["frame_denominator"] = feed_format.frame_denominator;
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
	// Override base class default (Y-flip) with identity.
	// Browser delivers frames already in the correct orientation.
	transform = Transform2D();

	for (int i = 0; i < info.formats.size(); i++) {
		FeedFormat feed_format;
		feed_format.width = info.formats[i].width;
		feed_format.height = info.formats[i].height;
		feed_format.format = String("RGBA");
		// Web API provides frame rate as integer (max fps).
		// Use frame_numerator/frame_denominator format consistent with other platforms.
		if (info.formats[i].frame_rate > 0) {
			feed_format.frame_numerator = info.formats[i].frame_rate;
			feed_format.frame_denominator = 1;
		}
		formats.append(feed_format);
	}
}

CameraFeedWeb::~CameraFeedWeb() {
	if (is_active()) {
		deactivate_feed();
	}
}

void CameraWeb::_on_get_cameras_callback(void *p_context, const Vector<CameraInfo> &p_camera_info) {
	CameraWeb *server = static_cast<CameraWeb *>(p_context);

	// Build a set of new device IDs for quick lookup.
	HashSet<String> new_device_ids;
	for (int i = 0; i < p_camera_info.size(); i++) {
		new_device_ids.insert(p_camera_info[i].device_id);
	}

	// Remove feeds that are no longer present.
	for (int i = server->feeds.size() - 1; i >= 0; i--) {
		Ref<CameraFeedWeb> feed = server->feeds[i];
		if (feed.is_valid() && !new_device_ids.has(feed->get_device_id())) {
			if (feed->is_active()) {
				feed->deactivate_feed();
			}
			server->remove_feed(feed);
		}
	}

	// Build a set of existing device IDs.
	HashSet<String> existing_device_ids;
	for (int i = 0; i < server->feeds.size(); i++) {
		Ref<CameraFeedWeb> feed = server->feeds[i];
		if (feed.is_valid()) {
			existing_device_ids.insert(feed->get_device_id());
		}
	}

	// Add new feeds that don't already exist.
	for (int i = 0; i < p_camera_info.size(); i++) {
		if (!existing_device_ids.has(p_camera_info[i].device_id)) {
			Ref<CameraFeedWeb> feed = memnew(CameraFeedWeb(p_camera_info[i]));
			server->add_feed(feed);
		}
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
			driver = memnew(CameraDriverWeb);
		}
		_update_feeds();
	} else {
		_cleanup();
	}
}

CameraWeb::~CameraWeb() {
	_cleanup();
}
