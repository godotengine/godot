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
#include "core/templates/hash_set.h"

#include "platform/web/camera_driver_web.h"

namespace {
// Pixel format codes - must match FORMAT_CODE_* in library_godot_camera.js.
constexpr int FORMAT_CODE_RGBA = 0;
constexpr int FORMAT_CODE_NV12 = 1;
constexpr int FORMAT_CODE_I420 = 2;

const String KEY_HEIGHT("height");
const String KEY_FORMATS("formats");
const String KEY_FRAME_RATE("frameRate");
const String KEY_FACING_MODE("facingMode");
const String KEY_CURRENT("current");
const String KEY_WIDTH("width");
const String FORMAT_BROWSER("Browser");

struct CameraWebRequest {
	ObjectID server_id;
	uint64_t request_id = 0;
};

int get_int_value(const Variant &p_value) {
	if (p_value.get_type() == Variant::INT) {
		return p_value;
	}
	if (p_value.get_type() == Variant::FLOAT) {
		return static_cast<int>(p_value.operator float());
	}
	return 0;
}
} // namespace

BufferDecoder *CameraFeedWeb::_create_buffer_decoder(CameraFeedWeb *p_feed, int p_pixel_format) {
	switch (p_pixel_format) {
		case FORMAT_CODE_NV12:
			return memnew(Nv12BufferDecoder(p_feed));
		case FORMAT_CODE_I420:
			return memnew(I420BufferDecoder(p_feed));
		case FORMAT_CODE_RGBA:
			return memnew(CopyBufferDecoder(p_feed, CopyBufferDecoder::rgba));
		default:
			// Unknown format - drop frames to avoid mis-sizing the destination image.
			ERR_FAIL_V_MSG(memnew(NullBufferDecoder(p_feed)), vformat("Camera feed error: Unknown pixel format code %d.", p_pixel_format));
	}
}

String CameraFeedWeb::_get_format_name(int p_pixel_format) {
	switch (p_pixel_format) {
		case FORMAT_CODE_NV12:
			return String("NV12");
		case FORMAT_CODE_I420:
			return String("I420");
		case FORMAT_CODE_RGBA:
			return String("RGBA");
		default:
			return String("Unknown");
	}
}

Size2i CameraFeedWeb::_get_requested_size() const {
	if (selected_format >= 0 && selected_format < formats.size()) {
		const CameraFeed::FeedFormat &format = formats[selected_format];
		return Size2i(format.width, format.height);
	}
	return Size2i();
}

void CameraFeedWeb::_on_get_pixel_data(void *p_context, const uint8_t *p_data, const int p_length, const int p_width, const int p_height, const int p_pixel_format, const int p_facing_mode, const char *p_error) {
	// Validate context first to avoid dereferencing null on error paths.
	ERR_FAIL_NULL_MSG(p_context, "Camera feed error: Null context received.");

	CameraFeedWeb *feed = reinterpret_cast<CameraFeedWeb *>(p_context);

	if (p_error) {
		String error_str = String::utf8(p_error);
		if (feed->is_active()) {
			feed->set_active(false);
		}
		feed->call_deferred("emit_signal", SNAME("activation_failed"), error_str);
		ERR_FAIL_MSG(vformat("Camera feed error from JS: %s.", error_str));
	}

	if (p_data == nullptr || p_length <= 0 || p_width <= 0 || p_height <= 0) {
		if (feed->is_active()) {
			feed->set_active(false);
		}
		ERR_FAIL_MSG("Camera feed error: Invalid pixel data received.");
	}
	if (feed->get_activation_status() == CameraFeed::FEED_ACTIVATING) {
		feed->_set_activation_status(CameraFeed::FEED_ACTIVE);
	}

	// Update feed position based on facing mode.
	// p_facing_mode: 0=unknown, 1=user/front, 2=environment/back
	if (p_facing_mode == 1) {
		feed->position = CameraFeed::FEED_FRONT;
	} else if (p_facing_mode == 2) {
		feed->position = CameraFeed::FEED_BACK;
	}

	// (Re)create the decoder when the pixel format or actual frame size changes.
	// Browser streams may not match the requested format exactly, so keep the
	// delivered size separately from the stable list returned by get_formats().
	const bool size_changed = feed->current_frame_width != p_width || feed->current_frame_height != p_height;
	if (feed->buffer_decoder == nullptr || feed->current_pixel_format != p_pixel_format || size_changed) {
		feed->detected_format_name = _get_format_name(p_pixel_format);
		feed->current_frame_width = p_width;
		feed->current_frame_height = p_height;

		// The browser may deliver frames before it reports a format list.
		// Synthesize one so the feed stays selectable and can be switched/stopped.
		if (feed->formats.is_empty()) {
			FeedFormat synthesized;
			synthesized.width = p_width;
			synthesized.height = p_height;
			synthesized.format = feed->detected_format_name;
			feed->formats.push_back(synthesized);
			feed->selected_format = 0;
			feed->synthesized_format = true;
			feed->call_deferred("emit_signal", SNAME("format_changed"));
		} else {
			for (FeedFormat &feed_format : feed->formats) {
				feed_format.format = feed->detected_format_name;
			}
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
	if (feed->is_active()) {
		feed->set_active(false);
	}
}

void CameraFeedWeb::_on_formats_callback(void *p_context, const char *p_result) {
	ERR_FAIL_NULL_MSG(p_context, "Camera feed error: Null context received in formats callback.");
	ERR_FAIL_NULL_MSG(p_result, "Camera feed error: Null formats result received.");

	CameraFeedWeb *feed = reinterpret_cast<CameraFeedWeb *>(p_context);
	Variant json_variant = JSON::parse_string(String::utf8(p_result));
	ERR_FAIL_COND_MSG(json_variant.get_type() != Variant::DICTIONARY, "Camera feed error: Formats result is not a Dictionary.");

	Dictionary json_dict = json_variant;
	int current_width = 0;
	int current_height = 0;
	int current_frame_rate = 0;
	Variant current_variant = json_dict.get(KEY_CURRENT, Variant());
	if (current_variant.get_type() == Variant::DICTIONARY) {
		Dictionary current_dict = current_variant;
		current_width = get_int_value(current_dict.get(KEY_WIDTH, Variant()));
		current_height = get_int_value(current_dict.get(KEY_HEIGHT, Variant()));
		current_frame_rate = get_int_value(current_dict.get(KEY_FRAME_RATE, Variant()));

		const int facing_mode = get_int_value(current_dict.get(KEY_FACING_MODE, Variant()));
		if (facing_mode == 1) {
			feed->position = CameraFeed::FEED_FRONT;
		} else if (facing_mode == 2) {
			feed->position = CameraFeed::FEED_BACK;
		}
	}

	if (!feed->formats.is_empty() && !feed->synthesized_format) {
		return;
	}

	Variant v_formats = json_dict.get(KEY_FORMATS, Variant());
	ERR_FAIL_COND_MSG(v_formats.get_type() != Variant::ARRAY, "Camera feed error: Formats result has no formats array.");

	Vector<FeedFormat> new_formats;
	Array formats_array = v_formats;
	for (Variant format_variant : formats_array) {
		if (format_variant.get_type() != Variant::DICTIONARY) {
			continue;
		}

		Dictionary format_dict = format_variant;
		const int width = get_int_value(format_dict.get(KEY_WIDTH, Variant()));
		const int height = get_int_value(format_dict.get(KEY_HEIGHT, Variant()));
		if (width <= 0 || height <= 0) {
			continue;
		}

		FeedFormat feed_format;
		feed_format.width = width;
		feed_format.height = height;
		feed_format.format = feed->detected_format_name.is_empty() ? FORMAT_BROWSER : feed->detected_format_name;
		const int frame_rate = get_int_value(format_dict.get(KEY_FRAME_RATE, Variant()));
		if (frame_rate > 0) {
			feed_format.frame_numerator = frame_rate;
			feed_format.frame_denominator = 1;
		}
		new_formats.push_back(feed_format);
	}

	if (new_formats.is_empty()) {
		return;
	}

	int new_selected_format = feed->selected_format;
	if (current_width > 0 && current_height > 0) {
		new_selected_format = -1;
		for (int i = 0; i < new_formats.size(); i++) {
			const FeedFormat &format = new_formats[i];
			if ((format.width == current_width && format.height == current_height) ||
					(format.width == current_height && format.height == current_width)) {
				new_selected_format = i;
				break;
			}
		}
		if (new_selected_format == -1) {
			FeedFormat current_format;
			current_format.width = current_width;
			current_format.height = current_height;
			current_format.format = feed->detected_format_name.is_empty() ? FORMAT_BROWSER : feed->detected_format_name;
			if (current_frame_rate > 0) {
				current_format.frame_numerator = current_frame_rate;
				current_format.frame_denominator = 1;
			}
			new_formats.push_back(current_format);
			new_selected_format = new_formats.size() - 1;
		}
	}
	if (new_selected_format < 0 || new_selected_format >= new_formats.size()) {
		new_selected_format = 0;
	}

	feed->formats = new_formats;
	feed->selected_format = new_selected_format;
	feed->synthesized_format = false;
	feed->call_deferred("emit_signal", SNAME("format_changed"));
}

bool CameraFeedWeb::activate_feed() {
	if (is_active()) {
		WARN_PRINT("Camera feed is already active.");
		return true;
	}

	CameraDriverWeb *driver = CameraDriverWeb::get_singleton();
	ERR_FAIL_NULL_V_MSG(driver, false, "CameraDriverWeb singleton is not initialized.");

	const Size2i requested_size = _get_requested_size();
	_set_activation_status(CameraFeed::FEED_ACTIVATING);

	// 'this' is passed as a raw context pointer to JS. This is safe because
	// CameraWeb aborts pending operations before its feeds can be destroyed.
	driver->get_pixel_data(this, device_id, requested_size.x, requested_size.y, &_on_get_pixel_data, &_on_denied_callback, &_on_formats_callback);
	return true;
}

void CameraFeedWeb::deactivate_feed() {
	CameraDriverWeb *driver = CameraDriverWeb::get_singleton();
	ERR_FAIL_NULL_MSG(driver, "CameraDriverWeb singleton is not initialized.");
	// Abort any in-flight open for this feed before stopping.
	driver->abort_stream(this);
	driver->stop_stream(device_id);

	if (buffer_decoder) {
		memdelete(buffer_decoder);
		buffer_decoder = nullptr;
	}
	current_pixel_format = -1;
	current_frame_width = 0;
	current_frame_height = 0;
}

bool CameraFeedWeb::set_format(int p_index, const Dictionary &p_parameters) {
	ERR_FAIL_COND_V_MSG(active, false, "Feed is active.");
	ERR_FAIL_INDEX_V_MSG(p_index, formats.size(), false, "Invalid format index.");

	selected_format = p_index;
	return true;
}

Array CameraFeedWeb::get_formats() const {
	Array result;
	for (const FeedFormat &feed_format : formats) {
		Dictionary dictionary;
		dictionary["width"] = feed_format.width;
		dictionary["height"] = feed_format.height;
		dictionary["format"] = detected_format_name.is_empty() ? feed_format.format : detected_format_name;
		dictionary["frame_numerator"] = feed_format.frame_numerator;
		dictionary["frame_denominator"] = feed_format.frame_denominator;
		result.push_back(dictionary);
	}
	return result;
}

CameraFeed::FeedFormat CameraFeedWeb::get_format() const {
	CameraFeed::FeedFormat feed_format = {};
	if (selected_format >= 0 && selected_format < formats.size()) {
		feed_format = formats[selected_format];
	}
	// If the browser has not reported selectable formats, fall back to the
	// delivered frame size so the decoder is never sized 0x0.
	if (current_frame_width > 0 && current_frame_height > 0) {
		feed_format.width = current_frame_width;
		feed_format.height = current_frame_height;
	}
	return feed_format;
}

CameraFeedWeb::CameraFeedWeb(const CameraInfo &info) {
	name = info.label;
	device_id = info.device_id;
	// Override base class default (Y-flip) with identity.
	// Browser delivers frames already in the correct orientation.
	transform = Transform2D();
}

CameraFeedWeb::~CameraFeedWeb() {
	if (is_active()) {
		deactivate_feed();
	}
}

void CameraWeb::_on_get_cameras_callback(void *p_context, const Vector<CameraInfo> &p_camera_info) {
	CameraWebRequest *request = static_cast<CameraWebRequest *>(p_context);
	ERR_FAIL_NULL(request);

	Object *server_object = ObjectDB::get_instance(request->server_id);
	CameraWeb *server = Object::cast_to<CameraWeb>(server_object);
	if (server == nullptr || request->request_id != server->request_id || !server->monitoring_feeds) {
		memdelete(request);
		return;
	}
	memdelete(request);

	// Build a set of new device IDs for quick lookup.
	HashSet<String> new_device_ids;
	for (const CameraInfo &camera_info : p_camera_info) {
		new_device_ids.insert(camera_info.device_id);
	}

	// Remove feeds that are no longer present.
	for (int i = server->feeds.size() - 1; i >= 0; i--) {
		Ref<CameraFeedWeb> feed = server->feeds[i];
		if (feed.is_valid() && !new_device_ids.has(feed->get_device_id())) {
			if (feed->is_active()) {
				feed->set_active(false);
			}
			server->remove_feed(feed);
		}
	}

	// Build a set of existing device IDs.
	HashSet<String> existing_device_ids;
	for (Ref<CameraFeedWeb> feed : server->feeds) {
		if (feed.is_valid()) {
			existing_device_ids.insert(feed->get_device_id());
		}
	}

	// Add new feeds that don't already exist.
	for (const CameraInfo &camera_info : p_camera_info) {
		if (!existing_device_ids.has(camera_info.device_id)) {
			Ref<CameraFeedWeb> feed = memnew(CameraFeedWeb(camera_info));
			server->add_feed(feed);
		}
	}

	server->call_deferred("emit_signal", SNAME(CameraServer::feeds_updated_signal_name));
}

void CameraWeb::_update_feeds() {
	CameraWebRequest *request = memnew(CameraWebRequest);
	request->server_id = get_instance_id();
	request->request_id = request_id;
	driver->get_cameras((void *)request, &_on_get_cameras_callback);
}

void CameraWeb::_cleanup() {
	request_id++;
	if (driver != nullptr) {
		// Abort in-flight opens for each active feed before stopping all streams.
		for (int i = 0; i < feeds.size(); i++) {
			CameraFeedWeb *feed = Object::cast_to<CameraFeedWeb>(feeds[i].ptr());
			if (feed) {
				driver->abort_stream(feed);
			}
		}
		driver->stop_stream();
		memdelete(driver);
		driver = nullptr;
	}
}

void CameraWeb::set_monitoring_feeds(bool p_monitoring_feeds) {
	if (p_monitoring_feeds == monitoring_feeds) {
		return;
	}

	CameraServer::set_monitoring_feeds(p_monitoring_feeds);
	if (p_monitoring_feeds) {
		if (driver == nullptr) {
			driver = memnew(CameraDriverWeb);
		}
		request_id++;
		_update_feeds();
	} else {
		_cleanup();
	}
}

CameraWeb::~CameraWeb() {
	_cleanup();
}
