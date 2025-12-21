/**************************************************************************/
/*  camera_android.cpp                                                    */
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

#include "camera_android.h"

#include "core/os/os.h"
#include "platform/android/display_server_android.h"
#include "platform/android/java_godot_io_wrapper.h"
#include "platform/android/os_android.h"

//////////////////////////////////////////////////////////////////////////
// Helper functions
//
// The following code enables you to view the contents of a media type while
// debugging.

#ifndef IF_EQUAL_RETURN
#define MAKE_FORMAT_CONST(suffix) AIMAGE_FORMAT_##suffix
#define IF_EQUAL_RETURN(param, val)      \
	if (MAKE_FORMAT_CONST(val) == param) \
	return #val
#endif

String GetFormatName(const int32_t &format) {
	IF_EQUAL_RETURN(format, YUV_420_888);
	IF_EQUAL_RETURN(format, RGB_888);
	IF_EQUAL_RETURN(format, RGBA_8888);

	return "Unsupported";
}

//////////////////////////////////////////////////////////////////////////
// CameraFeedAndroid - Subclass for our camera feed on Android

CameraFeedAndroid::CameraFeedAndroid(ACameraManager *manager, ACameraMetadata *metadata, const char *id,
		CameraFeed::FeedPosition position, int32_t orientation) :
		CameraFeed() {
	this->manager = manager;
	this->metadata = metadata;
	this->orientation = orientation;
	_add_formats();
	camera_id = id;
	set_position(position);

	// Position
	switch (position) {
		case CameraFeed::FEED_BACK:
			name = vformat("%s | BACK", id);
			break;
		case CameraFeed::FEED_FRONT:
			name = vformat("%s | FRONT", id);
			break;
		default:
			name = vformat("%s", id);
			break;
	}

	image_y.instantiate();
	image_uv.instantiate();
}

CameraFeedAndroid::~CameraFeedAndroid() {
	if (is_active()) {
		deactivate_feed();
	}
	if (metadata != nullptr) {
		ACameraMetadata_free(metadata);
	}
}

void CameraFeedAndroid::refresh_camera_metadata() {
	ERR_FAIL_NULL_MSG(manager, vformat("Camera %s: Cannot refresh metadata, manager is null.", camera_id));

	if (metadata != nullptr) {
		ACameraMetadata_free(metadata);
		metadata = nullptr;
	}

	camera_status_t status = ACameraManager_getCameraCharacteristics(manager, camera_id.utf8().get_data(), &metadata);
	if (status != ACAMERA_OK || metadata == nullptr) {
		ERR_FAIL_MSG(vformat("Camera %s: Failed to refresh metadata (status: %d).", camera_id, status));
	}

	ACameraMetadata_const_entry orientation_entry;
	status = ACameraMetadata_getConstEntry(metadata, ACAMERA_SENSOR_ORIENTATION, &orientation_entry);
	if (status == ACAMERA_OK) {
		orientation = orientation_entry.data.i32[0];
		print_verbose(vformat("Camera %s: Orientation updated to %d.", camera_id, orientation));
	} else {
		ERR_PRINT(vformat("Camera %s: Failed to get sensor orientation after refresh (status: %d).", camera_id, status));
	}

	formats.clear();
	_add_formats();

	print_verbose(vformat("Camera %s: Metadata refreshed successfully.", camera_id));
}

void CameraFeedAndroid::_set_rotation() {
	if (!metadata) {
		print_verbose(vformat("Camera %s: Metadata is null in _set_rotation, attempting refresh.", camera_id));
		refresh_camera_metadata();
	}

	float image_rotation = 0.0f;
	std::optional<int> result;

	if (metadata) {
		CameraRotationParams params;
		params.sensor_orientation = orientation;
		params.camera_facing = (position == CameraFeed::FEED_FRONT) ? CameraFacing::FRONT : CameraFacing::BACK;
		params.display_rotation = get_app_orientation();

		result = calculate_rotation(params);
	} else {
		ERR_PRINT(vformat("Camera %s: Cannot update rotation, metadata unavailable after refresh, using fallback.", camera_id));
	}

	if (result.has_value()) {
		image_rotation = static_cast<float>(result.value());
	} else {
		int display_rotation = DisplayServerAndroid::get_singleton()->get_display_rotation();
		switch (display_rotation) {
			case 90:
				display_rotation = 270;
				break;
			case 270:
				display_rotation = 90;
				break;
			default:
				break;
		}

		int sign = position == CameraFeed::FEED_FRONT ? 1 : -1;
		image_rotation = (orientation - display_rotation * sign + 360) % 360;
	}

	transform = Transform2D();
	transform = transform.rotated(Math::deg_to_rad(image_rotation));
}

void CameraFeedAndroid::_add_formats() {
	// Get supported formats
	ACameraMetadata_const_entry formats;
	camera_status_t status = ACameraMetadata_getConstEntry(metadata, ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS, &formats);

	if (status == ACAMERA_OK) {
		for (uint32_t f = 0; f < formats.count; f += 4) {
			// Only support output streams
			int32_t input = formats.data.i32[f + 3];
			if (input) {
				continue;
			}

			// Get format and resolution
			int32_t format = formats.data.i32[f + 0];
			if (format == AIMAGE_FORMAT_YUV_420_888 ||
					format == AIMAGE_FORMAT_RGBA_8888 ||
					format == AIMAGE_FORMAT_RGB_888) {
				CameraFeed::FeedFormat feed_format;
				feed_format.width = formats.data.i32[f + 1];
				feed_format.height = formats.data.i32[f + 2];
				feed_format.format = GetFormatName(format);
				feed_format.pixel_format = format;
				this->formats.append(feed_format);
			}
		}
	}
}

bool CameraFeedAndroid::activate_feed() {
	ERR_FAIL_COND_V_MSG(formats.is_empty(), false, "No camera formats available.");
	ERR_FAIL_INDEX_V_MSG(selected_format, formats.size(), false,
			vformat("CameraFeed format needs to be set before activating. Selected format index: %d (formats size: %d)", selected_format, formats.size()));
	if (is_active()) {
		deactivate_feed();
	};

	// Request permission
	if (!OS::get_singleton()->request_permission("CAMERA")) {
		return false;
	}

	// Open device
	static ACameraDevice_stateCallbacks deviceCallbacks = {
		.context = this,
		.onDisconnected = onDisconnected,
		.onError = onError,
	};
	camera_status_t c_status = ACameraManager_openCamera(manager, camera_id.utf8().get_data(), &deviceCallbacks, &device);
	if (c_status != ACAMERA_OK) {
		onError(this, device, c_status);
		return false;
	}

	// Create image reader
	const FeedFormat &feed_format = formats[selected_format];
	media_status_t m_status = AImageReader_new(feed_format.width, feed_format.height, feed_format.pixel_format, 1, &reader);
	if (m_status != AMEDIA_OK) {
		onError(this, device, m_status);
		return false;
	}

	// Get image listener
	static AImageReader_ImageListener listener{
		.context = this,
		.onImageAvailable = onImage,
	};
	m_status = AImageReader_setImageListener(reader, &listener);
	if (m_status != AMEDIA_OK) {
		onError(this, device, m_status);
		return false;
	}

	// Get image surface
	ANativeWindow *surface;
	m_status = AImageReader_getWindow(reader, &surface);
	if (m_status != AMEDIA_OK) {
		onError(this, device, m_status);
		return false;
	}

	// Prepare session outputs
	ACaptureSessionOutput *output = nullptr;
	c_status = ACaptureSessionOutput_create(surface, &output);
	if (c_status != ACAMERA_OK) {
		onError(this, device, c_status);
		return false;
	}

	ACaptureSessionOutputContainer *outputs = nullptr;
	c_status = ACaptureSessionOutputContainer_create(&outputs);
	if (c_status != ACAMERA_OK) {
		onError(this, device, c_status);
		return false;
	}

	c_status = ACaptureSessionOutputContainer_add(outputs, output);
	if (c_status != ACAMERA_OK) {
		onError(this, device, c_status);
		return false;
	}

	// Create capture session
	static ACameraCaptureSession_stateCallbacks sessionStateCallbacks{
		.context = this,
		.onClosed = onSessionClosed,
		.onReady = onSessionReady,
		.onActive = onSessionActive
	};
	c_status = ACameraDevice_createCaptureSession(device, outputs, &sessionStateCallbacks, &session);
	if (c_status != ACAMERA_OK) {
		onError(this, device, c_status);
		return false;
	}

	// Create capture request
	c_status = ACameraDevice_createCaptureRequest(device, TEMPLATE_PREVIEW, &request);
	if (c_status != ACAMERA_OK) {
		onError(this, device, c_status);
		return false;
	}

	// Set capture target
	ACameraOutputTarget *imageTarget = nullptr;
	c_status = ACameraOutputTarget_create(surface, &imageTarget);
	if (c_status != ACAMERA_OK) {
		onError(this, device, c_status);
		return false;
	}

	c_status = ACaptureRequest_addTarget(request, imageTarget);
	if (c_status != ACAMERA_OK) {
		onError(this, device, c_status);
		return false;
	}

	// Start capture
	c_status = ACameraCaptureSession_setRepeatingRequest(session, nullptr, 1, &request, nullptr);
	if (c_status != ACAMERA_OK) {
		onError(this, device, c_status);
		return false;
	}

	return true;
}

bool CameraFeedAndroid::set_format(int p_index, const Dictionary &p_parameters) {
	ERR_FAIL_COND_V_MSG(active, false, "Feed is active.");
	ERR_FAIL_INDEX_V_MSG(p_index, formats.size(), false, "Invalid format index.");

	selected_format = p_index;
	return true;
}

Array CameraFeedAndroid::get_formats() const {
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

CameraFeed::FeedFormat CameraFeedAndroid::get_format() const {
	CameraFeed::FeedFormat feed_format = {};
	ERR_FAIL_INDEX_V_MSG(selected_format, formats.size(), feed_format,
			vformat("Invalid format index: %d (formats size: %d)", selected_format, formats.size()));
	return formats[selected_format];
}

void CameraFeedAndroid::handle_pause() {
	if (is_active()) {
		was_active_before_pause = true;
		print_verbose(vformat("Camera %s: Pausing (was active).", camera_id));
		deactivate_feed();
	} else {
		was_active_before_pause = false;
	}
}

void CameraFeedAndroid::handle_resume() {
	if (was_active_before_pause) {
		print_verbose(vformat("Camera %s: Resuming.", camera_id));
		activate_feed();
		was_active_before_pause = false;
	}
}

void CameraFeedAndroid::handle_rotation_change() {
	if (!is_active()) {
		return;
	}

	print_verbose(vformat("Camera %s: Handling rotation change.", camera_id));
	refresh_camera_metadata();
	_set_rotation();
}

void CameraFeedAndroid::onImage(void *context, AImageReader *p_reader) {
	CameraFeedAndroid *feed = static_cast<CameraFeedAndroid *>(context);

	MutexLock lock(feed->callback_mutex);

	if (!feed->is_active()) {
		AImage *pending_image = nullptr;
		if (AImageReader_acquireNextImage(p_reader, &pending_image) == AMEDIA_OK) {
			AImage_delete(pending_image);
		}
		return;
	}

	Vector<uint8_t> data_y = feed->data_y;
	Vector<uint8_t> data_uv = feed->data_uv;
	Ref<Image> image_y = feed->image_y;
	Ref<Image> image_uv = feed->image_uv;

	// Get image
	AImage *image = nullptr;
	media_status_t status = AImageReader_acquireNextImage(p_reader, &image);
	ERR_FAIL_COND(status != AMEDIA_OK);

	// Get image data
	uint8_t *data = nullptr;
	int len = 0;
	int32_t pixel_stride, row_stride;
	FeedFormat format = feed->get_format();
	int width = format.width;
	int height = format.height;
	switch (format.pixel_format) {
		case AIMAGE_FORMAT_YUV_420_888:
			AImage_getPlaneData(image, 0, &data, &len);
			if (len <= 0) {
				return;
			}
			if (len != data_y.size()) {
				int64_t size = Image::get_image_data_size(width, height, Image::FORMAT_R8, false);
				data_y.resize(len > size ? len : size);
			}
			memcpy(data_y.ptrw(), data, len);

			AImage_getPlanePixelStride(image, 1, &pixel_stride);
			AImage_getPlaneRowStride(image, 1, &row_stride);
			AImage_getPlaneData(image, 1, &data, &len);
			if (len <= 0) {
				return;
			}
			if (len != data_uv.size()) {
				int64_t size = Image::get_image_data_size(width / 2, height / 2, Image::FORMAT_RG8, false);
				data_uv.resize(len > size ? len : size);
			}
			memcpy(data_uv.ptrw(), data, len);

			image_y->initialize_data(width, height, false, Image::FORMAT_R8, data_y);
			image_uv->initialize_data(width / 2, height / 2, false, Image::FORMAT_RG8, data_uv);

			feed->set_ycbcr_images(image_y, image_uv);
			break;
		case AIMAGE_FORMAT_RGBA_8888:
			AImage_getPlaneData(image, 0, &data, &len);
			if (len <= 0) {
				return;
			}
			if (len != data_y.size()) {
				int64_t size = Image::get_image_data_size(width, height, Image::FORMAT_RGBA8, false);
				data_y.resize(len > size ? len : size);
			}
			memcpy(data_y.ptrw(), data, len);

			image_y->initialize_data(width, height, false, Image::FORMAT_RGBA8, data_y);

			feed->set_rgb_image(image_y);
			break;
		case AIMAGE_FORMAT_RGB_888:
			AImage_getPlaneData(image, 0, &data, &len);
			if (len <= 0) {
				return;
			}
			if (len != data_y.size()) {
				int64_t size = Image::get_image_data_size(width, height, Image::FORMAT_RGB8, false);
				data_y.resize(len > size ? len : size);
			}
			memcpy(data_y.ptrw(), data, len);

			image_y->initialize_data(width, height, false, Image::FORMAT_RGB8, data_y);

			feed->set_rgb_image(image_y);
			break;
		default:
			return;
	}

	if (!feed->formats.is_empty()) {
		if (feed->metadata != nullptr) {
			feed->_set_rotation();
		} else {
			print_verbose(vformat("Camera %s: Metadata invalidated in onImage, attempting refresh.", feed->camera_id));
			feed->refresh_camera_metadata();
			if (feed->metadata != nullptr && !feed->formats.is_empty()) {
				feed->_set_rotation();
			}
		}
	}

	// Release image
	AImage_delete(image);
}

void CameraFeedAndroid::onSessionReady(void *context, ACameraCaptureSession *session) {
	print_verbose("Capture session ready");
}

void CameraFeedAndroid::onSessionActive(void *context, ACameraCaptureSession *session) {
	print_verbose("Capture session active");
}

void CameraFeedAndroid::onSessionClosed(void *context, ACameraCaptureSession *session) {
	print_verbose("Capture session closed");
}

void CameraFeedAndroid::deactivate_feed() {
	// First, remove image listener to prevent new callbacks.
	if (reader != nullptr) {
		AImageReader_setImageListener(reader, nullptr);
	}

	// Stop and close capture session.
	// These calls may wait for pending callbacks to complete.
	if (session != nullptr) {
		ACameraCaptureSession_stopRepeating(session);
		ACameraCaptureSession_close(session);
		session = nullptr;
	}

	// Now safe to acquire lock and clean up resources.
	// No new callbacks will be triggered after this point.
	MutexLock lock(callback_mutex);

	if (device != nullptr) {
		ACameraDevice_close(device);
		device = nullptr;
	}

	if (reader != nullptr) {
		AImageReader_delete(reader);
		reader = nullptr;
	}

	if (request != nullptr) {
		ACaptureRequest_free(request);
		request = nullptr;
	}
}

void CameraFeedAndroid::onError(void *context, ACameraDevice *p_device, int error) {
	print_error(vformat("Camera error: %d", error));
	onDisconnected(context, p_device);
}

void CameraFeedAndroid::onDisconnected(void *context, ACameraDevice *p_device) {
	print_verbose("Camera disconnected");
	auto *feed = static_cast<CameraFeedAndroid *>(context);
	feed->set_active(false);
}

//////////////////////////////////////////////////////////////////////////
// CameraAndroid - Subclass for our camera server on Android

void CameraAndroid::update_feeds() {
	ACameraIdList *cameraIds = nullptr;
	camera_status_t c_status = ACameraManager_getCameraIdList(cameraManager, &cameraIds);
	ERR_FAIL_COND(c_status != ACAMERA_OK);

	// remove existing devices
	for (int i = feeds.size() - 1; i >= 0; i--) {
		remove_feed(feeds[i]);
	}

	for (int c = 0; c < cameraIds->numCameras; ++c) {
		const char *id = cameraIds->cameraIds[c];
		ACameraMetadata *metadata = nullptr;
		ACameraManager_getCameraCharacteristics(cameraManager, id, &metadata);
		if (!metadata) {
			continue;
		}

		// Get sensor orientation
		ACameraMetadata_const_entry orientation;
		c_status = ACameraMetadata_getConstEntry(metadata, ACAMERA_SENSOR_ORIENTATION, &orientation);
		int32_t cameraOrientation;
		if (c_status == ACAMERA_OK) {
			cameraOrientation = orientation.data.i32[0];
		} else {
			cameraOrientation = 0;
			print_error(vformat("Unable to get sensor orientation: %s", id));
		}

		// Get position
		ACameraMetadata_const_entry lensInfo;
		CameraFeed::FeedPosition position = CameraFeed::FEED_UNSPECIFIED;
		camera_status_t status;
		status = ACameraMetadata_getConstEntry(metadata, ACAMERA_LENS_FACING, &lensInfo);
		if (status != ACAMERA_OK) {
			ACameraMetadata_free(metadata);
			continue;
		}
		uint8_t lens_facing = static_cast<acamera_metadata_enum_android_lens_facing_t>(lensInfo.data.u8[0]);
		if (lens_facing == ACAMERA_LENS_FACING_FRONT) {
			position = CameraFeed::FEED_FRONT;
		} else if (lens_facing == ACAMERA_LENS_FACING_BACK) {
			position = CameraFeed::FEED_BACK;
		} else {
			ACameraMetadata_free(metadata);
			continue;
		}

		Ref<CameraFeedAndroid> feed = memnew(CameraFeedAndroid(cameraManager, metadata, id, position, cameraOrientation));
		add_feed(feed);
	}

	ACameraManager_deleteCameraIdList(cameraIds);
	emit_signal(SNAME(CameraServer::feeds_updated_signal_name));
}

void CameraAndroid::remove_all_feeds() {
	// remove existing devices
	for (int i = feeds.size() - 1; i >= 0; i--) {
		remove_feed(feeds[i]);
	}

	if (cameraManager != nullptr) {
		ACameraManager_delete(cameraManager);
		cameraManager = nullptr;
	}
}

void CameraAndroid::set_monitoring_feeds(bool p_monitoring_feeds) {
	if (p_monitoring_feeds == monitoring_feeds) {
		return;
	}

	CameraServer::set_monitoring_feeds(p_monitoring_feeds);
	if (p_monitoring_feeds) {
		if (cameraManager == nullptr) {
			cameraManager = ACameraManager_create();
		}

		// Update feeds
		update_feeds();
	} else {
		remove_all_feeds();
	}
}

void CameraAndroid::handle_application_pause() {
	for (int i = 0; i < feeds.size(); i++) {
		Ref<CameraFeedAndroid> feed = feeds[i];
		if (feed.is_valid()) {
			feed->handle_pause();
		}
	}
}

void CameraAndroid::handle_application_resume() {
	for (int i = 0; i < feeds.size(); i++) {
		Ref<CameraFeedAndroid> feed = feeds[i];
		if (feed.is_valid()) {
			feed->handle_resume();
		}
	}
}

void CameraAndroid::handle_display_rotation_change(int) {
	for (int i = 0; i < feeds.size(); i++) {
		Ref<CameraFeedAndroid> feed = feeds[i];
		if (feed.is_valid()) {
			feed->handle_rotation_change();
		}
	}
}

CameraAndroid::~CameraAndroid() {
	remove_all_feeds();
}

std::optional<int> CameraFeedAndroid::calculate_rotation(const CameraRotationParams &p_params) {
	if (p_params.sensor_orientation < 0 || p_params.sensor_orientation > 270 || p_params.sensor_orientation % 90 != 0) {
		return std::nullopt;
	}

	int rotation_angle = p_params.sensor_orientation - p_params.display_rotation;
	return normalize_angle(rotation_angle);
}

int CameraFeedAndroid::normalize_angle(int p_angle) {
	while (p_angle < 0) {
		p_angle += 360;
	}
	return p_angle % 360;
}

int CameraFeedAndroid::get_display_rotation() {
	return DisplayServerAndroid::get_singleton()->get_display_rotation();
}

int CameraFeedAndroid::get_app_orientation() {
	GodotIOJavaWrapper *godot_io_java = OS_Android::get_singleton()->get_godot_io_java();
	ERR_FAIL_NULL_V(godot_io_java, 0);

	int orientation = godot_io_java->get_screen_orientation();
	switch (orientation) {
		case 0: // SCREEN_LANDSCAPE
			return 90;
		case 1: // SCREEN_PORTRAIT
			return 0;
		case 2: // SCREEN_REVERSE_LANDSCAPE
			return 270;
		case 3: // SCREEN_REVERSE_PORTRAIT
			return 180;
		case 4: // SCREEN_SENSOR_LANDSCAPE
		case 5: // SCREEN_SENSOR_PORTRAIT
		case 6: // SCREEN_SENSOR
		default:
			return get_display_rotation();
	}
}
