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

// Scope guard to ensure AImage instances are always deleted.
class ScopedAImage {
	AImage *image = nullptr;

public:
	ScopedAImage() = default;
	~ScopedAImage() {
		reset();
	}

	ScopedAImage(const ScopedAImage &) = delete;
	ScopedAImage &operator=(const ScopedAImage &) = delete;

	AImage **out() {
		return &image;
	}

	AImage *get() const {
		return image;
	}

	void reset(AImage *p_image = nullptr) {
		if (image != nullptr) {
			AImage_delete(image);
		}
		image = p_image;
	}
};

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
	device_callbacks = {
		.context = this,
		.onDisconnected = onDisconnected,
		.onError = onError,
	};
	camera_status_t c_status = ACameraManager_openCamera(manager, camera_id.utf8().get_data(), &device_callbacks, &device);
	if (c_status != ACAMERA_OK) {
		print_error(vformat("Camera %s: Failed to open camera (status: %d)", camera_id, c_status));
		deactivate_feed();
		return false;
	}

	// Create image reader
	const FeedFormat &feed_format = formats[selected_format];
	media_status_t m_status = AImageReader_new(feed_format.width, feed_format.height, feed_format.pixel_format, 1, &reader);
	if (m_status != AMEDIA_OK) {
		print_error(vformat("Camera %s: Failed to create image reader (status: %d)", camera_id, m_status));
		deactivate_feed();
		return false;
	}

	// Get image listener
	image_listener = {
		.context = this,
		.onImageAvailable = onImage,
	};
	m_status = AImageReader_setImageListener(reader, &image_listener);
	if (m_status != AMEDIA_OK) {
		print_error(vformat("Camera %s: Failed to set image listener (status: %d)", camera_id, m_status));
		deactivate_feed();
		return false;
	}

	// Get image surface
	ANativeWindow *surface;
	m_status = AImageReader_getWindow(reader, &surface);
	if (m_status != AMEDIA_OK) {
		print_error(vformat("Camera %s: Failed to get image surface (status: %d)", camera_id, m_status));
		deactivate_feed();
		return false;
	}

	// Prepare session outputs
	c_status = ACaptureSessionOutput_create(surface, &session_output);
	if (c_status != ACAMERA_OK) {
		print_error(vformat("Camera %s: Failed to create session output (status: %d)", camera_id, c_status));
		deactivate_feed();
		return false;
	}

	c_status = ACaptureSessionOutputContainer_create(&session_output_container);
	if (c_status != ACAMERA_OK) {
		print_error(vformat("Camera %s: Failed to create session output container (status: %d)", camera_id, c_status));
		deactivate_feed();
		return false;
	}

	c_status = ACaptureSessionOutputContainer_add(session_output_container, session_output);
	if (c_status != ACAMERA_OK) {
		print_error(vformat("Camera %s: Failed to add session output to container (status: %d)", camera_id, c_status));
		deactivate_feed();
		return false;
	}

	// Create capture session
	session_callbacks = {
		.context = this,
		.onClosed = onSessionClosed,
		.onReady = onSessionReady,
		.onActive = onSessionActive
	};
	c_status = ACameraDevice_createCaptureSession(device, session_output_container, &session_callbacks, &session);
	if (c_status != ACAMERA_OK) {
		print_error(vformat("Camera %s: Failed to create capture session (status: %d)", camera_id, c_status));
		deactivate_feed();
		return false;
	}

	// Create capture request
	c_status = ACameraDevice_createCaptureRequest(device, TEMPLATE_PREVIEW, &request);
	if (c_status != ACAMERA_OK) {
		print_error(vformat("Camera %s: Failed to create capture request (status: %d)", camera_id, c_status));
		deactivate_feed();
		return false;
	}

	// Set capture target
	c_status = ACameraOutputTarget_create(surface, &camera_output_target);
	if (c_status != ACAMERA_OK) {
		print_error(vformat("Camera %s: Failed to create camera output target (status: %d)", camera_id, c_status));
		deactivate_feed();
		return false;
	}

	c_status = ACaptureRequest_addTarget(request, camera_output_target);
	if (c_status != ACAMERA_OK) {
		print_error(vformat("Camera %s: Failed to add target to capture request (status: %d)", camera_id, c_status));
		deactivate_feed();
		return false;
	}

	// Start capture
	c_status = ACameraCaptureSession_setRepeatingRequest(session, nullptr, 1, &request, nullptr);
	if (c_status != ACAMERA_OK) {
		print_error(vformat("Camera %s: Failed to start repeating request (status: %d)", camera_id, c_status));
		deactivate_feed();
		return false;
	}

	return true;
}

bool CameraFeedAndroid::set_format(int p_index, const Dictionary &p_parameters) {
	ERR_FAIL_COND_V_MSG(active, false, "Feed is active.");
	ERR_FAIL_INDEX_V_MSG(p_index, formats.size(), false, "Invalid format index.");

	selected_format = p_index;

	// Reset base dimensions to force texture recreation on next frame.
	// This ensures proper handling when switching between different resolutions.
	base_width = 0;
	base_height = 0;

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

// In-place stride compaction (handles row padding).
void CameraFeedAndroid::compact_stride_inplace(uint8_t *data, size_t width, int height, size_t stride) {
	if (stride <= width) {
		return;
	}

	uint8_t *src_row = data + stride;
	uint8_t *dst_row = data + width;

	for (int y = 1; y < height; y++) {
		memmove(dst_row, src_row, width);
		src_row += stride;
		dst_row += width;
	}
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

	Vector<uint8_t> data_y;
	Vector<uint8_t> data_uv;

	// Get image
	ScopedAImage image_guard;
	media_status_t status = AImageReader_acquireNextImage(p_reader, image_guard.out());
	ERR_FAIL_COND(status != AMEDIA_OK);
	AImage *image = image_guard.get();

	// Get image data
	uint8_t *data = nullptr;
	int len = 0;
	int32_t pixel_stride, row_stride;
	FeedFormat format = feed->get_format();
	int width = format.width;
	int height = format.height;

	switch (format.pixel_format) {
		case AIMAGE_FORMAT_YUV_420_888: {
			int32_t y_row_stride;
			AImage_getPlaneRowStride(image, 0, &y_row_stride);
			AImage_getPlaneData(image, 0, &data, &len);

			if (len <= 0) {
				return;
			}

			int64_t y_size = Image::get_image_data_size(width, height, Image::FORMAT_R8, false);

			if (y_row_stride == width && len == y_size) {
				data_y.resize(y_size);
				memcpy(data_y.ptrw(), data, y_size);
			} else {
				// Validate buffer size before compaction to prevent out-of-bounds read.
				int64_t required_y_len = (int64_t)y_row_stride * (height - 1) + width;
				if (len < required_y_len) {
					return;
				}
				if (feed->scratch_y.size() < len) {
					feed->scratch_y.resize(len);
				}
				memcpy(feed->scratch_y.ptrw(), data, len);
				CameraFeedAndroid::compact_stride_inplace(feed->scratch_y.ptrw(), width, height, y_row_stride);
				data_y.resize(y_size);
				memcpy(data_y.ptrw(), feed->scratch_y.ptr(), y_size);
			}

			AImage_getPlanePixelStride(image, 1, &pixel_stride);
			AImage_getPlaneRowStride(image, 1, &row_stride);
			AImage_getPlaneData(image, 1, &data, &len);

			if (len <= 0) {
				return;
			}

			int64_t uv_size = Image::get_image_data_size(width / 2, height / 2, Image::FORMAT_RG8, false);

			int uv_width = width / 2;
			int uv_height = height / 2;

			uint8_t *data_v = nullptr;
			int32_t v_pixel_stride = 0;
			int32_t v_row_stride = 0;
			int len_v = 0;

			if (pixel_stride != 2) {
				AImage_getPlanePixelStride(image, 2, &v_pixel_stride);
				AImage_getPlaneRowStride(image, 2, &v_row_stride);
				AImage_getPlaneData(image, 2, &data_v, &len_v);
				if (len_v <= 0) {
					return;
				}
			}

			if (pixel_stride == 2 && row_stride == uv_width * 2 && len == uv_size) {
				data_uv.resize(uv_size);
				memcpy(data_uv.ptrw(), data, uv_size);
			} else if (pixel_stride == 2) {
				// Allow 1-2 byte tolerance for UV buffer (some devices omit final bytes).
				int64_t required_uv_len = (int64_t)row_stride * (uv_height - 1) + uv_width * 2;
				const int64_t UV_TOLERANCE = 2;
				if (len < required_uv_len - UV_TOLERANCE) {
					return;
				}

				if (feed->scratch_uv.size() < required_uv_len) {
					feed->scratch_uv.resize(required_uv_len);
				}
				if (len < required_uv_len) {
					memset(feed->scratch_uv.ptrw() + len, 128, required_uv_len - len);
				}
				memcpy(feed->scratch_uv.ptrw(), data, len);
				if (row_stride != uv_width * 2) {
					CameraFeedAndroid::compact_stride_inplace(feed->scratch_uv.ptrw(), uv_width * 2, uv_height, row_stride);
				}
				data_uv.resize(uv_size);
				memcpy(data_uv.ptrw(), feed->scratch_uv.ptr(), uv_size);
			} else {
				if (data_v && len_v > 0) {
					data_uv.resize(uv_size);
					uint8_t *dst = data_uv.ptrw();
					uint8_t *src_u = data;
					uint8_t *src_v = data_v;
					for (int row = 0; row < uv_height; row++) {
						for (int col = 0; col < uv_width; col++) {
							dst[col * 2] = src_u[col * pixel_stride];
							dst[col * 2 + 1] = src_v[col * v_pixel_stride];
						}
						dst += uv_width * 2;
						src_u += row_stride;
						src_v += v_row_stride;
					}
				} else {
					data_uv.resize(uv_size);
					memset(data_uv.ptrw(), 128, uv_size);
				}
			}

			// Defer to main thread to avoid race conditions with RenderingServer.
			feed->image_y.instantiate();
			feed->image_y->initialize_data(width, height, false, Image::FORMAT_R8, data_y);

			feed->image_uv.instantiate();
			feed->image_uv->initialize_data(width / 2, height / 2, false, Image::FORMAT_RG8, data_uv);

			feed->call_deferred("set_ycbcr_images", feed->image_y, feed->image_uv);
			break;
		}
		case AIMAGE_FORMAT_RGBA_8888: {
			AImage_getPlaneData(image, 0, &data, &len);
			if (len <= 0) {
				return;
			}
			int64_t size = Image::get_image_data_size(width, height, Image::FORMAT_RGBA8, false);
			data_y.resize(len > size ? len : size);
			memcpy(data_y.ptrw(), data, len);

			feed->image_y.instantiate();
			feed->image_y->initialize_data(width, height, false, Image::FORMAT_RGBA8, data_y);

			feed->call_deferred("set_rgb_image", feed->image_y);
			break;
		}
		case AIMAGE_FORMAT_RGB_888: {
			AImage_getPlaneData(image, 0, &data, &len);
			if (len <= 0) {
				return;
			}
			int64_t size = Image::get_image_data_size(width, height, Image::FORMAT_RGB8, false);
			data_y.resize(len > size ? len : size);
			memcpy(data_y.ptrw(), data, len);

			feed->image_y.instantiate();
			feed->image_y->initialize_data(width, height, false, Image::FORMAT_RGB8, data_y);

			feed->call_deferred("set_rgb_image", feed->image_y);
			break;
		}
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

	// Release image happens automatically via ScopedAImage.
}

void CameraFeedAndroid::onSessionReady(void *context, ACameraCaptureSession *session) {
	print_verbose("Capture session ready");
}

void CameraFeedAndroid::onSessionActive(void *context, ACameraCaptureSession *session) {
	print_verbose("Capture session active");
}

void CameraFeedAndroid::onSessionClosed(void *context, ACameraCaptureSession *p_session) {
	CameraFeedAndroid *feed = static_cast<CameraFeedAndroid *>(context);
	feed->session_closed_semaphore.post();
}

void CameraFeedAndroid::deactivate_feed() {
	// First, remove image listener to prevent new callbacks.
	if (reader != nullptr) {
		AImageReader_setImageListener(reader, nullptr);
	}

	// Stop and close capture session.
	// Must wait for session to fully close before closing device.
	if (session != nullptr) {
		ACameraCaptureSession_stopRepeating(session);
		ACameraCaptureSession_close(session);
		// Wait for onSessionClosed callback to ensure session is fully closed
		// before proceeding to close the device.
		session_closed_semaphore.wait();
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

	if (camera_output_target != nullptr) {
		ACameraOutputTarget_free(camera_output_target);
		camera_output_target = nullptr;
	}

	if (session_output_container != nullptr) {
		ACaptureSessionOutputContainer_free(session_output_container);
		session_output_container = nullptr;
	}

	if (session_output != nullptr) {
		ACaptureSessionOutput_free(session_output);
		session_output = nullptr;
	}
}

void CameraFeedAndroid::onError(void *context, ACameraDevice *p_device, int error) {
	print_error(vformat("Camera error: %d", error));
	onDisconnected(context, p_device);
}

void CameraFeedAndroid::onDisconnected(void *context, ACameraDevice *p_device) {
	auto *feed = static_cast<CameraFeedAndroid *>(context);
	// Defer to main thread to avoid reentrancy issues when called from
	// ACameraDevice_close() during deactivate_feed().
	feed->call_deferred("set_active", false);
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
