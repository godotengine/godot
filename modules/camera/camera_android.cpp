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

//////////////////////////////////////////////////////////////////////////
// Helper functions
//
// The following code enables you to view the contents of a media type while
// debugging.

#ifndef IF_EQUAL_RETURN
#define IF_EQUAL_RETURN(param, val) \
	if (val == param)               \
	return #val
#endif

String GetFormatName(const int32_t &format) {
	IF_EQUAL_RETURN(format, AIMAGE_FORMAT_YUV_420_888);
	IF_EQUAL_RETURN(format, AIMAGE_FORMAT_RGB_888);
	IF_EQUAL_RETURN(format, AIMAGE_FORMAT_RGBA_8888);

	return "Unsupported";
}

//////////////////////////////////////////////////////////////////////////
// CameraFeedAndroid - Subclass for our camera feed on Android

CameraFeedAndroid::CameraFeedAndroid(ACameraManager *manager, const char *id, int32_t position, int32_t width,
		int32_t height, int32_t format, int32_t orientation) {
	this->manager = manager;
	this->camera_id = id;
	this->width = width;
	this->height = height;

	// Name
	name = vformat("%s | %d x %d", id, width, height);

	// Data type
	this->format = format;
	if (format == AIMAGE_FORMAT_RGB_888) {
		this->datatype = FEED_RGB;
		name += " | RGB";
	}
	if (format == AIMAGE_FORMAT_RGBA_8888) {
		this->datatype = FEED_RGBA;
		name += " | RGBA";
	}
	if (format == AIMAGE_FORMAT_YUV_420_888) {
		this->datatype = FEED_YCBCR;
		name += " | YCBCR";
	}

	// Position
	if (position == ACAMERA_LENS_FACING_BACK) {
		this->position = CameraFeed::FEED_BACK;
		name += " | BACK";
	}
	if (position == ACAMERA_LENS_FACING_FRONT) {
		this->position = CameraFeed::FEED_FRONT;
		name += " | FRONT";
	}

	// Orientation
	int32_t imageRotation = 0;
	if (position == ACAMERA_LENS_FACING_FRONT) {
		imageRotation = orientation % 360;
		imageRotation = (360 - imageRotation) % 360;
	} else {
		imageRotation = (orientation + 360) % 360;
	}
	transform.rotate(real_t(imageRotation) * 0.015707963267949F);
}

CameraFeedAndroid::~CameraFeedAndroid() {
	if (is_active()) {
		deactivate_feed();
	};
}

bool CameraFeedAndroid::activate_feed() {
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
	camera_status_t c_status = ACameraManager_openCamera(manager, camera_id.utf8(), &deviceCallbacks, &device);
	if (c_status != ACAMERA_OK) {
		onError(this, device, c_status);
		return false;
	}

	// Create image reader
	media_status_t m_status = AImageReader_new(width, height, format, 1, &reader);
	if (m_status != AMEDIA_OK) {
		onError(this, device, m_status);
		return false;
	}

	// Create image buffers
	set_image(RenderingServer::CANVAS_TEXTURE_CHANNEL_DIFFUSE,
			Image::create_empty(width, height, false, Image::FORMAT_R8));
	set_image(RenderingServer::CANVAS_TEXTURE_CHANNEL_NORMAL,
			Image::create_empty(width / 2, height / 2, false, Image::FORMAT_RG8));
	//    set_image(RenderingServer::CANVAS_TEXTURE_CHANNEL_SPECULAR,
	//              Image::create_empty(width, height, false, Image::FORMAT_R8));

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

void CameraFeedAndroid::onImage(void *context, AImageReader *p_reader) {
	auto *feed = static_cast<CameraFeedAndroid *>(context);

	// Get image
	AImage *image = nullptr;
	media_status_t status = AImageReader_acquireNextImage(p_reader, &image);
	ERR_FAIL_COND(status != AMEDIA_OK);

	// Get image data
	uint8_t *data = nullptr;
	int len = 0;
	int32_t pixel_stride, row_stride;
	AImage_getPlaneData(image, 0, &data, &len);
	feed->set_image(RenderingServer::CANVAS_TEXTURE_CHANNEL_DIFFUSE, data, 0, len);
	AImage_getPlanePixelStride(image, 1, &pixel_stride);
	AImage_getPlaneRowStride(image, 1, &row_stride);
	AImage_getPlaneData(image, 1, &data, &len);
	feed->set_image(RenderingServer::CANVAS_TEXTURE_CHANNEL_NORMAL, data, 0, len);
	//    AImage_getPlaneData(image, 2, &data, &len);
	//    feed->set_image(RenderingServer::CANVAS_TEXTURE_CHANNEL_SPECULAR, data, 0, len);

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
	print_verbose("Capture session active");
}

void CameraFeedAndroid::deactivate_feed() {
	if (session != nullptr) {
		ACameraCaptureSession_stopRepeating(session);
		ACameraCaptureSession_close(session);
		session = nullptr;
	}

	if (request != nullptr) {
		ACaptureRequest_free(request);
		request = nullptr;
	}

	if (reader != nullptr) {
		AImageReader_delete(reader);
		reader = nullptr;
	}

	if (device != nullptr) {
		ACameraDevice_close(device);
		device = nullptr;
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
	if (c_status != ACAMERA_OK) {
		ERR_PRINT("Unable to retrieve supported cameras");
		return;
	}

	for (int c = 0; c < cameraIds->numCameras; ++c) {
		const char *id = cameraIds->cameraIds[c];
		ACameraMetadata *metadata;
		ACameraManager_getCameraCharacteristics(cameraManager, id, &metadata);

		// Get position
		ACameraMetadata_const_entry lensInfo;
		ACameraMetadata_getConstEntry(metadata, ACAMERA_LENS_FACING, &lensInfo);
		uint8_t position = static_cast<acamera_metadata_enum_android_lens_facing_t>(lensInfo.data.u8[0]);

		// Get sensor orientation
		ACameraMetadata_const_entry orientation;
		ACameraMetadata_getConstEntry(metadata, ACAMERA_SENSOR_ORIENTATION, &orientation);
		int32_t cameraOrientation = orientation.data.i32[0];

		// Get supported formats
		ACameraMetadata_const_entry formats;
		ACameraMetadata_getConstEntry(metadata, ACAMERA_SCALER_AVAILABLE_STREAM_CONFIGURATIONS, &formats);
		for (uint32_t f = 0; f < formats.count; f += 4) {
			// Only support output streams
			int32_t input = formats.data.i32[f + 3];
			if (input) {
				continue;
			}

			// Get format and resolution
			int32_t format = formats.data.i32[f + 0];
			if (format == AIMAGE_FORMAT_YUV_420_888 || format == AIMAGE_FORMAT_RGB_888 ||
					format == AIMAGE_FORMAT_RGBA_8888) {
				int32_t width = formats.data.i32[f + 1];
				int32_t height = formats.data.i32[f + 2];
				Ref<CameraFeedAndroid> feed = new CameraFeedAndroid(cameraManager, id,
						position,
						width,
						height,
						format,
						cameraOrientation);
				add_feed(feed);
				print_line("Added camera feed: ", feed->get_name());
			}
		}

		ACameraMetadata_free(metadata);
	}

	ACameraManager_deleteCameraIdList(cameraIds);
}

CameraAndroid::CameraAndroid() {
	cameraManager = ACameraManager_create();

	// Update feeds
	update_feeds();
}

CameraAndroid::~CameraAndroid() {
	if (cameraManager != nullptr) {
		ACameraManager_delete(cameraManager);
	}
}
