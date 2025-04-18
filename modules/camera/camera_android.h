/**************************************************************************/
/*  camera_android.h                                                      */
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

#ifndef CAMERA_ANDROID_H
#define CAMERA_ANDROID_H

#include "servers/camera/camera_feed.h"
#include "servers/camera_server.h"

#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraError.h>
#include <camera/NdkCameraManager.h>
#include <camera/NdkCameraMetadataTags.h>
#include <media/NdkImageReader.h>

class CameraFeedAndroid : public CameraFeed {
private:
    String camera_id;
    int32_t format;

	ACameraManager *manager = nullptr;
	ACameraDevice *device = nullptr;
	AImageReader *reader = nullptr;
	ACameraCaptureSession *session = nullptr;
	ACaptureRequest *request = nullptr;

    static void onError(void *context, ACameraDevice *p_device, int error);
	static void onDisconnected(void *context, ACameraDevice *p_device);
    static void onImage(void *context, AImageReader *p_reader);
    static void onSessionReady(void *context, ACameraCaptureSession *session);
    static void onSessionActive(void *context, ACameraCaptureSession *session);
    static void onSessionClosed(void *context, ACameraCaptureSession *session);

protected:
public:
	CameraFeedAndroid(ACameraManager *manager, const char *id, int32_t position, int32_t width, int32_t height,
                      int32_t format, int32_t orientation);
	virtual ~CameraFeedAndroid();

	bool activate_feed();
	void deactivate_feed();
};

class CameraAndroid : public CameraServer {
private:
	ACameraManager *cameraManager;

	void update_feeds();

public:
	CameraAndroid();
	~CameraAndroid();
};

#endif // CAMERA_ANDROID_H
