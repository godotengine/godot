/**************************************************************************/
/*  camera_apple.mm                                                       */
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

///@TODO this is a near duplicate of CameraIOS, we should find a way to combine those to minimize code duplication!!!!
// If you fix something here, make sure you fix it there as well!

#import "camera_apple.h"

#include "servers/camera/camera_feed.h"

#import <AVFoundation/AVFoundation.h>
#ifdef IOS_ENABLED
#import <UIKit/UIKit.h>
#endif

//////////////////////////////////////////////////////////////////////////
// MyCaptureSession - This is a little helper class so we can capture our frames

@interface MyCaptureSession : AVCaptureSession <AVCaptureVideoDataOutputSampleBufferDelegate> {
	Ref<CameraFeed> feed;
	size_t width[2];
	size_t height[2];
	Vector<uint8_t> img_data[2];

	AVCaptureDeviceInput *input;
	AVCaptureVideoDataOutput *output;
}

@end

@implementation MyCaptureSession

- (id)initForFeed:(Ref<CameraFeed>)p_feed andDevice:(AVCaptureDevice *)p_device {
	if (self = [super init]) {
		NSError *error;
		feed = p_feed;
		width[0] = 0;
		height[0] = 0;
		width[1] = 0;
		height[1] = 0;

#ifdef IOS_ENABLED
		if ([p_device lockForConfiguration:&error]) {
			if ([p_device isFocusModeSupported:AVCaptureFocusModeContinuousAutoFocus]) {
				[p_device setFocusMode:AVCaptureFocusModeContinuousAutoFocus];
			}
			if ([p_device isExposureModeSupported:AVCaptureExposureModeContinuousAutoExposure]) {
				[p_device setExposureMode:AVCaptureExposureModeContinuousAutoExposure];
			}
			if ([p_device isWhiteBalanceModeSupported:AVCaptureWhiteBalanceModeContinuousAutoWhiteBalance]) {
				[p_device setWhiteBalanceMode:AVCaptureWhiteBalanceModeContinuousAutoWhiteBalance];
			}

			[p_device unlockForConfiguration];
		}
#endif // IOS_ENABLED

		[self beginConfiguration];

#ifdef IOS_ENABLED
		self.sessionPreset = AVCaptureSessionPreset1280x720;
#endif // IOS_ENABLED

		input = [[AVCaptureDeviceInput alloc] initWithDevice:p_device error:&error];

		if (!input) {
			print_line("Couldn't get input device for camera");
			[self commitConfiguration];
			return nil;
		}
		if (![self canAddInput:input]) {
			print_line("Couldn't add input to capture session");
			input = nullptr;
			[self commitConfiguration];
			return nil;
		}
		[self addInput:input];

		output = [AVCaptureVideoDataOutput new];
		if (!output) {
			print_line("Couldn't get output device for camera");
			[self removeInput:input];
			input = nullptr;
			[self commitConfiguration];
			return nil;
		}
		if (![self canAddOutput:output]) {
			print_line("Couldn't add output to capture session");
			[self removeInput:input];
			input = nullptr;
			output = nullptr;
			[self commitConfiguration];
			return nil;
		}

		NSDictionary *settings = @{ (NSString *)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) };
		output.videoSettings = settings;

		// discard if the data output queue is blocked (as we process the still image)
		[output setAlwaysDiscardsLateVideoFrames:YES];

		// now set ourselves as the delegate to receive new frames.
		[output setSampleBufferDelegate:self queue:dispatch_get_main_queue()];

		// this takes ownership
		[self addOutput:output];

		[self commitConfiguration];

		// kick off our session..
		[self startRunning];
	};
	return self;
}

- (void)cleanup {
	// stop running
	[self stopRunning];

	// cleanup
	[self beginConfiguration];

	// remove input
	if (input) {
		[self removeInput:input];
		// don't release this
		input = nullptr;
	}

	// free up our output
	if (output) {
		[self removeOutput:output];
		[output setSampleBufferDelegate:nil queue:nullptr];
		output = nullptr;
	}

	[self commitConfiguration];
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
	// This gets called every time our camera has a new image for us to process.
	// May need to investigate in a way to throttle this if we get more images then we're rendering frames..

	// For now, version 1, we're just doing the bare minimum to make this work...
	CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
	if (pixelBuffer == nullptr) {
		return;
	}

	// It says that we need to lock this on the documentation pages but it's not in the samples
	// need to lock our base address so we can access our pixel buffers, better safe then sorry?
	CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

	// Check if we have the expected number of planes (Y and CbCr).
	size_t planeCount = CVPixelBufferGetPlaneCount(pixelBuffer);
	if (planeCount < 2) {
		static bool plane_count_error_logged = false;
		if (!plane_count_error_logged) {
			ERR_PRINT("Unexpected plane count in pixel buffer (expected 2, got " + itos(planeCount) + ")");
			plane_count_error_logged = true;
		}
		CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
		return;
	}

	// get our buffers
	unsigned char *dataY = (unsigned char *)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
	unsigned char *dataCbCr = (unsigned char *)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1);
	if (dataY == nullptr || dataCbCr == nullptr) {
		static bool buffer_access_error_logged = false;
		if (!buffer_access_error_logged) {
			ERR_PRINT("Couldn't access pixel buffer plane data");
			buffer_access_error_logged = true;
		}
		CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
		return;
	}

	Ref<Image> img[2];

	{
		// do Y
		size_t new_width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0);
		size_t new_height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0);
		size_t row_stride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);

		if ((width[0] != new_width) || (height[0] != new_height)) {
			width[0] = new_width;
			height[0] = new_height;
			img_data[0].resize(new_width * new_height);
		}

		uint8_t *w = img_data[0].ptrw();
		if (new_width == row_stride) {
			memcpy(w, dataY, new_width * new_height);
		} else {
			for (size_t i = 0; i < new_height; i++) {
				memcpy(w, dataY, new_width);
				w += new_width;
				dataY += row_stride;
			}
		}

		img[0].instantiate();
		img[0]->set_data(new_width, new_height, 0, Image::FORMAT_R8, img_data[0]);
	}

	{
		// do CbCr
		size_t new_width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 1);
		size_t new_height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 1);
		size_t row_stride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1);

		if ((width[1] != new_width) || (height[1] != new_height)) {
			width[1] = new_width;
			height[1] = new_height;
			img_data[1].resize(2 * new_width * new_height);
		}

		uint8_t *w = img_data[1].ptrw();
		if (new_width * 2 == row_stride) {
			memcpy(w, dataCbCr, 2 * new_width * new_height);
		} else {
			for (size_t i = 0; i < new_height; i++) {
				memcpy(w, dataCbCr, new_width * 2);
				w += new_width * 2;
				dataCbCr += row_stride;
			}
		}

		///TODO OpenGL doesn't support FORMAT_RG8, need to do some form of conversion
		img[1].instantiate();
		img[1]->set_data(new_width, new_height, 0, Image::FORMAT_RG8, img_data[1]);
	}

	// set our texture...
	feed->set_ycbcr_images(img[0], img[1]);

	// and unlock
	CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
}

@end

//////////////////////////////////////////////////////////////////////////
// CameraFeedApple - Subclass for camera feeds in macOS

class CameraFeedApple : public CameraFeed {
	GDSOFTCLASS(CameraFeedApple, CameraFeed);

private:
	AVCaptureDevice *device;
	MyCaptureSession *capture_session;
	bool device_locked;
	bool was_active_before_pause = false;

public:
	AVCaptureDevice *get_device() const;

	CameraFeedApple();
	~CameraFeedApple();

	void set_device(AVCaptureDevice *p_device);

	void handle_rotation_change(int p_orientation);
	void handle_pause();
	void handle_resume();

	bool activate_feed() override;
	void deactivate_feed() override;
};

AVCaptureDevice *CameraFeedApple::get_device() const {
	return device;
}

CameraFeedApple::CameraFeedApple() {
	device = nullptr;
	capture_session = nullptr;
	device_locked = false;
	transform = Transform2D(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
}

CameraFeedApple::~CameraFeedApple() {
	if (is_active()) {
		deactivate_feed();
	}
}

void CameraFeedApple::set_device(AVCaptureDevice *p_device) {
	device = p_device;

	// get some info
	NSString *device_name = p_device.localizedName;
	name = String::utf8(device_name.UTF8String);
	position = CameraFeed::FEED_UNSPECIFIED;
	if ([p_device position] == AVCaptureDevicePositionBack) {
		position = CameraFeed::FEED_BACK;
	} else if ([p_device position] == AVCaptureDevicePositionFront) {
		position = CameraFeed::FEED_FRONT;
	};
}

void CameraFeedApple::handle_rotation_change(int p_orientation) {
	// UIInterfaceOrientation values:
	// 1 = UIInterfaceOrientationPortrait
	// 2 = UIInterfaceOrientationPortraitUpsideDown
	// 3 = UIInterfaceOrientationLandscapeLeft
	// 4 = UIInterfaceOrientationLandscapeRight
	int display_rotation = 0;
	switch (p_orientation) {
		case 1:
			display_rotation = 0;
			break;
		case 2:
			display_rotation = 180;
			break;
		case 3:
			display_rotation = 270;
			break;
		case 4:
			display_rotation = 90;
			break;
		default:
			display_rotation = 0;
			break;
	}

	// iOS camera sensor orientation is 90 degrees.
	int sensor_orientation = 90;
	int sign = position == CameraFeed::FEED_FRONT ? 1 : -1;
	int image_rotation = (sensor_orientation - display_rotation * sign + 360) % 360;

	transform = Transform2D();
	transform = transform.rotated(Math::deg_to_rad((float)image_rotation));
}

void CameraFeedApple::handle_pause() {
	if (capture_session) {
		was_active_before_pause = true;
		deactivate_feed();
	} else {
		was_active_before_pause = false;
	}
}

void CameraFeedApple::handle_resume() {
	if (was_active_before_pause) {
		activate_feed();
		was_active_before_pause = false;
	}
}

bool CameraFeedApple::activate_feed() {
	if (capture_session) {
		// Already recording.
		return true;
	}

	// Start camera capture, check permission.
	if (@available(macOS 10.14, iOS 14.0, visionOS 1.0, *)) {
		AVAuthorizationStatus status = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
		if (status == AVAuthorizationStatusAuthorized) {
			capture_session = [[MyCaptureSession alloc] initForFeed:this andDevice:device];
			return capture_session != nullptr;
		} else if (status == AVAuthorizationStatusNotDetermined) {
			// Request permission asynchronously.
			[AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo
									 completionHandler:^(BOOL granted) {
										 if (granted) {
											 capture_session = [[MyCaptureSession alloc] initForFeed:this andDevice:device];
										 }
									 }];
			return false;
		} else if (status == AVAuthorizationStatusDenied) {
			print_line("Camera permission denied by user.");
			return false;
		} else if (status == AVAuthorizationStatusRestricted) {
			print_line("Camera access restricted.");
			return false;
		}
		return false;
	} else {
		capture_session = [[MyCaptureSession alloc] initForFeed:this andDevice:device];
		return capture_session != nullptr;
	}
}

void CameraFeedApple::deactivate_feed() {
	// end camera capture if we have one
	if (capture_session) {
		[capture_session cleanup];
		capture_session = nullptr;
	};
	if (device_locked) {
		[device unlockForConfiguration];
		device_locked = false;
	}
}

//////////////////////////////////////////////////////////////////////////
// MyDeviceNotifications - This is a little helper class gets notifications
// when devices are connected/disconnected

@interface MyDeviceNotifications : NSObject {
	CameraApple *camera_server;
}

@end

@implementation MyDeviceNotifications

- (void)devices_changed:(NSNotification *)notification {
	camera_server->update_feeds();
}

- (id)initForServer:(CameraApple *)p_server {
	if (self = [super init]) {
		camera_server = p_server;

		[[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(devices_changed:) name:AVCaptureDeviceWasConnectedNotification object:nil];
		[[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(devices_changed:) name:AVCaptureDeviceWasDisconnectedNotification object:nil];
	};
	return self;
}

- (void)dealloc {
	// remove notifications
	[[NSNotificationCenter defaultCenter] removeObserver:self name:AVCaptureDeviceWasConnectedNotification object:nil];
	[[NSNotificationCenter defaultCenter] removeObserver:self name:AVCaptureDeviceWasDisconnectedNotification object:nil];
}

@end

MyDeviceNotifications *device_notifications = nil;

//////////////////////////////////////////////////////////////////////////
// CameraApple - Subclass for our camera server on macOS

void CameraApple::update_feeds() {
	NSArray<AVCaptureDevice *> *devices = nullptr;
#ifdef APPLE_EMBEDDED_ENABLED
	{
		NSMutableArray *deviceTypes = [NSMutableArray array];
		if (@available(iOS 14.0, visionOS 2.1, *)) {
			[deviceTypes addObject:AVCaptureDeviceTypeBuiltInWideAngleCamera];
		}
#ifdef IOS_ENABLED
		[deviceTypes addObject:AVCaptureDeviceTypeBuiltInTelephotoCamera];
		[deviceTypes addObject:AVCaptureDeviceTypeBuiltInDualCamera];
		[deviceTypes addObject:AVCaptureDeviceTypeBuiltInTrueDepthCamera];
		[deviceTypes addObject:AVCaptureDeviceTypeBuiltInUltraWideCamera];
		[deviceTypes addObject:AVCaptureDeviceTypeBuiltInDualWideCamera];
		[deviceTypes addObject:AVCaptureDeviceTypeBuiltInTripleCamera];
#endif // IOS_ENABLED
		AVCaptureDeviceDiscoverySession *session = [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:deviceTypes mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionUnspecified];
		devices = session.devices;
	}
#else // APPLE_EMBEDDED_ENABLED
#if defined(__x86_64__)
	if (@available(macOS 10.15, *)) {
#endif // __x86_64__
		AVCaptureDeviceDiscoverySession *session;
		if (@available(macOS 14.0, *)) {
			session = [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:[NSArray arrayWithObjects:AVCaptureDeviceTypeExternal, AVCaptureDeviceTypeBuiltInWideAngleCamera, AVCaptureDeviceTypeContinuityCamera, nil] mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionUnspecified];
		} else {
			session = [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:[NSArray arrayWithObjects:AVCaptureDeviceTypeExternalUnknown, AVCaptureDeviceTypeBuiltInWideAngleCamera, nil] mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionUnspecified];
		}
		devices = session.devices;
#if defined(__x86_64__)
	} else {
		devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
	}
#endif // __x86_64__
#endif // APPLE_EMBEDDED_ENABLED

	// Deactivate feeds that are gone before removing them.
	for (int i = feeds.size() - 1; i >= 0; i--) {
		Ref<CameraFeedApple> feed = (Ref<CameraFeedApple>)feeds[i];
		if (feed.is_null()) {
			continue;
		}

		if (![devices containsObject:feed->get_device()]) {
			if (feed->is_active()) {
				feed->deactivate_feed();
			}
			remove_feed(feed);
		};
	};

	for (AVCaptureDevice *device in devices) {
		bool found = false;
		for (int i = 0; i < feeds.size() && !found; i++) {
			Ref<CameraFeedApple> feed = (Ref<CameraFeedApple>)feeds[i];
			if (feed.is_null()) {
				continue;
			}
			if (feed->get_device() == device) {
				found = true;
			};
		};

		if (!found) {
			Ref<CameraFeedApple> newfeed;
			newfeed.instantiate();
			newfeed->set_device(device);

			add_feed(newfeed);
		};
	};

#ifdef IOS_ENABLED
	// Update rotation for all feeds.
	UIInterfaceOrientation orientation = UIInterfaceOrientationUnknown;
	UIWindow *window = [UIApplication sharedApplication].delegate.window;
	UIWindowScene *windowScene = window.windowScene;
	if (windowScene) {
		orientation = windowScene.interfaceOrientation;
	}
	handle_display_rotation_change((int)orientation);
#endif // IOS_ENABLED

	emit_signal(SNAME(CameraServer::feeds_updated_signal_name));
}

void CameraApple::set_monitoring_feeds(bool p_monitoring_feeds) {
	if (p_monitoring_feeds == monitoring_feeds) {
		return;
	}

	CameraServer::set_monitoring_feeds(p_monitoring_feeds);
	if (p_monitoring_feeds) {
		// Find available cameras we have at this time.
		update_feeds();

		// Get notified on feed changes.
		device_notifications = [[MyDeviceNotifications alloc] initForServer:this];
	} else {
		// Stop monitoring feed changes.
		device_notifications = nil;
	}
}

void CameraApple::handle_display_rotation_change(int p_orientation) {
	for (int i = 0; i < feeds.size(); i++) {
		Ref<CameraFeedApple> feed = (Ref<CameraFeedApple>)feeds[i];
		if (feed.is_valid()) {
			feed->handle_rotation_change(p_orientation);
		}
	}
}

void CameraApple::handle_application_pause() {
	for (int i = 0; i < feeds.size(); i++) {
		Ref<CameraFeedApple> feed = (Ref<CameraFeedApple>)feeds[i];
		if (feed.is_valid()) {
			feed->handle_pause();
		}
	}
}

void CameraApple::handle_application_resume() {
	for (int i = 0; i < feeds.size(); i++) {
		Ref<CameraFeedApple> feed = (Ref<CameraFeedApple>)feeds[i];
		if (feed.is_valid()) {
			feed->handle_resume();
		}
	}
}

#ifdef APPLE_EMBEDDED_ENABLED

void register_camera_external_module() {
	CameraServer::make_default<CameraApple>();
}

void unregister_camera_external_module() {
}

#endif // APPLE_EMBEDDED_ENABLED
