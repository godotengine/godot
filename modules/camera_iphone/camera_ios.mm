/*************************************************************************/
/*  camera_ios.mm                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

///@TODO this is a near duplicate of CameraOSX, we should find a way to combine those to minimize code duplication!!!!
// If you fix something here, make sure you fix it there as wel!

#include "camera_ios.h"
#include "servers/camera/camera_feed.h"

#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>

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

		// prepare our device
		[p_device lockForConfiguration:&error];

		[p_device setFocusMode:AVCaptureFocusModeLocked];
		[p_device setExposureMode:AVCaptureExposureModeLocked];
		[p_device setWhiteBalanceMode:AVCaptureWhiteBalanceModeLocked];

		[p_device unlockForConfiguration];

		[self beginConfiguration];

		// setup our capture
		self.sessionPreset = AVCaptureSessionPreset1280x720;

		input = [AVCaptureDeviceInput deviceInputWithDevice:p_device error:&error];
		if (!input) {
			print_line("Couldn't get input device for camera");
		} else {
			[self addInput:input];
		}

		output = [AVCaptureVideoDataOutput new];
		if (!output) {
			print_line("Couldn't get output device for camera");
		} else {
			NSDictionary *settings = @{ (NSString *)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) };
			output.videoSettings = settings;

			// discard if the data output queue is blocked (as we process the still image)
			[output setAlwaysDiscardsLateVideoFrames:YES];

			// now set ourselves as the delegate to receive new frames. Note that we're doing this on the main thread at the moment, we may need to change this..
			[output setSampleBufferDelegate:self queue:dispatch_get_main_queue()];

			[self addOutput:output];
		}

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

	if (input) {
		[self removeInput:input];
		// don't release this
		input = nil;
	}

	if (output) {
		[self removeOutput:output];
		[output setSampleBufferDelegate:nil queue:NULL];
		output = nil;
	}

	[self commitConfiguration];
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
	// This gets called every time our camera has a new image for us to process.
	// May need to investigate in a way to throttle this if we get more images then we're rendering frames..

	// For now, version 1, we're just doing the bare minimum to make this work...

	CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
	// int width = CVPixelBufferGetWidth(pixelBuffer);
	// int height = CVPixelBufferGetHeight(pixelBuffer);

	// It says that we need to lock this on the documentation pages but it's not in the samples
	// need to lock our base address so we can access our pixel buffers, better safe then sorry?
	CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

	// get our buffers
	unsigned char *dataY = (unsigned char *)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
	unsigned char *dataCbCr = (unsigned char *)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1);
	if (dataY == NULL) {
		print_line("Couldn't access Y pixel buffer data");
	} else if (dataCbCr == NULL) {
		print_line("Couldn't access CbCr pixel buffer data");
	} else {
		UIInterfaceOrientation orientation = UIInterfaceOrientationUnknown;

		if (@available(iOS 13, *)) {
			orientation = [UIApplication sharedApplication].delegate.window.windowScene.interfaceOrientation;
#if !defined(TARGET_OS_SIMULATOR) || !TARGET_OS_SIMULATOR
		} else {
			orientation = [[UIApplication sharedApplication] statusBarOrientation];
#endif
		}

		Ref<Image> img[2];

		{
			// do Y
			size_t new_width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0);
			size_t new_height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0);

			if ((width[0] != new_width) || (height[0] != new_height)) {
				width[0] = new_width;
				height[0] = new_height;
				img_data[0].resize(new_width * new_height);
			}

			uint8_t *w = img_data[0].ptrw();
			memcpy(w, dataY, new_width * new_height);

			img[0].instance();
			img[0]->create(new_width, new_height, 0, Image::FORMAT_R8, img_data[0]);
		}

		{
			// do CbCr
			size_t new_width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 1);
			size_t new_height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 1);

			if ((width[1] != new_width) || (height[1] != new_height)) {
				width[1] = new_width;
				height[1] = new_height;
				img_data[1].resize(2 * new_width * new_height);
			}

			uint8_t *w = img_data[1].ptrw();
			memcpy(w, dataCbCr, 2 * new_width * new_height);

			///TODO GLES2 doesn't support FORMAT_RG8, need to do some form of conversion
			img[1].instance();
			img[1]->create(new_width, new_height, 0, Image::FORMAT_RG8, img_data[1]);
		}

		// set our texture...
		feed->set_YCbCr_imgs(img[0], img[1]);

		// update our matrix to match the orientation, note, before changing anything
		// here, be aware that the project orientation settings must match your xcode
		// settings or this will go wrong!
		Transform2D display_transform;
		switch (orientation) {
			case UIInterfaceOrientationPortrait: {
				display_transform = Transform2D(0.0, -1.0, -1.0, 0.0, 1.0, 1.0);
			} break;
			case UIInterfaceOrientationLandscapeRight: {
				display_transform = Transform2D(1.0, 0.0, 0.0, -1.0, 0.0, 1.0);
			} break;
			case UIInterfaceOrientationLandscapeLeft: {
				display_transform = Transform2D(-1.0, 0.0, 0.0, 1.0, 1.0, 0.0);
			} break;
			default: {
				display_transform = Transform2D(0.0, 1.0, 1.0, 0.0, 0.0, 0.0);
			} break;
		}

		//TODO: this is correct for the camera on the back, I have a feeling this needs to be inversed for the camera on the front!
		feed->set_transform(display_transform);
	}

	// and unlock
	CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
}

@end

//////////////////////////////////////////////////////////////////////////
// CameraFeedIOS - Subclass for camera feeds in iOS

class CameraFeedIOS : public CameraFeed {
private:
	AVCaptureDevice *device;
	MyCaptureSession *capture_session;

public:
	bool get_is_arkit() const;
	AVCaptureDevice *get_device() const;

	CameraFeedIOS();
	~CameraFeedIOS();

	void set_device(AVCaptureDevice *p_device);

	bool activate_feed();
	void deactivate_feed();
};

AVCaptureDevice *CameraFeedIOS::get_device() const {
	return device;
};

CameraFeedIOS::CameraFeedIOS() {
	capture_session = NULL;
	device = NULL;
	transform = Transform2D(1.0, 0.0, 0.0, 1.0, 0.0, 0.0); /* should re-orientate this based on device orientation */
};

void CameraFeedIOS::set_device(AVCaptureDevice *p_device) {
	device = p_device;

	// get some info
	NSString *device_name = p_device.localizedName;
	name = device_name.UTF8String;
	position = CameraFeed::FEED_UNSPECIFIED;
	if ([p_device position] == AVCaptureDevicePositionBack) {
		position = CameraFeed::FEED_BACK;
	} else if ([p_device position] == AVCaptureDevicePositionFront) {
		position = CameraFeed::FEED_FRONT;
	};
};

CameraFeedIOS::~CameraFeedIOS() {
	if (capture_session) {
		capture_session = nil;
	};

	if (device) {
		device = nil;
	};
};

bool CameraFeedIOS::activate_feed() {
	if (capture_session) {
		// already recording!
	} else {
		// start camera capture
		capture_session = [[MyCaptureSession alloc] initForFeed:this andDevice:device];
	};

	return true;
};

void CameraFeedIOS::deactivate_feed() {
	// end camera capture if we have one
	if (capture_session) {
		[capture_session cleanup];
		capture_session = nil;
	};
};

//////////////////////////////////////////////////////////////////////////
// MyDeviceNotifications - This is a little helper class gets notifications
// when devices are connected/disconnected

@interface MyDeviceNotifications : NSObject {
	CameraIOS *camera_server;
}

@end

@implementation MyDeviceNotifications

- (void)devices_changed:(NSNotification *)notification {
	camera_server->update_feeds();
}

- (id)initForServer:(CameraIOS *)p_server {
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
// CameraIOS - Subclass for our camera server on iPhone

void CameraIOS::update_feeds() {
	// this way of doing things is deprecated but still works,
	// rewrite to using AVCaptureDeviceDiscoverySession

	NSMutableArray *deviceTypes = [NSMutableArray array];

	if (@available(iOS 10, *)) {
		[deviceTypes addObject:AVCaptureDeviceTypeBuiltInWideAngleCamera];
		[deviceTypes addObject:AVCaptureDeviceTypeBuiltInTelephotoCamera];

		if (@available(iOS 10.2, *)) {
			[deviceTypes addObject:AVCaptureDeviceTypeBuiltInDualCamera];
		}

		if (@available(iOS 11.1, *)) {
			[deviceTypes addObject:AVCaptureDeviceTypeBuiltInTrueDepthCamera];
		}

		AVCaptureDeviceDiscoverySession *session = [AVCaptureDeviceDiscoverySession
				discoverySessionWithDeviceTypes:deviceTypes
									  mediaType:AVMediaTypeVideo
									   position:AVCaptureDevicePositionUnspecified];

		// remove devices that are gone..
		for (int i = feeds.size() - 1; i >= 0; i--) {
			Ref<CameraFeedIOS> feed(feeds[i]);

			if (feed.is_null()) {
				// feed not managed by us
			} else if (![session.devices containsObject:feed->get_device()]) {
				// remove it from our array, this will also destroy it ;)
				remove_feed(feed);
			};
		};

		// add new devices..
		for (AVCaptureDevice *device in session.devices) {
			bool found = false;

			for (int i = 0; i < feeds.size() && !found; i++) {
				Ref<CameraFeedIOS> feed(feeds[i]);

				if (feed.is_null()) {
					// feed not managed by us
				} else if (feed->get_device() == device) {
					found = true;
				};
			};

			if (!found) {
				Ref<CameraFeedIOS> newfeed;
				newfeed.instance();
				newfeed->set_device(device);
				add_feed(newfeed);
			};
		};
	}
};

CameraIOS::CameraIOS() {
	// check if we have our usage description
	NSString *usage_desc = [[NSBundle mainBundle] objectForInfoDictionaryKey:@"NSCameraUsageDescription"];
	if (usage_desc == NULL) {
		// don't initialise if we don't get anything
		print_line("No NSCameraUsageDescription key in pList, no access to cameras.");
		return;
	} else if (usage_desc.length == 0) {
		// don't initialise if we don't get anything
		print_line("Empty NSCameraUsageDescription key in pList, no access to cameras.");
		return;
	}

	// now we'll request access.
	// If this is the first time the user will be prompted with the string (iOS will read it).
	// Once a decision is made it is returned. If the user wants to change it later on they
	// need to go into setting.
	print_line("Requesting Camera permissions");

	[AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo
							 completionHandler:^(BOOL granted) {
								 if (granted) {
									 print_line("Access to cameras granted!");

									 // Find available cameras we have at this time
									 update_feeds();

									 // should only have one of these....
									 device_notifications = [[MyDeviceNotifications alloc] initForServer:this];
								 } else {
									 print_line("No access to cameras!");
								 }
							 }];
};

CameraIOS::~CameraIOS() {
	device_notifications = nil;
};
