/*************************************************************************/
/*  camera_osx.mm                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

///@TODO this is a near duplicate of CameraIOS, we should find a way to combine those to minimise code duplication!!!!
// If you fix something here, make sure you fix it there as wel!

#include "camera_osx.h"
#import <AVFoundation/AVFoundation.h>

//////////////////////////////////////////////////////////////////////////
// MyCaptureSession - This is a little helper class so we can capture our frames

@interface MyCaptureSession : AVCaptureSession <AVCaptureVideoDataOutputSampleBufferDelegate> {
	CameraFeed *feed;
	size_t width;
	size_t height;

	AVCaptureDeviceInput *input;
	AVCaptureVideoDataOutput *output;
}

@end

@implementation MyCaptureSession

- (id)initForFeed:(CameraFeed *)p_feed andDevice:(AVCaptureDevice *)p_device {
	if (self = [super init]) {
		NSError *error;
		feed = p_feed;

		[self beginConfiguration];

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
			NSDictionary *settings = @{(NSString *)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) };
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

- (void)dealloc {
	// stop running
	[self stopRunning];

	if (output) {
		[self removeOutput:output];
		[output release];
		output = nil;
	}

	if (input) {
		[self removeInput:input];
		[input release];
		input = nil;
	}
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
	// This gets called every time our camera has a new image for us to process. We need to give this to our camera server
	// That in turn locks our texture which may adversely effect our performance. Apple uses a texture cache here but I'm not sure if we can use that with Godot
	// Otherwise this needs to be enhanced by implementing something similar ourselves, load the camera image into a new texture and then give the new texture
	// to Godot.

	if (feed->is_waiting()) {
		// For now, version 1, we're just doing the bare minimum to make this work...
		CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
		width = CVPixelBufferGetWidth(pixelBuffer);
		height = CVPixelBufferGetHeight(pixelBuffer);

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
			// set our texture...
			feed->set_texture_data_YCbCr(
					dataY, CVPixelBufferGetWidthOfPlane(pixelBuffer, 0), CVPixelBufferGetHeightOfPlane(pixelBuffer, 0), dataCbCr, CVPixelBufferGetWidthOfPlane(pixelBuffer, 1), CVPixelBufferGetHeightOfPlane(pixelBuffer, 1));
		}

		// and unlock
		CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
	}
}

@end

//////////////////////////////////////////////////////////////////////////
// CameraFeedOSX - Subclass for camera feeds in OSX

class CameraFeedOSX : public CameraFeed {
private:
	AVCaptureDevice *device;
	MyCaptureSession *capture_session;

public:
	AVCaptureDevice *get_device() const;

	CameraFeedOSX(AVCaptureDevice *p_device);
	~CameraFeedOSX();

	bool activate_feed();
	void deactivate_feed();
};

AVCaptureDevice *CameraFeedOSX::get_device() const {
	return device;
};

CameraFeedOSX::CameraFeedOSX(AVCaptureDevice *p_device) {
	capture_session = NULL;
	device = p_device;
	[device retain];

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

CameraFeedOSX::~CameraFeedOSX() {
	if (capture_session != NULL) {
		[capture_session release];
		capture_session = NULL;
	};

	if (device != NULL) {
		[device release];
		device = NULL;
	};
};

bool CameraFeedOSX::activate_feed() {
	if (capture_session) {
		// already recording!
	} else {
		// start camera capture
		capture_session = [[MyCaptureSession alloc] initForFeed:this andDevice:device];
	};

	return true;
};

void CameraFeedOSX::deactivate_feed() {
	// end camera capture if we have one
	if (capture_session) {
		[capture_session release];
		capture_session = NULL;
	};
};

//////////////////////////////////////////////////////////////////////////
// MyDeviceNotifications - This is a little helper class gets notifications
// when devices are connected/disconnected

@interface MyDeviceNotifications : NSObject {
	CameraOSX *camera_server;
}

@end

@implementation MyDeviceNotifications

- (void)devices_changed:(NSNotification *)notification {
	camera_server->update_feeds();
}

- (id)initForServer:(CameraOSX *)p_server {
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
// CameraOSX - Subclass for our camera server on OSX

void CameraOSX::update_feeds() {
	NSArray *devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];

	// remove devices that are gone..
	for (int i = feeds.size() - 1; i >= 0; i--) {
		CameraFeedOSX *feed = (CameraFeedOSX *)feeds[i];

		if (![devices containsObject:feed->get_device()]) {
			// remove it from our array, this will also destroy it ;)
			remove_feed(feed->get_id());
		};
	};

	// add new devices..
	for (AVCaptureDevice *device in devices) {
		bool found = false;
		for (int i = 0; i < feeds.size() && !found; i++) {
			CameraFeedOSX *feed = (CameraFeedOSX *)feeds[i];
			if (feed->get_device() == device) {
				found = true;
			};
		};

		if (!found) {
			CameraFeedOSX *newfeed = new CameraFeedOSX(device);
			add_feed(newfeed);
		};
	};
};

CameraOSX::CameraOSX() {
	// Find available cameras we have at this time
	update_feeds();

	// should only have one of these....
	device_notifications = [[MyDeviceNotifications alloc] initForServer:this];
};

CameraOSX::~CameraOSX() {
	[device_notifications release];
};
