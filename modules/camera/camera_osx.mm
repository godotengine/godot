/*************************************************************************/
/*  camera_osx.mm                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

///@TODO this is a near duplicate of CameraIOS, we should find a way to combine those to minimize code duplication!!!!
// If you fix something here, make sure you fix it there as wel!

#include "camera_osx.h"
#include "servers/camera/camera_feed.h"
#import <AVFoundation/AVFoundation.h>

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
			NSDictionary *settings = @{ (NSString *)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) };
			output.videoSettings = settings;

			// discard if the data output queue is blocked (as we process the still image)
			[output setAlwaysDiscardsLateVideoFrames:YES];

			// now set ourselves as the delegate to receive new frames.
			[output setSampleBufferDelegate:self queue:dispatch_get_main_queue()];

			// this takes ownership
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
		[output release];
		output = nullptr;
	}

	[self commitConfiguration];
}

- (void)dealloc {
	// bye bye
	[super dealloc];
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
	// This gets called every time our camera has a new image for us to process.
	// May need to investigate in a way to throttle this if we get more images then we're rendering frames..

	// For now, version 1, we're just doing the bare minimum to make this work...
	CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
	// int _width = CVPixelBufferGetWidth(pixelBuffer);
	// int _height = CVPixelBufferGetHeight(pixelBuffer);

	// It says that we need to lock this on the documentation pages but it's not in the samples
	// need to lock our base address so we can access our pixel buffers, better safe then sorry?
	CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

	// get our buffers
	unsigned char *dataY = (unsigned char *)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
	unsigned char *dataCbCr = (unsigned char *)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1);
	if (dataY == nullptr) {
		print_line("Couldn't access Y pixel buffer data");
	} else if (dataCbCr == nullptr) {
		print_line("Couldn't access CbCr pixel buffer data");
	} else {
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
	}

	// and unlock
	CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
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

	CameraFeedOSX();
	~CameraFeedOSX();

	void set_device(AVCaptureDevice *p_device);

	bool activate_feed();
	void deactivate_feed();
};

AVCaptureDevice *CameraFeedOSX::get_device() const {
	return device;
};

CameraFeedOSX::CameraFeedOSX() {
	device = nullptr;
	capture_session = nullptr;
};

void CameraFeedOSX::set_device(AVCaptureDevice *p_device) {
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
	if (capture_session != nullptr) {
		[capture_session release];
		capture_session = nullptr;
	};

	if (device != nullptr) {
		[device release];
		device = nullptr;
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
		[capture_session cleanup];
		[capture_session release];
		capture_session = nullptr;
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

	[super dealloc];
}

@end

MyDeviceNotifications *device_notifications = nil;

//////////////////////////////////////////////////////////////////////////
// CameraOSX - Subclass for our camera server on OSX

void CameraOSX::update_feeds() {
#if MAC_OS_X_VERSION_MIN_REQUIRED >= 101500
	AVCaptureDeviceDiscoverySession *session = [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:[NSArray arrayWithObjects:AVCaptureDeviceTypeExternalUnknown, AVCaptureDeviceTypeBuiltInWideAngleCamera, nil] mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionUnspecified];
	NSArray *devices = session.devices;
#else
	NSArray *devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
#endif

	// remove devices that are gone..
	for (int i = feeds.size() - 1; i >= 0; i--) {
		Ref<CameraFeedOSX> feed = (Ref<CameraFeedOSX>)feeds[i];

		if (![devices containsObject:feed->get_device()]) {
			// remove it from our array, this will also destroy it ;)
			remove_feed(feed);
		};
	};

	for (AVCaptureDevice *device in devices) {
		bool found = false;
		for (int i = 0; i < feeds.size() && !found; i++) {
			Ref<CameraFeedOSX> feed = (Ref<CameraFeedOSX>)feeds[i];
			if (feed->get_device() == device) {
				found = true;
			};
		};

		if (!found) {
			Ref<CameraFeedOSX> newfeed;
			newfeed.instance();
			newfeed->set_device(device);

			// assume display camera so inverse
			Transform2D transform = Transform2D(-1.0, 0.0, 0.0, -1.0, 1.0, 1.0);
			newfeed->set_transform(transform);

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
