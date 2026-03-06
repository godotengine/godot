/**************************************************************************/
/*  camera_macos.mm                                                       */
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

#import "camera_macos.h"

#include "servers/camera/camera_feed.h"

#import <AVFoundation/AVFoundation.h>

//////////////////////////////////////////////////////////////////////////
// MyCaptureSession - This is a little helper class so we can capture our frames

@interface MyCaptureSession : AVCaptureSession <AVCaptureVideoDataOutputSampleBufferDelegate> {
	Ref<CameraFeed> feed;
	size_t width[2];
	size_t height[2];
	size_t channel[2];
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
		channel[0] = 0;
		width[1] = 0;
		height[1] = 0;
		channel[1] = 0;

#ifdef APPLE_EMBEDDED_ENABLED
		[p_device lockForConfiguration:&error];

		[p_device setFocusMode:AVCaptureFocusModeLocked];
		[p_device setExposureMode:AVCaptureExposureModeLocked];
		[p_device setWhiteBalanceMode:AVCaptureWhiteBalanceModeLocked];

		[p_device unlockForConfiguration];
#endif // APPLE_EMBEDDED_ENABLED

		[self beginConfiguration];

#ifdef APPLE_EMBEDDED_ENABLED
		self.sessionPreset = AVCaptureSessionPreset1280x720;
#endif // APPLE_EMBEDDED_ENABLED

		input = [AVCaptureDeviceInput deviceInputWithDevice:p_device error:&error];
		if (!input) {
			print_line("Couldn't get input device for camera");
			[self commitConfiguration];
			return nil;
		}
		[self addInput:input];

		output = [AVCaptureVideoDataOutput new];
		if (!output) {
			print_line("Couldn't get output device for camera");
			[self commitConfiguration];
			return nil;
		}

		// set videoSettings to empty dictionary to receive device native formats.
		output.videoSettings = @{};

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

	CMFormatDescriptionRef format = CMSampleBufferGetFormatDescription(sampleBuffer);
	FourCharCode fourcc = CMFormatDescriptionGetMediaSubType(format);

	// It says that we need to lock this on the documentation pages but it's not in the samples
	// need to lock our base address so we can access our pixel buffers, better safe then sorry?
	CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

	Ref<Image> img[2];

	if (fourcc == kCVPixelFormatType_24RGB) {
		img[0] = [self createImage:pixelBuffer plane:0 channels:3];
		if (img[0].is_valid()) {
			feed->set_rgb_image(img[0]);
		}
	} else if (fourcc == kCVPixelFormatType_32RGBA) {
		img[0] = [self createImage:pixelBuffer plane:0 channels:4];
		if (img[0].is_valid()) {
			feed->set_rgb_image(img[0]);
		}
	} else if (fourcc == kCVPixelFormatType_422YpCbCr8_yuvs) {
		img[0] = [self createImage:pixelBuffer plane:0 channels:2];
		if (img[0].is_valid()) {
			feed->set_ycbcr_image(img[0]);
		}
	} else if (fourcc == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange || fourcc == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) {
		img[0] = [self createImage:pixelBuffer plane:0 channels:1];
		img[1] = [self createImage:pixelBuffer plane:1 channels:2];
		if (img[0].is_valid() && img[1].is_valid()) {
			feed->set_ycbcr_images(img[0], img[1]);
		}
	} else {
		ERR_PRINT_ONCE(vformat("Unexpected pixel format: %x", fourcc));
	}

	// and unlock
	CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
}

- (Ref<Image>)createImage:(CVImageBufferRef)pixelBuffer plane:(uint32_t)plane channels:(uint32_t)channels {
	uint8_t *data;
	size_t new_width;
	size_t new_height;
	size_t row_stride;
	Image::Format image_format;
	Ref<Image> img;

	if (channels == 1) {
		image_format = Image::FORMAT_R8;
	} else if (channels == 2) {
		image_format = Image::FORMAT_RG8;
	} else if (channels == 3) {
		image_format = Image::FORMAT_RGB8;
	} else if (channels == 4) {
		image_format = Image::FORMAT_RGBA8;
	} else {
		ERR_PRINT_ONCE("Unexpected channel count.");
		return img;
	}

	size_t planeCount = CVPixelBufferGetPlaneCount(pixelBuffer);
	if (planeCount > 0) { // planar format
		ERR_FAIL_INDEX_V(plane, planeCount, img);
		data = (uint8_t *)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, plane);
		new_width = CVPixelBufferGetWidthOfPlane(pixelBuffer, plane);
		new_height = CVPixelBufferGetHeightOfPlane(pixelBuffer, plane);
		row_stride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, plane);
	} else { // packed format
		plane = 0;
		data = (uint8_t *)CVPixelBufferGetBaseAddress(pixelBuffer);
		new_width = CVPixelBufferGetWidth(pixelBuffer);
		new_height = CVPixelBufferGetHeight(pixelBuffer);
		row_stride = CVPixelBufferGetBytesPerRow(pixelBuffer);
	}

	if (data == nullptr) {
		ERR_PRINT_ONCE("Couldn't access pixel buffer data");
		return img;
	}

	if ((width[plane] != new_width) || (height[plane] != new_height) || (channel[plane] != channels)) {
		width[plane] = new_width;
		height[plane] = new_height;
		channel[plane] = channels;
		img_data[plane].resize(channels * new_width * new_height);
	}

	uint8_t *w = img_data[plane].ptrw();
	if (channels * new_width == row_stride) {
		memcpy(w, data, channels * new_width * new_height);
	} else {
		for (size_t i = 0; i < new_height; i++) {
			memcpy(w, data, channels * new_width);
			w += channels * new_width;
			data += row_stride;
		}
	}

	img.instantiate();
	img->set_data(new_width, new_height, false, image_format, img_data[plane]);
	return img;
}

@end

//////////////////////////////////////////////////////////////////////////
// CameraFeedMacOS - Subclass for camera feeds in macOS

class CameraFeedMacOS : public CameraFeed {
	GDSOFTCLASS(CameraFeedMacOS, CameraFeed);

private:
	struct FeedFormat {
		AVCaptureDeviceFormat *format = nullptr;
		AVFrameRateRange *range = nullptr;
	};

	AVCaptureDevice *device;
	MyCaptureSession *capture_session;
	Vector<FeedFormat> feed_formats;
	bool device_locked;

public:
	static String get_format_name(const FourCharCode &p_fourcc);

	AVCaptureDevice *get_device() const;

	CameraFeedMacOS();
	~CameraFeedMacOS();

	void set_device(AVCaptureDevice *p_device);

	bool activate_feed() override;
	void deactivate_feed() override;

	bool set_format(int p_index, const Dictionary &p_parameters) override;
	Array get_formats() const override;
};

AVCaptureDevice *CameraFeedMacOS::get_device() const {
	return device;
}

CameraFeedMacOS::CameraFeedMacOS() {
	device = nullptr;
	capture_session = nullptr;
	device_locked = false;
}

CameraFeedMacOS::~CameraFeedMacOS() {
	if (is_active()) {
		deactivate_feed();
	}
}

void CameraFeedMacOS::set_device(AVCaptureDevice *p_device) {
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
	for (AVCaptureDeviceFormat *format in p_device.formats) {
		CMFormatDescriptionRef formatDescription = format.formatDescription;
		FourCharCode fourcc = CMFormatDescriptionGetMediaSubType(formatDescription);
		switch (fourcc) {
			case kCVPixelFormatType_24RGB:
			case kCVPixelFormatType_32RGBA:
			case kCVPixelFormatType_422YpCbCr8_yuvs:
			case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
			case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
				break;
			default:
				continue;
		}
		for (AVFrameRateRange *range in format.videoSupportedFrameRateRanges) {
			FeedFormat feed_format = {};
			feed_format.format = format;
			feed_format.range = range;
			feed_formats.append(feed_format);
		}
	}
}

bool CameraFeedMacOS::activate_feed() {
	if (capture_session) {
		// Already recording.
		return true;
	}

	// Configure device format if specified.
	if (selected_format != -1) {
		NSError *error;
		if (!device_locked) {
			device_locked = [device lockForConfiguration:&error];
			ERR_FAIL_COND_V_MSG(!device_locked, false, error.localizedFailureReason.UTF8String);
		}
		const FeedFormat &feed_format = feed_formats[selected_format];
		[device setActiveFormat:feed_format.format];
		[device setActiveVideoMinFrameDuration:feed_format.range.minFrameDuration];
		[device setActiveVideoMaxFrameDuration:feed_format.range.maxFrameDuration];
	}

	// Start camera capture, check permission.
	if (@available(macOS 10.14, *)) {
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

void CameraFeedMacOS::deactivate_feed() {
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

bool CameraFeedMacOS::set_format(int p_index, const Dictionary &p_parameters) {
	if (p_index == -1) {
		selected_format = p_index;
		if (capture_session) {
			[capture_session beginConfiguration];
		}
		if (device_locked) {
			[device unlockForConfiguration];
			device_locked = false;
		}
		if (capture_session) {
			[capture_session commitConfiguration];
		}
		return true;
	}
	ERR_FAIL_INDEX_V((unsigned int)p_index, feed_formats.size(), false);
	if (capture_session) {
		if (!device_locked) {
			NSError *error;
			device_locked = [device lockForConfiguration:&error];
			ERR_FAIL_COND_V_MSG(!device_locked, false, error.localizedFailureReason.UTF8String);
		}
		[capture_session beginConfiguration];
		const FeedFormat &feed_format = feed_formats[p_index];
		[device setActiveFormat:feed_format.format];
		[device setActiveVideoMinFrameDuration:feed_format.range.minFrameDuration];
		[device setActiveVideoMaxFrameDuration:feed_format.range.maxFrameDuration];
		[capture_session commitConfiguration];
	}
	selected_format = p_index;
	return true;
}

Array CameraFeedMacOS::get_formats() const {
	Array result;
	for (const FeedFormat &feed_format : feed_formats) {
		Dictionary dictionary;
		CMFormatDescriptionRef formatDescription = feed_format.format.formatDescription;
		CMVideoDimensions dimension = CMVideoFormatDescriptionGetDimensions(formatDescription);
		FourCharCode fourcc = CMFormatDescriptionGetMediaSubType(formatDescription);
		dictionary["width"] = dimension.width;
		dictionary["height"] = dimension.height;
		dictionary["format"] = get_format_name(fourcc);
		dictionary["frame_numerator"] = feed_format.range.minFrameDuration.value;
		dictionary["frame_denominator"] = feed_format.range.minFrameDuration.timescale;
		result.push_back(dictionary);
	}
	return result;
}

String CameraFeedMacOS::get_format_name(const FourCharCode &p_fourcc) {
	switch (p_fourcc) {
		case kCVPixelFormatType_24RGB:
			return "RGB8";
		default:
			return String::chr((char)(p_fourcc >> 24) & 0xFF) +
					String::chr((char)(p_fourcc >> 16) & 0xFF) +
					String::chr((char)(p_fourcc >> 8) & 0xFF) +
					String::chr((char)(p_fourcc >> 0) & 0xFF);
	}
}

//////////////////////////////////////////////////////////////////////////
// MyDeviceNotifications - This is a little helper class gets notifications
// when devices are connected/disconnected

@interface MyDeviceNotifications : NSObject {
	CameraMacOS *camera_server;
}

@end

@implementation MyDeviceNotifications

- (void)devices_changed:(NSNotification *)notification {
	camera_server->update_feeds();
}

- (id)initForServer:(CameraMacOS *)p_server {
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
// CameraMacOS - Subclass for our camera server on macOS

void CameraMacOS::update_feeds() {
	NSArray<AVCaptureDevice *> *devices = nullptr;
#if defined(__x86_64__)
	if (@available(macOS 10.15, *)) {
#endif
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
#endif

	// Deactivate feeds that are gone before removing them.
	for (int i = feeds.size() - 1; i >= 0; i--) {
		Ref<CameraFeedMacOS> feed = (Ref<CameraFeedMacOS>)feeds[i];
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
			Ref<CameraFeedMacOS> feed = (Ref<CameraFeedMacOS>)feeds[i];
			if (feed.is_null()) {
				continue;
			}
			if (feed->get_device() == device) {
				found = true;
			};
		};

		if (!found) {
			Ref<CameraFeedMacOS> newfeed;
			newfeed.instantiate();
			newfeed->set_device(device);

			add_feed(newfeed);
		};
	};
	emit_signal(SNAME(CameraServer::feeds_updated_signal_name));
}

void CameraMacOS::set_monitoring_feeds(bool p_monitoring_feeds) {
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
