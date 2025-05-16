/**************************************************************************/
/*  camera_win.cpp                                                        */
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

#include "camera_win.h"
#include <strsafe.h>

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(x) \
	if (x != nullptr) { \
		x->Release();   \
		x = nullptr;    \
	}
#endif

#ifndef defer
template <typename Fn>
struct _deferClass {
	Fn fn;
	_deferClass(Fn &&fn) :
			fn(fn) {}
	~_deferClass() { fn(); }
};

#define DEFER_CONCAT_INTERNAL(a, b) a##b
#define DEFER_CONCAT(a, b) DEFER_CONCAT_INTERNAL(a, b)
#define defer const _deferClass DEFER_CONCAT(anon_deferVar_, __COUNTER__) = [&]()
#endif

//////////////////////////////////////////////////////////////////////////
// CameraFeedWindows - Subclass for our camera feed on windows

Ref<CameraFeedWindows> CameraFeedWindows::create(IMFActivate *pDevice) {
	Ref<CameraFeedWindows> feed = memnew(CameraFeedWindows);

	UINT32 len;
	HRESULT hr;

	// Get camera id
	wchar_t *szCameraID = nullptr;
	hr = pDevice->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &szCameraID, &len);
	ERR_FAIL_COND_V_MSG(FAILED(hr), {}, "Unable to get camera id");
	defer {
		CoTaskMemFree(szCameraID);
	}

	// Get name
	wchar_t *szFriendlyName = nullptr;
	hr = pDevice->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &szFriendlyName, &len);
	ERR_FAIL_COND_V_MSG(FAILED(hr), {}, "Unable to get camera name");
	defer {
		CoTaskMemFree(szFriendlyName);
	}

	feed->device_id = szCameraID;
	feed->name = szFriendlyName;

	// Get media imf_media_source
	IMFMediaSource *pSource = nullptr;
	hr = pDevice->ActivateObject(IID_PPV_ARGS(&pSource));
	defer {
		SAFE_RELEASE(pSource);
	}
	ERR_FAIL_COND_V_MSG(FAILED(hr), {}, "Unable to activate device");

	// Get information about device
	IMFPresentationDescriptor *imf_presentation_descriptor = nullptr;
	hr = pSource->CreatePresentationDescriptor(&imf_presentation_descriptor);
	defer {
		SAFE_RELEASE(imf_presentation_descriptor);
	}
	ERR_FAIL_COND_V_MSG(FAILED(hr), {}, "Unable to create presentation descriptor");

	// Get information about video stream
	BOOL fSelected;
	IMFStreamDescriptor *imf_stream_descriptor = nullptr;
	hr = imf_presentation_descriptor->GetStreamDescriptorByIndex(0, &fSelected, &imf_stream_descriptor);
	defer {
		SAFE_RELEASE(imf_stream_descriptor);
	}
	ERR_FAIL_COND_V_MSG(FAILED(hr), {}, "Unable to get stream descriptor");

	// Get information about supported media types
	IMFMediaTypeHandler *imf_media_type_handler = nullptr;
	hr = imf_stream_descriptor->GetMediaTypeHandler(&imf_media_type_handler);
	defer {
		SAFE_RELEASE(imf_media_type_handler);
	}
	ERR_FAIL_COND_V_MSG(FAILED(hr), {}, "Unable to get media type handler");

	// Get supported media types
	DWORD cTypes = 0;
	hr = imf_media_type_handler->GetMediaTypeCount(&cTypes);
	ERR_FAIL_COND_V_MSG(FAILED(hr), {}, "Unable to get media type count");

	for (DWORD i = 0; i < cTypes; i++) {
		// Get media type
		IMFMediaType *imf_media_type = nullptr;
		hr = imf_media_type_handler->GetMediaTypeByIndex(i, &imf_media_type);
		defer {
			SAFE_RELEASE(imf_media_type);
		}
		ERR_CONTINUE_MSG(FAILED(hr), "Unable to get media type at index");

		// Get video_format
		GUID video_format;
		hr = imf_media_type->GetGUID(MF_MT_SUBTYPE, &video_format);
		ERR_CONTINUE_MSG(FAILED(hr), "Unable to get video format for media type");
		bool supported_fmt = (video_format == MFVideoFormat_RGB24) || (video_format == MFVideoFormat_NV12);
		ERR_CONTINUE_MSG(!supported_fmt, "Skipping unsupported video format");

		// Get image size
		UINT32 width, height = 0;
		hr = MFGetAttributeSize(imf_media_type, MF_MT_FRAME_SIZE, &width, &height);
		ERR_CONTINUE_MSG(FAILED(hr), "Unable to get frame size for media type");

		UINT32 numerator, denominator = 0;
		hr = MFGetAttributeRatio(imf_media_type, MF_MT_FRAME_RATE, &numerator, &denominator);
		ERR_CONTINUE_MSG(FAILED(hr), "Unable to get frame rate for media type");

		// Add supported formats
		FeedFormat format;
		if (video_format == MFVideoFormat_RGB24) {
			format.format = "RGB24";
		} else if (video_format == MFVideoFormat_NV12) {
			format.format = "NV12";
		}
		format.width = width;
		format.height = height;
		format.frame_numerator = numerator;
		format.frame_denominator = denominator;

		feed->formats.append(format);
		feed->format_guids.append(video_format);
		feed->format_mediatypes.append(imf_media_type);
	}
	return feed;
}

CameraFeedWindows::~CameraFeedWindows() {
	if (is_active()) {
		deactivate_feed();
	};
}

bool CameraFeedWindows::activate_feed() {
	ERR_FAIL_COND_V_MSG(selected_format == -1, false, "CameraFeed format needs to be set before activating.");
	ERR_FAIL_INDEX_V_MSG(selected_format, formats.size(), false, "Invalid format index for CameraFeed");

	HRESULT hr;

	IMFAttributes *imf_attributes = nullptr;
	hr = MFCreateAttributes(&imf_attributes, 2);
	defer {
		SAFE_RELEASE(imf_attributes);
	}
	ERR_FAIL_COND_V(FAILED(hr), false);

	// Set the device type to video.
	hr = imf_attributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
	ERR_FAIL_COND_V(FAILED(hr), false);

	Vector<uint8_t> device_id_wchar = device_id.to_wchar_buffer();
	hr = imf_attributes->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, (const wchar_t *)device_id_wchar.ptr());
	ERR_FAIL_COND_V(FAILED(hr), false);

	// Create media imf_media_source
	hr = MFCreateDeviceSource(imf_attributes, &imf_media_source);
	ERR_FAIL_COND_V(FAILED(hr), false);

	// Get information about device
	IMFPresentationDescriptor *imf_presentation_descriptor = nullptr;
	hr = imf_media_source->CreatePresentationDescriptor(&imf_presentation_descriptor);
	defer {
		SAFE_RELEASE(imf_presentation_descriptor);
	}
	ERR_FAIL_COND_V(FAILED(hr), false);

	// Get information about video stream
	BOOL fSelected;
	IMFStreamDescriptor *imf_stream_descriptor;
	hr = imf_presentation_descriptor->GetStreamDescriptorByIndex(0, &fSelected, &imf_stream_descriptor);
	defer {
		SAFE_RELEASE(imf_stream_descriptor);
	}
	ERR_FAIL_COND_V(FAILED(hr), false);

	// Get information about supported media types
	IMFMediaTypeHandler *imf_media_type_handler;
	hr = imf_stream_descriptor->GetMediaTypeHandler(&imf_media_type_handler);
	defer {
		SAFE_RELEASE(imf_media_type_handler);
	}
	ERR_FAIL_COND_V(FAILED(hr), false);

	IMFMediaType *imf_media_type = format_mediatypes[selected_format];
	// Set media type
	hr = imf_media_type_handler->SetCurrentMediaType(imf_media_type);
	ERR_FAIL_COND_V(FAILED(hr), false);

	// Create media imf_source_reader
	hr = MFCreateSourceReaderFromMediaSource(imf_media_source, nullptr, &imf_source_reader);
	ERR_FAIL_COND_V(FAILED(hr), false);

	const FeedFormat &format = formats[selected_format];
	const GUID &video_format = format_guids[selected_format];

	// Prepare images and textures
	if (video_format == MFVideoFormat_RGB24) {
		data_y.resize(format.width * format.height * 3);
		image_y.instantiate(format.width, format.height, false, Image::FORMAT_RGB8);
		set_rgb_image(image_y);
	} else if (video_format == MFVideoFormat_NV12) {
		data_y.resize(format.width * format.height);
		data_uv.resize(format.width * format.height / 2); // half_width * half_height * 2 channels
		image_y.instantiate(format.width, format.height, false, Image::FORMAT_R8);
		image_uv.instantiate(format.width / 2, format.height / 2, false, Image::FORMAT_RG8);
		set_ycbcr_images(image_y, image_uv);
	}

	// Start reading
	worker = memnew(std::thread(capture, this));

	return true;
}

void CameraFeedWindows::deactivate_feed() {
	if (worker != nullptr) {
		active = false;
		worker->join();
		memdelete(worker);
		worker = nullptr;
	}

	SAFE_RELEASE(imf_source_reader);
	SAFE_RELEASE(imf_media_source);
}

void CameraFeedWindows::capture(CameraFeedWindows *feed) {
	print_verbose("Camera feed is now streaming");
	feed->active = true;
	while (feed->active) {
		feed->read();
		print_verbose("read")
	}
}

void CameraFeedWindows::read() {
	HRESULT hr = S_OK;
	DWORD streamIndex, flags;
	LONGLONG llTimeStamp;
	IMFSample *pSample = nullptr;

	hr = imf_source_reader->ReadSample(
			MF_SOURCE_READER_FIRST_VIDEO_STREAM, // Stream index.
			0, // Flags.
			&streamIndex, // Receives the actual stream index.
			&flags, // Receives status flags.
			&llTimeStamp, // Receives the time stamp.
			&pSample // Receives the sample or nullptr.
	);

	if (FAILED(hr)) {
		return;
	}

	// End of stream
	if (flags & MF_SOURCE_READERF_ENDOFSTREAM) {
		print_verbose("\tEnd of stream");
		active = false;
	}
	if (flags & MF_SOURCE_READERF_NEWSTREAM) {
		print_verbose("\tNew stream");
	}
	if (flags & MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED) {
		print_verbose("\tNative type changed");
	}
	if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED) {
		print_verbose("\tCurrent type changed");
	}
	if (flags & MF_SOURCE_READERF_STREAMTICK) {
		print_verbose("\tStream tick");
	}

	// Process sample
	if (pSample) {
		IMFMediaBuffer *buffer;
		hr = pSample->GetBufferByIndex(0, &buffer);
		if (SUCCEEDED(hr)) {
			// Get image buffer
			BYTE *data;
			buffer->Lock(&data, nullptr, nullptr);
			const GUID &format = format_guids[selected_format];
			if (format == MFVideoFormat_RGB24) {
				memcpy(data_y.ptrw(), data, data_y.size());
			} else if (format == MFVideoFormat_NV12) {
				memcpy(data_y.ptrw(), data, data_y.size());
				memcpy(data_uv.ptrw(), data + data_y.size(), data_uv.size());
			}

			buffer->Unlock();
			buffer->Release();

			if (format == MFVideoFormat_RGB24) {
				image_y->set_data(image_y->get_width(), image_y->get_height(), false, image_y->get_format(), data_y);
				set_rgb_image(image_y);
			} else if (format == MFVideoFormat_NV12) {
				image_y->set_data(image_y->get_width(), image_y->get_height(), false, image_y->get_format(), data_y);
				image_uv->set_data(image_uv->get_width(), image_uv->get_height(), false, image_uv->get_format(), data_uv);
				set_ycbcr_images(image_y, image_uv);
			}
		}
		pSample->Release();
	}
}

Array CameraFeedWindows::get_formats() const {
	Array result;
	for (const FeedFormat &format : formats) {
		Dictionary dictionary;
		dictionary["width"] = format.width;
		dictionary["height"] = format.height;
		dictionary["format"] = format.format;
		dictionary["frame_numerator"] = format.frame_numerator;
		dictionary["frame_denominator"] = format.frame_denominator;
		result.push_back(dictionary);
	}
	return result;
}

bool CameraFeedWindows::set_format(int p_index, const Dictionary &p_parameters) {
	selected_format = p_index;
	return true;
}

//////////////////////////////////////////////////////////////////////////
// CameraWindows - Subclass for our camera server on windows

void CameraWindows::update_feeds() {
	// remove existing devices
	for (int i = feeds.size() - 1; i >= 0; i--) {
		remove_feed(feeds[i]);
	};

	// Create an attribute store to hold the search criteria.
	IMFAttributes *pConfig = nullptr;
	HRESULT hr = MFCreateAttributes(&pConfig, 1);
	defer {
		SAFE_RELEASE(pConfig);
	}
	ERR_FAIL_COND(FAILED(hr));

	// Request video capture devices.
	hr = pConfig->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
	ERR_FAIL_COND_MSG(FAILED(hr), "Error setting attribute.");

	// Process devices
	UINT32 count = 0;
	IMFActivate **ppDevices = nullptr;
	hr = MFEnumDeviceSources(pConfig, &ppDevices, &count);
	ERR_FAIL_COND_MSG(FAILED(hr), "Error enumerating devices.");
	defer {
		CoTaskMemFree(ppDevices);
	}

	// Create feeds for all supported media sources
	for (DWORD i = 0; i < count; i++) {
		IMFActivate *pDevice = ppDevices[i];
		Ref<CameraFeedWindows> feed = CameraFeedWindows::create(pDevice);
		if (feed.is_valid()) {
			add_feed(feed);
		}
		pDevice->Release();
	}
}

CameraWindows::CameraWindows() {
	// Initialize the Media Foundation platform.
	HRESULT hr = MFStartup(MF_VERSION);
	if (FAILED(hr)) {
		print_error("Unable to initialize Media Foundation platform");
		return;
	}

	// Update feeds
	update_feeds();
}

CameraWindows::~CameraWindows() {
	MFShutdown();
}
