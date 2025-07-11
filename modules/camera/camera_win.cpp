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

//////////////////////////////////////////////////////////////////////////
// CameraFeedWindows - Subclass for our camera feed on windows

Ref<CameraFeedWindows> CameraFeedWindows::create(IMFActivate *imf_camera_device) {
	Ref<CameraFeedWindows> feed = memnew(CameraFeedWindows);

	UINT32 len;
	HRESULT hr;

	// Get camera id
	wchar_t *camera_id = nullptr;
	hr = imf_camera_device->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &camera_id, &len);
	ERR_FAIL_COND_V_MSG(FAILED(hr), {}, "Unable to get camera id");
	feed->device_id = camera_id;
	CoTaskMemFree(camera_id);

	// Get name
	wchar_t *camera_name = nullptr;
	hr = imf_camera_device->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &camera_name, &len);
	ERR_FAIL_COND_V_MSG(FAILED(hr), {}, "Unable to get camera name");
	feed->name = camera_name;
	CoTaskMemFree(camera_name);

	// Get media imf_media_source
	IMFMediaSource *imf_media_source = nullptr;
	hr = imf_camera_device->ActivateObject(IID_PPV_ARGS(&imf_media_source));
	ERR_FAIL_COND_V_MSG(FAILED(hr), {}, "Unable to activate device");

	// Get information about device
	IMFPresentationDescriptor *imf_presentation_descriptor = nullptr;
	hr = imf_media_source->CreatePresentationDescriptor(&imf_presentation_descriptor);
	if (FAILED(hr)) {
		ERR_PRINT("Unable to create presentation descriptor");
		imf_media_source->Release();
		return {};
	}

	// Get information about video stream
	BOOL _selected;
	IMFStreamDescriptor *imf_stream_descriptor = nullptr;
	hr = imf_presentation_descriptor->GetStreamDescriptorByIndex(0, &_selected, &imf_stream_descriptor);
	if (FAILED(hr)) {
		ERR_PRINT("Unable to get stream descriptor");
		imf_presentation_descriptor->Release();
		imf_media_source->Release();
		return {};
	}

	// Get information about supported media types
	IMFMediaTypeHandler *imf_media_type_handler = nullptr;
	hr = imf_stream_descriptor->GetMediaTypeHandler(&imf_media_type_handler);
	if (FAILED(hr)) {
		ERR_PRINT("Unable to get media type handler");
		imf_stream_descriptor->Release();
		imf_presentation_descriptor->Release();
		imf_media_source->Release();
		return {};
	}

	// Actually fill the feed formats
	feed->fill_formats(imf_media_type_handler);

	// Release all COM objects
	imf_media_type_handler->Release();
	imf_stream_descriptor->Release();
	imf_presentation_descriptor->Release();
	imf_media_source->Release();

	return feed;
}

void CameraFeedWindows::fill_formats(IMFMediaTypeHandler *imf_media_type_handler) {
	HRESULT hr;
	// Get supported media types
	DWORD media_type_count = 0;
	hr = imf_media_type_handler->GetMediaTypeCount(&media_type_count);
	ERR_FAIL_COND_MSG(FAILED(hr), "Unable to get media type count");

	for (DWORD i = 0; i < media_type_count; i++) {
		// Get media type
		IMFMediaType *imf_media_type;
		hr = imf_media_type_handler->GetMediaTypeByIndex(i, &imf_media_type);
		ERR_CONTINUE_MSG(FAILED(hr), "Unable to get media type at index");

		// Get video_format
		GUID video_format;
		hr = imf_media_type->GetGUID(MF_MT_SUBTYPE, &video_format);
		if (FAILED(hr)) {
			ERR_PRINT("Unable to get video format for media type");
			imf_media_type->Release();
			continue;
		}

		bool supported_fmt = (video_format == MFVideoFormat_RGB24);
		supported_fmt = supported_fmt || (video_format == MFVideoFormat_NV12);
		supported_fmt = supported_fmt || (video_format == MFVideoFormat_YUY2);
		supported_fmt = supported_fmt || (video_format == MFVideoFormat_MJPG);
		if (!supported_fmt) {
			uint32_t format = video_format.Data1;
			if (!warned_formats.has(format)) {
				if (format <= 255) {
					WARN_PRINT_ED(vformat("Skipping unsupported video format %d", format));
				} else {
					uint8_t *chars = (uint8_t *)&format;
					WARN_PRINT_ED(vformat("Skipping unsupported video format %c%c%c%c", chars[0], chars[1], chars[2], chars[3]));
				}
				warned_formats.append(format);
			}

			imf_media_type->Release();
			continue;
		}

		// Get image size
		UINT32 width, height = 0;
		hr = MFGetAttributeSize(imf_media_type, MF_MT_FRAME_SIZE, &width, &height);
		if (FAILED(hr)) {
			ERR_PRINT("Unable to get frame size for media type");
			imf_media_type->Release();
			continue;
		}

		UINT32 numerator, denominator = 0;
		hr = MFGetAttributeRatio(imf_media_type, MF_MT_FRAME_RATE, &numerator, &denominator);
		if (FAILED(hr)) {
			ERR_PRINT("Unable to get frame rate for media type");
			imf_media_type->Release();
			continue;
		}

		// Add supported formats
		FeedFormat format;
		if (video_format == MFVideoFormat_RGB24) {
			format.format = "RGB24";
		} else if (video_format == MFVideoFormat_NV12) {
			format.format = "NV12";
		} else if (video_format == MFVideoFormat_YUY2) {
			format.format = "YUY2";
		} else if (video_format == MFVideoFormat_MJPG) {
			format.format = "MJPG";
		}
		format.width = width;
		format.height = height;
		format.frame_numerator = numerator;
		format.frame_denominator = denominator;

		this->formats.append(format);
		this->format_guids.append(video_format);
		this->format_mediatypes.append(i);

		imf_media_type->Release();
	}
}

CameraFeedWindows::~CameraFeedWindows() {
	if (is_active()) {
		deactivate_feed();
	};
}

bool IMFMediaSource_set_media_type(IMFMediaSource *imf_media_source, uint32_t media_type_index) {
	bool result = false;
	HRESULT hr;

	IMFPresentationDescriptor *imf_presentation_descriptor;
	hr = imf_media_source->CreatePresentationDescriptor(&imf_presentation_descriptor);
	if (SUCCEEDED(hr)) {
		BOOL _selected;
		IMFStreamDescriptor *imf_stream_descriptor;
		hr = imf_presentation_descriptor->GetStreamDescriptorByIndex(0, &_selected, &imf_stream_descriptor);
		if (SUCCEEDED(hr)) {
			// Get information about supported media types
			IMFMediaTypeHandler *imf_media_type_handler;
			hr = imf_stream_descriptor->GetMediaTypeHandler(&imf_media_type_handler);
			if (SUCCEEDED(hr)) {
				IMFMediaType *imf_media_type;
				hr = imf_media_type_handler->GetMediaTypeByIndex(media_type_index, &imf_media_type);
				if (SUCCEEDED(hr)) {
					// Set media type
					hr = imf_media_type_handler->SetCurrentMediaType(imf_media_type);
					if (SUCCEEDED(hr)) {
						result = true;
					}
					imf_media_type->Release();
				}

				imf_media_type_handler->Release();
			}

			imf_stream_descriptor->Release();
		}

		imf_presentation_descriptor->Release();
	}

	return result;
}

bool CameraFeedWindows::activate_feed() {
	ERR_FAIL_COND_V_MSG(selected_format == -1, false, "CameraFeed format needs to be set before activating.");
	ERR_FAIL_INDEX_V_MSG(selected_format, formats.size(), false, "Invalid format index for CameraFeed");

	bool result = false;
	HRESULT hr;

	IMFAttributes *imf_attributes = nullptr;
	hr = MFCreateAttributes(&imf_attributes, 2);
	if (SUCCEEDED(hr)) {
		Vector<uint8_t> device_id_wchar;
		// Set the device type to video.
		hr = imf_attributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
		if (FAILED(hr)) {
			goto release_attributes;
		}

		device_id_wchar = device_id.to_wchar_buffer();
		hr = imf_attributes->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, (const wchar_t *)device_id_wchar.ptr());
		if (FAILED(hr)) {
			goto release_attributes;
		}

		// Create media imf_media_source
		hr = MFCreateDeviceSource(imf_attributes, &imf_media_source);
		if (SUCCEEDED(hr)) {
			bool media_type_set = IMFMediaSource_set_media_type(imf_media_source, format_mediatypes[selected_format]);

			if (media_type_set) {
				// Create media imf_source_reader
				hr = MFCreateSourceReaderFromMediaSource(imf_media_source, nullptr, &imf_source_reader);
				if (SUCCEEDED(hr)) {
					result = true;

					// Create buffer decoder
					buffer_decoder = _create_buffer_decoder();

					// Start reading
					worker = memnew(std::thread(capture, this));
				}
			}
		}
	release_attributes:
		imf_attributes->Release();
	}

	return result;
}

void CameraFeedWindows::deactivate_feed() {
	if (worker != nullptr) {
		active = false;
		worker->join();
		memdelete(worker);
		worker = nullptr;
	}

	if (buffer_decoder != nullptr) {
		memdelete(buffer_decoder);
		buffer_decoder = nullptr;
	}

	if (imf_media_source != nullptr) {
		imf_media_source->Release();
		imf_media_source = nullptr;
	}

	if (imf_source_reader != nullptr) {
		imf_source_reader->Release();
		imf_source_reader = nullptr;
	}
}

void CameraFeedWindows::capture(CameraFeedWindows *feed) {
	print_verbose("Camera feed is now streaming");
	feed->active = true;
	while (feed->active) {
		feed->read();
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
			DWORD buffer_length;
			buffer->Lock(&data, nullptr, &buffer_length);

			// Use buffer decoder to process the frame
			StreamingBuffer streaming_buffer;
			streaming_buffer.start = data;
			streaming_buffer.length = buffer_length;

			if (buffer_decoder) {
				buffer_decoder->decode(streaming_buffer);
			}

			buffer->Unlock();
			buffer->Release();
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

CameraFeed::FeedFormat CameraFeedWindows::get_format() const {
	FeedFormat feed_format = {};
	return selected_format == -1 ? feed_format : formats[selected_format];
}

bool CameraFeedWindows::set_format(int p_index, const Dictionary &p_parameters) {
	selected_format = p_index;
	parameters = p_parameters.duplicate();

	return true;
}

BufferDecoder *CameraFeedWindows::_create_buffer_decoder() {
	const GUID &video_format = format_guids[selected_format];

	if (video_format == MFVideoFormat_MJPG) {
		return memnew(JpegBufferDecoder(this));
	} else if (video_format == MFVideoFormat_NV12) {
		return memnew(Nv12BufferDecoder(this));
	} else if (video_format == MFVideoFormat_YUY2) {
		return memnew(SeparateYuyvBufferDecoder(this));
	} else if (video_format == MFVideoFormat_RGB24) {
		return memnew(CopyBufferDecoder(this, CopyBufferDecoder::rgb));
	}

	// Default to null decoder
	return memnew(NullBufferDecoder(this));
}

//////////////////////////////////////////////////////////////////////////
// CameraWindows - Subclass for our camera server on windows

void CameraWindows::update_feeds() {
	// remove existing devices
	for (int i = feeds.size() - 1; i >= 0; i--) {
		remove_feed(feeds[i]);
	};

	// Create an attribute store to hold the search criteria.
	IMFAttributes *source_attributes;
	HRESULT hr = MFCreateAttributes(&source_attributes, 1);

	if (SUCCEEDED(hr)) {
		// Request video capture devices.
		hr = source_attributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);

		if (SUCCEEDED(hr)) {
			// Process devices
			UINT32 count = 0;
			IMFActivate **devices;
			hr = MFEnumDeviceSources(source_attributes, &devices, &count);
			if (FAILED(hr)) {
				ERR_PRINT("Error enumerating devices.");
			} else {
				// Create feeds for all supported media sources
				for (DWORD i = 0; i < count; i++) {
					IMFActivate *device = devices[i];
					Ref<CameraFeedWindows> feed = CameraFeedWindows::create(device);
					if (feed.is_valid()) {
						add_feed(feed);
					}
					device->Release();
				}

				CoTaskMemFree(devices);
			}
		}

		source_attributes->Release();
	}

	emit_signal(SNAME(CameraServer::feeds_updated_signal_name));
}

CameraWindows::CameraWindows() {
	// Initialize the Media Foundation platform.
	HRESULT hr = MFStartup(MF_VERSION);
	ERR_FAIL_COND_MSG(FAILED(hr), "Unable to initialize Media Foundation platform");
}

CameraWindows::~CameraWindows() {
	MFShutdown();
}

void CameraWindows::set_monitoring_feeds(bool p_monitoring_feeds) {
	monitoring_feeds = p_monitoring_feeds;
	if (monitoring_feeds) {
		update_feeds();
	}
}
