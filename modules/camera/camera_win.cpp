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

// Supported formats for BufferDecoder (manual conversion).
static const GUID BUFFER_DECODER_FORMATS[] = {
	MFVideoFormat_RGB24,
	MFVideoFormat_NV12,
	MFVideoFormat_YUY2,
	MFVideoFormat_MJPG,
};

// Additional formats supported via Media Foundation automatic conversion.
static const GUID MF_CONVERSION_FORMATS[] = {
	MFVideoFormat_RGB32,
	MFVideoFormat_ARGB32,
	MFVideoFormat_UYVY,
	MFVideoFormat_YVYU,
	MFVideoFormat_I420,
	MFVideoFormat_IYUV,
	MFVideoFormat_YV12,
	MFVideoFormat_H264,
};

static bool _is_format_supported(const GUID &p_format) {
	for (const GUID &fmt : BUFFER_DECODER_FORMATS) {
		if (p_format == fmt) {
			return true;
		}
	}
	for (const GUID &fmt : MF_CONVERSION_FORMATS) {
		if (p_format == fmt) {
			return true;
		}
	}
	return false;
}

static bool _is_buffer_decoder_format(const GUID &p_format) {
	for (const GUID &fmt : BUFFER_DECODER_FORMATS) {
		if (p_format == fmt) {
			return true;
		}
	}
	return false;
}

static String _get_format_name(const GUID &p_format) {
	if (p_format == MFVideoFormat_RGB24) {
		return "RGB24";
	} else if (p_format == MFVideoFormat_RGB32) {
		return "RGB32";
	} else if (p_format == MFVideoFormat_ARGB32) {
		return "ARGB32";
	} else if (p_format == MFVideoFormat_NV12) {
		return "NV12";
	} else if (p_format == MFVideoFormat_YUY2) {
		return "YUY2";
	} else if (p_format == MFVideoFormat_UYVY) {
		return "UYVY";
	} else if (p_format == MFVideoFormat_YVYU) {
		return "YVYU";
	} else if (p_format == MFVideoFormat_I420) {
		return "I420";
	} else if (p_format == MFVideoFormat_IYUV) {
		return "IYUV";
	} else if (p_format == MFVideoFormat_YV12) {
		return "YV12";
	} else if (p_format == MFVideoFormat_MJPG) {
		return "MJPG";
	} else if (p_format == MFVideoFormat_H264) {
		return "H264";
	}
	return "Unknown";
}

//////////////////////////////////////////////////////////////////////////
// CameraFeedWindows - Subclass for our camera feed on windows

Ref<CameraFeedWindows> CameraFeedWindows::create(IMFActivate *imf_camera_device) {
	Ref<CameraFeedWindows> feed = memnew(CameraFeedWindows);

	UINT32 len;
	HRESULT hr;

	// Store IMFActivate for reactivation.
	feed->imf_activate = imf_camera_device;
	feed->imf_activate->AddRef();

	// Get camera ID.
	wchar_t *camera_id = nullptr;
	hr = imf_camera_device->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &camera_id, &len);
	ERR_FAIL_COND_V_MSG(FAILED(hr), {}, "Unable to get camera id.");
	feed->device_id = camera_id;
	CoTaskMemFree(camera_id);

	// Get name.
	wchar_t *camera_name = nullptr;
	hr = imf_camera_device->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &camera_name, &len);
	ERR_FAIL_COND_V_MSG(FAILED(hr), {}, "Unable to get camera name.");
	feed->name = camera_name;
	CoTaskMemFree(camera_name);

	// Get media imf_media_source.
	IMFMediaSource *imf_media_source = nullptr;
	hr = imf_camera_device->ActivateObject(IID_PPV_ARGS(&imf_media_source));
	ERR_FAIL_COND_V_MSG(FAILED(hr), {}, "Unable to activate device.");

	// Get information about device.
	IMFPresentationDescriptor *imf_presentation_descriptor = nullptr;
	hr = imf_media_source->CreatePresentationDescriptor(&imf_presentation_descriptor);
	if (FAILED(hr)) {
		ERR_PRINT("Unable to create presentation descriptor.");
		imf_media_source->Release();
		return {};
	}

	// Get information about video stream.
	BOOL _selected;
	IMFStreamDescriptor *imf_stream_descriptor = nullptr;
	hr = imf_presentation_descriptor->GetStreamDescriptorByIndex(0, &_selected, &imf_stream_descriptor);
	if (FAILED(hr)) {
		ERR_PRINT("Unable to get stream descriptor.");
		imf_presentation_descriptor->Release();
		imf_media_source->Release();
		return {};
	}

	// Get information about supported media types.
	IMFMediaTypeHandler *imf_media_type_handler = nullptr;
	hr = imf_stream_descriptor->GetMediaTypeHandler(&imf_media_type_handler);
	if (FAILED(hr)) {
		ERR_PRINT("Unable to get media type handler.");
		imf_stream_descriptor->Release();
		imf_presentation_descriptor->Release();
		imf_media_source->Release();
		return {};
	}

	// Actually fill the feed formats.
	feed->fill_formats(imf_media_type_handler);

	// Release all COM objects.
	imf_media_type_handler->Release();
	imf_stream_descriptor->Release();
	imf_presentation_descriptor->Release();
	imf_media_source->Release();

	return feed;
}

void CameraFeedWindows::fill_formats(IMFMediaTypeHandler *imf_media_type_handler) {
	HRESULT hr;
	// Get supported media types.
	DWORD media_type_count = 0;
	hr = imf_media_type_handler->GetMediaTypeCount(&media_type_count);
	ERR_FAIL_COND_MSG(FAILED(hr), "Unable to get media type count.");

	// Track unique format combinations after RGB24 conversion.
	Vector<FormatKey> seen_formats;

	// First pass: collect all formats and check for RGB24.
	Vector<TempFormat> temp_formats;

	for (DWORD i = 0; i < media_type_count; i++) {
		// Get media type.
		IMFMediaType *imf_media_type;
		hr = imf_media_type_handler->GetMediaTypeByIndex(i, &imf_media_type);
		ERR_CONTINUE_MSG(FAILED(hr), "Unable to get media type at index.");

		// Get video_format.
		GUID video_format;
		hr = imf_media_type->GetGUID(MF_MT_SUBTYPE, &video_format);
		if (FAILED(hr)) {
			ERR_PRINT("Unable to get video format for media type.");
			imf_media_type->Release();
			continue;
		}

		if (!_is_format_supported(video_format)) {
			uint32_t format = video_format.Data1;
			if (!warned_formats.has(format)) {
				if (format <= 255) {
					WARN_PRINT_ED(vformat("Skipping unsupported video format %d.", format));
				} else {
					uint8_t *chars = (uint8_t *)&format;
					WARN_PRINT_ED(vformat("Skipping unsupported video format %c%c%c%c.", chars[0], chars[1], chars[2], chars[3]));
				}
				warned_formats.append(format);
			}

			imf_media_type->Release();
			continue;
		}

		// Get image size.
		UINT32 width, height = 0;
		hr = MFGetAttributeSize(imf_media_type, MF_MT_FRAME_SIZE, &width, &height);
		if (FAILED(hr)) {
			ERR_PRINT("Unable to get frame size for media type.");
			imf_media_type->Release();
			continue;
		}

		UINT32 numerator, denominator = 0;
		hr = MFGetAttributeRatio(imf_media_type, MF_MT_FRAME_RATE, &numerator, &denominator);
		if (FAILED(hr)) {
			ERR_PRINT("Unable to get frame rate for media type.");
			imf_media_type->Release();
			continue;
		}

		// Store format information temporarily.
		TempFormat temp;
		temp.video_format = video_format;
		temp.media_type_index = i;
		temp.is_rgb24 = (video_format == MFVideoFormat_RGB24);

		FeedFormat format;
		format.format = _get_format_name(video_format);
		format.width = width;
		format.height = height;
		format.frame_numerator = numerator;
		format.frame_denominator = denominator;

		temp.format = format;
		temp_formats.append(temp);

		imf_media_type->Release();
	}

	// Second pass: add formats, prioritizing RGB24.
	for (const TempFormat &temp : temp_formats) {
		FormatKey key;
		key.width = temp.format.width;
		key.height = temp.format.height;
		key.frame_numerator = temp.format.frame_numerator;
		key.frame_denominator = temp.format.frame_denominator;
		key.is_rgb24 = temp.is_rgb24;

		// Check if this combination already exists.
		bool should_add = true;
		for (const FormatKey &seen : seen_formats) {
			if (seen == key) {
				// If we already have this combination and it was RGB24, skip non-RGB24.
				if (seen.is_rgb24 && !temp.is_rgb24) {
					should_add = false;
					break;
				} else if (!seen.is_rgb24 && temp.is_rgb24) {
					// If we have non-RGB24 and this is RGB24, we'll replace it
					// Remove the existing non-RGB24 format.
					for (int j = formats.size() - 1; j >= 0; j--) {
						if (formats[j].width == key.width &&
								formats[j].height == key.height &&
								formats[j].frame_numerator == key.frame_numerator &&
								formats[j].frame_denominator == key.frame_denominator) {
							formats.remove_at(j);
							format_guids.remove_at(j);
							format_mediatypes.remove_at(j);
							break;
						}
					}
					// Update seen_formats.
					for (int j = 0; j < seen_formats.size(); j++) {
						if (seen_formats[j] == key) {
							seen_formats.write[j].is_rgb24 = true;
							break;
						}
					}
				}
				// Both are the same type, skip.
				else {
					should_add = false;
					break;
				}
			}
		}

		if (should_add) {
			// Add to seen formats.
			seen_formats.append(key);

			// Add format.
			formats.append(temp.format);
			format_guids.append(temp.video_format);
			format_mediatypes.append(temp.media_type_index);
		}
	}
}

CameraFeedWindows::~CameraFeedWindows() {
	if (is_active()) {
		deactivate_feed();
	}

	// Release IMFActivate.
	if (imf_activate != nullptr) {
		imf_activate->Release();
		imf_activate = nullptr;
	}
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
			// Get information about supported media types.
			IMFMediaTypeHandler *imf_media_type_handler;
			hr = imf_stream_descriptor->GetMediaTypeHandler(&imf_media_type_handler);
			if (SUCCEEDED(hr)) {
				IMFMediaType *imf_media_type;
				hr = imf_media_type_handler->GetMediaTypeByIndex(media_type_index, &imf_media_type);
				if (SUCCEEDED(hr)) {
					// Set media type.
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
	ERR_FAIL_INDEX_V_MSG(selected_format, formats.size(), false, "Invalid format index for CameraFeed.");
	ERR_FAIL_COND_V_MSG(imf_activate == nullptr, false, "IMFActivate is null, cannot activate camera feed.");

	bool result = false;
	HRESULT hr;

	// Create media source.
	hr = imf_activate->ActivateObject(IID_PPV_ARGS(&imf_media_source));
	if (SUCCEEDED(hr)) {
		bool media_type_set = IMFMediaSource_set_media_type(imf_media_source, format_mediatypes[selected_format]);

		if (media_type_set) {
			// Create media imf_source_reader.
			IMFAttributes *reader_attributes = nullptr;
			hr = MFCreateAttributes(&reader_attributes, 2);
			if (SUCCEEDED(hr)) {
				// Enable hardware acceleration if available.
				hr = reader_attributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);
				// Hint to disconnect media source on shutdown to help unblock.
				reader_attributes->SetUINT32(MF_SOURCE_READER_DISCONNECT_MEDIASOURCE_ON_SHUTDOWN, TRUE);

				hr = MFCreateSourceReaderFromMediaSource(imf_media_source, reader_attributes, &imf_source_reader);
				reader_attributes->Release();

				if (SUCCEEDED(hr)) {
					// Ensure we are reading the first video stream.
					imf_source_reader->SetStreamSelection(MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE);

					// Configure source reader to convert to RGB24.
					IMFMediaType *output_type = nullptr;
					hr = MFCreateMediaType(&output_type);
					if (SUCCEEDED(hr)) {
						hr = output_type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
						hr = output_type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB24);

						// Try to set RGB24 as output format.
						hr = imf_source_reader->SetCurrentMediaType(
								MF_SOURCE_READER_FIRST_VIDEO_STREAM,
								nullptr,
								output_type);

						if (SUCCEEDED(hr)) {
							result = true;
							use_mf_conversion = true;
							// Windows RGB24 uses BGR byte order. See:
							// https://learn.microsoft.com/en-us/windows/win32/directshow/uncompressed-rgb-video-subtypes
							buffer_decoder = memnew(CopyBufferDecoder(this, CopyBufferDecoder::bgr));
						} else {
							// Fallback to manual conversion for formats that support it.
							const GUID &video_format = format_guids[selected_format];
							if (_is_buffer_decoder_format(video_format)) {
								result = true;
								use_mf_conversion = false;
								buffer_decoder = _create_buffer_decoder();
							} else {
								ERR_PRINT("Format not supported by Media Foundation conversion or manual decoder.");
								result = false;
							}
						}

						output_type->Release();
					}

					if (result) {
						// Start reading.
						worker = std::thread(capture, this);
					}
				}
			}
		}
	} else {
		ERR_PRINT(vformat("IMFActivate::ActivateObject failed: 0x%08x.", (uint32_t)hr));
		// If another application is using the device, provide a human-readable hint.
		if (hr == HRESULT_FROM_WIN32(ERROR_SHARING_VIOLATION)) {
			ERR_PRINT("Check that no other applications are currently using the camera.");
		}
	}

	return result;
}

void CameraFeedWindows::deactivate_feed() {
	if (worker.joinable()) {
		active = false;
		// Attempt to unblock ReadSample by deselecting and flushing the stream.
		if (imf_source_reader) {
			imf_source_reader->SetStreamSelection(MF_SOURCE_READER_FIRST_VIDEO_STREAM, FALSE);
			imf_source_reader->Flush(MF_SOURCE_READER_FIRST_VIDEO_STREAM);
		}
		worker.join();
	}

	if (buffer_decoder != nullptr) {
		memdelete(buffer_decoder);
		buffer_decoder = nullptr;
	}

	// Release in safe order: reader first, then media source.
	if (imf_source_reader != nullptr) {
		imf_source_reader->Release();
		imf_source_reader = nullptr;
	}

	if (imf_media_source != nullptr) {
		imf_media_source->Release();
		imf_media_source = nullptr;
	}

	// Shutdown the device.
	if (imf_activate != nullptr) {
		imf_activate->ShutdownObject();
	}
}

void CameraFeedWindows::capture(CameraFeedWindows *feed) {
	print_verbose("Camera feed is now streaming.");
	// Initialize COM on this worker thread to safely use MF objects here.
	HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
	feed->active = true;
	while (feed->active) {
		feed->read();
	}
	if (SUCCEEDED(hr)) {
		CoUninitialize();
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
		ERR_PRINT(vformat("ReadSample failed: 0x%08x.", (uint32_t)hr));
		return;
	}

	// End of stream.
	if (flags & MF_SOURCE_READERF_ENDOFSTREAM) {
		print_verbose("\tEnd of stream.");
		active = false;
	}
	if (flags & MF_SOURCE_READERF_NEWSTREAM) {
		print_verbose("\tNew stream.");
	}
	if (flags & MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED) {
		print_verbose("\tNative type changed.");
	}
	if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED) {
		print_verbose("\tCurrent type changed.");
	}
	if (flags & MF_SOURCE_READERF_STREAMTICK) {
		print_verbose("\tStream tick.");
	}

	// Process sample.
	if (pSample) {
		IMFMediaBuffer *buffer = nullptr;
		hr = pSample->ConvertToContiguousBuffer(&buffer);
		if (SUCCEEDED(hr) && buffer) {
			BYTE *data = nullptr;
			DWORD buffer_length = 0;
			bool flip_detected = false;
			bool used_2d_buffer = false;

			// Try IMF2DBuffer for stride information (faster).
			IMF2DBuffer *buffer2d = nullptr;
			if (SUCCEEDED(buffer->QueryInterface(IID_PPV_ARGS(&buffer2d)))) {
				LONG pitch = 0;
				BYTE *scan_line = nullptr;
				hr = buffer2d->Lock2D(&scan_line, &pitch);
				if (SUCCEEDED(hr)) {
					data = scan_line;
					buffer_length = abs(pitch) * get_format().height;
					flip_detected = (pitch < 0);
					used_2d_buffer = true;

					if (buffer_decoder) {
						buffer_decoder->set_flip_vertical(flip_detected);
					}

					StreamingBuffer streaming_buffer;
					streaming_buffer.start = data;
					streaming_buffer.length = buffer_length;
					streaming_buffer.stride = abs(pitch);

					if (buffer_decoder) {
						buffer_decoder->decode(streaming_buffer);
					}

					buffer2d->Unlock2D();
				}
				buffer2d->Release();
			}

			// Fallback: Use standard buffer lock.
			if (!used_2d_buffer) {
				hr = buffer->Lock(&data, nullptr, &buffer_length);
				if (SUCCEEDED(hr)) {
					StreamingBuffer streaming_buffer;
					streaming_buffer.start = data;
					streaming_buffer.length = buffer_length;

					if (buffer_decoder) {
						buffer_decoder->set_flip_vertical(true);
						buffer_decoder->decode(streaming_buffer);
					}

					buffer->Unlock();
				} else {
					ERR_PRINT(vformat("IMFMediaBuffer::Lock failed: 0x%08x.", (uint32_t)hr));
				}
			}

			buffer->Release();
		} else {
			ERR_PRINT(vformat("ConvertToContiguousBuffer failed: 0x%08x.", (uint32_t)hr));
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
	if (selected_format == -1 || selected_format >= formats.size()) {
		return feed_format;
	}
	return formats[selected_format];
}

bool CameraFeedWindows::set_format(int p_index, const Dictionary &p_parameters) {
	selected_format = p_index;
	parameters = p_parameters.duplicate();

	return true;
}

BufferDecoder *CameraFeedWindows::_create_buffer_decoder() {
	const GUID &video_format = format_guids[selected_format];

	// Only create decoder for formats that Media Foundation can't convert.
	if (video_format == MFVideoFormat_MJPG) {
		return memnew(JpegBufferDecoder(this));
	} else if (video_format == MFVideoFormat_NV12) {
		return memnew(Nv12BufferDecoder(this));
	} else if (video_format == MFVideoFormat_YUY2) {
		return memnew(SeparateYuyvBufferDecoder(this));
	} else if (video_format == MFVideoFormat_RGB24) {
		// Windows RGB24 uses BGR byte order. See:
		// https://learn.microsoft.com/en-us/windows/win32/directshow/uncompressed-rgb-video-subtypes
		return memnew(CopyBufferDecoder(this, CopyBufferDecoder::bgr));
	}

	// Default to null decoder for unsupported formats.
	// These formats will rely on Media Foundation conversion.
	return memnew(NullBufferDecoder(this));
}

//////////////////////////////////////////////////////////////////////////
// CameraWindows - Subclass for our camera server on windows

void CameraWindows::update_feeds() {
	remove_all_feeds();

	// Create an attribute store to hold the search criteria.
	IMFAttributes *source_attributes;
	HRESULT hr = MFCreateAttributes(&source_attributes, 1);

	if (SUCCEEDED(hr)) {
		// Request video capture devices.
		hr = source_attributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);

		if (SUCCEEDED(hr)) {
			// Process devices.
			UINT32 count = 0;
			IMFActivate **devices;
			hr = MFEnumDeviceSources(source_attributes, &devices, &count);
			if (FAILED(hr)) {
				ERR_PRINT("Error enumerating devices.");
			} else {
				// Create feeds for all supported media sources.
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

void CameraWindows::remove_all_feeds() {
	// remove existing devices.
	for (int i = feeds.size() - 1; i >= 0; i--) {
		remove_feed(feeds[i]);
	}
}

CameraWindows::CameraWindows() {
	// Initialize the Media Foundation platform.
	HRESULT hr = MFStartup(MF_VERSION);
	ERR_FAIL_COND_MSG(FAILED(hr), "Unable to initialize Media Foundation platform.");
}

CameraWindows::~CameraWindows() {
	remove_all_feeds();
	MFShutdown();
}

void CameraWindows::set_monitoring_feeds(bool p_monitoring_feeds) {
	if (p_monitoring_feeds == monitoring_feeds) {
		return;
	}

	CameraServer::set_monitoring_feeds(p_monitoring_feeds);
	if (monitoring_feeds) {
		update_feeds();
	} else {
		remove_all_feeds();
	}
}
