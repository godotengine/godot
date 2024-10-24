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
// Helper functions
//
// The following code enables you to view the contents of a media type while
// debugging.

#ifndef IF_EQUAL_RETURN
#define IF_EQUAL_RETURN(param, val) \
	if (val == param)               \
	return #val
#endif

String GetGUIDNameConst(const GUID &guid) {
	IF_EQUAL_RETURN(guid, MF_MT_MAJOR_TYPE);
	IF_EQUAL_RETURN(guid, MF_MT_MAJOR_TYPE);
	IF_EQUAL_RETURN(guid, MF_MT_SUBTYPE);
	IF_EQUAL_RETURN(guid, MF_MT_ALL_SAMPLES_INDEPENDENT);
	IF_EQUAL_RETURN(guid, MF_MT_FIXED_SIZE_SAMPLES);
	IF_EQUAL_RETURN(guid, MF_MT_COMPRESSED);
	IF_EQUAL_RETURN(guid, MF_MT_SAMPLE_SIZE);
	IF_EQUAL_RETURN(guid, MF_MT_WRAPPED_TYPE);
	IF_EQUAL_RETURN(guid, MF_MT_AUDIO_NUM_CHANNELS);
	IF_EQUAL_RETURN(guid, MF_MT_AUDIO_SAMPLES_PER_SECOND);
	IF_EQUAL_RETURN(guid, MF_MT_AUDIO_FLOAT_SAMPLES_PER_SECOND);
	IF_EQUAL_RETURN(guid, MF_MT_AUDIO_AVG_BYTES_PER_SECOND);
	IF_EQUAL_RETURN(guid, MF_MT_AUDIO_BLOCK_ALIGNMENT);
	IF_EQUAL_RETURN(guid, MF_MT_AUDIO_BITS_PER_SAMPLE);
	IF_EQUAL_RETURN(guid, MF_MT_AUDIO_VALID_BITS_PER_SAMPLE);
	IF_EQUAL_RETURN(guid, MF_MT_AUDIO_SAMPLES_PER_BLOCK);
	IF_EQUAL_RETURN(guid, MF_MT_AUDIO_CHANNEL_MASK);
	IF_EQUAL_RETURN(guid, MF_MT_AUDIO_FOLDDOWN_MATRIX);
	IF_EQUAL_RETURN(guid, MF_MT_AUDIO_WMADRC_PEAKREF);
	IF_EQUAL_RETURN(guid, MF_MT_AUDIO_WMADRC_PEAKTARGET);
	IF_EQUAL_RETURN(guid, MF_MT_AUDIO_WMADRC_AVGREF);
	IF_EQUAL_RETURN(guid, MF_MT_AUDIO_WMADRC_AVGTARGET);
	IF_EQUAL_RETURN(guid, MF_MT_AUDIO_PREFER_WAVEFORMATEX);
	IF_EQUAL_RETURN(guid, MF_MT_AAC_PAYLOAD_TYPE);
	IF_EQUAL_RETURN(guid, MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION);
	IF_EQUAL_RETURN(guid, MF_MT_FRAME_SIZE);
	IF_EQUAL_RETURN(guid, MF_MT_FRAME_RATE);
	IF_EQUAL_RETURN(guid, MF_MT_FRAME_RATE_RANGE_MAX);
	IF_EQUAL_RETURN(guid, MF_MT_FRAME_RATE_RANGE_MIN);
	IF_EQUAL_RETURN(guid, MF_MT_PIXEL_ASPECT_RATIO);
	IF_EQUAL_RETURN(guid, MF_MT_DRM_FLAGS);
	IF_EQUAL_RETURN(guid, MF_MT_PAD_CONTROL_FLAGS);
	IF_EQUAL_RETURN(guid, MF_MT_SOURCE_CONTENT_HINT);
	IF_EQUAL_RETURN(guid, MF_MT_VIDEO_CHROMA_SITING);
	IF_EQUAL_RETURN(guid, MF_MT_INTERLACE_MODE);
	IF_EQUAL_RETURN(guid, MF_MT_TRANSFER_FUNCTION);
	IF_EQUAL_RETURN(guid, MF_MT_VIDEO_PRIMARIES);
	IF_EQUAL_RETURN(guid, MF_MT_CUSTOM_VIDEO_PRIMARIES);
	IF_EQUAL_RETURN(guid, MF_MT_YUV_MATRIX);
	IF_EQUAL_RETURN(guid, MF_MT_VIDEO_LIGHTING);
	IF_EQUAL_RETURN(guid, MF_MT_VIDEO_NOMINAL_RANGE);
	IF_EQUAL_RETURN(guid, MF_MT_GEOMETRIC_APERTURE);
	IF_EQUAL_RETURN(guid, MF_MT_MINIMUM_DISPLAY_APERTURE);
	IF_EQUAL_RETURN(guid, MF_MT_PAN_SCAN_APERTURE);
	IF_EQUAL_RETURN(guid, MF_MT_PAN_SCAN_ENABLED);
	IF_EQUAL_RETURN(guid, MF_MT_AVG_BITRATE);
	IF_EQUAL_RETURN(guid, MF_MT_AVG_BIT_ERROR_RATE);
	IF_EQUAL_RETURN(guid, MF_MT_MAX_KEYFRAME_SPACING);
	IF_EQUAL_RETURN(guid, MF_MT_DEFAULT_STRIDE);
	IF_EQUAL_RETURN(guid, MF_MT_PALETTE);
	IF_EQUAL_RETURN(guid, MF_MT_USER_DATA);
	IF_EQUAL_RETURN(guid, MF_MT_AM_FORMAT_TYPE);
	IF_EQUAL_RETURN(guid, MF_MT_MPEG_START_TIME_CODE);
	IF_EQUAL_RETURN(guid, MF_MT_MPEG2_PROFILE);
	IF_EQUAL_RETURN(guid, MF_MT_MPEG2_LEVEL);
	IF_EQUAL_RETURN(guid, MF_MT_MPEG2_FLAGS);
	IF_EQUAL_RETURN(guid, MF_MT_MPEG_SEQUENCE_HEADER);
	IF_EQUAL_RETURN(guid, MF_MT_DV_AAUX_SRC_PACK_0);
	IF_EQUAL_RETURN(guid, MF_MT_DV_AAUX_CTRL_PACK_0);
	IF_EQUAL_RETURN(guid, MF_MT_DV_AAUX_SRC_PACK_1);
	IF_EQUAL_RETURN(guid, MF_MT_DV_AAUX_CTRL_PACK_1);
	IF_EQUAL_RETURN(guid, MF_MT_DV_VAUX_SRC_PACK);
	IF_EQUAL_RETURN(guid, MF_MT_DV_VAUX_CTRL_PACK);
	IF_EQUAL_RETURN(guid, MF_MT_ARBITRARY_HEADER);
	IF_EQUAL_RETURN(guid, MF_MT_ARBITRARY_FORMAT);
	IF_EQUAL_RETURN(guid, MF_MT_IMAGE_LOSS_TOLERANT);
	IF_EQUAL_RETURN(guid, MF_MT_MPEG4_SAMPLE_DESCRIPTION);
	IF_EQUAL_RETURN(guid, MF_MT_MPEG4_CURRENT_SAMPLE_ENTRY);
	IF_EQUAL_RETURN(guid, MF_MT_ORIGINAL_4CC);
	IF_EQUAL_RETURN(guid, MF_MT_ORIGINAL_WAVE_FORMAT_TAG);

	// Media types
	IF_EQUAL_RETURN(guid, MFMediaType_Audio);
	IF_EQUAL_RETURN(guid, MFMediaType_Video);
	IF_EQUAL_RETURN(guid, MFMediaType_Protected);
	IF_EQUAL_RETURN(guid, MFMediaType_SAMI);
	IF_EQUAL_RETURN(guid, MFMediaType_Script);
	IF_EQUAL_RETURN(guid, MFMediaType_Image);
	IF_EQUAL_RETURN(guid, MFMediaType_HTML);
	IF_EQUAL_RETURN(guid, MFMediaType_Binary);
	IF_EQUAL_RETURN(guid, MFMediaType_FileTransfer);

	IF_EQUAL_RETURN(guid, MFVideoFormat_AI44); //     FCC('AI44')
	IF_EQUAL_RETURN(guid, MFVideoFormat_ARGB32); //   D3DFMT_A8R8G8B8
	IF_EQUAL_RETURN(guid, MFVideoFormat_AYUV); //     FCC('AYUV')
	IF_EQUAL_RETURN(guid, MFVideoFormat_DV25); //     FCC('dv25')
	IF_EQUAL_RETURN(guid, MFVideoFormat_DV50); //     FCC('dv50')
	IF_EQUAL_RETURN(guid, MFVideoFormat_DVH1); //     FCC('dvh1')
	IF_EQUAL_RETURN(guid, MFVideoFormat_DVSD); //     FCC('dvsd')
	IF_EQUAL_RETURN(guid, MFVideoFormat_DVSL); //     FCC('dvsl')
	IF_EQUAL_RETURN(guid, MFVideoFormat_H264); //     FCC('H264')
	IF_EQUAL_RETURN(guid, MFVideoFormat_I420); //     FCC('I420')
	IF_EQUAL_RETURN(guid, MFVideoFormat_IYUV); //     FCC('IYUV')
	IF_EQUAL_RETURN(guid, MFVideoFormat_M4S2); //     FCC('M4S2')
	IF_EQUAL_RETURN(guid, MFVideoFormat_MJPG);
	IF_EQUAL_RETURN(guid, MFVideoFormat_MP43); //     FCC('MP43')
	IF_EQUAL_RETURN(guid, MFVideoFormat_MP4S); //     FCC('MP4S')
	IF_EQUAL_RETURN(guid, MFVideoFormat_MP4V); //     FCC('MP4V')
	IF_EQUAL_RETURN(guid, MFVideoFormat_MPG1); //     FCC('MPG1')
	IF_EQUAL_RETURN(guid, MFVideoFormat_MSS1); //     FCC('MSS1')
	IF_EQUAL_RETURN(guid, MFVideoFormat_MSS2); //     FCC('MSS2')
	IF_EQUAL_RETURN(guid, MFVideoFormat_NV11); //     FCC('NV11')
	IF_EQUAL_RETURN(guid, MFVideoFormat_NV12); //     FCC('NV12')
	IF_EQUAL_RETURN(guid, MFVideoFormat_P010); //     FCC('P010')
	IF_EQUAL_RETURN(guid, MFVideoFormat_P016); //     FCC('P016')
	IF_EQUAL_RETURN(guid, MFVideoFormat_P210); //     FCC('P210')
	IF_EQUAL_RETURN(guid, MFVideoFormat_P216); //     FCC('P216')
	IF_EQUAL_RETURN(guid, MFVideoFormat_RGB24); //    D3DFMT_R8G8B8
	IF_EQUAL_RETURN(guid, MFVideoFormat_RGB32); //    D3DFMT_X8R8G8B8
	IF_EQUAL_RETURN(guid, MFVideoFormat_RGB555); //   D3DFMT_X1R5G5B5
	IF_EQUAL_RETURN(guid, MFVideoFormat_RGB565); //   D3DFMT_R5G6B5
	IF_EQUAL_RETURN(guid, MFVideoFormat_RGB8);
	IF_EQUAL_RETURN(guid, MFVideoFormat_UYVY); //     FCC('UYVY')
	IF_EQUAL_RETURN(guid, MFVideoFormat_v210); //     FCC('v210')
	IF_EQUAL_RETURN(guid, MFVideoFormat_v410); //     FCC('v410')
	IF_EQUAL_RETURN(guid, MFVideoFormat_WMV1); //     FCC('WMV1')
	IF_EQUAL_RETURN(guid, MFVideoFormat_WMV2); //     FCC('WMV2')
	IF_EQUAL_RETURN(guid, MFVideoFormat_WMV3); //     FCC('WMV3')
	IF_EQUAL_RETURN(guid, MFVideoFormat_WVC1); //     FCC('WVC1')
	IF_EQUAL_RETURN(guid, MFVideoFormat_Y210); //     FCC('Y210')
	IF_EQUAL_RETURN(guid, MFVideoFormat_Y216); //     FCC('Y216')
	IF_EQUAL_RETURN(guid, MFVideoFormat_Y410); //     FCC('Y410')
	IF_EQUAL_RETURN(guid, MFVideoFormat_Y416); //     FCC('Y416')
	IF_EQUAL_RETURN(guid, MFVideoFormat_Y41P);
	IF_EQUAL_RETURN(guid, MFVideoFormat_Y41T);
	IF_EQUAL_RETURN(guid, MFVideoFormat_YUY2); //     FCC('YUY2')
	IF_EQUAL_RETURN(guid, MFVideoFormat_YV12); //     FCC('YV12')
	IF_EQUAL_RETURN(guid, MFVideoFormat_YVYU);

	IF_EQUAL_RETURN(guid, MFAudioFormat_PCM); //              WAVE_FORMAT_PCM
	IF_EQUAL_RETURN(guid, MFAudioFormat_Float); //            WAVE_FORMAT_IEEE_FLOAT
	IF_EQUAL_RETURN(guid, MFAudioFormat_DTS); //              WAVE_FORMAT_DTS
	IF_EQUAL_RETURN(guid, MFAudioFormat_Dolby_AC3_SPDIF); //  WAVE_FORMAT_DOLBY_AC3_SPDIF
	IF_EQUAL_RETURN(guid, MFAudioFormat_DRM); //              WAVE_FORMAT_DRM
	IF_EQUAL_RETURN(guid, MFAudioFormat_WMAudioV8); //        WAVE_FORMAT_WMAUDIO2
	IF_EQUAL_RETURN(guid, MFAudioFormat_WMAudioV9); //        WAVE_FORMAT_WMAUDIO3
	IF_EQUAL_RETURN(guid, MFAudioFormat_WMAudio_Lossless); // WAVE_FORMAT_WMAUDIO_LOSSLESS
	IF_EQUAL_RETURN(guid, MFAudioFormat_WMASPDIF); //         WAVE_FORMAT_WMASPDIF
	IF_EQUAL_RETURN(guid, MFAudioFormat_MSP1); //             WAVE_FORMAT_WMAVOICE9
	IF_EQUAL_RETURN(guid, MFAudioFormat_MP3); //              WAVE_FORMAT_MPEGLAYER3
	IF_EQUAL_RETURN(guid, MFAudioFormat_MPEG); //             WAVE_FORMAT_MPEG
	IF_EQUAL_RETURN(guid, MFAudioFormat_AAC); //              WAVE_FORMAT_MPEG_HEAAC
	IF_EQUAL_RETURN(guid, MFAudioFormat_ADTS); //             WAVE_FORMAT_MPEG_ADTS_AAC

	return "Unknown";
}

//////////////////////////////////////////////////////////////////////////
// CameraFeedWindows - Subclass for our camera feed on windows

CameraFeedWindows::CameraFeedWindows(LPCWSTR camera_id, IMFMediaType *type, String name, int width, int height, GUID format) {
	this->camera_id = camera_id;
	this->name = name;
	this->width = width;
	this->height = height;
	this->type = type;
	this->format = format;
}

CameraFeedWindows::~CameraFeedWindows() {
	if (is_active()) {
		deactivate_feed();
	};

	SafeRelease(&type);
}

bool CameraFeedWindows::activate_feed() {
	IMFAttributes *pAttributes = NULL;
	HRESULT hr = MFCreateAttributes(&pAttributes, 2);
	if (FAILED(hr)) {
		goto done;
	}

	// Set the device type to video.
	hr = pAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
	if (FAILED(hr)) {
		goto done;
	}

	// Set the symbolic link.
	hr = pAttributes->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, camera_id);
	if (FAILED(hr)) {
		goto done;
	}

	// Create media source
	hr = MFCreateDeviceSource(pAttributes, &source);
	if (FAILED(hr)) {
		goto done;
	}

	// Get information about device
	IMFPresentationDescriptor *pPD;
	hr = source->CreatePresentationDescriptor(&pPD);
	if (FAILED(hr)) {
		goto done;
	}

	// Get information about video stream
	BOOL fSelected;
	IMFStreamDescriptor *pSD;
	hr = pPD->GetStreamDescriptorByIndex(0, &fSelected, &pSD);
	if (FAILED(hr)) {
		goto done;
	}

	// Get information about supported media types
	IMFMediaTypeHandler *pHandler;
	hr = pSD->GetMediaTypeHandler(&pHandler);
	if (FAILED(hr)) {
		goto done;
	}

	// Set media type
	hr = pHandler->SetCurrentMediaType(type);
	if (FAILED(hr)) {
		goto done;
	}

	// Create media reader
	hr = MFCreateSourceReaderFromMediaSource(source, NULL, &reader);
	if (FAILED(hr)) {
		goto done;
	}

	// Prepare images and textures
	if (format == MFVideoFormat_RGB24) {
		set_image(RenderingServer::CANVAS_TEXTURE_CHANNEL_DIFFUSE,
				Image::create_empty(width, height, false, Image::FORMAT_RGB8));
	}

	if (format == MFVideoFormat_NV12) {
		set_image(RenderingServer::CANVAS_TEXTURE_CHANNEL_DIFFUSE,
				Image::create_empty(width, height, false, Image::FORMAT_R8));

		set_image(RenderingServer::CANVAS_TEXTURE_CHANNEL_NORMAL,
				Image::create_empty(width / 2, height / 2, false, Image::FORMAT_RG8));
	}

	// Start reading
	worker = memnew(std::thread(capture, this));

done:
	SafeRelease(&pAttributes);
	SafeRelease(&pPD);
	SafeRelease(&pSD);
	SafeRelease(&pHandler);

	if FAILED (hr) {
		print_error(vformat("Unable to activate camera feed (%d)", hr));
		return false;
	}
	return true;
}

void CameraFeedWindows::deactivate_feed() {
	if (worker != NULL) {
		active = false;
		worker->join();
		memdelete(worker);
		worker = NULL;
	}

	SafeRelease(&reader);
	SafeRelease(&source);
}

void CameraFeedWindows::capture(CameraFeedWindows *feed) {
	print_verbose("Camera feed is now streaming");
	feed->active = true;
	while (feed->active) {
		feed->read();
		Sleep(100);
	}
}

void CameraFeedWindows::read() {
	HRESULT hr = S_OK;
	IMFSample *pSample = NULL;
	BYTE *data;
	DWORD streamIndex, flags, len;
	LONGLONG llTimeStamp;
	IMFMediaBuffer *buffer;

	hr = reader->ReadSample(
			MF_SOURCE_READER_FIRST_VIDEO_STREAM, // Stream index.
			0, // Flags.
			&streamIndex, // Receives the actual stream index.
			&flags, // Receives status flags.
			&llTimeStamp, // Receives the time stamp.
			&pSample // Receives the sample or NULL.
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
	if (flags & MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED) {
		print_verbose("\tOutput format changed");
	}

	// Process sample
	if (pSample) {
		hr = pSample->GetBufferByIndex(0, &buffer);
		if (FAILED(hr)) {
			return;
		}

		// Get image buffer
		buffer->Lock(&data, NULL, &len);

		// Get RGB or Y plane
		Ref<Image> yImage = get_image(RenderingServer::CANVAS_TEXTURE_CHANNEL_DIFFUSE);
		set_image(RenderingServer::CANVAS_TEXTURE_CHANNEL_DIFFUSE, data, 0, yImage->get_data().size());

		// Get UV plane
		if (format == MFVideoFormat_NV12) {
			Ref<Image> uvImage = get_image(RenderingServer::CANVAS_TEXTURE_CHANNEL_NORMAL);
			set_image(RenderingServer::CANVAS_TEXTURE_CHANNEL_NORMAL, data, yImage->get_data().size(), uvImage->get_data().size());
		}

		buffer->Unlock();
		buffer->Release();
		pSample->Release();
	}
}

//////////////////////////////////////////////////////////////////////////
// CameraWindows - Subclass for our camera server on windows

void CameraWindows::update_feeds() {
	// remove existing devices
	for (int i = feeds.size() - 1; i >= 0; i--) {
		Ref<CameraFeedWindows> feed = (Ref<CameraFeedWindows>)feeds[i];
		remove_feed(feed);
	};

	// Create an attribute store to hold the search criteria.
	IMFAttributes *pConfig = NULL;
	HRESULT hr = MFCreateAttributes(&pConfig, 1);
	if (FAILED(hr)) {
		goto done_all;
	}

	// Request video capture devices.
	hr = pConfig->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
	if (FAILED(hr)) {
		goto done_all;
	}

	// Process devices
	UINT32 count = 0;
	IMFActivate **ppDevices = NULL;
	hr = MFEnumDeviceSources(pConfig, &ppDevices, &count);
	if (FAILED(hr)) {
		goto done_all;
	}

	// Create feeds for all supported media sources
	for (DWORD i = 0; i < count; i++) {
		IMFActivate *pDevice = ppDevices[i];

		// Get camera id
		WCHAR *szCameraID = NULL;
		UINT32 len;
		hr = pDevice->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &szCameraID, &len);
		if (FAILED(hr)) {
			goto done_device;
		}

		// Get name
		WCHAR *szFriendlyName = NULL;
		hr = pDevice->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &szFriendlyName, &len);
		if (FAILED(hr)) {
			goto done_device;
		}

		// Get media source
		IMFMediaSource *pSource = NULL;
		hr = pDevice->ActivateObject(IID_PPV_ARGS(&pSource));
		if (FAILED(hr)) {
			goto done_device;
		}

		// Get information about device
		IMFPresentationDescriptor *pPD = NULL;
		hr = pSource->CreatePresentationDescriptor(&pPD);
		if (FAILED(hr)) {
			goto done_device;
		}

		// Get information about video stream
		BOOL fSelected;
		IMFStreamDescriptor *pSD = NULL;
		hr = pPD->GetStreamDescriptorByIndex(0, &fSelected, &pSD);
		if (FAILED(hr)) {
			goto done_device;
		}

		// Get information about supported media types
		IMFMediaTypeHandler *pHandler = NULL;
		hr = pSD->GetMediaTypeHandler(&pHandler);
		if (FAILED(hr)) {
			goto done_device;
		}

		// Get supported media types
		DWORD cTypes = 0;
		hr = pHandler->GetMediaTypeCount(&cTypes);
		if (FAILED(hr)) {
			goto done_device;
		}

		for (DWORD i = 0; i < cTypes; i++) {
			// Get media type
			IMFMediaType *pType = NULL;
			hr = pHandler->GetMediaTypeByIndex(i, &pType);
			if (FAILED(hr)) {
				SafeRelease(&pType);
				break;
			}

			// Get subtype
			GUID subType;
			hr = pType->GetGUID(MF_MT_SUBTYPE, &subType);
			if (FAILED(hr)) {
				SafeRelease(&pType);
				break;
			}

			// Get image size
			UINT32 width, height = 0;
			hr = MFGetAttributeSize(pType, MF_MT_FRAME_SIZE, &width, &height);
			if (FAILED(hr)) {
				SafeRelease(&pType);
				break;
			}

			// Add feed for supported formats
			if (subType == MFVideoFormat_RGB24 || subType == MFVideoFormat_NV12) {
				String format = GetGUIDNameConst(subType);
				format = format.replace("MFVideoFormat_", "");
				String name = szFriendlyName + vformat(" (%d x %d, %s)", width, height, format);
				Ref<CameraFeedWindows> feed = new CameraFeedWindows(szCameraID, pType, name, width, height, subType);
				add_feed(feed);

				print_line("Added camera feed: ", name);
			}
		}

	done_device:
		SafeRelease(&pPD);
		SafeRelease(&pSD);
		SafeRelease(&pHandler);
		SafeRelease(&pSource);
		SafeRelease(&pDevice);
	}

done_all:
	SafeRelease(&pConfig);

	if (FAILED(hr)) {
		print_error(vformat("Error updating feeds (%d)", hr));
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
