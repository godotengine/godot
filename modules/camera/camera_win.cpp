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

String GetGUIDNameConst(const GUID &guid);
String GetGUIDName(const GUID &guid);
String GetGUIDName(const GUID &guid);
String LogAttributeValueByIndex(IMFAttributes *pAttr, DWORD index);
String SpecialCaseAttributeValue(GUID guid, const PROPVARIANT &var);

String LogMediaType(IMFMediaType *pType) {
	String retval = "Media format:\n";
	UINT32 count = 0;
	HRESULT hr = pType->GetCount(&count);
	if (FAILED(hr) || count == 0) {
		retval += "No attributes";
	} else {
		for (UINT32 i = 0; i < count; i++) {
			retval += LogAttributeValueByIndex(pType, i) + "\n";
		}
	}
	return retval;
}

String LogAttributeValueByIndex(IMFAttributes *pAttr, DWORD index) {
	GUID guid = { 0 };
	PROPVARIANT var;
	PropVariantInit(&var);

	HRESULT hr = pAttr->GetItemByIndex(index, &guid, &var);
	if (FAILED(hr)) {
		return "Unknown";
	}
	String retval = vformat("\t%s = ", GetGUIDName(guid));

	String scCase = SpecialCaseAttributeValue(guid, var);
	if (!scCase.is_empty()) {
		retval += scCase;
	} else {
		switch (var.vt) {
			case VT_UI4:
			case VT_UI8:
				retval += itos(var.ulVal);
				break;

			case VT_R8:
				retval += rtos(var.dblVal);
				break;

			case VT_CLSID:
				retval += GetGUIDName(*var.puuid);
				break;

			case VT_LPWSTR:
				retval += var.pwszVal;
				break;

			case VT_VECTOR | VT_UI1:
				retval += "<<byte array>>";
				break;

			case VT_UNKNOWN:
				retval += "IUnknown";
				break;

			default:
				retval += vformat("Unexpected attribute type (vt = %d)", var.vt);
				break;
		}
	}
	return retval;
}

String GetGUIDName(const GUID &guid) {
	return GetGUIDNameConst(guid);
}

String LogUINT32AsUINT64(const PROPVARIANT &var) {
	UINT32 uHigh = 0, uLow = 0;
	Unpack2UINT32AsUINT64(var.uhVal.QuadPart, &uHigh, &uLow);
	return vformat("%d x %d", uHigh, uLow);
}

float OffsetToFloat(const MFOffset &offset) {
	return offset.value + (static_cast<float>(offset.fract) / 65536.0f);
}

String LogVideoArea(const PROPVARIANT &var) {
	if (var.caub.cElems < sizeof(MFVideoArea)) {
		return "unknown";
	}
	MFVideoArea *pArea = (MFVideoArea *)var.caub.pElems;
	return vformat("(%f,%f) (%d,%d)", OffsetToFloat(pArea->OffsetX), OffsetToFloat(pArea->OffsetY),
			pArea->Area.cx, pArea->Area.cy);
}

// Handle certain known special cases.
String SpecialCaseAttributeValue(GUID guid, const PROPVARIANT &var) {
	if ((guid == MF_MT_FRAME_RATE) || (guid == MF_MT_FRAME_RATE_RANGE_MAX) ||
			(guid == MF_MT_FRAME_RATE_RANGE_MIN) || (guid == MF_MT_FRAME_SIZE) ||
			(guid == MF_MT_PIXEL_ASPECT_RATIO)) {
		// Attributes that contain two packed 32-bit values.
		return LogUINT32AsUINT64(var);
	} else if ((guid == MF_MT_GEOMETRIC_APERTURE) ||
			(guid == MF_MT_MINIMUM_DISPLAY_APERTURE) ||
			(guid == MF_MT_PAN_SCAN_APERTURE)) {
		// Attributes that an MFVideoArea structure.
		return LogVideoArea(var);
	}
	return "";
}

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

CameraFeedWindows::CameraFeedWindows(IMFActivate *device) {
	UINT32 len;

	// Set camera id
	WCHAR *szCameraID = NULL;
	HRESULT hr = device->GetAllocatedString(
			MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
			&szCameraID, &len);
	if (SUCCEEDED(hr)) {
		camera_id = szCameraID;
	}

	// Set name
	WCHAR *szFriendlyName = NULL;
	hr = device->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &szFriendlyName, &len);
	if (SUCCEEDED(hr)) {
		name = szFriendlyName;
	}

	// Set transform
	transform = Transform2D(-1.0, 0.0, 0.0, -1.0, 1.0, 1.0);
}

CameraFeedWindows::~CameraFeedWindows() {
	if (is_active()) {
		deactivate_feed();
	};
}

HRESULT CameraFeedWindows::init() {
	IMFAttributes *pAttributes = NULL;
	HRESULT hr = MFCreateAttributes(&pAttributes, 2);

	// Set the device type to video.
	if (SUCCEEDED(hr)) {
		hr = pAttributes->SetGUID(
				MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
				MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
	}

	// Set the symbolic link.
	if (SUCCEEDED(hr)) {
		hr = pAttributes->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, camera_id);
	}

	if (SUCCEEDED(hr)) {
		hr = MFCreateDeviceSource(pAttributes, &source);
	}

	SafeRelease(&pAttributes);
	return hr;
}

void CameraFeedWindows::set_format(int p_format) {
	IMFPresentationDescriptor *pPD = NULL;
	IMFStreamDescriptor *pSD = NULL;
	IMFMediaTypeHandler *pHandler = NULL;
	IMFMediaType *pType = NULL;

	HRESULT hr = init();
	if (FAILED(hr)) {
		goto done;
	}

	hr = source->CreatePresentationDescriptor(&pPD);
	if (FAILED(hr)) {
		goto done;
	}

	BOOL fSelected;
	hr = pPD->GetStreamDescriptorByIndex(0, &fSelected, &pSD);
	if (FAILED(hr)) {
		goto done;
	}

	hr = pSD->GetMediaTypeHandler(&pHandler);
	if (FAILED(hr)) {
		goto done;
	}

	hr = pHandler->GetMediaTypeByIndex(p_format, &pType);
	if (FAILED(hr)) {
		print_error("Invalid media format");
		goto done;
	}

	// Get subtype
	GUID subType;
	hr = pType->GetGUID(MF_MT_SUBTYPE, &subType);
	if (FAILED(hr)) {
		print_error("Invalid media format");
		goto done;
	}

	// Get image size
	hr = MFGetAttributeSize(pType, MF_MT_FRAME_SIZE, &base_width, &base_height);
	if (FAILED(hr)) {
		print_error("Unable to retrieve frame size");
		goto done;
	}
	
	// Prepare images
	diffuse = Image::create_empty(base_width, base_height, false, Image::FORMAT_R8);
	normal = Image::create_empty(base_width / 2, base_height / 2, false, Image::FORMAT_RG8);
	set_texture(diffuse, normal);

	// Select media type
	hr = pHandler->SetCurrentMediaType(pType);

	// Feedback
	print_line(LogMediaType(pType));
done:
	SafeRelease(&pPD);
	SafeRelease(&pSD);
	SafeRelease(&pHandler);
	SafeRelease(&pType);
	SafeRelease(&source);
}

bool CameraFeedWindows::activate_feed() {
	HRESULT hr = init();
	if (FAILED(hr)) {
		goto error;
	}

	hr = MFCreateSourceReaderFromMediaSource(source, NULL, &reader);
	if (SUCCEEDED(hr)) {
		worker = memnew(std::thread(capture, this));
		return true;
	}

error:
	print_error("Unable to activate camera feed");
	return false;
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
	while (feed->is_active()) {
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

		// Set Y layer
		Vector<uint8_t> dataY = diffuse->get_data();
		uint8_t *inY = dataY.ptrw();
		CopyMemory(inY, data, dataY.size());
		diffuse->set_data(diffuse->get_width(), diffuse->get_height(), false, diffuse->get_format(), dataY);
		RenderingServer::get_singleton()->texture_2d_update(diffuse_texture, diffuse);

		// Set UV layer
		Vector<uint8_t> dataUV = normal->get_data();
		uint8_t *inUV = dataUV.ptrw();
		CopyMemory(inUV, data + dataY.size(), dataUV.size());
		normal->set_data(normal->get_width(), normal->get_height(), false, normal->get_format(), dataUV);
		RenderingServer::get_singleton()->texture_2d_update(normal_texture, normal);
		
		buffer->Unlock();
		buffer->Release();
		pSample->Release();
	}
}

//////////////////////////////////////////////////////////////////////////
// CameraWindows - Subclass for our camera server on windows

void CameraWindows::add_active_cameras() {
	UINT32 count = 0;
	IMFAttributes *pConfig = NULL;
	IMFActivate **ppDevices = NULL;

	// remove existing devices
	for (int i = feeds.size() - 1; i >= 0; i--) {
		Ref<CameraFeedWindows> feed = (Ref<CameraFeedWindows>)feeds[i];
		remove_feed(feed);
	};

	// Create an attribute store to hold the search criteria.
	HRESULT hr = MFCreateAttributes(&pConfig, 1);

	// Request video capture devices.
	if (SUCCEEDED(hr)) {
		hr = pConfig->SetGUID(
				MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
				MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
	}

	// Enumerate the devices,
	if (SUCCEEDED(hr)) {
		hr = MFEnumDeviceSources(pConfig, &ppDevices, &count);
	}

	if (FAILED(hr)) {
		print_error("Error detecting device cameras");
		return;
	}

	if (count == 0) {
		print_line("No device cameras available");
		return;
	}

	// Store cameras
	for (DWORD i = 0; i < count; i++) {
		Ref<CameraFeedWindows> newfeed = new CameraFeedWindows(ppDevices[i]);
		add_feed(newfeed);
		ppDevices[i]->Release();
	};
	CoTaskMemFree(ppDevices);
}

CameraWindows::CameraWindows() {
	// Initialize the Media Foundation platform.
	HRESULT hr = MFStartup(MF_VERSION);
	if (FAILED(hr)) {
		print_error("Unable to initialize Media Foundation platform");
		return;
	}

	// Find cameras active right now
	add_active_cameras();

	// need to add something that will react to devices being connected/removed...
}

CameraWindows::~CameraWindows() {
	MFShutdown();
}
