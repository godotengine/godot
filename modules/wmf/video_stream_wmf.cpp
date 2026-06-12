/**************************************************************************/
/*  video_stream_wmf.cpp                                                  */
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

#include "video_stream_wmf.h"

#include "core/config/project_settings.h"
#include "core/error/error_list.h"
#include "core/error/error_macros.h"
#include "core/io/file_access.h"
#include "core/object/object.h"
#include "sample_grabber_callback.h"
#include "scene/resources/image_texture.h"
#include <mfapi.h>
#include <mferror.h>
#include <mftransform.h>
#include <shlwapi.h>
#include <wmcodecdsp.h>

#ifdef __MINGW32__
#include <initguid.h>

#if defined(__GNUC__) && __GNUC__ < 13
DEFINE_GUID(CLSID_VideoProcessorMFT, 0x88753808, 0x311a, 0x4a01, 0x83, 0xd0, 0x4, 0x58, 0xb2, 0xc4, 0x25, 0xf);
#endif // __GNUC__ < 12

#ifndef IID_IMFVideoProcessorControl
DEFINE_GUID(IID_IMFVideoProcessorControl, 0xA3F675D7, 0x2744, 0x46dd, 0x95, 0x67, 0x94, 0xBB, 0x04, 0x20, 0x6F, 0x4C);
#endif
#endif // __MINGW32__

#define CHECK_HR(func)                                                           \
	if (SUCCEEDED(hr)) {                                                         \
		hr = (func);                                                             \
		if (FAILED(hr)) {                                                        \
			print_line(vformat("%s failed, return:%s", __FUNCTION__, itos(hr))); \
		}                                                                        \
	}
#define SafeRelease(p)      \
	{                       \
		if (p) {            \
			(p)->Release(); \
			(p) = nullptr;  \
		}                   \
	}

HRESULT AddSourceNode(IMFTopology *pTopology, IMFMediaSource *pSource,
		IMFPresentationDescriptor *pPD, IMFStreamDescriptor *pSD,
		IMFTopologyNode **ppNode) {
	IMFTopologyNode *pNode = NULL;

	HRESULT hr = S_OK;
	CHECK_HR(MFCreateTopologyNode(MF_TOPOLOGY_SOURCESTREAM_NODE, &pNode));
	CHECK_HR(pNode->SetUnknown(MF_TOPONODE_SOURCE, pSource));
	CHECK_HR(pNode->SetUnknown(MF_TOPONODE_PRESENTATION_DESCRIPTOR, pPD));
	CHECK_HR(pNode->SetUnknown(MF_TOPONODE_STREAM_DESCRIPTOR, pSD));
	CHECK_HR(pTopology->AddNode(pNode));

	if (SUCCEEDED(hr)) {
		*ppNode = pNode;
		(*ppNode)->AddRef();
	}
	SafeRelease(pNode);
	return hr;
}

HRESULT AddOutputNode(IMFTopology *pTopology, // Topology.
		IMFActivate *pActivate, // Media sink activation object.
		DWORD dwId, // Identifier of the stream sink.
		IMFTopologyNode **ppNode) // Receives the node pointer.
{
	IMFTopologyNode *pNode = NULL;

	HRESULT hr = S_OK;
	CHECK_HR(MFCreateTopologyNode(MF_TOPOLOGY_OUTPUT_NODE, &pNode));
	CHECK_HR(pNode->SetObject(pActivate));
	CHECK_HR(pNode->SetUINT32(MF_TOPONODE_STREAMID, dwId));
	CHECK_HR(pNode->SetUINT32(MF_TOPONODE_NOSHUTDOWN_ON_REMOVE, FALSE));
	CHECK_HR(pTopology->AddNode(pNode));

	// Return the pointer to the caller.
	if (SUCCEEDED(hr)) {
		*ppNode = pNode;
		(*ppNode)->AddRef();
	}

	SafeRelease(pNode);
	return hr;
}

HRESULT AddColourConversionNode(IMFTopology *pTopology,
		IMFMediaType *inputType,
		IMFTransform **ppColorTransform) {
	HRESULT hr = S_OK;

	IMFTransform *colorTransform = nullptr;
	CHECK_HR(CoCreateInstance(CLSID_VideoProcessorMFT, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&colorTransform)));

	UINT32 uWidth = 0, uHeight = 0;
	MFGetAttributeSize(inputType, MF_MT_FRAME_SIZE, &uWidth, &uHeight);
	print_line("Video Processor MFT setup for: " + itos(uWidth) + "x" + itos(uHeight));

	IMFMediaType *pInType = nullptr;
	CHECK_HR(MFCreateMediaType(&pInType));
	CHECK_HR(pInType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video));
	CHECK_HR(pInType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12));
	CHECK_HR(MFSetAttributeSize(pInType, MF_MT_FRAME_SIZE, uWidth, uHeight));
	CHECK_HR(pInType->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive));

	hr = colorTransform->SetInputType(0, pInType, 0);
	if (FAILED(hr)) {
		print_line("Failed to set input type: " + itos(hr));
		SafeRelease(pInType);
		SafeRelease(colorTransform);
		return hr;
	}

	IMFMediaType *pOutType = nullptr;
	CHECK_HR(MFCreateMediaType(&pOutType));
	CHECK_HR(pOutType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video));

	hr = pOutType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_ARGB32);

	CHECK_HR(MFSetAttributeSize(pOutType, MF_MT_FRAME_SIZE, uWidth, uHeight));
	CHECK_HR(pOutType->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive));

	// Set positive stride for top-down output (standard for RGB formats)
	LONG outStride = (LONG)(uWidth * 4); // 4 bytes per pixel for ARGB32/RGB32
	CHECK_HR(pOutType->SetUINT32(MF_MT_DEFAULT_STRIDE, (UINT32)outStride));

	hr = colorTransform->SetOutputType(0, pOutType, 0);
	if (FAILED(hr)) {
		print_line("Failed to set output type: " + itos(hr));
		SafeRelease(pInType);
		SafeRelease(pOutType);
		SafeRelease(colorTransform);
		return hr;
	}

	print_line("Video Processor MFT setup successful");
	*ppColorTransform = colorTransform;

	SafeRelease(pInType);
	SafeRelease(pOutType);
	return S_OK;
}

HRESULT CreateSampleGrabber(UINT width, UINT height, SampleGrabberCallback *pSampleGrabber, IMFActivate **pSinkActivate) {
	HRESULT hr = S_OK;
	IMFMediaType *pType = NULL;

	CHECK_HR(MFCreateMediaType(&pType));
	CHECK_HR(pType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video));
	CHECK_HR(pType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12));
	CHECK_HR(MFSetAttributeSize(pType, MF_MT_FRAME_SIZE, width, height));
	CHECK_HR(pType->SetUINT32(MF_MT_FIXED_SIZE_SAMPLES, TRUE));
	CHECK_HR(pType->SetUINT32(MF_MT_ALL_SAMPLES_INDEPENDENT, TRUE));
	CHECK_HR(MFSetAttributeRatio(pType, MF_MT_PIXEL_ASPECT_RATIO, 1, 1));
	CHECK_HR(MFCreateSampleGrabberSinkActivate(pType, pSampleGrabber, pSinkActivate));

	SafeRelease(pType);
	return hr;
}

HRESULT CreateTopology(IMFMediaSource *pSource, SampleGrabberCallback *pSampleGrabber, IMFTopology **ppTopo, VideoStreamPlaybackWMF::StreamInfo *info) {
	IMFTopology *pTopology = NULL;
	IMFPresentationDescriptor *pPD = NULL;
	IMFStreamDescriptor *pSD = NULL;
	IMFMediaTypeHandler *pHandler = NULL;
	IMFTopologyNode *inputNode = NULL;
	IMFTopologyNode *outputNode = NULL;
	IMFTopologyNode *inputNodeAudio = NULL;
	IMFTopologyNode *outputNodeAudio = NULL;
	IMFActivate *audioActivate = NULL;

	HRESULT hr = S_OK;

	CHECK_HR(MFCreateTopology(&pTopology));
	CHECK_HR(pSource->CreatePresentationDescriptor(&pPD));

	DWORD cStreams = 0;
	CHECK_HR(pPD->GetStreamDescriptorCount(&cStreams));

	print_line(itos(cStreams) + " streams");

	for (DWORD i = 0; i < cStreams; i++) {
		BOOL bSelected = FALSE;
		GUID majorType;

		CHECK_HR(pPD->GetStreamDescriptorByIndex(i, &bSelected, &pSD));
		CHECK_HR(pSD->GetMediaTypeHandler(&pHandler));
		CHECK_HR(pHandler->GetMajorType(&majorType));

		if (majorType == MFMediaType_Video && bSelected) {
			print_line("Video Stream");

			IMFMediaType *pType = NULL;
			CHECK_HR(pHandler->GetMediaTypeByIndex(0, &pType));
			UINT32 width = 0, height = 0;
			MFGetAttributeSize(pType, MF_MT_FRAME_SIZE, &width, &height);

			IMFActivate *pSinkActivate = NULL;
			CHECK_HR(CreateSampleGrabber(width, height, pSampleGrabber, &pSinkActivate));
			IMFTransform *pColorTransform = NULL;
			CHECK_HR(AddColourConversionNode(pTopology, pType, &pColorTransform));
			pSampleGrabber->set_color_transform(pColorTransform);

			CHECK_HR(AddSourceNode(pTopology, pSource, pPD, pSD, &inputNode));
			CHECK_HR(AddOutputNode(pTopology, pSinkActivate, 0, &outputNode));

			CHECK_HR(inputNode->ConnectOutput(0, outputNode, 0));

			info->size.x = width;
			info->size.y = height;
			print_line("Width & Height " + itos(width) + "x" + itos(height));

			UINT32 numerator = 0, denominator = 1;
			MFGetAttributeRatio(pType, MF_MT_FRAME_RATE, &numerator, &denominator);
			if (denominator > 0) {
				info->fps = (float)numerator / (float)denominator;
			} else {
				info->fps = 30.0f; // Default fallback
			}
			print_line("Frame Rate: " + itos(numerator) + "/" + itos(denominator) + " = " + rtos(info->fps) + " fps");

			UINT64 duration;
			pPD->GetUINT64(MF_PD_DURATION, &duration);
			info->duration = duration / 10000000.f;
			print_line("Duration: " + rtos(info->duration) + " secs");

			break;
		} else if (majorType == MFMediaType_Audio && bSelected) {
			CHECK_HR(MFCreateAudioRendererActivate(&audioActivate));
			CHECK_HR(AddSourceNode(pTopology, pSource, pPD, pSD, &inputNodeAudio));
			CHECK_HR(AddOutputNode(pTopology, audioActivate, 0, &outputNodeAudio));
			CHECK_HR(inputNodeAudio->ConnectOutput(0, outputNodeAudio, 0));
		} else {
			print_line("Stream deselected");
			CHECK_HR(pPD->DeselectStream(i));
		}
		SafeRelease(pSD);
		SafeRelease(pHandler);
	}

	if (SUCCEEDED(hr)) {
		*ppTopo = pTopology;
		(*ppTopo)->AddRef();
	}
	SafeRelease(pTopology);
	SafeRelease(inputNode);
	SafeRelease(outputNode);
	SafeRelease(pPD);
	SafeRelease(pHandler);
	SafeRelease(audioActivate);
	return hr;
}

HRESULT CreateMediaSource(const String &p_file, IMFMediaSource **pMediaSource) {
	ERR_FAIL_COND_V(p_file.is_empty(), E_FAIL);

	IMFSourceResolver *pSourceResolver = nullptr;
	IUnknown *pSource = nullptr;

	// Create the source resolver.
	HRESULT hr = S_OK;
	CHECK_HR(MFCreateSourceResolver(&pSourceResolver));

	print_line("Original File:" + p_file);

	String file_path = p_file;

	// Handle Godot resource paths
	if (p_file.begins_with("res://")) {
		// Convert resource path to global path
		file_path = ProjectSettings::get_singleton()->globalize_path(p_file);
		print_line("Globalized Path: " + file_path);
	}

	// Verify the file exists
	Error e;
	Ref<FileAccess> fa = FileAccess::open(file_path, FileAccess::READ, &e);
	if (e != OK) {
		print_line("Failed to open file: " + file_path + " Error: " + itos(e));
		SafeRelease(pSourceResolver);
		return E_FAIL;
	}
	fa.unref();

	// Convert to Windows path format if needed
	file_path = file_path.replace("/", "\\");
	print_line("Final Path: " + file_path);

	MF_OBJECT_TYPE ObjectType;
	CHECK_HR(pSourceResolver->CreateObjectFromURL((LPCWSTR)file_path.utf16().ptrw(),
			MF_RESOLUTION_MEDIASOURCE, nullptr, &ObjectType, &pSource));
	CHECK_HR(pSource->QueryInterface(IID_PPV_ARGS(pMediaSource)));

	SafeRelease(pSourceResolver);
	SafeRelease(pSource);

	return hr;
}

void VideoStreamPlaybackWMF::shutdown_stream() {
	if (media_source) {
		media_source->Stop();
		media_source->Shutdown();
	}
	if (media_session) {
		media_session->Stop();
		media_session->Shutdown();
	}

	SafeRelease(topology);
	SafeRelease(media_source);
	SafeRelease(media_session);
	SafeRelease(presentation_clock);
	//SafeRelease(sample_grabber_callback);

	is_video_playing = false;
	is_video_paused = false;
	is_video_seekable = false;

	stream_info.size = Point2i(0, 0);
	stream_info.fps = 0;
	stream_info.duration = 0;
}

void VideoStreamPlaybackWMF::play() {
	if (is_video_playing) {
		return;
	}

	if (media_session) {
		HRESULT hr = S_OK;

		PROPVARIANT var;
		PropVariantInit(&var);
		CHECK_HR(media_session->Start(&GUID_NULL, &var));

		if (SUCCEEDED(hr)) {
			is_video_playing = true;
		}
	}
}

void VideoStreamPlaybackWMF::stop() {
	if (media_session) {
		HRESULT hr = S_OK;
		CHECK_HR(media_session->Stop());

		if (SUCCEEDED(hr)) {
			is_video_playing = false;
		}
	}
}

bool VideoStreamPlaybackWMF::is_playing() const {
	return is_video_playing;
}

void VideoStreamPlaybackWMF::set_paused(bool p_paused) {
	print_line(String(__FUNCTION__) + ": " + itos(p_paused));
	is_video_paused = p_paused;

	if (media_session) {
		HRESULT hr = S_OK;
		if (p_paused) {
			CHECK_HR(media_session->Pause());
		} else {
			PROPVARIANT var;
			PropVariantInit(&var);
			CHECK_HR(media_session->Start(&GUID_NULL, &var));
		}
	}
}

bool VideoStreamPlaybackWMF::is_paused() const {
	return is_video_paused;
}

double VideoStreamPlaybackWMF::get_length() const {
	return stream_info.duration;
}

String VideoStreamPlaybackWMF::get_stream_name() const {
	return String("A video file");
}

int VideoStreamPlaybackWMF::get_loop_count() const {
	return 0;
}

double VideoStreamPlaybackWMF::get_playback_position() const {
	return time;
}

void VideoStreamPlaybackWMF::seek(double p_time) {
	print_line(String(__FUNCTION__) + ": " + rtos(p_time));

	time = p_time;

	if (media_session) {
		double wmf_time = p_time * 10000000.0;

		HRESULT hr = S_OK;
		PROPVARIANT varStart;
		varStart.vt = VT_I8;
		varStart.hVal.QuadPart = (MFTIME)wmf_time;
		CHECK_HR(media_session->Start(NULL, &varStart));

		if (is_video_paused) {
			media_session->Pause();
		}
	}
}

void VideoStreamPlaybackWMF::set_file(const String &p_file) {
	ERR_FAIL_COND(p_file.is_empty());

	shutdown_stream();

	HRESULT hr = S_OK;

	CHECK_HR(CreateMediaSource(p_file, &media_source));
	CHECK_HR(MFCreateMediaSession(nullptr, &media_session));

	CHECK_HR(SampleGrabberCallback::CreateInstance(&sample_grabber_callback, this, mtx));
	CHECK_HR(CreateTopology(media_source, sample_grabber_callback, &topology, &stream_info));

	CHECK_HR(media_session->SetTopology(0, topology));

	if (SUCCEEDED(hr)) {
		IMFRateControl *m_pRate;
		HRESULT hrRate = MFGetService(media_session, MF_RATE_CONTROL_SERVICE, IID_PPV_ARGS(&m_pRate));

		if (SUCCEEDED(hrRate)) {
			BOOL bThin = false;
			float fRate = 0.f;
			CHECK_HR(m_pRate->GetRate(&bThin, &fRate));
			//print_line("Thin = " + itos(bThin) + ", Playback Rate:" + rtos(fRate));
		}

		DWORD caps = 0;
		CHECK_HR(media_session->GetSessionCapabilities(&caps));
		if ((caps & MFSESSIONCAP_SEEK) != 0) {
			is_video_seekable = true;
		}

		IMFClock *clock;
		if (SUCCEEDED(media_session->GetClock(&clock))) {
			CHECK_HR(clock->QueryInterface(IID_PPV_ARGS(&presentation_clock)));
		}

		sample_grabber_callback->set_frame_size(stream_info.size.x, stream_info.size.y);

		const int rgba32_frame_size = stream_info.size.x * stream_info.size.y * 4; // RGB32 = 4 bytes per pixel (RGBA)
		cache_frames.resize(24);
		for (int i = 0; i < cache_frames.size(); ++i) {
			cache_frames.write[i].data.resize(rgba32_frame_size);
		}
		read_frame_idx = write_frame_idx = 0;
		Ref<Image> img = Image::create_empty(stream_info.size.x, stream_info.size.y, false, Image::FORMAT_RGBA8);
		texture->set_image(img);
	} else {
		SafeRelease(media_session);
	}
}

Ref<Texture2D> VideoStreamPlaybackWMF::get_texture() const {
	return texture;
}

void VideoStreamPlaybackWMF::update(double p_delta) {
	if (!is_video_playing || is_video_paused) {
		return;
	}

	time += p_delta;
	double current_time = time;

	if (media_session) {
		HRESULT hr = S_OK;
		HRESULT hrStatus = S_OK;
		MediaEventType met = 0;
		IMFMediaEvent *event = nullptr;

		hr = media_session->GetEvent(MF_EVENT_FLAG_NO_WAIT, &event);
		if (SUCCEEDED(hr)) {
			hr = event->GetStatus(&hrStatus);
			if (SUCCEEDED(hr)) {
				hr = event->GetType(&met);
				if (SUCCEEDED(hr)) {
					if (met == MESessionEnded) {
						// We're done playing
						media_session->Stop();
						is_video_playing = false;
						SafeRelease(event);
						return;
					}
				}
			}
		}
		SafeRelease(event);

		// Check if we have frames available and if it's time to display the next frame
		if (read_frame_idx != write_frame_idx) {
			mtx.lock();
			FrameData &the_frame = cache_frames.write[read_frame_idx];
			double frame_time = the_frame.sample_time / 10000000.0; // Convert from 100ns units to seconds
			mtx.unlock();

			// If it's time to display this frame
			if (current_time >= frame_time) {
				present();
			}
		}
	}
}

void VideoStreamPlaybackWMF::set_mix_callback(AudioMixCallback p_callback, void *p_userdata) {
	mix_callback = p_callback;
	mix_udata = p_userdata;
}

int VideoStreamPlaybackWMF::get_channels() const {
	return audio_channels;
}

int VideoStreamPlaybackWMF::get_mix_rate() const {
	return audio_sample_rate;
}

void VideoStreamPlaybackWMF::set_audio_track(int p_idx) {
	//print_line(__FUNCTION__ ": " + itos(p_idx));
}

FrameData *VideoStreamPlaybackWMF::get_next_writable_frame() {
	return &cache_frames.write[write_frame_idx];
}

void VideoStreamPlaybackWMF::write_frame_done() {
	MutexLock lock(mtx);
	int next_write_frame_idx = (write_frame_idx + 1) % cache_frames.size();

	// TODO: just ignore the buffer full case for now because sometimes one Player may hit this if forever
	// claiming all memory eventually...
	if (read_frame_idx == next_write_frame_idx) {
		//print_line(itos(id) + " Chase up! W:" + itos(write_frame_idx) + " R:" + itos(read_frame_idx) + " Size:" + itos(cache_frames.size()));
		// the time gap between videos is larger than the buffer size
		// need to extend the buffer size

		/*
		int current_size = cache_frames.size();
		cache_frames.resize(current_size + 10);

		const int rgb24_frame_size = stream_info.size.x * stream_info.size.y * 3;
		for (int i = 0; i < cache_frames.size(); ++i) {
			cache_frames.write[i].data.resize(rgb24_frame_size);
		}
		next_write_frame_idx = write_frame_idx + 1;
		*/
	}

	write_frame_idx = next_write_frame_idx;
}

void VideoStreamPlaybackWMF::present() {
	if (read_frame_idx == write_frame_idx) {
		return;
	}
	mtx.lock();
	FrameData &the_frame = cache_frames.write[read_frame_idx];
	read_frame_idx = (read_frame_idx + 1) % cache_frames.size();
	mtx.unlock();

	// Check if frame data is valid for RGBA format
	if (the_frame.data.size() != stream_info.size.x * stream_info.size.y * 4) {
		return;
	}

	Ref<Image> img = memnew(Image(stream_info.size.x, stream_info.size.y, 0, Image::FORMAT_RGBA8, the_frame.data));
	texture->update(img);
}

int64_t VideoStreamPlaybackWMF::next_sample_time() {
	MutexLock lock(mtx);
	int64_t sample_time = INT64_MAX;
	if (!cache_frames.is_empty()) {
		sample_time = cache_frames[read_frame_idx].sample_time;
	}
	return sample_time;
}

static int counter = 0;

VideoStreamPlaybackWMF::VideoStreamPlaybackWMF() :
		media_session(NULL), media_source(NULL), topology(NULL), presentation_clock(NULL), sample_grabber_callback(nullptr), read_frame_idx(0), write_frame_idx(0), is_video_playing(false), is_video_paused(false), is_video_seekable(false), time(0.0), next_frame_time(0.0), current_frame_time(-1.0), frame_ready(false), audio_channels(0), audio_sample_rate(0), audio_buffer_pos(0) {
	id = counter;
	counter++;

	texture = Ref<ImageTexture>(memnew(ImageTexture));
	// make sure cache_frames.size() is something more than 0
	cache_frames.resize(24);
}

VideoStreamPlaybackWMF::~VideoStreamPlaybackWMF() {
	shutdown_stream();
}

void VideoStreamWMF::_bind_methods() {}

Ref<Resource> ResourceFormatLoaderWMF::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	if (f.is_null()) {
		if (r_error) {
			*r_error = ERR_CANT_OPEN;
		}
		return Ref<Resource>();
	}

	VideoStreamWMF *stream = memnew(VideoStreamWMF);
	stream->set_file(p_path);

	Ref<VideoStreamWMF> wmf_stream = Ref<VideoStreamWMF>(stream);

	if (r_error) {
		*r_error = OK;
	}

	return wmf_stream;
}

void ResourceFormatLoaderWMF::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("mp4");
	p_extensions->push_back("avi");
	p_extensions->push_back("wmv");
	p_extensions->push_back("mov");
	p_extensions->push_back("mkv");
	p_extensions->push_back("webm");
	p_extensions->push_back("flv");
}

bool ResourceFormatLoaderWMF::handles_type(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "VideoStream");
}

String ResourceFormatLoaderWMF::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "mp4" || el == "avi" || el == "wmv" || el == "mov" || el == "mkv" || el == "webm" || el == "flv") {
		return "VideoStreamWMF";
	}
	return "";
}
