/*************************************************************************/
/*  video_stream_webm.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "video_stream_webm.h"

#include "OpusVorbisDecoder.hpp"
#include "VPXDecoder.hpp"

#include "mkvparser/mkvparser.h"

#include "os/file_access.h"
#include "project_settings.h"

#include "thirdparty/misc/yuv2rgb.h"

#include <string.h>

class MkvReader : public mkvparser::IMkvReader {

public:
	MkvReader(const String &p_file) {

		file = FileAccess::open(p_file, FileAccess::READ);
		ERR_FAIL_COND(!file);
	}
	~MkvReader() {

		if (file)
			memdelete(file);
	}

	virtual int Read(long long pos, long len, unsigned char *buf) {

		if (file) {

			if (file->get_position() != (size_t)pos)
				file->seek(pos);
			if (file->get_buffer(buf, len) == len)
				return 0;
		}
		return -1;
	}

	virtual int Length(long long *total, long long *available) {

		if (file) {

			const size_t len = file->get_len();
			if (total)
				*total = len;
			if (available)
				*available = len;
			return 0;
		}
		return -1;
	}

private:
	FileAccess *file;
};

/**/

VideoStreamPlaybackWebm::VideoStreamPlaybackWebm()
	: audio_track(0),
	  webm(NULL),
	  video(NULL),
	  audio(NULL),
	  video_frames(NULL), audio_frame(NULL),
	  video_frames_pos(0), video_frames_capacity(0),
	  num_decoded_samples(0), samples_offset(-1),
	  mix_callback(NULL),
	  mix_udata(NULL),
	  playing(false), paused(false),
	  delay_compensation(0.0),
	  time(0.0), video_frame_delay(0.0), video_pos(0.0),
	  texture(memnew(ImageTexture)),
	  pcm(NULL) {}
VideoStreamPlaybackWebm::~VideoStreamPlaybackWebm() {

	delete_pointers();
}

bool VideoStreamPlaybackWebm::open_file(const String &p_file) {

	file_name = p_file;
	webm = memnew(WebMDemuxer(new MkvReader(file_name), 0, audio_track));
	if (webm->isOpen()) {

		video = memnew(VPXDecoder(*webm, 8)); //TODO: Detect CPU threads
		if (video->isOpen()) {

			audio = memnew(OpusVorbisDecoder(*webm));
			if (audio->isOpen()) {

				audio_frame = memnew(WebMFrame);
				pcm = (int16_t *)memalloc(sizeof(int16_t) * audio->getBufferSamples() * webm->getChannels());
			} else {

				memdelete(audio);
				audio = NULL;
			}

			frame_data.resize((webm->getWidth() * webm->getHeight()) << 2);
			texture->create(webm->getWidth(), webm->getHeight(), Image::FORMAT_RGBA8, Texture::FLAG_FILTER | Texture::FLAG_VIDEO_SURFACE);

			return true;
		}
		memdelete(video);
		video = NULL;
	}
	memdelete(webm);
	webm = NULL;
	return false;
}

void VideoStreamPlaybackWebm::stop() {

	if (playing) {

		delete_pointers();

		pcm = NULL;

		audio_frame = NULL;
		video_frames = NULL;

		video = NULL;
		audio = NULL;

		open_file(file_name); //Should not fail here...

		video_frames_capacity = video_frames_pos = 0;
		num_decoded_samples = 0;
		samples_offset = -1;
		video_frame_delay = video_pos = 0.0;
	}
	time = 0.0;
	playing = false;
}
void VideoStreamPlaybackWebm::play() {

	stop();

	delay_compensation = ProjectSettings::get_singleton()->get("audio/video_delay_compensation_ms");
	delay_compensation /= 1000.0;

	playing = true;
}

bool VideoStreamPlaybackWebm::is_playing() const {

	return playing;
}

void VideoStreamPlaybackWebm::set_paused(bool p_paused) {

	paused = p_paused;
}
bool VideoStreamPlaybackWebm::is_paused(bool p_paused) const {

	return paused;
}

void VideoStreamPlaybackWebm::set_loop(bool p_enable) {

	//Empty
}
bool VideoStreamPlaybackWebm::has_loop() const {

	return false;
}

float VideoStreamPlaybackWebm::get_length() const {

	if (webm)
		return webm->getLength();
	return 0.0f;
}

float VideoStreamPlaybackWebm::get_playback_position() const {

	return video_pos;
}
void VideoStreamPlaybackWebm::seek(float p_time) {

	//Not implemented
}

void VideoStreamPlaybackWebm::set_audio_track(int p_idx) {

	audio_track = p_idx;
}

Ref<Texture> VideoStreamPlaybackWebm::get_texture() {

	return texture;
}
void VideoStreamPlaybackWebm::update(float p_delta) {

	if ((!playing || paused) || !video)
		return;

	bool audio_buffer_full = false;

	if (samples_offset > -1) {

		//Mix remaining samples
		const int to_read = num_decoded_samples - samples_offset;
		const int mixed = mix_callback(mix_udata, pcm + samples_offset * webm->getChannels(), to_read);
		if (mixed != to_read) {

			samples_offset += mixed;
			audio_buffer_full = true;
		} else {

			samples_offset = -1;
		}
	}

	const bool hasAudio = (audio && mix_callback);
	while ((hasAudio && (!audio_buffer_full || !has_enough_video_frames())) || (!hasAudio && video_frames_pos == 0)) {

		if (hasAudio && !audio_buffer_full && audio_frame->isValid() && audio->getPCMS16(*audio_frame, pcm, num_decoded_samples) && num_decoded_samples > 0) {

			const int mixed = mix_callback(mix_udata, pcm, num_decoded_samples);
			if (mixed != num_decoded_samples) {

				samples_offset = mixed;
				audio_buffer_full = true;
			}
		}

		WebMFrame *video_frame;
		if (video_frames_pos >= video_frames_capacity) {

			WebMFrame **video_frames_new = (WebMFrame **)memrealloc(video_frames, ++video_frames_capacity * sizeof(void *));
			ERR_FAIL_COND(!video_frames_new); //Out of memory
			(video_frames = video_frames_new)[video_frames_capacity - 1] = memnew(WebMFrame);
		}
		video_frame = video_frames[video_frames_pos];

		if (!webm->readFrame(video_frame, audio_frame)) //This will invalidate frames
			break; //Can't demux, EOS?

		if (video_frame->isValid())
			++video_frames_pos;
	};

	const double video_delay = video->getFramesDelay() * video_frame_delay;

	bool want_this_frame = false;
	while (video_frames_pos > 0 && !want_this_frame) {

		WebMFrame *video_frame = video_frames[0];
		if (video_frame->time <= time + video_delay) {

			if (video->decode(*video_frame)) {

				VPXDecoder::IMAGE_ERROR err;
				VPXDecoder::Image image;

				while ((err = video->getImage(image)) != VPXDecoder::NO_FRAME) {

					want_this_frame = (time - video_frame->time <= video_frame_delay);

					if (want_this_frame) {

						if (err == VPXDecoder::NO_ERROR && image.w == webm->getWidth() && image.h == webm->getHeight()) {

							PoolVector<uint8_t>::Write w = frame_data.write();
							bool converted = false;

							if (image.chromaShiftW == 1 && image.chromaShiftH == 1) {

								yuv420_2_rgb8888(w.ptr(), image.planes[0], image.planes[2], image.planes[1], image.w, image.h, image.linesize[0], image.linesize[1], image.w << 2, 0);
								// 								libyuv::I420ToARGB(image.planes[0], image.linesize[0], image.planes[2], image.linesize[2], image.planes[1], image.linesize[1], w.ptr(), image.w << 2, image.w, image.h);
								converted = true;
							} else if (image.chromaShiftW == 1 && image.chromaShiftH == 0) {

								yuv422_2_rgb8888(w.ptr(), image.planes[0], image.planes[2], image.planes[1], image.w, image.h, image.linesize[0], image.linesize[1], image.w << 2, 0);
								// 								libyuv::I422ToARGB(image.planes[0], image.linesize[0], image.planes[2], image.linesize[2], image.planes[1], image.linesize[1], w.ptr(), image.w << 2, image.w, image.h);
								converted = true;
							} else if (image.chromaShiftW == 0 && image.chromaShiftH == 0) {

								yuv444_2_rgb8888(w.ptr(), image.planes[0], image.planes[2], image.planes[1], image.w, image.h, image.linesize[0], image.linesize[1], image.w << 2, 0);
								// 								libyuv::I444ToARGB(image.planes[0], image.linesize[0], image.planes[2], image.linesize[2], image.planes[1], image.linesize[1], w.ptr(), image.w << 2, image.w, image.h);
								converted = true;
							} else if (image.chromaShiftW == 2 && image.chromaShiftH == 0) {

								// 								libyuv::I411ToARGB(image.planes[0], image.linesize[0], image.planes[2], image.linesize[2], image.planes[1], image.linesize[1], w.ptr(), image.w << 2, image.w, image.h);
								// 								converted = true;
							}

							if (converted)
								texture->set_data(Image(image.w, image.h, 0, Image::FORMAT_RGBA8, frame_data)); //Zero copy send to visual server
						}

						break;
					}
				}
			}

			video_frame_delay = video_frame->time - video_pos;
			video_pos = video_frame->time;

			memmove(video_frames, video_frames + 1, (--video_frames_pos) * sizeof(void *));
			video_frames[video_frames_pos] = video_frame;
		} else {

			break;
		}
	}

	time += p_delta;

	if (video_frames_pos == 0 && webm->isEOS())
		stop();
}

void VideoStreamPlaybackWebm::set_mix_callback(VideoStreamPlayback::AudioMixCallback p_callback, void *p_userdata) {

	mix_callback = p_callback;
	mix_udata = p_userdata;
}
int VideoStreamPlaybackWebm::get_channels() const {

	if (audio)
		return webm->getChannels();
	return 0;
}
int VideoStreamPlaybackWebm::get_mix_rate() const {

	if (audio)
		return webm->getSampleRate();
	return 0;
}

inline bool VideoStreamPlaybackWebm::has_enough_video_frames() const {
	if (video_frames_pos > 0) {

		const double audio_delay = AudioServer::get_singleton()->get_output_delay();
		const double video_time = video_frames[video_frames_pos - 1]->time;
		return video_time >= time + audio_delay + delay_compensation;
	}
	return false;
}

void VideoStreamPlaybackWebm::delete_pointers() {

	if (pcm)
		memfree(pcm);

	if (audio_frame)
		memdelete(audio_frame);
	for (int i = 0; i < video_frames_capacity; ++i)
		memdelete(video_frames[i]);
	if (video_frames)
		memfree(video_frames);

	if (video)
		memdelete(video);
	if (audio)
		memdelete(audio);

	if (webm)
		memdelete(webm);
}

/**/

RES ResourceFormatLoaderVideoStreamWebm::load(const String &p_path, const String &p_original_path, Error *r_error) {

	Ref<VideoStreamWebm> stream = memnew(VideoStreamWebm);
	stream->set_file(p_path);
	if (r_error)
		*r_error = OK;
	return stream;
}

void ResourceFormatLoaderVideoStreamWebm::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("webm");
}
bool ResourceFormatLoaderVideoStreamWebm::handles_type(const String &p_type) const {

	return (p_type == "VideoStream" || p_type == "VideoStreamWebm");
}

String ResourceFormatLoaderVideoStreamWebm::get_resource_type(const String &p_path) const {

	const String exl = p_path.get_extension().to_lower();
	if (exl == "webm")
		return "VideoStreamWebm";
	return "";
}

/**/

VideoStreamWebm::VideoStreamWebm()
	: audio_track(0) {}

Ref<VideoStreamPlayback> VideoStreamWebm::instance_playback() {

	Ref<VideoStreamPlaybackWebm> pb = memnew(VideoStreamPlaybackWebm);
	pb->set_audio_track(audio_track);
	if (pb->open_file(file))
		return pb;
	return NULL;
}

void VideoStreamWebm::set_file(const String &p_file) {

	file = p_file;
}
void VideoStreamWebm::set_audio_track(int p_track) {

	audio_track = p_track;
}
