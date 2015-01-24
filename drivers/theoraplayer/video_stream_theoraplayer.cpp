/*************************************************************************/
/*  video_stream.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "video_stream_theoraplayer.h"

#include "core/os/file_access.h"

#include "include/theoraplayer/TheoraPlayer.h"
#include "include/theoraplayer/TheoraTimer.h"
#include "include/theoraplayer/TheoraAudioInterface.h"
#include "include/theoraplayer/TheoraDataSource.h"
#include "include/theoraplayer/TheoraException.h"

#include "core/ring_buffer.h"
#include "core/os/thread_safe.h"

#include "core/globals.h"

static TheoraVideoManager* mgr = NULL;

class TPDataFA : public TheoraDataSource {

	FileAccess* fa;
	String data_name;

public:

	int read(void* output,int nBytes) {

		if (!fa)
			return -1;

		return fa->get_buffer((uint8_t*)output, nBytes);
	};

	//! returns a string representation of the DataSource, eg 'File: source.ogg'
	virtual std::string repr() {
		return data_name.utf8().get_data();
	};

	//! position the source pointer to byte_index from the start of the source
	virtual void seek(unsigned long byte_index) {

		if (!fa)
			return;

		fa->seek(byte_index);
	};


	//! return the size of the stream in bytes
	virtual unsigned long size() {

		if (!fa)
			return 0;

		return fa->get_len();
	};

	//! return the current position of the source pointer
	virtual unsigned long tell() {

		if (!fa)
			return 0;

		return fa->get_pos();
	};

	TPDataFA(const String& p_path) {

		fa = FileAccess::open(p_path, FileAccess::READ);
		data_name = "File: " + p_path;
	};

	TPDataFA(FileAccess* p_fa, const String& p_path) {

		fa = p_fa;
		data_name = "File: " + p_path;
	};

	~TPDataFA() {

		if (fa)
			memdelete(fa);
	};
};

class AudioStreamInput : public AudioStreamResampled {

	_THREAD_SAFE_CLASS_;

	int channels;
	int freq;

	RID stream_rid;
	mutable RingBuffer<float> rb;
	int rb_power;
	int total_wrote;
	bool playing;
	bool paused;

public:

	virtual void play() {

		_THREAD_SAFE_METHOD_
		_setup(channels, freq, 256);
		stream_rid=AudioServer::get_singleton()->audio_stream_create(get_audio_stream());
		AudioServer::get_singleton()->stream_set_active(stream_rid,true);
		AudioServer::get_singleton()->stream_set_volume_scale(stream_rid,1);
		playing = true;
		paused = false;
	};
	virtual void stop() {

		_THREAD_SAFE_METHOD_

		AudioServer::get_singleton()->stream_set_active(stream_rid,false);
		//_clear_stream();
		playing=false;
		_clear();
	};

	virtual bool is_playing() const { return true; };

	virtual void set_paused(bool p_paused) { paused = p_paused; };
	virtual bool is_paused(bool p_paused) const { return paused; };

	virtual void set_loop(bool p_enable) {};
	virtual bool has_loop() const { return false; };

	virtual float get_length() const { return 0; };

	virtual String get_stream_name() const { return "Theora Audio Stream"; };

	virtual int get_loop_count() const { return 1; };

	virtual float get_pos() const { return 0; };
	virtual void seek_pos(float p_time) {};

	virtual UpdateMode get_update_mode() const { return UPDATE_THREAD; };

	virtual bool _can_mix() const { return true; };

	void input(float* p_data, int p_samples) {


		_THREAD_SAFE_METHOD_;
		//printf("input %i samples from %p\n", p_samples, p_data);
		if (rb.space_left() < p_samples) {
			rb_power += 1;
			rb.resize(rb_power);
		}
		rb.write(p_data, p_samples);

		update(); //update too here for less latency
	};

	void update() {

		_THREAD_SAFE_METHOD_;
		int todo = get_todo();
		int16_t* buffer = get_write_buffer();
		int frames = rb.data_left()/channels;
		const int to_write = MIN(todo, frames);

		for (int i=0; i<to_write*channels; i++) {

			int v = rb.read() * 32767;
			int16_t sample = CLAMP(v,-32768,32767);
			buffer[i] = sample;
		};
		write(to_write);
		total_wrote += to_write;
	};

	int get_pending() const {
		return rb.data_left();
	};

	int get_total_wrote() {

		return total_wrote - (get_total() - get_todo());
	};

	AudioStreamInput(int p_channels, int p_freq) {

		playing = false;
		paused = true;
		channels = p_channels;
		freq = p_freq;
		total_wrote = 0;
		rb_power = 22;
		rb.resize(rb_power);
	};

	~AudioStreamInput() {

		stop();
	};
};

class TPAudioGodot : public TheoraAudioInterface, TheoraTimer {

	Ref<AudioStreamInput> stream;
	int sample_count;
	int channels;
	int freq;

public:

	void insertData(float* data, int nSamples) {

		stream->input(data, nSamples);
	};

	TPAudioGodot(TheoraVideoClip* owner, int nChannels, int p_freq)
		: TheoraAudioInterface(owner, nChannels, p_freq), TheoraTimer() {

		printf("***************** audio interface constructor freq %i\n", p_freq);
		channels = nChannels;
		freq = p_freq;
		stream = Ref<AudioStreamInput>(memnew(AudioStreamInput(nChannels, p_freq)));
		stream->play();
		sample_count = 0;
		owner->setTimer(this);
	};

	void stop() {

		stream->stop();
	};

	void update(float time_increase)
	{
		float prev_time = mTime;
		//mTime = (float)(stream->get_total_wrote()) / freq;
		//mTime = MAX(0,mTime-AudioServer::get_singleton()->get_output_delay());
		//mTime = (float)sample_count / channels / freq;
		mTime += time_increase;
		if (mTime - prev_time > .02) printf("time increase %f secs\n", mTime - prev_time);
		//float duration=mClip->getDuration();
		//if (mTime > duration) mTime=duration;
		//printf("time at timer is %f, %f, samples %i\n", mTime, time_increase, sample_count);
	}
};

class TPAudioGodotFactory : public TheoraAudioInterfaceFactory {

public:
	TheoraAudioInterface* createInstance(TheoraVideoClip* owner, int nChannels, int freq) {

		printf("************** creating audio output\n");
		TheoraAudioInterface* ta = new TPAudioGodot(owner, nChannels, freq);
		return ta;
	};
};

static TPAudioGodotFactory* audio_factory = NULL;

void VideoStreamTheoraplayer::stop() {

	playing = false;
	if (clip) {
		clip->stop();
		clip->seek(0);
	};
	started = true;
};

void VideoStreamTheoraplayer::play() {
	if (clip)
		playing = true;
};

bool VideoStreamTheoraplayer::is_playing() const {

	return playing;
};

void VideoStreamTheoraplayer::set_paused(bool p_paused) {

	paused = p_paused;
	if (paused) {
		clip->pause();
	} else {
		if (clip && playing && !started)
			clip->play();
	}
};

bool VideoStreamTheoraplayer::is_paused(bool p_paused) const {

	return !playing;
};

void VideoStreamTheoraplayer::set_loop(bool p_enable) {

	loop = p_enable;
};

bool VideoStreamTheoraplayer::has_loop() const {

	return loop;
};

float VideoStreamTheoraplayer::get_length() const {

	if (!clip)
		return 0;

	return clip->getDuration();
};


float VideoStreamTheoraplayer::get_pos() const {

	if (!clip)
		return 0;

	return clip->getTimer()->getTime();
};

void VideoStreamTheoraplayer::seek_pos(float p_time) {

	if (!clip)
		return;

	clip->seek(p_time);
};

int VideoStreamTheoraplayer::get_pending_frame_count() const {

	if (!clip)
		return 0;

	TheoraVideoFrame* f = clip->getNextFrame();
	return f ? 1 : 0;
};


void VideoStreamTheoraplayer::pop_frame(Ref<ImageTexture> p_tex) {

	if (!clip)
		return;

	TheoraVideoFrame* f = clip->getNextFrame();
	if (!f) {
		return;
	};

#ifdef GLES2_ENABLED
//	RasterizerGLES2* r = RasterizerGLES2::get_singleton();
//	r->_texture_set_data(p_tex, f->mBpp == 3 ? Image::Format_RGB : Image::Format_RGBA, f->mBpp, w, h, f->getBuffer());

#endif

	float w=clip->getWidth(),h=clip->getHeight();
	int imgsize = w * h * f->mBpp;

	int size = f->getStride() * f->getHeight() * f->mBpp;
	data.resize(imgsize);
	{
		DVector<uint8_t>::Write wr = data.write();
		uint8_t* ptr = wr.ptr();
		memcpy(ptr, f->getBuffer(), imgsize);
	}
    /*
    for (int i=0; i<h; i++) {
        int dstofs = i * w * f->mBpp;
        int srcofs = i * f->getStride() * f->mBpp;
        copymem(ptr + dstofs, f->getBuffer() + dstofs, w * f->mBpp);
    };
     */
	Image frame = Image();
	frame.create(w, h, 0, f->mBpp == 3 ? Image::FORMAT_RGB : Image::FORMAT_RGBA, data);

	clip->popFrame();

	if (p_tex->get_width() == 0) {
		p_tex->create(frame.get_width(),frame.get_height(),frame.get_format(),Texture::FLAG_VIDEO_SURFACE|Texture::FLAG_FILTER);
		p_tex->set_data(frame);
	} else {

		p_tex->set_data(frame);
	};
};

/*
Image VideoStreamTheoraplayer::pop_frame() {

	Image ret = frame;
	frame = Image();
	return ret;
};
*/

Image VideoStreamTheoraplayer::peek_frame() const {

	return Image();
};

void VideoStreamTheoraplayer::update(float p_time) {

	if (!mgr)
		return;

	if (!clip)
		return;

	if (!playing || paused)
		return;

	//printf("video update!\n");
	if (started) {
		if (clip->getNumReadyFrames() < 2) {
			printf("frames not ready, returning!\n");
			return;
		};
		started = false;
		//printf("playing clip!\n");
		clip->play();
	} else if (clip->isDone()) {
		playing = false;
	};

	mgr->update(p_time);
};


void VideoStreamTheoraplayer::set_audio_track(int p_idx) {
	audio_track=p_idx;
	if (clip)
		clip->set_audio_track(audio_track);
}

void VideoStreamTheoraplayer::set_file(const String& p_file) {

	FileAccess* f = FileAccess::open(p_file, FileAccess::READ);
	if (!f || !f->is_open())
		return;

	if (!audio_factory) {
		audio_factory = memnew(TPAudioGodotFactory);
	};

	if (mgr == NULL) {
		mgr = memnew(TheoraVideoManager);
		mgr->setAudioInterfaceFactory(audio_factory);
	};

	int track = GLOBAL_DEF("theora/audio_track", 0); // hack

	if (p_file.find(".mp4") != -1) {
		
		std::string file = p_file.replace("res://", "").utf8().get_data();
		clip = mgr->createVideoClip(file, TH_RGBX, 2, false, track);
		//clip->set_audio_track(audio_track);
		memdelete(f);

	} else {

		TheoraDataSource* ds = memnew(TPDataFA(f, p_file));

		try {
			clip = mgr->createVideoClip(ds);
			clip->set_audio_track(audio_track);
		} catch (_TheoraGenericException e) {
			printf("exception ocurred! %s\n", e.repr().c_str());
			clip = NULL;
		};
	};

	clip->pause();
	started = true;
};

VideoStreamTheoraplayer::~VideoStreamTheoraplayer() {

	stop();
	//if (mgr) { // this should be a singleton or static or something
	//	memdelete(mgr);
	//};
	//mgr = NULL;
	if (clip) {
		mgr->destroyVideoClip(clip);
		clip = NULL;
	};
};

VideoStreamTheoraplayer::VideoStreamTheoraplayer() {

	//mgr = NULL;
	clip = NULL;
	started = false;
	playing = false;
	paused = false;
	loop = false;
	audio_track=0;
};


RES ResourceFormatLoaderVideoStreamTheoraplayer::load(const String &p_path,const String& p_original_path) {

	VideoStreamTheoraplayer *stream = memnew(VideoStreamTheoraplayer);
	stream->set_file(p_path);
	return Ref<VideoStreamTheoraplayer>(stream);
}

void ResourceFormatLoaderVideoStreamTheoraplayer::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("ogm");
	p_extensions->push_back("ogv");
	p_extensions->push_back("mp4");
}
bool ResourceFormatLoaderVideoStreamTheoraplayer::handles_type(const String& p_type) const {
	return p_type=="VideoStream" || p_type == "VideoStreamTheoraplayer";
}

String ResourceFormatLoaderVideoStreamTheoraplayer::get_resource_type(const String &p_path) const {

	String exl=p_path.extension().to_lower();
	if (exl=="ogm" || exl=="ogv" || exl=="mp4")
		return "VideoStream";
	return "";
}



