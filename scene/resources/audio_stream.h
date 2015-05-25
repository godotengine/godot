/*************************************************************************/
/*  audio_stream.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef AUDIO_STREAM_H
#define AUDIO_STREAM_H

#include "resource.h"
#include "servers/audio_server.h"
#include "scene/resources/audio_stream.h"

class AudioStream : public Resource {

	OBJ_TYPE( AudioStream, Resource );
	OBJ_SAVE_TYPE( AudioStream ); //children are all saved as AudioStream, so they can be exchanged

	friend class InternalAudioStream;

	struct InternalAudioStream : public AudioServer::AudioStream {

		::AudioStream *owner;
		virtual int get_channel_count() const;
		virtual void set_mix_rate(int p_rate); //notify the stream of the mix rate
		virtual bool mix(int32_t *p_buffer,int p_frames);
		virtual bool can_update_mt() const;
		virtual void update();
	};


	int _mix_rate;
	InternalAudioStream *internal_audio_stream;
protected:

	_FORCE_INLINE_ int get_mix_rate() const { return _mix_rate; }
	virtual int get_channel_count() const=0;
	virtual bool mix(int32_t *p_buffer, int p_frames)=0;

	static void _bind_methods();
public:

	enum UpdateMode {
		UPDATE_NONE,
		UPDATE_IDLE,
		UPDATE_THREAD
	};

	AudioServer::AudioStream *get_audio_stream();

	virtual void play()=0;
	virtual void stop()=0;
	virtual bool is_playing() const=0;

	virtual void set_paused(bool p_paused)=0;
	virtual bool is_paused(bool p_paused) const=0;

	virtual void set_loop(bool p_enable)=0;
	virtual bool has_loop() const=0;

	virtual float get_length() const=0;

	virtual String get_stream_name() const=0;

	virtual int get_loop_count() const=0;

	virtual float get_pos() const=0;
	virtual void seek_pos(float p_time)=0;

	virtual UpdateMode get_update_mode() const=0;
	virtual void update()=0;

	AudioStream();
	~AudioStream();
};


VARIANT_ENUM_CAST( AudioStream::UpdateMode );

#endif // AUDIO_STREAM_H
