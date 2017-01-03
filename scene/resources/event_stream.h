/*************************************************************************/
/*  event_stream.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef EVENT_STREAM_H
#define EVENT_STREAM_H

#include "resource.h"
#include "servers/audio_server.h"

class EventStreamPlayback : public Reference {

	GDCLASS(EventStreamPlayback,Reference);

	class InternalEventStream : public AudioServer::EventStream {
	public:
		AudioMixer *_get_mixer(){ return get_mixer(); }
		EventStreamPlayback *playback;
		virtual void update(uint64_t p_usec) {

			playback->_update(get_mixer(),p_usec);
		}


		virtual ~InternalEventStream() {}
	};


	InternalEventStream estream;

	RID stream;
	bool playing;


protected:

	virtual AudioMixer* _get_mixer() { return estream._get_mixer(); }
	virtual Error _play()=0;
	virtual bool _update(AudioMixer* p_mixer, uint64_t p_usec)=0;
	virtual void _stop()=0;
public:

	virtual Error play();
	virtual void stop();
	virtual bool is_playing() const;

	virtual void set_paused(bool p_paused)=0;
	virtual bool is_paused() const=0;

	virtual void set_loop(bool p_loop)=0;
	virtual bool is_loop_enabled() const=0;

	virtual int get_loop_count() const=0;

	virtual float get_pos() const=0;
	virtual void seek_pos(float p_time)=0;

	virtual void set_volume(float p_vol)=0;
	virtual float get_volume() const=0;

	virtual void set_pitch_scale(float p_pitch_scale)=0;
	virtual float get_pitch_scale() const=0;

	virtual void set_tempo_scale(float p_tempo_scale)=0;
	virtual float get_tempo_scale() const=0;

	virtual void set_channel_volume(int p_channel,float p_volume)=0;
	virtual float get_channel_volume(int p_channel) const=0;

	virtual float get_last_note_time(int p_channel) const=0;
	EventStreamPlayback();
	~EventStreamPlayback();

};

class EventStream : public Resource {

	GDCLASS(EventStream,Resource);
	OBJ_SAVE_TYPE( EventStream ); //children are all saved as EventStream, so they can be exchanged

public:

	virtual Ref<EventStreamPlayback> instance_playback()=0;

	virtual String get_stream_name() const=0;
	virtual float get_length() const=0;
	virtual int get_channel_count() const=0;



	EventStream();
};

#endif // EVENT_STREAM_H
