/*************************************************************************/
/*  spatial_sound_2d_server_sw.cpp                                       */
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
#include "spatial_sound_2d_server_sw.h"

#include "os/os.h"
#include "servers/audio/audio_filter_sw.h"



int SpatialSound2DServerSW::InternalAudioStream::get_channel_count() const {

	return AudioServer::get_singleton()->get_default_channel_count();
}

void SpatialSound2DServerSW::InternalAudioStream::set_mix_rate(int p_rate) {


}

void SpatialSound2DServerSW::InternalAudioStream::update() {

	owner->_update_sources();
}

bool SpatialSound2DServerSW::InternalAudioStream::mix(int32_t *p_buffer,int p_frames) {

	return owner->internal_buffer_mix(p_buffer,p_frames);
}

void SpatialSound2DServerSW::_update_sources() {

	_THREAD_SAFE_METHOD_
	for (Set<Source*>::Element *E=streaming_sources.front();E;E=E->next()) {

		Source *s=E->get();
		ERR_CONTINUE(!s->stream);
		s->stream->update();
	}
}


SpatialSound2DServerSW::Room::Room() {

	//params[ROOM_PARAM_SPEED_OF_SOUND]=343.0;
	params[ROOM_PARAM_PITCH_SCALE]=1.0;
	params[ROOM_PARAM_VOLUME_SCALE_DB]=0;
	params[ROOM_PARAM_REVERB_SEND]=0;
	params[ROOM_PARAM_CHORUS_SEND]=0;
	params[ROOM_PARAM_ATTENUATION_SCALE]=1.0;
	params[ROOM_PARAM_ATTENUATION_HF_CUTOFF]=5000;
	params[ROOM_PARAM_ATTENUATION_HF_FLOOR_DB]=-24.0;
	params[ROOM_PARAM_ATTENUATION_HF_RATIO_EXP]=1.0;
	params[ROOM_PARAM_ATTENUATION_REVERB_SCALE]=0.0;
	override_other_sources=false;
	reverb=ROOM_REVERB_HALL;
	//octree_id=0;
	level=-1;


}


SpatialSound2DServerSW::Source::Source() {

	params[SOURCE_PARAM_VOLUME_DB]=0.0;
	params[SOURCE_PARAM_PITCH_SCALE]=1.0;
	params[SOURCE_PARAM_ATTENUATION_MIN_DISTANCE]=1;
	params[SOURCE_PARAM_ATTENUATION_MAX_DISTANCE]=100;
	params[SOURCE_PARAM_ATTENUATION_DISTANCE_EXP]=1.0; //linear (and not really good)
	stream=NULL;
	voices.resize(1);
	last_voice=0;
}

SpatialSound2DServerSW::Source::Voice::Voice() {

	active=false;
	restart=false;
	pitch_scale=1.0;
	volume_scale=0.0;
	voice_rid=AudioServer::get_singleton()->voice_create();

}
SpatialSound2DServerSW::Source::Voice::~Voice() {

	AudioServer::get_singleton()->free(voice_rid);
}


SpatialSound2DServerSW::Listener::Listener() {

	params[LISTENER_PARAM_VOLUME_SCALE_DB]=0.0;
	params[LISTENER_PARAM_PITCH_SCALE]=1.0;
	params[LISTENER_PARAM_ATTENUATION_SCALE]=1.0;
	params[LISTENER_PARAM_PAN_RANGE]=128;

}

/* SPACE */
RID SpatialSound2DServerSW::space_create() {

	Space* space = memnew( Space );
	RID space_rid = space_owner.make_rid(space);
	space->default_room=room_create();
	room_set_space(space->default_room,space_rid);
	return space_rid;
}

/* ROOM */

RID SpatialSound2DServerSW::room_create() {

	Room *room = memnew( Room );
	return room_owner.make_rid(room);
}

void SpatialSound2DServerSW::room_set_space(RID p_room,RID p_space) {

	Room *room = room_owner.get(p_room);
	ERR_FAIL_COND(!room);

	if (room->space.is_valid()) {

		Space *space = space_owner.get(room->space);
		space->rooms.erase(p_room);
		//space->octree.erase(room->octree_id);
		//room->octree_id=0;
	}

	room->space=RID();

	if (p_space.is_valid()) {

		Space *space = space_owner.get(p_space);
		ERR_FAIL_COND(!space);
		space->rooms.insert(p_room);
		//room->octree_id=space->octree.create(room,AABB());
		//set bounds
		//AABB aabb = room->bounds.is_empty()?AABB():room->bounds.get_aabb();
		//space->octree.move(room->octree_id,room->transform.xform(aabb));
		room->space=p_space;
	}


}

RID SpatialSound2DServerSW::room_get_space(RID p_room) const {

	Room *room = room_owner.get(p_room);
	ERR_FAIL_COND_V(!room,RID());


	return room->space;
}



void SpatialSound2DServerSW::room_set_bounds(RID p_room, const PoolVector<Point2>& p_bounds) {

	Room *room = room_owner.get(p_room);
	ERR_FAIL_COND(!room);

	room->bounds=p_bounds;

	if (!room->space.is_valid())
		return;

	//AABB aabb = room->bounds.is_empty()?AABB():room->bounds.get_aabb();
	//Space* space = space_owner.get(room->space);
	//ERR_FAIL_COND(!space);

	//space->octree.move(room->octree_id,room->transform.xform(aabb));

}
PoolVector<Point2> SpatialSound2DServerSW::room_get_bounds(RID p_room) const {
	Room *room = room_owner.get(p_room);
	ERR_FAIL_COND_V(!room,PoolVector<Point2>());

	return room->bounds;
}

void SpatialSound2DServerSW::room_set_transform(RID p_room, const Transform2D& p_transform) {

	if (space_owner.owns(p_room))
		p_room=space_owner.get(p_room)->default_room;

	Room *room = room_owner.get(p_room);
	ERR_FAIL_COND(!room);
	room->transform=p_transform;
	room->inverse_transform=p_transform.affine_inverse(); // needs to be done to unscale BSP properly

	if (!room->space.is_valid())
		return;

	/*
	if (!room->bounds.is_empty()) {

		Space* space = space_owner.get(room->space);
		ERR_FAIL_COND(!space);

		//space->octree.move(room->octree_id,room->transform.xform(room->bounds.get_aabb()));
	}*/
}

Transform2D SpatialSound2DServerSW::room_get_transform(RID p_room) const {

	if (space_owner.owns(p_room))
		p_room=space_owner.get(p_room)->default_room;

	Room *room = room_owner.get(p_room);
	ERR_FAIL_COND_V(!room,Transform2D());
	return room->transform;
}


void SpatialSound2DServerSW::room_set_param(RID p_room, RoomParam p_param, float p_value) {

	if (space_owner.owns(p_room))
		p_room=space_owner.get(p_room)->default_room;

	ERR_FAIL_INDEX(p_param,ROOM_PARAM_MAX);
	Room *room = room_owner.get(p_room);
	ERR_FAIL_COND(!room);
	room->params[p_param]=p_value;

}
float SpatialSound2DServerSW::room_get_param(RID p_room, RoomParam p_param) const {

	if (space_owner.owns(p_room))
		p_room=space_owner.get(p_room)->default_room;

	ERR_FAIL_INDEX_V(p_param,ROOM_PARAM_MAX,0);
	Room *room = room_owner.get(p_room);
	ERR_FAIL_COND_V(!room,0);
	return room->params[p_param];
}

void SpatialSound2DServerSW::room_set_level(RID p_room, int p_level) {

	Room *room = room_owner.get(p_room);
	ERR_FAIL_COND(!room);
	room->level =p_level;

}

int SpatialSound2DServerSW::room_get_level(RID p_room) const {

	Room *room = room_owner.get(p_room);
	ERR_FAIL_COND_V(!room,0);
	return room->level;

}


void SpatialSound2DServerSW::room_set_reverb(RID p_room, RoomReverb p_reverb) {

	if (space_owner.owns(p_room))
		p_room=space_owner.get(p_room)->default_room;

	Room *room = room_owner.get(p_room);
	ERR_FAIL_COND(!room);
	room->reverb=p_reverb;

}
SpatialSound2DServerSW::RoomReverb SpatialSound2DServerSW::room_get_reverb(RID p_room) const {

	if (space_owner.owns(p_room))
		p_room=space_owner.get(p_room)->default_room;

	Room *room = room_owner.get(p_room);
	ERR_FAIL_COND_V(!room,ROOM_REVERB_SMALL);
	return room->reverb;
}

//useful for underwater or rooms with very strange conditions
void SpatialSound2DServerSW::room_set_force_params_to_all_sources(RID p_room, bool p_force) {

	if (space_owner.owns(p_room))
		p_room=space_owner.get(p_room)->default_room;

	Room *room = room_owner.get(p_room);
	ERR_FAIL_COND(!room);
	room->override_other_sources=p_force;

}
bool SpatialSound2DServerSW::room_is_forcing_params_to_all_sources(RID p_room) const {

	if (space_owner.owns(p_room))
		p_room=space_owner.get(p_room)->default_room;

	Room *room = room_owner.get(p_room);
	ERR_FAIL_COND_V(!room,false);
	return room->override_other_sources;
}

/* SOURCE */

RID SpatialSound2DServerSW::source_create(RID p_space) {

	Space *space = space_owner.get(p_space);
	ERR_FAIL_COND_V(!space,RID());

	Source *source = memnew( Source );
	source->space=p_space;
	RID source_rid = source_owner.make_rid(source);
	space->sources.insert(source_rid);

	return source_rid;
}


void SpatialSound2DServerSW::source_set_polyphony(RID p_source,int p_voice_count) {


	ERR_FAIL_COND(p_voice_count<=0); // more than 32 is too much, change this if you really need more
	if (p_voice_count>32) {

		ERR_PRINT("Voices will be clipped to 32");
		p_voice_count=32;
	}
	Source *source = source_owner.get(p_source);
	ERR_FAIL_COND(!source);

	if (p_voice_count<source->voices.size()) {

		for(int i=p_voice_count;i<source->voices.size();i++) {
			active_voices.erase(ActiveVoice(source,i)); //erase from active voices
		}
	}
	source->voices.resize(p_voice_count);

}

int SpatialSound2DServerSW::source_get_polyphony(RID p_source) const {

	Source *source = source_owner.get(p_source);
	ERR_FAIL_COND_V(!source,-1);
	return source->voices.size();

}

void SpatialSound2DServerSW::source_set_transform(RID p_source, const Transform2D& p_transform) {

	Source *source = source_owner.get(p_source);
	ERR_FAIL_COND(!source);
	source->transform=p_transform;
	source->transform.orthonormalize();
}
Transform2D SpatialSound2DServerSW::source_get_transform(RID p_source) const {

	Source *source = source_owner.get(p_source);
	ERR_FAIL_COND_V(!source,Transform2D());
	return source->transform;
}

void SpatialSound2DServerSW::source_set_param(RID p_source, SourceParam p_param, float p_value) {

	ERR_FAIL_INDEX(p_param,SOURCE_PARAM_MAX);
	Source *source = source_owner.get(p_source);
	ERR_FAIL_COND(!source);
	source->params[p_param]=p_value;

}
float SpatialSound2DServerSW::source_get_param(RID p_source, SourceParam p_param) const {
	ERR_FAIL_INDEX_V(p_param,SOURCE_PARAM_MAX,0);
	Source *source = source_owner.get(p_source);
	ERR_FAIL_COND_V(!source,0);
	return source->params[p_param];


}

void SpatialSound2DServerSW::source_set_audio_stream(RID p_source, AudioServer::AudioStream *p_stream) {

	Source *source = source_owner.get(p_source);
	ERR_FAIL_COND(!source);
	AudioServer::get_singleton()->lock();
	source->stream=p_stream;
	_THREAD_SAFE_METHOD_

	if (!p_stream) {
		streaming_sources.erase(source);
		active_voices.erase(ActiveVoice(source,VOICE_IS_STREAM));
	} else {
		streaming_sources.insert(source);
		active_voices.insert(ActiveVoice(source,VOICE_IS_STREAM));
		zeromem(source->stream_data.filter_state,sizeof(Source::StreamData::FilterState)*4); //reset filter for safetyness
		p_stream->set_mix_rate(AudioServer::get_singleton()->get_default_mix_rate());
	}

	AudioServer::get_singleton()->unlock();

} //null to unset

SpatialSound2DServer::SourceVoiceID SpatialSound2DServerSW::source_play_sample(RID p_source, RID p_sample, int p_mix_rate, int p_voice) {

	Source *source = source_owner.get(p_source);
	ERR_FAIL_COND_V(!source,SOURCE_INVALID_VOICE);

	int to_play=0;

	if (p_voice==SOURCE_NEXT_VOICE) {
		to_play=source->last_voice+1;
		if (to_play>=source->voices.size())
			to_play=0;

	} else
		to_play=p_voice;

	ERR_FAIL_INDEX_V(to_play,source->voices.size(),SOURCE_INVALID_VOICE);

	source->voices[to_play].restart=true;
	source->voices[to_play].sample_rid=p_sample;
	source->voices[to_play].sample_mix_rate=p_mix_rate;
	source->voices[to_play].pitch_scale=1;
	source->voices[to_play].volume_scale=0;
	source->last_voice=to_play;
	active_voices.insert(ActiveVoice(source,to_play));
	return to_play;
}

/* VOICES */
void SpatialSound2DServerSW::source_voice_set_pitch_scale(RID p_source, SourceVoiceID p_voice, float p_pitch_scale) {

	Source *source = source_owner.get(p_source);
	ERR_FAIL_COND(!source);
	ERR_FAIL_INDEX(p_voice,source->voices.size());
	source->voices[p_voice].pitch_scale=p_pitch_scale;

}
void SpatialSound2DServerSW::source_voice_set_volume_scale_db(RID p_source, SourceVoiceID p_voice, float p_db) {

	Source *source = source_owner.get(p_source);
	ERR_FAIL_COND(!source);
	ERR_FAIL_INDEX(p_voice,source->voices.size());
	source->voices[p_voice].volume_scale=p_db;

}

bool SpatialSound2DServerSW::source_is_voice_active(RID p_source, SourceVoiceID p_voice) const {

	Source *source = source_owner.get(p_source);
	ERR_FAIL_COND_V(!source,false);
	ERR_FAIL_INDEX_V(p_voice,source->voices.size(),false);
	return source->voices[p_voice].active || source->voices[p_voice].restart;

}
void SpatialSound2DServerSW::source_stop_voice(RID p_source, SourceVoiceID p_voice) {

	Source *source = source_owner.get(p_source);
	ERR_FAIL_COND(!source);
	ERR_FAIL_INDEX(p_voice,source->voices.size());
	if (source->voices[p_voice].active) {
		AudioServer::get_singleton()->voice_stop(source->voices[p_voice].voice_rid);
	}
	source->voices[p_voice].active=false;
	source->voices[p_voice].restart=false;
	active_voices.erase(ActiveVoice(source,p_voice));
}

/* LISTENER */

RID SpatialSound2DServerSW::listener_create() {

	Listener *listener = memnew( Listener );
	RID listener_rid = listener_owner.make_rid(listener);
	return listener_rid;

}

void SpatialSound2DServerSW::listener_set_space(RID p_listener,RID p_space) {

	Listener *listener = listener_owner.get(p_listener);
	ERR_FAIL_COND(!listener);

	if (listener->space.is_valid()) {

		Space *lspace = space_owner.get(listener->space);
		ERR_FAIL_COND(!lspace);
		lspace->listeners.erase(p_listener);
	}

	listener->space=RID();

	if (p_space.is_valid()) {
		Space *space = space_owner.get(p_space);
		ERR_FAIL_COND(!space);

		listener->space=p_space;
		space->listeners.insert(p_listener);
	}

}

void SpatialSound2DServerSW::listener_set_transform(RID p_listener, const Transform2D& p_transform) {

	Listener *listener = listener_owner.get(p_listener);
	ERR_FAIL_COND(!listener);
	listener->transform=p_transform;
	listener->transform.orthonormalize(); //must be done..
}
Transform2D SpatialSound2DServerSW::listener_get_transform(RID p_listener) const {

	Listener *listener = listener_owner.get(p_listener);
	ERR_FAIL_COND_V(!listener,Transform2D());
	return listener->transform;
}

void SpatialSound2DServerSW::listener_set_param(RID p_listener, ListenerParam p_param, float p_value) {

	ERR_FAIL_INDEX(p_param,LISTENER_PARAM_MAX);
	Listener *listener = listener_owner.get(p_listener);
	ERR_FAIL_COND(!listener);
	listener->params[p_param]=p_value;
}

float SpatialSound2DServerSW::listener_get_param(RID p_listener, ListenerParam p_param) const {

	ERR_FAIL_INDEX_V(p_param,LISTENER_PARAM_MAX,0);
	Listener *listener = listener_owner.get(p_listener);
	ERR_FAIL_COND_V(!listener,0);
	return listener->params[p_param];
}


/* MISC */

void SpatialSound2DServerSW::free(RID p_id) {


	if (space_owner.owns(p_id)) {

		Space *space = space_owner.get(p_id);
		free(space->default_room);

		while(space->listeners.size()) {
			listener_set_space(space->listeners.front()->get(),RID());
		}
		while(space->sources.size()) {
			free(space->sources.front()->get());
		}
		while(space->rooms.size()) {
			room_set_space(space->rooms.front()->get(),RID());
		}
		space_owner.free(p_id);
		memdelete(space);

	} else if (source_owner.owns(p_id)) {

		Source *source = source_owner.get(p_id);
		if (source->stream)
			source_set_audio_stream(p_id,NULL);

		Space *space = space_owner.get(source->space);
		ERR_FAIL_COND(!space);
		space->sources.erase(p_id);
		for(int i=0;i<source->voices.size();i++) {
			active_voices.erase(ActiveVoice(source,i));
		}
		source_owner.free(p_id);
		memdelete(source);
	} else if (listener_owner.owns(p_id)) {

		Listener *listener = listener_owner.get(p_id);
		if (listener->space.is_valid()) {
			Space *space = space_owner.get(listener->space);
			ERR_FAIL_COND(!space);
			space->listeners.erase(p_id);
		}
		listener_owner.free(p_id);
		memdelete(listener);

	} else if (room_owner.owns(p_id)) {

		Room *room = room_owner.get(p_id);

		if (room->space.is_valid()) {
			Space *space = space_owner.get(room->space);
			ERR_FAIL_COND(!space);
			//space->octree.erase(room->octree_id);
			space->rooms.erase(p_id);
		}
		room_owner.free(p_id);
		memdelete(room);
	} else {
		ERR_PRINT("Attempt to free invalid ID")	;
	}

}

void SpatialSound2DServerSW::_clean_up_owner(RID_OwnerBase *p_owner, const char *p_area) {

	List<RID> rids;
	p_owner->get_owned_list(&rids);

	for(List<RID>::Element *I=rids.front();I;I=I->next()) {
		if (OS::get_singleton()->is_stdout_verbose()) {

			print_line("Leaked RID ("+itos(I->get().get_id())+") of type "+String(p_area));
		}
		free(I->get());
	}
}

void SpatialSound2DServerSW::init() {

	internal_buffer = memnew_arr(int32_t, INTERNAL_BUFFER_SIZE*INTERNAL_BUFFER_MAX_CHANNELS);
	internal_buffer_channels=AudioServer::get_singleton()->get_default_channel_count();

	internal_audio_stream = memnew( InternalAudioStream );
	internal_audio_stream->owner=this;
	internal_audio_stream_rid = AudioServer::get_singleton()->audio_stream_create(internal_audio_stream);

	AudioServer::get_singleton()->stream_set_active(internal_audio_stream_rid,true);

}



bool SpatialSound2DServerSW::internal_buffer_mix(int32_t *p_buffer,int p_frames) {

	if (streaming_sources.size()==0)
		return false; //nothing to mix


	for (Set<Source*>::Element *E=streaming_sources.front();E;E=E->next()) {

		Source *s=E->get();
		ERR_CONTINUE(!s->stream);

		int channels = s->stream->get_channel_count();
		Source::StreamData &sd=s->stream_data;

		int todo=p_frames;

		AudioFilterSW filter;
		filter.set_sampling_rate(AudioServer::get_singleton()->get_default_mix_rate());
		filter.set_cutoff(sd.filter_cutoff);
		filter.set_gain(sd.filter_gain);
		filter.set_resonance(1);
		filter.set_mode(AudioFilterSW::HIGHSHELF);
		filter.set_stages(1);
		AudioFilterSW::Coeffs coefs;
		filter.prepare_coefficients(&coefs);

		int32_t in[4];
#ifndef SPATIAL_SOUND_SERVER_NO_FILTER
#define DO_FILTER(m_c)\
		{\
			float val = in[m_c];\
			float pre=val;\
			val = val*coefs.b0 + sd.filter_state[m_c].hb[0]*coefs.b1 + sd.filter_state[m_c].hb[1]*coefs.b2 + sd.filter_state[m_c].ha[0]*coefs.a1 + sd.filter_state[m_c].ha[1]*coefs.a2;\
			sd.filter_state[m_c].ha[1]=sd.filter_state[m_c].ha[0];\
			sd.filter_state[m_c].hb[1]=sd.filter_state[m_c].hb[0];			\
			sd.filter_state[m_c].hb[0]=pre;\
			sd.filter_state[m_c].ha[0]=val;\
			in[m_c]=Math::fast_ftoi(val);\
		}
#else
#define DO_FILTER(m_c)
#endif

		while(todo) {

			int to_mix=MIN(todo,INTERNAL_BUFFER_SIZE);

			s->stream->mix(internal_buffer,to_mix);

			switch(internal_buffer_channels) {

				case 2: {

					float p = sd.panning.x*0.5+0.5;
					float panf[2]={ (1.0-p),p };
					panf[0]*=sd.volume;
					panf[1]*=sd.volume;

					int32_t pan[2]={Math::fast_ftoi(panf[0]*(1<<16)),Math::fast_ftoi(panf[1]*(1<<16))};

					switch(channels) {
						case 1: {

							for(int i=0;i<to_mix;i++) {

								in[0]=internal_buffer[i];
								in[1]=internal_buffer[i];
								DO_FILTER(0);
								DO_FILTER(1);
								p_buffer[(i<<1)+0]=((in[0]>>16)*pan[0]);
								p_buffer[(i<<1)+1]=((in[1]>>16)*pan[1]);
							}
						} break;
						case 2: {

							for(int i=0;i<to_mix;i++) {

								in[0]=internal_buffer[(i<<1)+0];
								in[1]=internal_buffer[(i<<1)+1];
								DO_FILTER(0);
								DO_FILTER(1);
								p_buffer[(i<<1)+0]=((in[0]>>16)*pan[0]);
								p_buffer[(i<<1)+1]=((in[1]>>16)*pan[1]);
							}
						} break;
						case 4: {

							for(int i=0;i<to_mix;i++) {

								in[0]=(internal_buffer[(i<<2)+0]+internal_buffer[(i<<2)+2])>>1;
								in[1]=(internal_buffer[(i<<2)+1]+internal_buffer[(i<<2)+3])>>1;
								DO_FILTER(0);
								DO_FILTER(1);
								p_buffer[(i<<1)+0]=((in[0]>>16)*pan[0]);
								p_buffer[(i<<1)+1]=((in[1]>>16)*pan[1]);
							}
						} break;

					} break;

				} break;
				case 4: {

					float xp = sd.panning.x*0.5+0.5;
					float yp = sd.panning.y*0.5+0.5;
					float panf[4]={ (1.0-xp)*(1.0-yp),(xp)*(1.0-yp),(1.0-xp)*(yp),(xp)*(yp) };
					panf[0]*=sd.volume;
					panf[1]*=sd.volume;
					panf[2]*=sd.volume;
					panf[3]*=sd.volume;

					int32_t pan[4]={
						Math::fast_ftoi(panf[0]*(1<<16)),
						Math::fast_ftoi(panf[1]*(1<<16)),
						Math::fast_ftoi(panf[2]*(1<<16)),
						Math::fast_ftoi(panf[3]*(1<<16))};

					switch(channels) {
						case 1: {

							for(int i=0;i<to_mix;i++) {

								in[0]=internal_buffer[i];
								in[1]=internal_buffer[i];
								in[2]=internal_buffer[i];
								in[3]=internal_buffer[i];
								DO_FILTER(0);
								DO_FILTER(1);
								DO_FILTER(2);
								DO_FILTER(3);
								p_buffer[(i<<2)+0]=((in[0]>>16)*pan[0]);
								p_buffer[(i<<2)+1]=((in[1]>>16)*pan[1]);
								p_buffer[(i<<2)+2]=((in[2]>>16)*pan[2]);
								p_buffer[(i<<2)+3]=((in[3]>>16)*pan[3]);
							}
						} break;
						case 2: {

							for(int i=0;i<to_mix;i++) {

								in[0]=internal_buffer[(i<<1)+0];
								in[1]=internal_buffer[(i<<1)+1];
								in[2]=internal_buffer[(i<<1)+0];
								in[3]=internal_buffer[(i<<1)+1];
								DO_FILTER(0);
								DO_FILTER(1);
								DO_FILTER(2);
								DO_FILTER(3);
								p_buffer[(i<<2)+0]=((in[0]>>16)*pan[0]);
								p_buffer[(i<<2)+1]=((in[1]>>16)*pan[1]);
								p_buffer[(i<<2)+2]=((in[2]>>16)*pan[2]);
								p_buffer[(i<<2)+3]=((in[3]>>16)*pan[3]);
							}
						} break;
						case 4: {

							for(int i=0;i<to_mix;i++) {

								in[0]=internal_buffer[(i<<2)+0];
								in[1]=internal_buffer[(i<<2)+1];
								in[2]=internal_buffer[(i<<2)+2];
								in[3]=internal_buffer[(i<<2)+3];
								DO_FILTER(0);
								DO_FILTER(1);
								DO_FILTER(2);
								DO_FILTER(3);
								p_buffer[(i<<2)+0]=((in[0]>>16)*pan[0]);
								p_buffer[(i<<2)+1]=((in[1]>>16)*pan[1]);
								p_buffer[(i<<2)+2]=((in[2]>>16)*pan[2]);
								p_buffer[(i<<2)+3]=((in[3]>>16)*pan[3]);
							}
						} break;

					} break;

				} break;
				case 6: {


				} break;
			}
			p_buffer+=to_mix*internal_buffer_channels;
			todo-=to_mix;

		}

	}

	return true;
}

void SpatialSound2DServerSW::update(float p_delta) {

	List<ActiveVoice> to_disable;

	for(Set<ActiveVoice>::Element *E=active_voices.front();E;E=E->next()) {

		Source *source = E->get().source;
		int voice = E->get().voice;

		if (voice!=VOICE_IS_STREAM) {
			Source::Voice &v=source->voices[voice];
			ERR_CONTINUE(!v.active && !v.restart); // likely a bug...
		}

		//this could be optimized at some point... am not sure
		Space *space=space_owner.get(source->space);
		Room *room=room_owner.get(space->default_room);

		//compute mixing weights (support for multiple listeners in the same output)
		float total_distance=0;
		for(Set<RID>::Element *L=space->listeners.front();L;L=L->next()) {
			Listener *listener=listener_owner.get(L->get());
			float d = listener->transform.get_origin().distance_to(source->transform.get_origin());
			if (d==0)
				d=0.1;
			total_distance+=d;
		}

		//compute spatialization variables, weighted according to distance
		float volume_attenuation = 0.0;
		float air_absorption_hf_cutoff = 0.0;
		float air_absorption = 0.0;
		float pitch_scale=0.0;
		Vector2 panning;

		for(Set<RID>::Element *L=space->listeners.front();L;L=L->next()) {

			Listener *listener=listener_owner.get(L->get());

			Vector2 rel_vector = -listener->transform.xform_inv(source->transform.get_origin());
			//Vector2 source_rel_vector = source->transform.xform_inv(listener->transform.get_origin()).normalized();
			float distance=rel_vector.length();
			float weight = distance/total_distance;
			float pscale=1.0;

			float distance_scale=listener->params[LISTENER_PARAM_ATTENUATION_SCALE]*room->params[ROOM_PARAM_ATTENUATION_SCALE];
			float distance_min=source->params[SOURCE_PARAM_ATTENUATION_MIN_DISTANCE]*distance_scale;
			float distance_max=source->params[SOURCE_PARAM_ATTENUATION_MAX_DISTANCE]*distance_scale;
			float attenuation_exp=source->params[SOURCE_PARAM_ATTENUATION_DISTANCE_EXP];
			float attenuation=1;

			if (distance_max>0) {
				distance = CLAMP(distance,distance_min,distance_max);
				attenuation = Math::pow(1.0 - ((distance - distance_min)/(distance_max-distance_min)),CLAMP(attenuation_exp,0.001,16));
			}

			float hf_attenuation_cutoff = room->params[ROOM_PARAM_ATTENUATION_HF_CUTOFF];
			float hf_attenuation_exp = room->params[ROOM_PARAM_ATTENUATION_HF_RATIO_EXP];
			float hf_attenuation_floor = room->params[ROOM_PARAM_ATTENUATION_HF_FLOOR_DB];
			float absorption=Math::db2linear(Math::lerp(hf_attenuation_floor,0,Math::pow(attenuation,hf_attenuation_exp)));

			// source emission cone
/* only for 3D
			float emission_deg=source->params[SOURCE_PARAM_EMISSION_CONE_DEGREES];
			float emission_attdb=source->params[SOURCE_PARAM_EMISSION_CONE_ATTENUATION_DB];
			absorption*=_get_attenuation(source_rel_vector.dot(Vector2(0,0,-1)),emission_deg,emission_attdb);
*/
			Vector2 vpanning=rel_vector.normalized();
			if (distance < listener->params[LISTENER_PARAM_PAN_RANGE])
				vpanning*=distance/listener->params[LISTENER_PARAM_PAN_RANGE];

			//listener stuff

			{

				// head cone
/* only for 3D
				float reception_deg=listener->params[LISTENER_PARAM_RECEPTION_CONE_DEGREES];
				float reception_attdb=listener->params[LISTENER_PARAM_RECEPTION_CONE_ATTENUATION_DB];

				absorption*=_get_attenuation(vpanning.dot(Vector2(0,0,-1)),reception_deg,reception_attdb);
*/

				// scale

				attenuation*=Math::db2linear(listener->params[LISTENER_PARAM_VOLUME_SCALE_DB]);
				pscale*=Math::db2linear(listener->params[LISTENER_PARAM_PITCH_SCALE]);


			}




			//add values

			volume_attenuation+=weight*attenuation; // plus other stuff i guess
			air_absorption+=weight*absorption;
			air_absorption_hf_cutoff+=weight*hf_attenuation_cutoff;
			panning+=vpanning*weight;
			pitch_scale+=pscale*weight;

		}

		RoomReverb reverb_room=ROOM_REVERB_HALL;
		float reverb_send=0;

		/* APPLY ROOM SETTINGS */

		{
			pitch_scale*=room->params[ROOM_PARAM_PITCH_SCALE];
			volume_attenuation*=Math::db2linear(room->params[ROOM_PARAM_VOLUME_SCALE_DB]);
			reverb_room=room->reverb;
			reverb_send=Math::lerp(1.0,volume_attenuation,room->params[ROOM_PARAM_ATTENUATION_REVERB_SCALE])*room->params[ROOM_PARAM_REVERB_SEND];

		}

		/* UPDATE VOICE & STREAM */



		if (voice==VOICE_IS_STREAM) {

			//update voice!!
			source->stream_data.panning=panning;
			source->stream_data.volume=volume_attenuation*Math::db2linear(source->params[SOURCE_PARAM_VOLUME_DB]);
			source->stream_data.reverb=reverb_room;
			source->stream_data.reverb_send=reverb_send;
			source->stream_data.filter_gain=air_absorption;
			source->stream_data.filter_cutoff=air_absorption_hf_cutoff;

			if (!source->stream) //stream is gone bye bye
				to_disable.push_back(ActiveVoice(source,voice)); // oh well..

		} else if (voice>=0) {
			//update stream!!
			Source::Voice &v=source->voices[voice];

			if (v.restart)
				AudioServer::get_singleton()->voice_play(v.voice_rid,v.sample_rid);

			float volume_scale = Math::db2linear(v.volume_scale)*Math::db2linear(source->params[SOURCE_PARAM_VOLUME_DB]);
			float volume = volume_attenuation*volume_scale;
			reverb_send*=volume_scale;
			int mix_rate = v.sample_mix_rate*v.pitch_scale*pitch_scale*source->params[SOURCE_PARAM_PITCH_SCALE];


			if (mix_rate<=0) {

				ERR_PRINT("Invalid mix rate for voice (0) check for invalid pitch_scale param.");
				to_disable.push_back(ActiveVoice(source,voice)); // oh well..
				continue; //invalid mix rate, disabling
			}
			if (v.restart || v.last_volume!=volume)
				AudioServer::get_singleton()->voice_set_volume(v.voice_rid,volume);
			if (v.restart || v.last_mix_rate!=mix_rate)
				AudioServer::get_singleton()->voice_set_mix_rate(v.voice_rid,mix_rate);
			if (v.restart || v.last_filter_gain!=air_absorption || v.last_filter_cutoff!=air_absorption_hf_cutoff)
				AudioServer::get_singleton()->voice_set_filter(v.voice_rid,AudioServer::FILTER_HIGH_SHELF,air_absorption_hf_cutoff,1.0,air_absorption);
			if (v.restart || v.last_panning!=panning) {
				AudioServer::get_singleton()->voice_set_pan(v.voice_rid,-panning.x,panning.y,0);
			}
			if (v.restart || v.last_reverb_room!=reverb_room || v.last_reverb_send!=reverb_send)
				AudioServer::get_singleton()->voice_set_reverb(v.voice_rid,AudioServer::ReverbRoomType(reverb_room),reverb_send);

			v.last_volume=volume;
			v.last_mix_rate=mix_rate;
			v.last_filter_gain=air_absorption;
			v.last_filter_cutoff=air_absorption_hf_cutoff;
			v.last_panning=panning;
			v.last_reverb_room=reverb_room;
			v.last_reverb_send=reverb_send;
			v.restart=false;
			v.active=true;

			if (!AudioServer::get_singleton()->voice_is_active(v.voice_rid))
				to_disable.push_back(ActiveVoice(source,voice)); // oh well..
		}
	}

	while(to_disable.size()) {

		ActiveVoice av = to_disable.front()->get();
		av.source->voices[av.voice].active=false;
		av.source->voices[av.voice].restart=false;
		active_voices.erase(av);
		to_disable.pop_front();
	}

}
void SpatialSound2DServerSW::finish() {

	AudioServer::get_singleton()->free(internal_audio_stream_rid);
	memdelete(internal_audio_stream);

	_clean_up_owner(&source_owner,"Source");
	_clean_up_owner(&listener_owner,"Listener");
	_clean_up_owner(&room_owner,"Room");
	_clean_up_owner(&space_owner,"Space");

	memdelete_arr(internal_buffer);
}

SpatialSound2DServerSW::SpatialSound2DServerSW() {

}
