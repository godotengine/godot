#ifndef AUDIO_STREAM_STB_VORBIS_H
#define AUDIO_STREAM_STB_VORBIS_H

#include "servers/audio/audio_stream.h"
#include "io/resource_loader.h"

#define STB_VORBIS_HEADER_ONLY
#include "thirdparty/stb_vorbis/stb_vorbis.c"
#undef STB_VORBIS_HEADER_ONLY


class AudioStreamOGGVorbis;

class AudioStreamPlaybackOGGVorbis : public AudioStreamPlaybackResampled {

	GDCLASS( AudioStreamPlaybackOGGVorbis, AudioStreamPlaybackResampled )

	stb_vorbis * ogg_stream;
	stb_vorbis_alloc ogg_alloc;
	uint32_t frames_mixed;
	bool active;
	int loops;

friend class AudioStreamOGGVorbis;

	Ref<AudioStreamOGGVorbis> vorbis_stream;
protected:

	virtual void _mix_internal(AudioFrame* p_buffer, int p_frames);
	virtual float get_stream_sampling_rate();

public:
	virtual void start(float p_from_pos=0.0);
	virtual void stop();
	virtual bool is_playing() const;

	virtual int get_loop_count() const; //times it looped

	virtual float get_pos() const;
	virtual void seek_pos(float p_time);

	virtual float get_length() const; //if supported, otherwise return 0

	AudioStreamPlaybackOGGVorbis() {   }
	~AudioStreamPlaybackOGGVorbis();
};

class AudioStreamOGGVorbis : public AudioStream {

	GDCLASS( AudioStreamOGGVorbis, AudioStream )
	OBJ_SAVE_TYPE( AudioStream ) //children are all saved as AudioStream, so they can be exchanged

friend class AudioStreamPlaybackOGGVorbis;

	void *data;
	uint32_t data_len;

	int decode_mem_size;
	float sample_rate;
	int channels;
	float length;

public:


	virtual Ref<AudioStreamPlayback> instance_playback();
	virtual String get_stream_name() const;

	Error setup(const uint8_t *p_data, uint32_t p_data_len);

	AudioStreamOGGVorbis();
};

class ResourceFormatLoaderAudioStreamOGGVorbis : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path,const String& p_original_path="",Error *r_error=NULL);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String& p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};



#endif
