#include "core/reference.h"
#include "core/resource.h"
#include "servers/audio/audio_stream.h"

class AudioStreamPlaybackTransitioner;

class AudioStreamTransitioner : AudioStream {
	GDCLASS(AudioStreamTransitioner, AudioStream)
	OBJ_SAVE_TYPE(AudioStream)

private:
	friend class AudioStreamPlaybackTransitioner;
	uint64_t pos;
	int sample_rate;
	bool stereo;
	int clip_count;
	int transition_count;
	int bpm;
	double time;

	enum {
		MAX_STREAMS = 64,
		MAX_TRANSITIONS = 64
	};


	struct Transition {
		int next_clip_number;

		int fade_in_beats;
		int fade_out_beats;

		char next_clip_name;
	};

	Transition transitions[MAX_TRANSITIONS];

	Ref<AudioStream> audio_streams[MAX_STREAMS];
	Set<AudioStreamPlaybackTransitioner *> playbacks;

public:
	void reset();
	void set_position(uint64_t pos);

	void set_bpm(int p_bpm);
	int get_bpm();

	void set_clip_count(int p_clip_count);
	int get_clip_count();
	void set_list_clip(int stream_number, Ref<AudioStream> p_stream);
	Ref<AudioStream> get_list_clip(int stream_number);

	void set_transition_count(int p_transition_count);
	int get_transition_count();

	void set_transition_fade_in(int transition_number, int fade_in);
	int get_transition_fade_in(int transition_number);

	void set_transition_fade_out(int transition_number, int fade_out);
	int get_transition_fade_out(int transition_number);

	void set_next_clip(int transition_number, Ref<AudioStream> next_clip);
	Ref<AudioStream> get_next_clip(int transition_number);

	void set_active_transition(Transition t_transition);

	virtual Ref<AudioStreamPlayback> instance_playback();
	virtual String get_stream_name() const;
	virtual float get_length() const { return 0; }
	AudioStreamTransitioner();

protected:
	static void _bind_methods();
	void _validate_property(PropertyInfo &property) const;
};

//consider whether or not it's worth only using transitions and importing the streams into them
//find out how to be able to click on the part where next stream is set on transition and it shows you a drop list of all the streams already in transitioner
//

class AudioStreamPlaybackTransitioner : AudioStreamPlayback {
	GDCLASS(AudioStreamPlaybackTransitioner, AudioStreamPlayback)
	friend class AudioStreamTransitioner;

private:
	enum {
		MIX_BUFFER_SIZE = 128
	};
	enum {
		MIX_FRAC_BITS = 13,
		MIX_FRAC_LEN = (1 << MIX_FRAC_BITS),
		MIX_FRAC_MASK = MIX_FRAC_LEN - 1,
	};
	AudioFrame pcm_buffer[MIX_BUFFER_SIZE];
	AudioFrame aux_buffer[MIX_BUFFER_SIZE];

	Ref<AudioStreamTransitioner> transitioner;
	Ref<AudioStreamPlayback> playbacks[AudioStreamTransitioner::MAX_STREAMS];

	int current;
	int next;
	int fade_in_samples_total;
	int fade_out_samples_total;
	int transition_samples_total;

	bool active;

	virtual void _update_playback_instances();

public:
	virtual void start(float p_from_pos = 0.0);
	virtual void stop();
	virtual bool is_playing() const;
	void clear_buffer(int samples);
	virtual int get_loop_count() const; // times it looped
	virtual float get_playback_position() const;
	virtual void seek(float p_time);
	void add_stream_to_buffer(Ref<AudioStreamTransitioner> playback, int samples, float p_rate_scale, float initial_volume, float final_volume);
	virtual void mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames);
	virtual float get_length() const;
	AudioStreamPlaybackTransitioner();
	~AudioStreamPlaybackTransitioner();
};
