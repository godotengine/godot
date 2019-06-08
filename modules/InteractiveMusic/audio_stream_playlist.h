#include "core/reference.h"
#include "core/resource.h"
#include "servers/audio/audio_stream.h"

class AudioStreamPlaylist : public AudioStream {
	GDCLASS(AudioStreamPlaylist, AudioStream)
	OBJ_SAVE_TYPE(AudioStream)

private:
	friend class AudioStreamPlaybackPlaylist;
	uint64_t pos;
	int sample_rate;
	bool stereo;
	int stream_count;
	
	int bpm;
	double beat_size;
	double time;
	
	enum OrderMode {
		Sequence,
		Shuffle
	};

	enum {
		MAX_STREAMS = 64
	};

	struct Streams {
		Ref<AudioStream> stream;
		Ref<AudioStreamPlayback> playback;

		int beat_count;
	};

	Streams audio_streams[MAX_STREAMS];

	OrderMode order_mode;

public:
	void reset();
	void set_position(uint64_t pos);
	void set_stereo();
	void set_stream_beats(int p_stream,int beats);
	int get_stream_beats(int p_stream);
	void set_bpm(int beats_per_minute);
	int get_bpm();
	void set_stream_count(int count);
	int get_stream_count();
	void set_list_stream(int stream_number, Ref<AudioStream> p_stream);
	Ref<AudioStream> get_list_stream(int stream_number);
	void set_order(OrderMode p_order);
	OrderMode get_order(){
		return order_mode;
	};
	
	virtual Ref<AudioStreamPlayback> instance_playback();
	virtual String get_stream_name() const;
	virtual float get_length() const { return 0; }
	AudioStreamPlaylist();

protected:
	static void _bind_methods();
	void _validate_property(PropertyInfo &property) const;
};

///////////////////////////////////////

class AudioStreamPlaybackPlaylist : public AudioStreamPlayback {
	GDCLASS(AudioStreamPlaybackPlaylist, AudioStreamPlayback)
	friend class AudioStreamPlaylist;

private:
	int buffer_size;
	enum {
		MIX_FRAC_BITS = 13,
		MIX_FRAC_LEN = (1 << MIX_FRAC_BITS),
		MIX_FRAC_MASK = MIX_FRAC_LEN - 1,
	};
	AudioFrame *pcm_buffer;
	AudioFrame *aux_buffer;
	
	Ref<AudioStreamPlaylist> instance;

	int current;
	bool fading;
	int fading_samples_total;
	int fading_time;
	int beat_amount_remaining;

	bool active;

public:
	virtual void start(float p_from_pos = 0.0);
	virtual void stop();
	virtual bool is_playing() const;
	void clear_buffer(int samples);
	virtual int get_loop_count() const; // times it looped
	virtual float get_playback_position() const;
	virtual void seek(float p_time);
	void add_stream_to_buffer(Ref<AudioStreamPlayback> playback, int samples, float p_rate_scale, float initial_volume, float final_volume);
	virtual void mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames);
	virtual float get_length() const;
	AudioStreamPlaybackPlaylist();
	~AudioStreamPlaybackPlaylist();
};
