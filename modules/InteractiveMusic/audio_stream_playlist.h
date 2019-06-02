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
	//double beat_size = 60 / bpm;
	//double time;
	enum OrderMode {
		Sequence,
		Shuffle
	};
	OrderMode order_mode;
	Vector<Ref<AudioStream> > audio_streams;

public:
	void reset();
	void set_position(uint64_t pos);
	void set_stereo();
	void set_bpm(int beats);
	void set_stream_count(int count);
	void set_order(OrderMode p_order);
	virtual void play(Vector<Ref<AudioStream> > audio_streams, int stream_count);
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
	int buffer_size = 256;
	enum {
		MIX_FRAC_BITS = 13,
		MIX_FRAC_LEN = (1 << MIX_FRAC_BITS),
		MIX_FRAC_MASK = MIX_FRAC_LEN - 1,
	};
	AudioFrame *pcm_buffer;
	Ref<AudioStreamPlaylist> instance;
	bool active;

public:
	virtual void start(float p_from_pos = 0.0);
	virtual void stop();
	virtual bool is_playing() const;
	virtual int get_loop_count() const; // times it looped
	virtual float get_playback_position() const;
	virtual void seek(float p_time);
	virtual void mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames);
	virtual float get_length() const;
	AudioStreamPlaybackPlaylist();
	~AudioStreamPlaybackPlaylist();
};
