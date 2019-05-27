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
	double bpm;
	enum order; 
	Vector<Ref<AudioStream> > audio_streams;

public:
	void reset();
	void set_position(uint64_t pos);
	void set_stereo();
	void set_stream_count(int count);
	virtual void play(Vector<Ref<AudioStream> > audio_streams, int stream_count);
	virtual Ref<AudioStreamPlayback> instance_playback();
	virtual String get_stream_name() const;
	virtual float get_length() const { return 0; }
	AudioStreamPlaylist();

protected:
	static void _get_property_list(List<PropertyInfo> *p_list);

};
