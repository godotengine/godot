#ifndef AUDIOPLAYER_H
#define AUDIOPLAYER_H

#include "scene/main/node.h"
#include "servers/audio/audio_stream.h"


class AudioPlayer : public Node {

	GDCLASS( AudioPlayer, Node )

public:

	enum MixTarget {
		MIX_TARGET_STEREO,
		MIX_TARGET_SURROUND,
		MIX_TARGET_CENTER
	};
private:
	Ref<AudioStreamPlayback> stream_playback;
	Ref<AudioStream> stream;
	Vector<AudioFrame> mix_buffer;

	volatile float setseek;
	volatile bool active;

	float mix_volume_db;
	float volume_db;
	bool autoplay;
	StringName bus;

	MixTarget mix_target;

	void _mix_audio();
	static void _mix_audios(void *self) { reinterpret_cast<AudioPlayer*>(self)->_mix_audio(); }

	void _set_playing(bool p_enable);
	bool _is_active() const;

	void _bus_layout_changed();

protected:

	void _validate_property(PropertyInfo& property) const;
	void _notification(int p_what);
	static void _bind_methods();
public:

	void set_stream(Ref<AudioStream> p_stream);
	Ref<AudioStream> get_stream() const;

	void set_volume_db(float p_volume);
	float get_volume_db() const;

	void play(float p_from_pos=0.0);
	void seek(float p_seconds);
	void stop();
	bool is_playing() const;
	float get_pos();

	void set_bus(const StringName& p_bus);
	StringName get_bus() const;

	void set_autoplay(bool p_enable);
	bool is_autoplay_enabled();

	void set_mix_target(MixTarget p_target);
	MixTarget get_mix_target() const;

	AudioPlayer();
	~AudioPlayer();
};

VARIANT_ENUM_CAST(AudioPlayer::MixTarget)
#endif // AUDIOPLAYER_H
