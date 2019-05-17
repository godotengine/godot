#include "core/reference.h"
#include "core/resource.h"
#include "servers/audio/audio_stream.h"

class AudioStreamTransitioner : public AudioStream {
	GDCLASS(AudioStreamTransitioner, AudioStream);

	private:
		friend class AudioStreamPlaybackMyTone;
		uint64_t pos;
		int mix_rate;
		bool stereo;
		int bpm;

};
