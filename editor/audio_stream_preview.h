#ifndef AUDIO_STREAM_PREVIEW_H
#define AUDIO_STREAM_PREVIEW_H

#include "os/thread.h"
#include "scene/main/node.h"
#include "servers/audio/audio_stream.h"

class AudioStreamPreview : public Reference {
	GDCLASS(AudioStreamPreview, Reference)
	friend class AudioStream;
	Vector<uint8_t> preview;
	float length;

	friend class AudioStreamPreviewGenerator;

public:
	float get_length() const;
	float get_max(float p_time, float p_time_next) const;
	float get_min(float p_time, float p_time_next) const;

	AudioStreamPreview();
};

class AudioStreamPreviewGenerator : public Node {
	GDCLASS(AudioStreamPreviewGenerator, Node)

	static AudioStreamPreviewGenerator *singleton;

	struct Preview {
		Ref<AudioStreamPreview> preview;
		Ref<AudioStream> base_stream;
		Ref<AudioStreamPlayback> playback;
		volatile bool generating;
		ObjectID id;
		Thread *thread;
	};

	Map<ObjectID, Preview> previews;

	static void _preview_thread(void *p_preview);

	void _update_emit(ObjectID p_id);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static AudioStreamPreviewGenerator *get_singleton() { return singleton; }

	Ref<AudioStreamPreview> generate_preview(const Ref<AudioStream> &p_preview);

	AudioStreamPreviewGenerator();
};

#endif // AUDIO_STREAM_PREVIEW_H
