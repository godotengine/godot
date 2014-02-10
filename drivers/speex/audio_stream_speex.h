#ifndef AUDIO_STREAM_SPEEX_H
#define AUDIO_STREAM_SPEEX_H

#include "scene/resources/audio_stream_resampled.h"
#include "speex/speex.h"
#include "os/file_access.h"
#include "io/resource_loader.h"
#include "os/thread_safe.h"

#include <speex/speex.h>
#include <speex/speex_header.h>
#include <speex/speex_stereo.h>
#include <speex/speex_callbacks.h>

#include <ogg/ogg.h>

class AudioStreamSpeex : public AudioStreamResampled {

	OBJ_TYPE(AudioStreamSpeex, AudioStreamResampled);
	_THREAD_SAFE_CLASS_

	void *st;
	SpeexBits bits;
	Vector<uint8_t> data;
	int read_ofs;
	bool active;
	String filename;
	int loop_count;
	bool loops;
	int page_size;
	bool playing;
	bool paused;
	bool packets_available;

	void unload();
	void reload();

	ogg_sync_state oy;
	ogg_page       og;
	ogg_packet     op;
	ogg_stream_state os;
	int nframes;
	int frame_size;
	int packet_no;

	ogg_int64_t page_granule, last_granule;
	int skip_samples, page_nb_packets;

	void* process_header(ogg_packet *op, int *frame_size, int *rate, int *nframes, int *channels, int *extra_headers);

	static void _bind_methods();

protected:

	virtual bool _can_mix() const;

	Dictionary _get_bundled() const;
	void _set_bundled(const Dictionary& dict);

public:


	void set_file(const String& p_file);
	String get_file() const;

	virtual void play();
	virtual void stop();
	virtual bool is_playing() const;

	virtual void set_paused(bool p_paused);
	virtual bool is_paused(bool p_paused) const;

	virtual void set_loop(bool p_enable);
	virtual bool has_loop() const;

	virtual float get_length() const;

	virtual String get_stream_name() const;

	virtual int get_loop_count() const;

	virtual float get_pos() const;
	virtual void seek_pos(float p_time);

	virtual UpdateMode get_update_mode() const;
	virtual void update();

	AudioStreamSpeex();
	~AudioStreamSpeex();
};

class ResourceFormatLoaderAudioStreamSpeex : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path,const String& p_original_path="");
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String& p_type) const;
	virtual String get_resource_type(const String &p_path) const;

};

#endif // AUDIO_STREAM_SPEEX_H
