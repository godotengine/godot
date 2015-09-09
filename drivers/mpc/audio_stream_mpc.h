#ifndef AUDIO_STREAM_MPC_H
#define AUDIO_STREAM_MPC_H

#include "scene/resources/audio_stream.h"
#include "os/file_access.h"
#include "mpc/mpcdec.h"
#include "os/thread_safe.h"
#include "io/resource_loader.h"
//#include "../libmpcdec/decoder.h"
//#include "../libmpcdec/internal.h"

class AudioStreamPlaybackMPC : public AudioStreamPlayback {

	OBJ_TYPE( AudioStreamPlaybackMPC, AudioStreamPlayback );

	bool preload;
	FileAccess *f;
	String file;
	DVector<uint8_t> data;
	int data_ofs;
	int streamlen;


	bool active;
	bool paused;
	bool loop;
	int loops;

	// mpc
	mpc_reader reader;
	mpc_demux* demux;
	mpc_streaminfo si;
	MPC_SAMPLE_FORMAT sample_buffer[MPC_DECODER_BUFFER_LENGTH];

	static mpc_int32_t _mpc_read(mpc_reader *p_reader,void *p_dst, mpc_int32_t p_bytes);
	static mpc_bool_t _mpc_seek(mpc_reader *p_reader,mpc_int32_t p_offset);
	static mpc_int32_t _mpc_tell(mpc_reader *p_reader);
	static mpc_int32_t _mpc_get_size(mpc_reader *p_reader);
	static mpc_bool_t _mpc_canseek(mpc_reader *p_reader);

	int stream_min_size;
	int stream_rate;
	int stream_channels;

protected:
	Error _open_file();
	void _close_file();
	int _read_file(void *p_dst,int p_bytes);
	bool _seek_file(int p_pos);
	int _tell_file()  const;
	int _sizeof_file() const;
	bool _canseek_file() const;


	Error _reload();
	static void _bind_methods();

public:

	void set_file(const String& p_file);
	String get_file() const;

	virtual void play(float p_offset=0);
	virtual void stop();
	virtual bool is_playing() const;


	virtual void set_loop(bool p_enable);
	virtual bool has_loop() const;

	virtual float get_length() const;

	virtual String get_stream_name() const;

	virtual int get_loop_count() const;

	virtual float get_pos() const;
	virtual void seek_pos(float p_time);

	virtual int get_channels() const { return stream_channels; }
	virtual int get_mix_rate() const { return stream_rate; }

	virtual int get_minimum_buffer_size() const { return stream_min_size; }
	virtual int mix(int16_t* p_bufer,int p_frames);

	virtual void set_loop_restart_time(float p_time) {  }

	AudioStreamPlaybackMPC();
	~AudioStreamPlaybackMPC();
};

class AudioStreamMPC : public AudioStream {

	OBJ_TYPE( AudioStreamMPC, AudioStream );

	String file;
public:

	Ref<AudioStreamPlayback> instance_playback() {
		Ref<AudioStreamPlaybackMPC> pb = memnew( AudioStreamPlaybackMPC );
		pb->set_file(file);
		return pb;
	}

	void set_file(const String& p_file) { file=p_file; }


};

class ResourceFormatLoaderAudioStreamMPC : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path,const String& p_original_path="",Error *r_error=NULL);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String& p_type) const;
	virtual String get_resource_type(const String &p_path) const;

};

#endif // AUDIO_STREAM_MPC_H
