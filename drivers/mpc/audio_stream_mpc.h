#ifndef AUDIO_STREAM_MPC_H
#define AUDIO_STREAM_MPC_H

#include "scene/resources/audio_stream_resampled.h"
#include "os/file_access.h"
#include "mpc/mpcdec.h"
#include "os/thread_safe.h"
#include "io/resource_loader.h"
//#include "../libmpcdec/decoder.h"
//#include "../libmpcdec/internal.h"
class AudioStreamMPC : public AudioStreamResampled {

	OBJ_TYPE( AudioStreamMPC, AudioStreamResampled );

	_THREAD_SAFE_CLASS_

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

	virtual bool _can_mix() const ;

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

	AudioStreamMPC();
	~AudioStreamMPC();
};


class ResourceFormatLoaderAudioStreamMPC : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path,const String& p_original_path="");
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String& p_type) const;
	virtual String get_resource_type(const String &p_path) const;

};

#endif // AUDIO_STREAM_MPC_H
