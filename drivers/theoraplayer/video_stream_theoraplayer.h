#ifndef VIDEO_STREAM_THEORAPLAYER_H
#define VIDEO_STREAM_THEORAPLAYER_H

#include "scene/resources/video_stream.h"
#include "io/resource_loader.h"

class TheoraVideoManager;
class TheoraVideoClip;

class VideoStreamTheoraplayer : public VideoStream {

	OBJ_TYPE(VideoStreamTheoraplayer,VideoStream);

	mutable Image frame;
	TheoraVideoManager* mgr;
	TheoraVideoClip* clip;
	bool started;
	bool playing;
	bool loop;

public:

	virtual void stop();
	virtual void play();

	virtual bool is_playing() const;

	virtual void set_paused(bool p_paused);
	virtual bool is_paused(bool p_paused) const;

	virtual void set_loop(bool p_enable);
	virtual bool has_loop() const;

	virtual float get_pos() const;
	virtual void seek_pos(float p_time);

	virtual float get_length() const;

	virtual int get_pending_frame_count() const;
	virtual Image pop_frame();
	virtual Image peek_frame() const;

	void update(float p_time);

	void set_file(const String& p_file);

	~VideoStreamTheoraplayer();
	VideoStreamTheoraplayer();
};

class ResourceFormatLoaderVideoStreamTheoraplayer : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path,const String& p_original_path="");
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String& p_type) const;
	virtual String get_resource_type(const String &p_path) const;

};


#endif

