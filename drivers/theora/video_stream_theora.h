#ifndef VIDEO_STREAM_THEORA_H
#define VIDEO_STREAM_THEORA_H

#ifdef THEORA_ENABLED

#include "theora/theoradec.h"
#include "vorbis/codec.h"
#include "os/file_access.h"

#include "io/resource_loader.h"
#include "scene/resources/video_stream.h"

class VideoStreamTheora : public VideoStream {

	OBJ_TYPE(VideoStreamTheora, VideoStream);

	enum {
		MAX_FRAMES = 4,
	};

	//Image frames[MAX_FRAMES];
	Image::Format format;
	DVector<uint8_t> frame_data;
	int frames_pending;
	FileAccess* file;
	String file_name;
	int audio_frames_wrote;
	Point2i size;

	int buffer_data();
	int queue_page(ogg_page *page);
	void video_write(void);
	float get_time() const;

	ogg_sync_state   oy;
	ogg_page         og;
	ogg_stream_state vo;
	ogg_stream_state to;
	th_info      ti;
	th_comment   tc;
	th_dec_ctx       *td;
	vorbis_info      vi;
	vorbis_dsp_state vd;
	vorbis_block     vb;
	vorbis_comment   vc;
	th_pixel_fmt     px_fmt;
	double videobuf_time;
	int pp_inc;

	int theora_p;
	int vorbis_p;
	int pp_level_max;
	int pp_level;
	int videobuf_ready;

	bool playing;
	bool buffering;

	double last_update_time;
	double time;

protected:

	virtual UpdateMode get_update_mode() const;
	virtual void update();

	void clear();

	virtual bool _can_mix() const;

public:

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


	void set_file(const String& p_file);

	int get_pending_frame_count() const;
	Image pop_frame();
	Image peek_frame() const;

	VideoStreamTheora();
	~VideoStreamTheora();
};

class ResourceFormatLoaderVideoStreamTheora : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path,const String& p_original_path="");
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String& p_type) const;
	virtual String get_resource_type(const String &p_path) const;

};



#endif

#endif
