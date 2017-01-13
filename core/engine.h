#ifndef ENGINE_H
#define ENGINE_H

#include "ustring.h"
#include "list.h"
#include "vector.h"
#include "os/main_loop.h"

class Engine {

friend class Main;

	String _custom_level;
	uint64_t frames_drawn;
	uint32_t _frame_delay;

	int ips;
	float _fps;
	int _target_fps;
	float _time_scale;
	bool _pixel_snap;
	uint64_t _fixed_frames;
	uint64_t _idle_frames;
	bool _in_fixed;

	static Engine *singleton;
public:

	static Engine *get_singleton();

	virtual void set_iterations_per_second(int p_ips);
	virtual int get_iterations_per_second() const;

	virtual void set_target_fps(int p_fps);
	virtual float get_target_fps() const;

	virtual float get_frames_per_second() const { return _fps; }

	String get_custom_level() const { return _custom_level; }

	uint64_t get_frames_drawn();

	uint64_t get_fixed_frames() const { return _fixed_frames; }
	uint64_t get_idle_frames() const { return _idle_frames; }
	bool is_in_fixed_frame() const { return _in_fixed; }

	void set_time_scale(float p_scale);
	float get_time_scale() const;

	void set_frame_delay(uint32_t p_msec);
	uint32_t get_frame_delay() const;

	_FORCE_INLINE_ bool get_use_pixel_snap() const { return _pixel_snap; }

	String get_version() const;
	String get_version_name() const;
	String get_version_short_name() const;
	int get_version_major() const;
	int get_version_minor() const;
	String get_version_revision() const;
	String get_version_status() const;
	int get_version_year() const;

	Engine();
};

#endif // ENGINE_H
