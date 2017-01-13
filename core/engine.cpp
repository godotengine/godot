#include "engine.h"
#include "version.h"

void Engine::set_iterations_per_second(int p_ips) {

	ips=p_ips;
}
int Engine::get_iterations_per_second() const {

	return ips;
}

void Engine::set_target_fps(int p_fps) {
	_target_fps=p_fps>0? p_fps : 0;
}

float Engine::get_target_fps() const {
	return _target_fps;
}

uint64_t Engine::get_frames_drawn() {

	return frames_drawn;
}

void Engine::set_frame_delay(uint32_t p_msec) {

	_frame_delay=p_msec;
}

uint32_t Engine::get_frame_delay() const {

	return _frame_delay;
}

void Engine::set_time_scale(float p_scale) {

	_time_scale=p_scale;
}

float Engine::get_time_scale() const {

	return _time_scale;
}


String Engine::get_version() const {

	return VERSION_FULL_NAME;
}
String Engine::get_version_name() const{

	return _MKSTR(VERSION_NAME);
}
String Engine::get_version_short_name() const{

	return _MKSTR(VERSION_SHORT_NAME);

}
int Engine::get_version_major() const{

	return VERSION_MAJOR;
}
int Engine::get_version_minor() const{

	return VERSION_MINOR;
}
String Engine::get_version_revision() const{

	return _MKSTR(VERSION_REVISION);
}
String Engine::get_version_status() const{

	return _MKSTR(VERSION_STATUS);
}
int Engine::get_version_year() const{

	return VERSION_YEAR;
}


Engine *Engine::singleton=NULL;

Engine *Engine::get_singleton() {
	return singleton;
}

Engine::Engine()
{

	singleton=this;
	frames_drawn=0;
	ips=60;
	_frame_delay=0;
	_fps=1;
	_target_fps=0;
	_time_scale=1.0;
	_pixel_snap=false;
	_fixed_frames=0;
	_idle_frames=0;
	_in_fixed=false;
}
