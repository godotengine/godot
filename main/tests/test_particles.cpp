/*************************************************************************/
/*  test_particles.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "test_particles.h"
#include "math_funcs.h"
#include "os/main_loop.h"
#include "print_string.h"
#include "servers/visual_server.h"

namespace TestParticles {

class TestMainLoop : public MainLoop {

	RID particles;
	RID instance;
	RID camera;
	RID viewport;
	RID light;
	RID scenario;

	struct InstanceInfo {

		RID instance;
		Transform base;
		Vector3 rot_axis;
	};

	List<InstanceInfo> instances;

	float ofs;
	bool quit;

public:
	virtual void input_event(const InputEvent &p_event) {
	}
	virtual void request_quit() {

		quit = true;
	}
	virtual void init() {

		VisualServer *vs = VisualServer::get_singleton();
		particles = vs->particles_create();
		vs->particles_set_amount(particles, 1000);

		instance = vs->instance_create2(particles, scenario);

		camera = vs->camera_create();

		// 		vs->camera_set_perspective( camera, 60.0,0.1, 100.0 );
		viewport = vs->viewport_create();
		vs->viewport_attach_camera(viewport, camera);
		vs->camera_set_transform(camera, Transform(Matrix3(), Vector3(0, 0, 20)));
		/*
		RID lightaux = vs->light_create( VisualServer::LIGHT_OMNI );
		vs->light_set_var( lightaux, VisualServer::LIGHT_VAR_RADIUS, 80 );
		vs->light_set_var( lightaux, VisualServer::LIGHT_VAR_ATTENUATION, 1 );
		vs->light_set_var( lightaux, VisualServer::LIGHT_VAR_ENERGY, 1.5 );
		light = vs->instance_create2( lightaux );
		*/
		RID lightaux = vs->light_create(VisualServer::LIGHT_DIRECTIONAL);
		//	vs->light_set_color( lightaux, VisualServer::LIGHT_COLOR_AMBIENT, Color(0.0,0.0,0.0) );
		light = vs->instance_create2(lightaux, scenario);

		ofs = 0;
		quit = false;
	}
	virtual bool idle(float p_time) {
		return false;
	}

	virtual bool iteration(float p_time) {

		//		VisualServer *vs=VisualServer::get_singleton();

		ofs += p_time;
		return quit;
	}
	virtual void finish() {
	}
};

MainLoop *test() {

	return memnew(TestMainLoop);
}
} // namespace TestParticles
