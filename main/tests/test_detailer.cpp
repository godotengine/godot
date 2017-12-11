/*************************************************************************/
/*  test_detailer.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "test_detailer.h"
#include "geometry.h"
#include "math_funcs.h"
#include "os/main_loop.h"
#include "print_string.h"
#include "quick_hull.h"
#include "servers/visual_server.h"
namespace TestMultiMesh {

class TestMainLoop : public MainLoop {

	RID instance;
	RID camera;
	RID viewport;
	RID light;
	RID mesh;
	RID scenario;

#define MULTIMESH_COUNT 1500

	float ofs_x, ofs_y;
	bool quit;

public:
	virtual void _update_qh() {

		VisualServer *vs = VisualServer::get_singleton();
		Vector<Vector3> vts;
		/*

		static const int s = 20;
		for(int i=0;i<s;i++) {
			Matrix3 rot(Vector3(0,1,0),i*Math_PI/s);

			for(int j=0;j<s;j++) {
				Vector3 v;
				v.x=Math::sin(j*Math_PI*2/s);
				v.y=Math::cos(j*Math_PI*2/s);

				vts.push_back( rot.xform(v*2 ) );
			}
		}*/
		/*
		Math::seed(0);
		for(int i=0;i<50;i++) {

			vts.push_back( Vector3(Math::randf()*2-1.0,Math::randf()*2-1.0,Math::randf()*2-1.0).normalized()*2);
		}*/
		/*
		vts.push_back(Vector3(0,0,1));
		vts.push_back(Vector3(0,0,-1));
		vts.push_back(Vector3(0,1,0));
		vts.push_back(Vector3(0,-1,0));
		vts.push_back(Vector3(1,0,0));
		vts.push_back(Vector3(-1,0,0));*/
		/*
		vts.push_back(Vector3(1,1,1));
		vts.push_back(Vector3(1,-1,1));
		vts.push_back(Vector3(-1,1,1));
		vts.push_back(Vector3(-1,-1,1));
		vts.push_back(Vector3(1,1,-1));
		vts.push_back(Vector3(1,-1,-1));
		vts.push_back(Vector3(-1,1,-1));
		vts.push_back(Vector3(-1,-1,-1));
*/

		DVector<Plane> convex_planes = Geometry::build_cylinder_planes(0.5, 0.7, 4, Vector3::AXIS_Z);
		Geometry::MeshData convex_data = Geometry::build_convex_mesh(convex_planes);
		vts = convex_data.vertices;

		Geometry::MeshData md;
		Error err = QuickHull::build(vts, md);
		print_line("ERR: " + itos(err));

		vs->mesh_remove_surface(mesh, 0);
		vs->mesh_add_surface_from_mesh_data(mesh, md);

		//vs->scenario_set_debug(scenario,VS::SCENARIO_DEBUG_WIREFRAME);

		/*
		RID sm = vs->shader_create();
		//vs->shader_set_fragment_code(sm,"OUT_ALPHA=mod(TIME,1);");
		//vs->shader_set_vertex_code(sm,"OUT_VERTEX=IN_VERTEX*mod(TIME,1);");
		vs->shader_set_fragment_code(sm,"OUT_DIFFUSE=vec3(1,0,1);OUT_GLOW=abs(sin(TIME));");
		RID tcmat = vs->mesh_surface_get_material(test_cube,0);
		vs->material_set_shader(tcmat,sm);
		*/
	}

	virtual void input_event(const InputEvent &p_event) {

		if (p_event.type == InputEvent::MOUSE_MOTION && p_event.mouse_motion.button_mask & 4) {

			ofs_x += p_event.mouse_motion.relative_y / 200.0;
			ofs_y += p_event.mouse_motion.relative_x / 200.0;
		}
		if (p_event.type == InputEvent::MOUSE_BUTTON && p_event.mouse_button.pressed && p_event.mouse_button.button_index == 1) {

			QuickHull::debug_stop_after++;
			_update_qh();
		}
		if (p_event.type == InputEvent::MOUSE_BUTTON && p_event.mouse_button.pressed && p_event.mouse_button.button_index == 2) {

			if (QuickHull::debug_stop_after > 0)
				QuickHull::debug_stop_after--;
			_update_qh();
		}
	}

	virtual void request_quit() {

		quit = true;
	}

	virtual void init() {

		VisualServer *vs = VisualServer::get_singleton();

		mesh = vs->mesh_create();

		scenario = vs->scenario_create();

		QuickHull::debug_stop_after = 0;
		_update_qh();

		instance = vs->instance_create2(mesh, scenario);

		camera = vs->camera_create();

		vs->camera_set_perspective(camera, 60.0, 0.1, 100.0);
		viewport = vs->viewport_create();
		vs->viewport_attach_camera(viewport, camera);
		vs->viewport_attach_to_screen(viewport);
		vs->viewport_set_scenario(viewport, scenario);

		vs->camera_set_transform(camera, Transform(Matrix3(), Vector3(0, 0, 2)));

		RID lightaux = vs->light_create(VisualServer::LIGHT_DIRECTIONAL);
		//vs->light_set_color( lightaux, VisualServer::LIGHT_COLOR_AMBIENT, Color(0.3,0.3,0.3) );
		light = vs->instance_create2(lightaux, scenario);
		vs->instance_set_transform(light, Transform(Matrix3(Vector3(0.1, 0.4, 0.7).normalized(), 0.9)));

		ofs_x = 0;
		ofs_y = 0;
		quit = false;
	}

	virtual bool idle(float p_time) {
		return false;
	}

	virtual bool iteration(float p_time) {

		VisualServer *vs = VisualServer::get_singleton();

		Transform tr_camera;
		tr_camera.rotate(Vector3(0, 1, 0), ofs_y);
		tr_camera.rotate(Vector3(1, 0, 0), ofs_x);
		tr_camera.translate(0, 0, 10);

		vs->camera_set_transform(camera, tr_camera);

		return quit;
	}
	virtual void finish() {
	}
};

MainLoop *test() {

	return memnew(TestMainLoop);
}
} // namespace TestMultiMesh
