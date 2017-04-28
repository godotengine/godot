/*************************************************************************/
/*  test_physics.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "test_physics.h"

#include "map.h"
#include "math_funcs.h"
#include "os/main_loop.h"
#include "os/os.h"
#include "print_string.h"
#include "quick_hull.h"
#include "servers/physics_server.h"
#include "servers/visual_server.h"

class TestPhysicsMainLoop : public MainLoop {

	GDCLASS(TestPhysicsMainLoop, MainLoop);

	enum {
		LINK_COUNT = 20,
	};

	RID test_cube;

	RID plane;
	RID sphere;
	RID light;
	RID camera;
	RID mover;
	RID scenario;
	RID space;

	RID character;

	float ofs_x, ofs_y;

	Point2 joy_direction;

	List<RID> bodies;
	Map<PhysicsServer::ShapeType, RID> type_shape_map;
	Map<PhysicsServer::ShapeType, RID> type_mesh_map;

	void body_changed_transform(Object *p_state, RID p_visual_instance) {

		PhysicsDirectBodyState *state = (PhysicsDirectBodyState *)p_state;
		VisualServer *vs = VisualServer::get_singleton();
		Transform t = state->get_transform();
		//t.basis.scale( Vector3(1.0,0.5,0.2) );
		vs->instance_set_transform(p_visual_instance, t);
	}

	bool quit;

protected:
	static void _bind_methods() {

		ClassDB::bind_method("body_changed_transform", &TestPhysicsMainLoop::body_changed_transform);
	}

	RID create_body(PhysicsServer::ShapeType p_shape, PhysicsServer::BodyMode p_body, const Transform p_location, bool p_active_default = true, const Transform &p_shape_xform = Transform()) {

		VisualServer *vs = VisualServer::get_singleton();
		PhysicsServer *ps = PhysicsServer::get_singleton();

		RID mesh_instance = vs->instance_create2(type_mesh_map[p_shape], scenario);
		RID body = ps->body_create(p_body, !p_active_default);
		ps->body_set_space(body, space);
		ps->body_set_param(body, PhysicsServer::BODY_PARAM_BOUNCE, 0.0);
		//todo set space
		ps->body_add_shape(body, type_shape_map[p_shape]);
		ps->body_set_force_integration_callback(body, this, "body_changed_transform", mesh_instance);

		ps->body_set_state(body, PhysicsServer::BODY_STATE_TRANSFORM, p_location);
		bodies.push_back(body);

		if (p_body == PhysicsServer::BODY_MODE_STATIC) {

			vs->instance_set_transform(mesh_instance, p_location);
		}
		return body;
	}

	RID create_static_plane(const Plane &p_plane) {

		PhysicsServer *ps = PhysicsServer::get_singleton();

		RID plane_shape = ps->shape_create(PhysicsServer::SHAPE_PLANE);
		ps->shape_set_data(plane_shape, p_plane);

		RID b = ps->body_create(PhysicsServer::BODY_MODE_STATIC);
		ps->body_set_space(b, space);
		//todo set space
		ps->body_add_shape(b, plane_shape);
		return b;
	}

	void configure_body(RID p_body, float p_mass, float p_friction, float p_bounce) {

		PhysicsServer *ps = PhysicsServer::get_singleton();
		ps->body_set_param(p_body, PhysicsServer::BODY_PARAM_MASS, p_mass);
		ps->body_set_param(p_body, PhysicsServer::BODY_PARAM_FRICTION, p_friction);
		ps->body_set_param(p_body, PhysicsServer::BODY_PARAM_BOUNCE, p_bounce);
	}

	void init_shapes() {

		VisualServer *vs = VisualServer::get_singleton();
		PhysicsServer *ps = PhysicsServer::get_singleton();

		/* SPHERE SHAPE */
		RID sphere_mesh = vs->make_sphere_mesh(10, 20, 0.5);
		type_mesh_map[PhysicsServer::SHAPE_SPHERE] = sphere_mesh;

		RID sphere_shape = ps->shape_create(PhysicsServer::SHAPE_SPHERE);
		ps->shape_set_data(sphere_shape, 0.5);
		type_shape_map[PhysicsServer::SHAPE_SPHERE] = sphere_shape;

		/* BOX SHAPE */

		PoolVector<Plane> box_planes = Geometry::build_box_planes(Vector3(0.5, 0.5, 0.5));
		RID box_mesh = vs->mesh_create();
		Geometry::MeshData box_data = Geometry::build_convex_mesh(box_planes);
		vs->mesh_add_surface_from_mesh_data(box_mesh, box_data);
		type_mesh_map[PhysicsServer::SHAPE_BOX] = box_mesh;

		RID box_shape = ps->shape_create(PhysicsServer::SHAPE_BOX);
		ps->shape_set_data(box_shape, Vector3(0.5, 0.5, 0.5));
		type_shape_map[PhysicsServer::SHAPE_BOX] = box_shape;

		/* CAPSULE SHAPE */

		PoolVector<Plane> capsule_planes = Geometry::build_capsule_planes(0.5, 0.7, 12, Vector3::AXIS_Z);

		RID capsule_mesh = vs->mesh_create();
		Geometry::MeshData capsule_data = Geometry::build_convex_mesh(capsule_planes);
		vs->mesh_add_surface_from_mesh_data(capsule_mesh, capsule_data);

		type_mesh_map[PhysicsServer::SHAPE_CAPSULE] = capsule_mesh;

		RID capsule_shape = ps->shape_create(PhysicsServer::SHAPE_CAPSULE);
		Dictionary capsule_params;
		capsule_params["radius"] = 0.5;
		capsule_params["height"] = 1.4;
		ps->shape_set_data(capsule_shape, capsule_params);
		type_shape_map[PhysicsServer::SHAPE_CAPSULE] = capsule_shape;

		/* CONVEX SHAPE */

		PoolVector<Plane> convex_planes = Geometry::build_cylinder_planes(0.5, 0.7, 5, Vector3::AXIS_Z);

		RID convex_mesh = vs->mesh_create();
		Geometry::MeshData convex_data = Geometry::build_convex_mesh(convex_planes);
		QuickHull::build(convex_data.vertices, convex_data);
		vs->mesh_add_surface_from_mesh_data(convex_mesh, convex_data);

		type_mesh_map[PhysicsServer::SHAPE_CONVEX_POLYGON] = convex_mesh;

		RID convex_shape = ps->shape_create(PhysicsServer::SHAPE_CONVEX_POLYGON);
		ps->shape_set_data(convex_shape, convex_data.vertices);
		type_shape_map[PhysicsServer::SHAPE_CONVEX_POLYGON] = convex_shape;
	}

	void make_trimesh(Vector<Vector3> p_faces, const Transform &p_xform = Transform()) {

		VisualServer *vs = VisualServer::get_singleton();
		PhysicsServer *ps = PhysicsServer::get_singleton();
		RID trimesh_shape = ps->shape_create(PhysicsServer::SHAPE_CONCAVE_POLYGON);
		ps->shape_set_data(trimesh_shape, p_faces);
		p_faces = ps->shape_get_data(trimesh_shape); // optimized one
		Vector<Vector3> normals; // for drawing
		for (int i = 0; i < p_faces.size() / 3; i++) {

			Plane p(p_faces[i * 3 + 0], p_faces[i * 3 + 1], p_faces[i * 3 + 2]);
			normals.push_back(p.normal);
			normals.push_back(p.normal);
			normals.push_back(p.normal);
		}

		RID trimesh_mesh = vs->mesh_create();
		Array d;
		d.resize(VS::ARRAY_MAX);
		d[VS::ARRAY_VERTEX] = p_faces;
		d[VS::ARRAY_NORMAL] = normals;
		vs->mesh_add_surface_from_arrays(trimesh_mesh, VS::PRIMITIVE_TRIANGLES, d);
		//vs->material_set_flag( trimesh_mat, VisualServer::MATERIAL_FLAG_UNSHADED,true);

		RID triins = vs->instance_create2(trimesh_mesh, scenario);

		RID tribody = ps->body_create(PhysicsServer::BODY_MODE_STATIC);
		ps->body_set_space(tribody, space);
		//todo set space
		ps->body_add_shape(tribody, trimesh_shape);
		Transform tritrans = p_xform;
		ps->body_set_state(tribody, PhysicsServer::BODY_STATE_TRANSFORM, tritrans);
		vs->instance_set_transform(triins, tritrans);
		//RID trimesh_material = vs->fixed_material_create();
		//vs->material_generate( trimesh_material, Color(0.2,0.4,0.6) );
		//vs->mesh_surface_set_material( trimesh_mesh, 0, trimesh_material );
	}

	void make_grid(int p_width, int p_height, float p_cellsize, float p_cellheight, const Transform &p_xform = Transform()) {

		Vector<Vector<float> > grid;

		grid.resize(p_width);

		for (int i = 0; i < p_width; i++) {

			grid[i].resize(p_height);

			for (int j = 0; j < p_height; j++) {

				grid[i][j] = 1.0 + Math::random(-p_cellheight, p_cellheight);
			}
		}

		Vector<Vector3> faces;

		for (int i = 1; i < p_width; i++) {

			for (int j = 1; j < p_height; j++) {

#define MAKE_VERTEX(m_x, m_z) \
	faces.push_back(Vector3((m_x - p_width / 2) * p_cellsize, grid[m_x][m_z], (m_z - p_height / 2) * p_cellsize))

				MAKE_VERTEX(i, j - 1);
				MAKE_VERTEX(i, j);
				MAKE_VERTEX(i - 1, j);

				MAKE_VERTEX(i - 1, j - 1);
				MAKE_VERTEX(i, j - 1);
				MAKE_VERTEX(i - 1, j);
			}
		}

		make_trimesh(faces, p_xform);
	}

public:
	virtual void input_event(const InputEvent &p_event) {

		if (p_event.type == InputEvent::MOUSE_MOTION && p_event.mouse_motion.button_mask & 4) {

			ofs_y -= p_event.mouse_motion.relative_y / 200.0;
			ofs_x += p_event.mouse_motion.relative_x / 200.0;
		}

		if (p_event.type == InputEvent::MOUSE_MOTION && p_event.mouse_motion.button_mask & 1) {

			float y = -p_event.mouse_motion.relative_y / 20.0;
			float x = p_event.mouse_motion.relative_x / 20.0;

			if (mover.is_valid()) {

				PhysicsServer *ps = PhysicsServer::get_singleton();
				Transform t = ps->body_get_state(mover, PhysicsServer::BODY_STATE_TRANSFORM);
				t.origin += Vector3(x, y, 0);

				ps->body_set_state(mover, PhysicsServer::BODY_STATE_TRANSFORM, t);
			}
		}

		if (p_event.type == InputEvent::JOYPAD_MOTION) {

			if (p_event.joy_motion.axis == 0) {

				joy_direction.x = p_event.joy_motion.axis_value;
			};

			if (p_event.joy_motion.axis == 1) {

				joy_direction.y = p_event.joy_motion.axis_value;
			};
		};
	}

	virtual void request_quit() {

		quit = true;
	}
	virtual void init() {

		ofs_x = ofs_y = 0;
		init_shapes();

		PhysicsServer *ps = PhysicsServer::get_singleton();
		space = ps->space_create();
		ps->space_set_active(space, true);

		VisualServer *vs = VisualServer::get_singleton();

		/* LIGHT */
		RID lightaux = vs->light_create(VisualServer::LIGHT_DIRECTIONAL);
		//vs->light_set_color( lightaux, VisualServer::LIGHT_COLOR_AMBIENT, Color(0.0,0.0,0.0) );
		scenario = vs->scenario_create();
		vs->light_set_shadow(lightaux, true);
		light = vs->instance_create2(lightaux, scenario);
		Transform t;
		t.rotate(Vector3(1.0, 0, 0), 0.6);
		vs->instance_set_transform(light, t);

		/* CAMERA */

		camera = vs->camera_create();

		RID viewport = vs->viewport_create();
		Size2i screen_size = OS::get_singleton()->get_window_size();
		vs->viewport_set_size(viewport, screen_size.x, screen_size.y);
		vs->viewport_attach_to_screen(viewport, Rect2(Vector2(), screen_size));
		vs->viewport_set_active(viewport, true);
		vs->viewport_attach_camera(viewport, camera);
		vs->viewport_set_scenario(viewport, scenario);

		vs->camera_set_perspective(camera, 60, 0.1, 40.0);
		vs->camera_set_transform(camera, Transform(Basis(), Vector3(0, 9, 12)));
		//vs->scenario_set_debug(scenario,VS::SCENARIO_DEBUG_WIREFRAME);

		Transform gxf;
		gxf.basis.scale(Vector3(1.4, 0.4, 1.4));
		gxf.origin = Vector3(-2, 1, -2);
		make_grid(5, 5, 2.5, 1, gxf);
		//create_body(PhysicsServer::SHAPE_BOX,PhysicsServer::BODY_MODE_STATIC,gxf);
		//create_static_plane( Plane( Vector3(0,1,0), -2) );
		//test_joint();
		test_fall();
		//test_joint();

		/*
		Vector<Vector3> faces;
		faces.push_back( Vector3(10,0,-5) );
		faces.push_back( Vector3(0,0,10) );
		faces.push_back( Vector3(-10,-0.2,-5) );
		make_trimesh(faces);
*/
		/* Make Trimesh */
		quit = false;
		return;

#if 0
#define GRID_SIZE 5

		float grid[GRID_SIZE][GRID_SIZE];

		for (int i=0;i<GRID_SIZE;i++) {

			for (int j=0;j<GRID_SIZE;j++) {

				grid[j][i]=Math::random(0.0, 1.0 );
			}
		}

		Vector<Vector3> faces;

		for (int i=1;i<GRID_SIZE;i++) {

			for (int j=1;j<GRID_SIZE;j++) {

#define MAKE_VERTEX(m_x, m_z) \
	faces.push_back(Vector3(m_x - GRID_SIZE / 2.0, grid[m_x][m_z], m_z - GRID_SIZE / 2.0) * 3.0)

				MAKE_VERTEX(i,j-1);
				MAKE_VERTEX(i,j);
				MAKE_VERTEX(i-1,j);

				MAKE_VERTEX(i-1,j-1);
				MAKE_VERTEX(i,j-1);
				MAKE_VERTEX(i-1,j);

			}
		}
		/*
		faces.clear();
		faces.push_back( Vector3(0,0,-5) );
		faces.push_back( Vector3(1,0,-1) );
		faces.push_back( Vector3(-1,-0,-1) );
		*/

		RID trimesh_shape = ps->shape_create();
		ps->shape_set_data(trimesh_shape, PhysicsServer::SHAPE_CONCAVE_POLYGON,faces);
		faces=ps->shape_get_shape(trimesh_shape, 0);
		Vector<Vector3> normals; // for drawing
		for (int i=0;i<faces.size()/3;i++) {

			Plane p( faces[i*3+0],faces[i*3+1], faces[i*3+2] );
			normals.push_back(p.normal);
			normals.push_back(p.normal);
			normals.push_back(p.normal);
		}

		RID trimesh_mesh = vs->mesh_create();
		vs->mesh_add_surface(trimesh_mesh, VS::PRIMITIVE_TRIANGLES, VS::ARRAY_FORMAT_VERTEX|VS::ARRAY_FORMAT_NORMAL, faces.size() );
		vs->mesh_surface_set_array(trimesh_mesh,0,VS::ARRAY_VERTEX, faces );
		vs->mesh_surface_set_array(trimesh_mesh,0,VS::ARRAY_NORMAL, normals );
		RID trimesh_mat = vs->fixed_material_create();
		vs->material_generate( trimesh_mat, Color(1.0,0.5,0.3) );
		vs->mesh_surface_set_material( trimesh_mesh, 0, trimesh_mat );

		RID triins = vs->instance_create2(trimesh_mesh);



		RID tribody = ps->body_create( PhysicsServer::BODY_MODE_STATIC, trimesh_shape);
		Transform tritrans = Transform( Basis(), Vector3(0,0,-2) );
		ps->body_set_state( tribody, PhysicsServer::BODY_STATE_TRANSFORM, tritrans );
		vs->instance_set_transform( triins, tritrans );
		RID trimesh_material = vs->fixed_material_create();
		vs->material_generate( trimesh_material, Color(0.2,0.4,0.6) );
		vs->mesh_surface_set_material( trimesh_mesh, 0, trimesh_material );
#endif
	}
	virtual bool iteration(float p_time) {

		if (mover.is_valid()) {
			static float joy_speed = 10;
			PhysicsServer *ps = PhysicsServer::get_singleton();
			Transform t = ps->body_get_state(mover, PhysicsServer::BODY_STATE_TRANSFORM);
			t.origin += Vector3(joy_speed * joy_direction.x * p_time, -joy_speed * joy_direction.y * p_time, 0);
			ps->body_set_state(mover, PhysicsServer::BODY_STATE_TRANSFORM, t);
		};

		Transform cameratr;
		cameratr.rotate(Vector3(0, 1, 0), ofs_x);
		cameratr.rotate(Vector3(1, 0, 0), -ofs_y);
		cameratr.translate(Vector3(0, 2, 8));
		VisualServer *vs = VisualServer::get_singleton();
		vs->camera_set_transform(camera, cameratr);

		return quit;
	}
	virtual void finish() {
	}

	void test_joint() {
#if 0
		PhysicsServer * ps = PhysicsServer::get_singleton();

		mover = create_body(PhysicsServer::SHAPE_BOX,PhysicsServer::BODY_MODE_STATIC,Transform(Basis(),Vector3(0,0,-24)));
		RID b = create_body(PhysicsServer::SHAPE_CAPSULE,PhysicsServer::BODY_MODE_RIGID,Transform());

		ps->joint_create_double_pin(b,Vector3(0,0,1.0),mover,Vector3(0,0,0));
		ps->body_add_collision_exception(mover,b);


		List<String> cmdline = OS::get_singleton()->get_cmdline_args();
		int link_count = LINK_COUNT;
		if (cmdline.size() > 0 && cmdline[cmdline.size()-1].to_int()) {
			link_count = cmdline[cmdline.size()-1].to_int();
		};

		for(int i=0;i<link_count;i++) {

			RID c = create_body(PhysicsServer::SHAPE_CAPSULE,PhysicsServer::BODY_MODE_RIGID,Transform());
			ps->joint_create_double_pin(b,Vector3(0,0,-0.7),c,Vector3(0,0,0.7));
			ps->body_add_collision_exception(c,b);
			b=c;
		}


		create_static_plane(Plane(Vector3(0,1,0),-8));
#endif
	}

	void test_hinge() {
#if 0
		PhysicsServer * ps = PhysicsServer::get_singleton();


		mover = create_body(PhysicsServer::SHAPE_BOX,PhysicsServer::BODY_MODE_STATIC,Transform(Basis(),Vector3(0,0,-24)));
		RID b = create_body(PhysicsServer::SHAPE_BOX,PhysicsServer::BODY_MODE_RIGID,Transform());

		ps->joint_create_double_hinge(b,Transform(Basis(),Vector3(1,1,1.0)),mover,Transform(Basis(),Vector3(0,0,0)));
		ps->body_add_collision_exception(mover,b);

/*
		for(int i=0;i<20;i++) {

			RID c = create_body(PhysicsServer::SHAPE_CAPSULE,PhysicsServer::BODY_MODE_RIGID,Transform());
			ps->joint_create_double_hinge(b,Transform(Basis(),Vector3(0,0,-0.7)),c,Transform(Basis(),Vector3(0,0,0.7)));
			ps->body_add_collision_exception(c,b);
			b=c;
		}

*/
		//create_static_plane(Plane(Vector3(0,1,0),-8));
#endif
	}

	void test_character() {

		VisualServer *vs = VisualServer::get_singleton();
		PhysicsServer *ps = PhysicsServer::get_singleton();

		PoolVector<Plane> capsule_planes = Geometry::build_capsule_planes(0.5, 1, 12, 5, Vector3::AXIS_Y);

		RID capsule_mesh = vs->mesh_create();
		Geometry::MeshData capsule_data = Geometry::build_convex_mesh(capsule_planes);
		vs->mesh_add_surface_from_mesh_data(capsule_mesh, capsule_data);
		type_mesh_map[PhysicsServer::SHAPE_CAPSULE] = capsule_mesh;

		RID capsule_shape = ps->shape_create(PhysicsServer::SHAPE_CAPSULE);
		Dictionary capsule_params;
		capsule_params["radius"] = 0.5;
		capsule_params["height"] = 1;
		Transform shape_xform;
		shape_xform.rotate(Vector3(1, 0, 0), Math_PI / 2.0);
		//shape_xform.origin=Vector3(1,1,1);
		ps->shape_set_data(capsule_shape, capsule_params);

		RID mesh_instance = vs->instance_create2(capsule_mesh, scenario);
		character = ps->body_create(PhysicsServer::BODY_MODE_CHARACTER);
		ps->body_set_space(character, space);
		//todo add space
		ps->body_add_shape(character, capsule_shape);

		ps->body_set_force_integration_callback(character, this, "body_changed_transform", mesh_instance);

		ps->body_set_state(character, PhysicsServer::BODY_STATE_TRANSFORM, Transform(Basis(), Vector3(-2, 5, -2)));
		bodies.push_back(character);
	}

	void test_fall() {

		for (int i = 0; i < 35; i++) {

			static const PhysicsServer::ShapeType shape_idx[] = {
				PhysicsServer::SHAPE_CAPSULE,
				PhysicsServer::SHAPE_BOX,
				PhysicsServer::SHAPE_SPHERE,
				PhysicsServer::SHAPE_CONVEX_POLYGON
			};

			PhysicsServer::ShapeType type = shape_idx[i % 4];
			//type=PhysicsServer::SHAPE_CONVEX_POLYGON;

			Transform t;

			t.origin = Vector3(0.0 * i, 3.5 + 1.1 * i, 0.7 + 0.0 * i);
			//t.origin=Vector3(-0.7+0.0*i,0.5+4.1*i,0);
			t.basis.rotate(Vector3(0.2, -1, 0), Math_PI / 2 * 0.6);
			//t.basis.rotate(Vector3(0,-1,0),Math_PI/4*i);
			//t.basis.rotate(Vector3(0,-1,0),Math_PI/4*i);
			//t.basis.rotate(Vector3(-1,0,0),Math_PI/4*i);

			create_body(type, PhysicsServer::BODY_MODE_RIGID, t);
			//RID b = create_body(type,i==0?PhysicsServer::BODY_MODE_STATIC:PhysicsServer::BODY_MODE_RIGID,t);
		}

		create_static_plane(Plane(Vector3(0, 1, 0), -1));

		/*
		create_static_plane( Plane( Vector3(1,0,0), -2) );
		create_static_plane( Plane( Vector3(-1,0,0), -2) );
		create_static_plane( Plane( Vector3(0,0,1), -2) );
		create_static_plane( Plane( Vector3(0,0,-1), -2) );
*/
	}

	void test_activate() {

		create_body(PhysicsServer::SHAPE_BOX, PhysicsServer::BODY_MODE_RIGID, Transform(Basis(), Vector3(0, 2, 0)), true);
		//create_body(PhysicsServer::SHAPE_SPHERE,PhysicsServer::BODY_MODE_RIGID,Transform(Basis(),Vector3(0,6,0)),true);
		create_static_plane(Plane(Vector3(0, 1, 0), -1));
	}

	virtual bool idle(float p_time) {
		return false;
	}

	TestPhysicsMainLoop() {
	}
};

namespace TestPhysics {

MainLoop *test() {

	return memnew(TestPhysicsMainLoop);
}
}
