/*************************************************************************/
/*  test_physics_3d.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "test_physics_3d.h"

#include "core/math/convex_hull.h"
#include "core/math/geometry_3d.h"
#include "core/os/main_loop.h"
#include "servers/physics_server_3d.h"
#include "servers/rendering_server.h"

class TestPhysics3DMainLoop : public MainLoop {
	GDCLASS(TestPhysics3DMainLoop, MainLoop);

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

	real_t ofs_x, ofs_y;

	Point2 joy_direction;

	List<RID> bodies;
	Map<PhysicsServer3D::ShapeType, RID> type_shape_map;
	Map<PhysicsServer3D::ShapeType, RID> type_mesh_map;

	void body_changed_transform(Object *p_state, RID p_visual_instance) {
		PhysicsDirectBodyState3D *state = (PhysicsDirectBodyState3D *)p_state;
		RenderingServer *vs = RenderingServer::get_singleton();
		Transform3D t = state->get_transform();
		vs->instance_set_transform(p_visual_instance, t);
	}

	bool quit;

protected:
	RID create_body(PhysicsServer3D::ShapeType p_shape, PhysicsServer3D::BodyMode p_body, const Transform3D p_location, bool p_active_default = true, const Transform3D &p_shape_xform = Transform3D()) {
		RenderingServer *vs = RenderingServer::get_singleton();
		PhysicsServer3D *ps = PhysicsServer3D::get_singleton();

		RID mesh_instance = vs->instance_create2(type_mesh_map[p_shape], scenario);
		RID body = ps->body_create();
		ps->body_set_mode(body, p_body);
		ps->body_set_state(body, PhysicsServer3D::BODY_STATE_SLEEPING, !p_active_default);
		ps->body_set_space(body, space);
		ps->body_set_param(body, PhysicsServer3D::BODY_PARAM_BOUNCE, 0.0);
		//todo set space
		ps->body_add_shape(body, type_shape_map[p_shape]);
		ps->body_set_force_integration_callback(body, callable_mp(this, &TestPhysics3DMainLoop::body_changed_transform), mesh_instance);

		ps->body_set_state(body, PhysicsServer3D::BODY_STATE_TRANSFORM, p_location);
		bodies.push_back(body);

		if (p_body == PhysicsServer3D::BODY_MODE_STATIC) {
			vs->instance_set_transform(mesh_instance, p_location);
		}
		return body;
	}

	RID create_world_boundary(const Plane &p_plane) {
		PhysicsServer3D *ps = PhysicsServer3D::get_singleton();

		RID world_boundary_shape = ps->shape_create(PhysicsServer3D::SHAPE_WORLD_BOUNDARY);
		ps->shape_set_data(world_boundary_shape, p_plane);

		RID b = ps->body_create();
		ps->body_set_mode(b, PhysicsServer3D::BODY_MODE_STATIC);

		ps->body_set_space(b, space);
		ps->body_add_shape(b, world_boundary_shape);
		return b;
	}

	void configure_body(RID p_body, real_t p_mass, real_t p_friction, real_t p_bounce) {
		PhysicsServer3D *ps = PhysicsServer3D::get_singleton();
		ps->body_set_param(p_body, PhysicsServer3D::BODY_PARAM_MASS, p_mass);
		ps->body_set_param(p_body, PhysicsServer3D::BODY_PARAM_FRICTION, p_friction);
		ps->body_set_param(p_body, PhysicsServer3D::BODY_PARAM_BOUNCE, p_bounce);
	}

	void initialize_shapes() {
		RenderingServer *vs = RenderingServer::get_singleton();
		PhysicsServer3D *ps = PhysicsServer3D::get_singleton();

		/* SPHERE SHAPE */
		RID sphere_mesh = vs->make_sphere_mesh(10, 20, 0.5);
		type_mesh_map[PhysicsServer3D::SHAPE_SPHERE] = sphere_mesh;

		RID sphere_shape = ps->shape_create(PhysicsServer3D::SHAPE_SPHERE);
		ps->shape_set_data(sphere_shape, 0.5);
		type_shape_map[PhysicsServer3D::SHAPE_SPHERE] = sphere_shape;

		/* BOX SHAPE */

		Vector<Plane> box_planes = Geometry3D::build_box_planes(Vector3(0.5, 0.5, 0.5));
		RID box_mesh = vs->mesh_create();
		Geometry3D::MeshData box_data = Geometry3D::build_convex_mesh(box_planes);
		vs->mesh_add_surface_from_mesh_data(box_mesh, box_data);
		type_mesh_map[PhysicsServer3D::SHAPE_BOX] = box_mesh;

		RID box_shape = ps->shape_create(PhysicsServer3D::SHAPE_BOX);
		ps->shape_set_data(box_shape, Vector3(0.5, 0.5, 0.5));
		type_shape_map[PhysicsServer3D::SHAPE_BOX] = box_shape;

		/* CAPSULE SHAPE */

		Vector<Plane> capsule_planes = Geometry3D::build_capsule_planes(0.5, 0.7, 12, Vector3::AXIS_Z);

		RID capsule_mesh = vs->mesh_create();
		Geometry3D::MeshData capsule_data = Geometry3D::build_convex_mesh(capsule_planes);
		vs->mesh_add_surface_from_mesh_data(capsule_mesh, capsule_data);

		type_mesh_map[PhysicsServer3D::SHAPE_CAPSULE] = capsule_mesh;

		RID capsule_shape = ps->shape_create(PhysicsServer3D::SHAPE_CAPSULE);
		Dictionary capsule_params;
		capsule_params["radius"] = 0.5;
		capsule_params["height"] = 1.4;
		ps->shape_set_data(capsule_shape, capsule_params);
		type_shape_map[PhysicsServer3D::SHAPE_CAPSULE] = capsule_shape;

		/* CONVEX SHAPE */

		Vector<Plane> convex_planes = Geometry3D::build_cylinder_planes(0.5, 0.7, 5, Vector3::AXIS_Z);

		RID convex_mesh = vs->mesh_create();
		Geometry3D::MeshData convex_data = Geometry3D::build_convex_mesh(convex_planes);
		ConvexHullComputer::convex_hull(convex_data.vertices, convex_data);
		vs->mesh_add_surface_from_mesh_data(convex_mesh, convex_data);

		type_mesh_map[PhysicsServer3D::SHAPE_CONVEX_POLYGON] = convex_mesh;

		RID convex_shape = ps->shape_create(PhysicsServer3D::SHAPE_CONVEX_POLYGON);
		ps->shape_set_data(convex_shape, convex_data.vertices);
		type_shape_map[PhysicsServer3D::SHAPE_CONVEX_POLYGON] = convex_shape;
	}

	void make_trimesh(Vector<Vector3> p_faces, const Transform3D &p_xform = Transform3D()) {
		RenderingServer *vs = RenderingServer::get_singleton();
		PhysicsServer3D *ps = PhysicsServer3D::get_singleton();
		RID trimesh_shape = ps->shape_create(PhysicsServer3D::SHAPE_CONCAVE_POLYGON);
		Dictionary trimesh_params;
		trimesh_params["faces"] = p_faces;
		trimesh_params["backface_collision"] = false;
		ps->shape_set_data(trimesh_shape, trimesh_params);
		Vector<Vector3> normals; // for drawing
		for (int i = 0; i < p_faces.size() / 3; i++) {
			Plane p(p_faces[i * 3 + 0], p_faces[i * 3 + 1], p_faces[i * 3 + 2]);
			normals.push_back(p.normal);
			normals.push_back(p.normal);
			normals.push_back(p.normal);
		}

		RID trimesh_mesh = vs->mesh_create();
		Array d;
		d.resize(RS::ARRAY_MAX);
		d[RS::ARRAY_VERTEX] = p_faces;
		d[RS::ARRAY_NORMAL] = normals;
		vs->mesh_add_surface_from_arrays(trimesh_mesh, RS::PRIMITIVE_TRIANGLES, d);

		RID triins = vs->instance_create2(trimesh_mesh, scenario);

		RID tribody = ps->body_create();
		ps->body_set_mode(tribody, PhysicsServer3D::BODY_MODE_STATIC);
		ps->body_set_space(tribody, space);
		//todo set space
		ps->body_add_shape(tribody, trimesh_shape);
		Transform3D tritrans = p_xform;
		ps->body_set_state(tribody, PhysicsServer3D::BODY_STATE_TRANSFORM, tritrans);
		vs->instance_set_transform(triins, tritrans);
	}

	void make_grid(int p_width, int p_height, real_t p_cellsize, real_t p_cellheight, const Transform3D &p_xform = Transform3D()) {
		Vector<Vector<real_t>> grid;

		grid.resize(p_width);

		for (int i = 0; i < p_width; i++) {
			grid.write[i].resize(p_height);

			for (int j = 0; j < p_height; j++) {
				grid.write[i].write[j] = 1.0 + Math::random(-p_cellheight, p_cellheight);
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
	virtual void input_event(const Ref<InputEvent> &p_event) {
		Ref<InputEventMouseMotion> mm = p_event;
		if (mm.is_valid() && (mm->get_button_mask() & MouseButton::MASK_MIDDLE) != MouseButton::NONE) {
			ofs_y -= mm->get_relative().y / 200.0;
			ofs_x += mm->get_relative().x / 200.0;
		}

		if (mm.is_valid() && (mm->get_button_mask() & MouseButton::MASK_LEFT) != MouseButton::NONE) {
			real_t y = -mm->get_relative().y / 20.0;
			real_t x = mm->get_relative().x / 20.0;

			if (mover.is_valid()) {
				PhysicsServer3D *ps = PhysicsServer3D::get_singleton();
				Transform3D t = ps->body_get_state(mover, PhysicsServer3D::BODY_STATE_TRANSFORM);
				t.origin += Vector3(x, y, 0);

				ps->body_set_state(mover, PhysicsServer3D::BODY_STATE_TRANSFORM, t);
			}
		}
	}

	virtual void request_quit() {
		quit = true;
	}
	virtual void initialize() override {
		ofs_x = ofs_y = 0;
		initialize_shapes();

		PhysicsServer3D *ps = PhysicsServer3D::get_singleton();
		space = ps->space_create();
		ps->space_set_active(space, true);

		RenderingServer *vs = RenderingServer::get_singleton();

		/* LIGHT */
		RID lightaux = vs->directional_light_create();
		scenario = vs->scenario_create();
		vs->light_set_shadow(lightaux, true);
		light = vs->instance_create2(lightaux, scenario);
		Transform3D t;
		t.rotate(Vector3(1.0, 0, 0), 0.6);
		vs->instance_set_transform(light, t);

		/* CAMERA */

		camera = vs->camera_create();

		RID viewport = vs->viewport_create();
		Size2i screen_size = DisplayServer::get_singleton()->window_get_size();
		vs->viewport_set_size(viewport, screen_size.x, screen_size.y);
		vs->viewport_attach_to_screen(viewport, Rect2(Vector2(), screen_size));
		vs->viewport_set_active(viewport, true);
		vs->viewport_attach_camera(viewport, camera);
		vs->viewport_set_scenario(viewport, scenario);

		vs->camera_set_perspective(camera, 60, 0.1, 40.0);
		vs->camera_set_transform(camera, Transform3D(Basis(), Vector3(0, 9, 12)));

		Transform3D gxf;
		gxf.basis.scale(Vector3(1.4, 0.4, 1.4));
		gxf.origin = Vector3(-2, 1, -2);
		make_grid(5, 5, 2.5, 1, gxf);
		test_fall();
		quit = false;
	}
	virtual bool physics_process(double p_time) override {
		if (mover.is_valid()) {
			static real_t joy_speed = 10;
			PhysicsServer3D *ps = PhysicsServer3D::get_singleton();
			Transform3D t = ps->body_get_state(mover, PhysicsServer3D::BODY_STATE_TRANSFORM);
			t.origin += Vector3(joy_speed * joy_direction.x * p_time, -joy_speed * joy_direction.y * p_time, 0);
			ps->body_set_state(mover, PhysicsServer3D::BODY_STATE_TRANSFORM, t);
		};

		Transform3D cameratr;
		cameratr.rotate(Vector3(0, 1, 0), ofs_x);
		cameratr.rotate(Vector3(1, 0, 0), -ofs_y);
		cameratr.translate(Vector3(0, 2, 8));
		RenderingServer *vs = RenderingServer::get_singleton();
		vs->camera_set_transform(camera, cameratr);

		return quit;
	}
	virtual void finalize() override {
	}

	void test_joint() {
	}

	void test_hinge() {
	}

	void test_character() {
		RenderingServer *vs = RenderingServer::get_singleton();
		PhysicsServer3D *ps = PhysicsServer3D::get_singleton();

		Vector<Plane> capsule_planes = Geometry3D::build_capsule_planes(0.5, 1, 12, 5, Vector3::AXIS_Y);

		RID capsule_mesh = vs->mesh_create();
		Geometry3D::MeshData capsule_data = Geometry3D::build_convex_mesh(capsule_planes);
		vs->mesh_add_surface_from_mesh_data(capsule_mesh, capsule_data);
		type_mesh_map[PhysicsServer3D::SHAPE_CAPSULE] = capsule_mesh;

		RID capsule_shape = ps->shape_create(PhysicsServer3D::SHAPE_CAPSULE);
		Dictionary capsule_params;
		capsule_params["radius"] = 0.5;
		capsule_params["height"] = 1;
		Transform3D shape_xform;
		shape_xform.rotate(Vector3(1, 0, 0), Math_PI / 2.0);
		//shape_xform.origin=Vector3(1,1,1);
		ps->shape_set_data(capsule_shape, capsule_params);

		RID mesh_instance = vs->instance_create2(capsule_mesh, scenario);
		character = ps->body_create();
		ps->body_set_mode(character, PhysicsServer3D::BODY_MODE_DYNAMIC_LINEAR);
		ps->body_set_space(character, space);
		//todo add space
		ps->body_add_shape(character, capsule_shape);
		ps->body_set_force_integration_callback(character, callable_mp(this, &TestPhysics3DMainLoop::body_changed_transform), mesh_instance);

		ps->body_set_state(character, PhysicsServer3D::BODY_STATE_TRANSFORM, Transform3D(Basis(), Vector3(-2, 5, -2)));
		bodies.push_back(character);
	}

	void test_fall() {
		for (int i = 0; i < 35; i++) {
			static const PhysicsServer3D::ShapeType shape_idx[] = {
				PhysicsServer3D::SHAPE_CAPSULE,
				PhysicsServer3D::SHAPE_BOX,
				PhysicsServer3D::SHAPE_SPHERE,
				PhysicsServer3D::SHAPE_CONVEX_POLYGON
			};

			PhysicsServer3D::ShapeType type = shape_idx[i % 4];

			Transform3D t;

			t.origin = Vector3(0.0 * i, 3.5 + 1.1 * i, 0.7 + 0.0 * i);
			t.basis.rotate(Vector3(0.2, -1, 0), Math_PI / 2 * 0.6);

			create_body(type, PhysicsServer3D::BODY_MODE_DYNAMIC, t);
		}

		create_world_boundary(Plane(Vector3(0, 1, 0), -1));
	}

	void test_activate() {
		create_body(PhysicsServer3D::SHAPE_BOX, PhysicsServer3D::BODY_MODE_DYNAMIC, Transform3D(Basis(), Vector3(0, 2, 0)), true);
		create_world_boundary(Plane(Vector3(0, 1, 0), -1));
	}

	virtual bool process(double p_time) override {
		return false;
	}

	TestPhysics3DMainLoop() {
	}
};

namespace TestPhysics3D {

MainLoop *test() {
	return memnew(TestPhysics3DMainLoop);
}
} // namespace TestPhysics3D
