/*************************************************************************/
/*  soft_body_bullet.cpp                                                 */
/*  Author: AndreaCatania                                                */
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

#include "soft_body_bullet.h"
#include "bullet_types_converter.h"
#include "bullet_utilities.h"
#include "space_bullet.h"

#include "scene/3d/immediate_geometry.h"

SoftBodyBullet::SoftBodyBullet() :
		CollisionObjectBullet(CollisionObjectBullet::TYPE_SOFT_BODY),
		mass(1),
		simulation_precision(5),
		stiffness(0.5f),
		pressure_coefficient(50),
		damping_coefficient(0.005),
		drag_coefficient(0.005),
		bt_soft_body(NULL),
		soft_shape_type(SOFT_SHAPETYPE_NONE),
		isScratched(false),
		soft_body_shape_data(NULL) {

	test_geometry = memnew(ImmediateGeometry);

	red_mat = Ref<SpatialMaterial>(memnew(SpatialMaterial));
	red_mat->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
	red_mat->set_line_width(20.0);
	red_mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
	red_mat->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	red_mat->set_flag(SpatialMaterial::FLAG_SRGB_VERTEX_COLOR, true);
	red_mat->set_albedo(Color(1, 0, 0, 1));
	test_geometry->set_material_override(red_mat);

	test_is_in_scene = false;
}

SoftBodyBullet::~SoftBodyBullet() {
	bulletdelete(soft_body_shape_data);
}

void SoftBodyBullet::reload_body() {
	if (space) {
		space->remove_soft_body(this);
		space->add_soft_body(this);
	}
}

void SoftBodyBullet::set_space(SpaceBullet *p_space) {
	if (space) {
		isScratched = false;

		// Remove this object from the physics world
		space->remove_soft_body(this);
	}

	space = p_space;

	if (space) {
		space->add_soft_body(this);
	}

	reload_soft_body();
}

void SoftBodyBullet::dispatch_callbacks() {
	if (!bt_soft_body) {
		return;
	}

	if (!test_is_in_scene) {
		test_is_in_scene = true;
		SceneTree::get_singleton()->get_current_scene()->add_child(test_geometry);
	}

	test_geometry->clear();
	test_geometry->begin(Mesh::PRIMITIVE_LINES, NULL);
	bool first = true;
	Vector3 pos;
	for (int i = 0; i < bt_soft_body->m_nodes.size(); ++i) {
		const btSoftBody::Node &n = bt_soft_body->m_nodes[i];
		B_TO_G(n.m_x, pos);
		test_geometry->add_vertex(pos);
		if (!first) {
			test_geometry->add_vertex(pos);
		} else {
			first = false;
		}
	}
	test_geometry->end();
}

void SoftBodyBullet::on_collision_filters_change() {
}

void SoftBodyBullet::on_collision_checker_start() {
}

void SoftBodyBullet::on_enter_area(AreaBullet *p_area) {
}

void SoftBodyBullet::on_exit_area(AreaBullet *p_area) {
}

void SoftBodyBullet::set_trimesh_body_shape(PoolVector<int> p_indices, PoolVector<Vector3> p_vertices, int p_triangles_num) {

	TrimeshSoftShapeData *shape_data = bulletnew(TrimeshSoftShapeData);
	shape_data->m_triangles_indices = p_indices;
	shape_data->m_vertices = p_vertices;
	shape_data->m_triangles_num = p_triangles_num;

	set_body_shape_data(shape_data, SOFT_SHAPE_TYPE_TRIMESH);
	reload_soft_body();
}

void SoftBodyBullet::set_body_shape_data(SoftShapeData *p_soft_shape_data, SoftShapeType p_type) {
	bulletdelete(soft_body_shape_data);
	soft_body_shape_data = p_soft_shape_data;
	soft_shape_type = p_type;
}

void SoftBodyBullet::set_transform(const Transform &p_transform) {
	transform = p_transform;
	if (bt_soft_body) {
		// TODO the softbody set new transform considering the current transform as center of world
		// like if it's local transform, so I must fix this by setting nwe transform considering the old
		btTransform bt_trans;
		G_TO_B(transform, bt_trans);
		//bt_soft_body->transform(bt_trans);
	}
}

const Transform &SoftBodyBullet::get_transform() const {
	return transform;
}

void SoftBodyBullet::get_first_node_origin(btVector3 &p_out_origin) const {
	if (bt_soft_body && bt_soft_body->m_nodes.size()) {
		p_out_origin = bt_soft_body->m_nodes[0].m_x;
	} else {
		p_out_origin.setZero();
	}
}

void SoftBodyBullet::set_activation_state(bool p_active) {
	if (p_active) {
		bt_soft_body->setActivationState(ACTIVE_TAG);
	} else {
		bt_soft_body->setActivationState(WANTS_DEACTIVATION);
	}
}

void SoftBodyBullet::set_mass(real_t p_val) {
	if (0 >= p_val) {
		p_val = 1;
	}
	mass = p_val;
	if (bt_soft_body) {
		bt_soft_body->setTotalMass(mass);
	}
}

void SoftBodyBullet::set_stiffness(real_t p_val) {
	stiffness = p_val;
	if (bt_soft_body) {
		mat0->m_kAST = stiffness;
		mat0->m_kLST = stiffness;
		mat0->m_kVST = stiffness;
	}
}

void SoftBodyBullet::set_simulation_precision(int p_val) {
	simulation_precision = p_val;
	if (bt_soft_body) {
		bt_soft_body->m_cfg.piterations = simulation_precision;
	}
}

void SoftBodyBullet::set_pressure_coefficient(real_t p_val) {
	pressure_coefficient = p_val;
	if (bt_soft_body) {
		bt_soft_body->m_cfg.kPR = pressure_coefficient;
	}
}

void SoftBodyBullet::set_damping_coefficient(real_t p_val) {
	damping_coefficient = p_val;
	if (bt_soft_body) {
		bt_soft_body->m_cfg.kDP = damping_coefficient;
	}
}

void SoftBodyBullet::set_drag_coefficient(real_t p_val) {
	drag_coefficient = p_val;
	if (bt_soft_body) {
		bt_soft_body->m_cfg.kDG = drag_coefficient;
	}
}

void SoftBodyBullet::reload_soft_body() {

	destroy_soft_body();
	create_soft_body();

	if (bt_soft_body) {

		// TODO the softbody set new transform considering the current transform as center of world
		// like if it's local transform, so I must fix this by setting nwe transform considering the old
		btTransform bt_trans;
		G_TO_B(transform, bt_trans);
		bt_soft_body->transform(bt_trans);

		bt_soft_body->generateBendingConstraints(2, mat0);
		mat0->m_kAST = stiffness;
		mat0->m_kLST = stiffness;
		mat0->m_kVST = stiffness;

		bt_soft_body->m_cfg.piterations = simulation_precision;
		bt_soft_body->m_cfg.kDP = damping_coefficient;
		bt_soft_body->m_cfg.kDG = drag_coefficient;
		bt_soft_body->m_cfg.kPR = pressure_coefficient;
		bt_soft_body->setTotalMass(mass);
	}
	if (space) {
		// TODO remove this please
		space->add_soft_body(this);
	}
}

void SoftBodyBullet::create_soft_body() {
	if (!space || !soft_body_shape_data) {
		return;
	}
	ERR_FAIL_COND(!space->is_using_soft_world());
	switch (soft_shape_type) {
		case SOFT_SHAPE_TYPE_TRIMESH: {
			TrimeshSoftShapeData *trimesh_data = static_cast<TrimeshSoftShapeData *>(soft_body_shape_data);

			Vector<int> indices;
			Vector<btScalar> vertices;

			int i;
			const int indices_size = trimesh_data->m_triangles_indices.size();
			const int vertices_size = trimesh_data->m_vertices.size();
			indices.resize(indices_size);
			vertices.resize(vertices_size * 3);

			PoolVector<int>::Read i_r = trimesh_data->m_triangles_indices.read();
			for (i = 0; i < indices_size; ++i) {
				indices[i] = i_r[i];
			}
			i_r = PoolVector<int>::Read();

			PoolVector<Vector3>::Read f_r = trimesh_data->m_vertices.read();
			for (int j = i = 0; i < vertices_size; ++i, j += 3) {
				vertices[j + 0] = f_r[i][0];
				vertices[j + 1] = f_r[i][1];
				vertices[j + 2] = f_r[i][2];
			}
			f_r = PoolVector<Vector3>::Read();

			bt_soft_body = btSoftBodyHelpers::CreateFromTriMesh(*space->get_soft_body_world_info(), vertices.ptr(), indices.ptr(), trimesh_data->m_triangles_num);
		} break;
		default:
			ERR_PRINT("Shape type not supported");
			return;
	}

	setupBulletCollisionObject(bt_soft_body);
	bt_soft_body->getCollisionShape()->setMargin(0.001f);
	bt_soft_body->setCollisionFlags(bt_soft_body->getCollisionFlags() & (~(btCollisionObject::CF_KINEMATIC_OBJECT | btCollisionObject::CF_STATIC_OBJECT)));
	mat0 = bt_soft_body->appendMaterial();
}

void SoftBodyBullet::destroy_soft_body() {
	if (space) {
		/// This step is required to assert that the body is not into the world during deletion
		/// This step is required since to change the body shape the body must be re-created.
		/// Here is handled the case when the body is assigned into a world and the body
		/// shape is changed.
		space->remove_soft_body(this);
	}
	destroyBulletCollisionObject();
	bt_soft_body = NULL;
}
