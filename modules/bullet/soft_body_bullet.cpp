/*************************************************************************/
/*  soft_body_bullet.cpp                                                 */
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

#include "soft_body_bullet.h"

#include "bullet_types_converter.h"
#include "bullet_utilities.h"
#include "space_bullet.h"

#include "servers/rendering_server.h"

SoftBodyBullet::SoftBodyBullet() :
		CollisionObjectBullet(CollisionObjectBullet::TYPE_SOFT_BODY) {}

SoftBodyBullet::~SoftBodyBullet() {
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
		space->remove_soft_body(this);
	}

	space = p_space;

	if (space) {
		space->add_soft_body(this);
	}
}

void SoftBodyBullet::on_enter_area(AreaBullet *p_area) {}

void SoftBodyBullet::on_exit_area(AreaBullet *p_area) {}

void SoftBodyBullet::update_rendering_server(RenderingServerHandler *p_rendering_server_handler) {
	if (!bt_soft_body) {
		return;
	}

	/// Update rendering server vertices
	const btSoftBody::tNodeArray &nodes(bt_soft_body->m_nodes);
	const int nodes_count = nodes.size();

	const Vector<int> *vs_indices;
	const void *vertex_position;
	const void *vertex_normal;

	for (int vertex_index = 0; vertex_index < nodes_count; ++vertex_index) {
		vertex_position = reinterpret_cast<const void *>(&nodes[vertex_index].m_x);
		vertex_normal = reinterpret_cast<const void *>(&nodes[vertex_index].m_n);

		vs_indices = &indices_table[vertex_index];

		const int vs_indices_size(vs_indices->size());
		for (int x = 0; x < vs_indices_size; ++x) {
			p_rendering_server_handler->set_vertex((*vs_indices)[x], vertex_position);
			p_rendering_server_handler->set_normal((*vs_indices)[x], vertex_normal);
		}
	}

	/// Generate AABB
	btVector3 aabb_min;
	btVector3 aabb_max;
	bt_soft_body->getAabb(aabb_min, aabb_max);

	btVector3 size(aabb_max - aabb_min);

	AABB aabb;
	B_TO_G(aabb_min, aabb.position);
	B_TO_G(size, aabb.size);

	p_rendering_server_handler->set_aabb(aabb);
}

void SoftBodyBullet::set_soft_mesh(RID p_mesh) {
	destroy_soft_body();

	soft_mesh = p_mesh;

	if (soft_mesh.is_null()) {
		return;
	}

	Array arrays = RenderingServer::get_singleton()->mesh_surface_get_arrays(soft_mesh, 0);
	ERR_FAIL_COND(arrays.is_empty());

	bool success = set_trimesh_body_shape(arrays[RS::ARRAY_INDEX], arrays[RS::ARRAY_VERTEX]);
	if (!success) {
		destroy_soft_body();
	}
}

void SoftBodyBullet::destroy_soft_body() {
	soft_mesh = RID();

	if (!bt_soft_body) {
		return;
	}

	if (space) {
		/// Remove from world before deletion
		space->remove_soft_body(this);
	}

	destroyBulletCollisionObject();
	bt_soft_body = nullptr;
}

void SoftBodyBullet::set_soft_transform(const Transform3D &p_transform) {
	reset_all_node_positions();
	move_all_nodes(p_transform);
}

AABB SoftBodyBullet::get_bounds() const {
	if (!bt_soft_body) {
		return AABB();
	}

	btVector3 aabb_min;
	btVector3 aabb_max;
	bt_soft_body->getAabb(aabb_min, aabb_max);

	btVector3 size(aabb_max - aabb_min);

	AABB aabb;
	B_TO_G(aabb_min, aabb.position);
	B_TO_G(size, aabb.size);

	return aabb;
}

void SoftBodyBullet::move_all_nodes(const Transform3D &p_transform) {
	if (!bt_soft_body) {
		return;
	}
	btTransform bt_transf;
	G_TO_B(p_transform, bt_transf);
	bt_soft_body->transform(bt_transf);
}

void SoftBodyBullet::set_node_position(int p_node_index, const Vector3 &p_global_position) {
	btVector3 bt_pos;
	G_TO_B(p_global_position, bt_pos);
	set_node_position(p_node_index, bt_pos);
}

void SoftBodyBullet::set_node_position(int p_node_index, const btVector3 &p_global_position) {
	if (bt_soft_body) {
		bt_soft_body->m_nodes[p_node_index].m_q = bt_soft_body->m_nodes[p_node_index].m_x;
		bt_soft_body->m_nodes[p_node_index].m_x = p_global_position;
	}
}

void SoftBodyBullet::get_node_position(int p_node_index, Vector3 &r_position) const {
	if (bt_soft_body) {
		B_TO_G(bt_soft_body->m_nodes[p_node_index].m_x, r_position);
	}
}

void SoftBodyBullet::set_node_mass(int p_node_index, btScalar p_mass) {
	if (0 >= p_mass) {
		pin_node(p_node_index);
	} else {
		unpin_node(p_node_index);
	}
	if (bt_soft_body) {
		ERR_FAIL_INDEX(p_node_index, bt_soft_body->m_nodes.size());
		bt_soft_body->setMass(p_node_index, p_mass);
	}
}

btScalar SoftBodyBullet::get_node_mass(int p_node_index) const {
	if (bt_soft_body) {
		ERR_FAIL_INDEX_V(p_node_index, bt_soft_body->m_nodes.size(), 1);
		return bt_soft_body->getMass(p_node_index);
	} else {
		return -1 == search_node_pinned(p_node_index) ? 1 : 0;
	}
}

void SoftBodyBullet::reset_all_node_mass() {
	if (bt_soft_body) {
		for (int i = pinned_nodes.size() - 1; 0 <= i; --i) {
			bt_soft_body->setMass(pinned_nodes[i], 1);
		}
	}
	pinned_nodes.resize(0);
}

void SoftBodyBullet::reset_all_node_positions() {
	if (soft_mesh.is_null()) {
		return;
	}

	Array arrays = soft_mesh->surface_get_arrays(0);
	Vector<Vector3> vs_vertices(arrays[RS::ARRAY_VERTEX]);
	const Vector3 *vs_vertices_read = vs_vertices.ptr();

	for (int vertex_index = bt_soft_body->m_nodes.size() - 1; 0 <= vertex_index; --vertex_index) {
		G_TO_B(vs_vertices_read[indices_table[vertex_index][0]], bt_soft_body->m_nodes[vertex_index].m_x);

		bt_soft_body->m_nodes[vertex_index].m_q = bt_soft_body->m_nodes[vertex_index].m_x;
		bt_soft_body->m_nodes[vertex_index].m_v = btVector3(0, 0, 0);
		bt_soft_body->m_nodes[vertex_index].m_f = btVector3(0, 0, 0);
	}
}

void SoftBodyBullet::set_activation_state(bool p_active) {
	if (p_active) {
		bt_soft_body->setActivationState(ACTIVE_TAG);
	} else {
		bt_soft_body->setActivationState(WANTS_DEACTIVATION);
	}
}

void SoftBodyBullet::set_total_mass(real_t p_val) {
	if (0 >= p_val) {
		p_val = 1;
	}
	total_mass = p_val;
	if (bt_soft_body) {
		bt_soft_body->setTotalMass(total_mass);
	}
}

void SoftBodyBullet::set_linear_stiffness(real_t p_val) {
	linear_stiffness = p_val;
	if (bt_soft_body) {
		mat0->m_kLST = linear_stiffness;
	}
}

void SoftBodyBullet::set_simulation_precision(int p_val) {
	simulation_precision = p_val;
	if (bt_soft_body) {
		bt_soft_body->m_cfg.piterations = simulation_precision;
		bt_soft_body->m_cfg.viterations = simulation_precision;
		bt_soft_body->m_cfg.diterations = simulation_precision;
		bt_soft_body->m_cfg.citerations = simulation_precision;
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

bool SoftBodyBullet::set_trimesh_body_shape(Vector<int> p_indices, Vector<Vector3> p_vertices) {
	ERR_FAIL_COND_V(p_indices.is_empty(), false);
	ERR_FAIL_COND_V(p_vertices.is_empty(), false);

	/// Parse rendering server indices to physical indices.
	/// Merge all overlapping vertices and create a map of physical vertices to rendering server

	{
		/// This is the map of rendering server indices to physics indices (So it's the inverse of idices_map), Thanks to it I don't need make a heavy search in the indices_map
		Vector<int> vs_indices_to_physics_table;

		{ // Map vertices
			indices_table.resize(0);

			int index = 0;
			Map<Vector3, int> unique_vertices;

			const int vs_vertices_size(p_vertices.size());

			const Vector3 *p_vertices_read = p_vertices.ptr();

			for (int vs_vertex_index = 0; vs_vertex_index < vs_vertices_size; ++vs_vertex_index) {
				Map<Vector3, int>::Element *e = unique_vertices.find(p_vertices_read[vs_vertex_index]);
				int vertex_id;
				if (e) {
					// Already existing
					vertex_id = e->value();
				} else {
					// Create new one
					unique_vertices[p_vertices_read[vs_vertex_index]] = vertex_id = index++;
					indices_table.push_back(Vector<int>());
				}

				indices_table.write[vertex_id].push_back(vs_vertex_index);
				vs_indices_to_physics_table.push_back(vertex_id);
			}
		}

		const int indices_map_size(indices_table.size());

		Vector<btScalar> bt_vertices;

		{ // Parse vertices to bullet

			bt_vertices.resize(indices_map_size * 3);
			const Vector3 *p_vertices_read = p_vertices.ptr();

			for (int i = 0; i < indices_map_size; ++i) {
				bt_vertices.write[3 * i + 0] = p_vertices_read[indices_table[i][0]].x;
				bt_vertices.write[3 * i + 1] = p_vertices_read[indices_table[i][0]].y;
				bt_vertices.write[3 * i + 2] = p_vertices_read[indices_table[i][0]].z;
			}
		}

		Vector<int> bt_triangles;
		const int triangles_size(p_indices.size() / 3);

		{ // Parse indices

			bt_triangles.resize(triangles_size * 3);

			const int *p_indices_read = p_indices.ptr();

			for (int i = 0; i < triangles_size; ++i) {
				bt_triangles.write[3 * i + 0] = vs_indices_to_physics_table[p_indices_read[3 * i + 2]];
				bt_triangles.write[3 * i + 1] = vs_indices_to_physics_table[p_indices_read[3 * i + 1]];
				bt_triangles.write[3 * i + 2] = vs_indices_to_physics_table[p_indices_read[3 * i + 0]];
			}
		}

		btSoftBodyWorldInfo fake_world_info;
		bt_soft_body = btSoftBodyHelpers::CreateFromTriMesh(fake_world_info, &bt_vertices[0], &bt_triangles[0], triangles_size, false);
		setup_soft_body();
	}

	return true;
}

void SoftBodyBullet::setup_soft_body() {
	if (!bt_soft_body) {
		return;
	}

	// Soft body setup
	setupBulletCollisionObject(bt_soft_body);
	bt_soft_body->m_worldInfo = nullptr; // Remove fake world info
	bt_soft_body->getCollisionShape()->setMargin(0.01);
	bt_soft_body->setCollisionFlags(bt_soft_body->getCollisionFlags() & (~(btCollisionObject::CF_KINEMATIC_OBJECT | btCollisionObject::CF_STATIC_OBJECT)));

	// Space setup
	if (space) {
		space->add_soft_body(this);
	}

	mat0 = bt_soft_body->appendMaterial();

	// Assign soft body data
	bt_soft_body->generateBendingConstraints(2, mat0);

	mat0->m_kLST = linear_stiffness;

	// Clusters allow to have Soft vs Soft collision but doesn't work well right now

	//bt_soft_body->m_cfg.kSRHR_CL = 1;// Soft vs rigid hardness [0,1] (cluster only)
	//bt_soft_body->m_cfg.kSKHR_CL = 1;// Soft vs kinematic hardness [0,1] (cluster only)
	//bt_soft_body->m_cfg.kSSHR_CL = 1;// Soft vs soft hardness [0,1] (cluster only)
	//bt_soft_body->m_cfg.kSR_SPLT_CL = 1; // Soft vs rigid impulse split [0,1] (cluster only)
	//bt_soft_body->m_cfg.kSK_SPLT_CL = 1; // Soft vs kinematic impulse split [0,1] (cluster only)
	//bt_soft_body->m_cfg.kSS_SPLT_CL = 1; // Soft vs Soft impulse split [0,1] (cluster only)
	//bt_soft_body->m_cfg.collisions = btSoftBody::fCollision::CL_SS + btSoftBody::fCollision::CL_RS + btSoftBody::fCollision::VF_SS;
	//bt_soft_body->generateClusters(64);

	bt_soft_body->m_cfg.piterations = simulation_precision;
	bt_soft_body->m_cfg.viterations = simulation_precision;
	bt_soft_body->m_cfg.diterations = simulation_precision;
	bt_soft_body->m_cfg.citerations = simulation_precision;
	bt_soft_body->m_cfg.kDP = damping_coefficient;
	bt_soft_body->m_cfg.kDG = drag_coefficient;
	bt_soft_body->m_cfg.kPR = pressure_coefficient;
	bt_soft_body->setTotalMass(total_mass);

	btSoftBodyHelpers::ReoptimizeLinkOrder(bt_soft_body);
	bt_soft_body->updateBounds();

	// Set pinned nodes
	for (int i = pinned_nodes.size() - 1; 0 <= i; --i) {
		const int node_index = pinned_nodes[i];
		ERR_CONTINUE(0 > node_index || bt_soft_body->m_nodes.size() <= node_index);
		bt_soft_body->setMass(node_index, 0);
	}
}

void SoftBodyBullet::pin_node(int p_node_index) {
	if (bt_soft_body) {
		ERR_FAIL_INDEX(p_node_index, bt_soft_body->m_nodes.size());
	}
	if (-1 == search_node_pinned(p_node_index)) {
		pinned_nodes.push_back(p_node_index);
	}
}

void SoftBodyBullet::unpin_node(int p_node_index) {
	if (bt_soft_body) {
		ERR_FAIL_INDEX(p_node_index, bt_soft_body->m_nodes.size());
	}
	const int id = search_node_pinned(p_node_index);
	if (-1 != id) {
		pinned_nodes.remove_at(id);
	}
}

int SoftBodyBullet::search_node_pinned(int p_node_index) const {
	for (int i = pinned_nodes.size() - 1; 0 <= i; --i) {
		if (p_node_index == pinned_nodes[i]) {
			return i;
		}
	}
	return -1;
}
