/*************************************************************************/
/*  mesh_storage.cpp                                                     */
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

#ifdef GLES3_ENABLED

#include "mesh_storage.h"
#include "../rasterizer_storage_gles3.h"
#include "material_storage.h"

using namespace GLES3;

MeshStorage *MeshStorage::singleton = nullptr;

MeshStorage *MeshStorage::get_singleton() {
	return singleton;
}

MeshStorage::MeshStorage() {
	singleton = this;
}

MeshStorage::~MeshStorage() {
	singleton = nullptr;
}

/* MESH API */

RID MeshStorage::mesh_allocate() {
	return mesh_owner.allocate_rid();
}

void MeshStorage::mesh_initialize(RID p_rid) {
	mesh_owner.initialize_rid(p_rid, Mesh());
}

void MeshStorage::mesh_free(RID p_rid) {
	mesh_clear(p_rid);
	mesh_set_shadow_mesh(p_rid, RID());
	Mesh *mesh = mesh_owner.get_or_null(p_rid);
	mesh->dependency.deleted_notify(p_rid);
	if (mesh->instances.size()) {
		ERR_PRINT("deleting mesh with active instances");
	}
	if (mesh->shadow_owners.size()) {
		for (Mesh *E : mesh->shadow_owners) {
			Mesh *shadow_owner = E;
			shadow_owner->shadow_mesh = RID();
			shadow_owner->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_MESH);
		}
	}
	mesh_owner.free(p_rid);
}

void MeshStorage::mesh_set_blend_shape_count(RID p_mesh, int p_blend_shape_count) {
	ERR_FAIL_COND(p_blend_shape_count < 0);

	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND(!mesh);

	ERR_FAIL_COND(mesh->surface_count > 0); //surfaces already exist
	WARN_PRINT_ONCE("blend shapes not supported by GLES3 renderer yet");
	mesh->blend_shape_count = p_blend_shape_count;
}

bool MeshStorage::mesh_needs_instance(RID p_mesh, bool p_has_skeleton) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND_V(!mesh, false);

	return mesh->blend_shape_count > 0 || (mesh->has_bone_weights && p_has_skeleton);
}

void MeshStorage::mesh_add_surface(RID p_mesh, const RS::SurfaceData &p_surface) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND(!mesh);

	ERR_FAIL_COND(mesh->surface_count == RS::MAX_MESH_SURFACES);

#ifdef DEBUG_ENABLED
	//do a validation, to catch errors first
	{
		uint32_t stride = 0;
		uint32_t attrib_stride = 0;
		uint32_t skin_stride = 0;

		// TODO: I think this should be <=, but it is copied from RendererRD, will have to verify later
		for (int i = 0; i < RS::ARRAY_WEIGHTS; i++) {
			if ((p_surface.format & (1 << i))) {
				switch (i) {
					case RS::ARRAY_VERTEX: {
						if (p_surface.format & RS::ARRAY_FLAG_USE_2D_VERTICES) {
							stride += sizeof(float) * 2;
						} else {
							stride += sizeof(float) * 3;
						}

					} break;
					case RS::ARRAY_NORMAL: {
						stride += sizeof(int32_t);

					} break;
					case RS::ARRAY_TANGENT: {
						stride += sizeof(int32_t);

					} break;
					case RS::ARRAY_COLOR: {
						attrib_stride += sizeof(uint32_t);
					} break;
					case RS::ARRAY_TEX_UV: {
						attrib_stride += sizeof(float) * 2;

					} break;
					case RS::ARRAY_TEX_UV2: {
						attrib_stride += sizeof(float) * 2;

					} break;
					case RS::ARRAY_CUSTOM0:
					case RS::ARRAY_CUSTOM1:
					case RS::ARRAY_CUSTOM2:
					case RS::ARRAY_CUSTOM3: {
						int idx = i - RS::ARRAY_CUSTOM0;
						uint32_t fmt_shift[RS::ARRAY_CUSTOM_COUNT] = { RS::ARRAY_FORMAT_CUSTOM0_SHIFT, RS::ARRAY_FORMAT_CUSTOM1_SHIFT, RS::ARRAY_FORMAT_CUSTOM2_SHIFT, RS::ARRAY_FORMAT_CUSTOM3_SHIFT };
						uint32_t fmt = (p_surface.format >> fmt_shift[idx]) & RS::ARRAY_FORMAT_CUSTOM_MASK;
						uint32_t fmtsize[RS::ARRAY_CUSTOM_MAX] = { 4, 4, 4, 8, 4, 8, 12, 16 };
						attrib_stride += fmtsize[fmt];

					} break;
					case RS::ARRAY_WEIGHTS:
					case RS::ARRAY_BONES: {
						//uses a separate array
						bool use_8 = p_surface.format & RS::ARRAY_FLAG_USE_8_BONE_WEIGHTS;
						skin_stride += sizeof(int16_t) * (use_8 ? 16 : 8);
					} break;
				}
			}
		}

		int expected_size = stride * p_surface.vertex_count;
		ERR_FAIL_COND_MSG(expected_size != p_surface.vertex_data.size(), "Size of vertex data provided (" + itos(p_surface.vertex_data.size()) + ") does not match expected (" + itos(expected_size) + ")");

		int bs_expected_size = expected_size * mesh->blend_shape_count;

		ERR_FAIL_COND_MSG(bs_expected_size != p_surface.blend_shape_data.size(), "Size of blend shape data provided (" + itos(p_surface.blend_shape_data.size()) + ") does not match expected (" + itos(bs_expected_size) + ")");

		int expected_attrib_size = attrib_stride * p_surface.vertex_count;
		ERR_FAIL_COND_MSG(expected_attrib_size != p_surface.attribute_data.size(), "Size of attribute data provided (" + itos(p_surface.attribute_data.size()) + ") does not match expected (" + itos(expected_attrib_size) + ")");

		if ((p_surface.format & RS::ARRAY_FORMAT_WEIGHTS) && (p_surface.format & RS::ARRAY_FORMAT_BONES)) {
			expected_size = skin_stride * p_surface.vertex_count;
			ERR_FAIL_COND_MSG(expected_size != p_surface.skin_data.size(), "Size of skin data provided (" + itos(p_surface.skin_data.size()) + ") does not match expected (" + itos(expected_size) + ")");
		}
	}

#endif

	Mesh::Surface *s = memnew(Mesh::Surface);

	s->format = p_surface.format;
	s->primitive = p_surface.primitive;

	glGenBuffers(1, &s->vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, s->vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, p_surface.vertex_data.size(), p_surface.vertex_data.ptr(), (s->format & RS::ARRAY_FLAG_USE_DYNAMIC_UPDATE) ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind
	s->vertex_buffer_size = p_surface.vertex_data.size();

	if (p_surface.attribute_data.size()) {
		glGenBuffers(1, &s->attribute_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, s->attribute_buffer);
		glBufferData(GL_ARRAY_BUFFER, p_surface.attribute_data.size(), p_surface.attribute_data.ptr(), (s->format & RS::ARRAY_FLAG_USE_DYNAMIC_UPDATE) ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind
		s->attribute_buffer_size = p_surface.attribute_data.size();
	}
	if (p_surface.skin_data.size()) {
		glGenBuffers(1, &s->skin_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, s->skin_buffer);
		glBufferData(GL_ARRAY_BUFFER, p_surface.skin_data.size(), p_surface.skin_data.ptr(), (s->format & RS::ARRAY_FLAG_USE_DYNAMIC_UPDATE) ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0); //unbind
		s->skin_buffer_size = p_surface.skin_data.size();
	}

	s->vertex_count = p_surface.vertex_count;

	if (p_surface.format & RS::ARRAY_FORMAT_BONES) {
		mesh->has_bone_weights = true;
	}

	if (p_surface.index_count) {
		bool is_index_16 = p_surface.vertex_count <= 65536;
		glGenBuffers(1, &s->index_buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->index_buffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, p_surface.index_data.size(), p_surface.index_data.ptr(), GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); //unbind
		s->index_count = p_surface.index_count;
		s->index_buffer_size = p_surface.index_data.size();

		if (p_surface.lods.size()) {
			s->lods = memnew_arr(Mesh::Surface::LOD, p_surface.lods.size());
			s->lod_count = p_surface.lods.size();

			for (int i = 0; i < p_surface.lods.size(); i++) {
				glGenBuffers(1, &s->lods[i].index_buffer);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->lods[i].index_buffer);
				glBufferData(GL_ELEMENT_ARRAY_BUFFER, p_surface.lods[i].index_data.size(), p_surface.lods[i].index_data.ptr(), GL_STATIC_DRAW);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); //unbind
				s->lods[i].edge_length = p_surface.lods[i].edge_length;
				s->lods[i].index_count = p_surface.lods[i].index_data.size() / (is_index_16 ? 2 : 4);
				s->lods[i].index_buffer_size = p_surface.lods[i].index_data.size();
			}
		}
	}

	s->aabb = p_surface.aabb;
	s->bone_aabbs = p_surface.bone_aabbs; //only really useful for returning them.

	if (mesh->blend_shape_count > 0) {
		//s->blend_shape_buffer = RD::get_singleton()->storage_buffer_create(p_surface.blend_shape_data.size(), p_surface.blend_shape_data);
	}

	if (mesh->surface_count == 0) {
		mesh->bone_aabbs = p_surface.bone_aabbs;
		mesh->aabb = p_surface.aabb;
	} else {
		if (mesh->bone_aabbs.size() < p_surface.bone_aabbs.size()) {
			// ArrayMesh::_surface_set_data only allocates bone_aabbs up to max_bone
			// Each surface may affect different numbers of bones.
			mesh->bone_aabbs.resize(p_surface.bone_aabbs.size());
		}
		for (int i = 0; i < p_surface.bone_aabbs.size(); i++) {
			mesh->bone_aabbs.write[i].merge_with(p_surface.bone_aabbs[i]);
		}
		mesh->aabb.merge_with(p_surface.aabb);
	}

	s->material = p_surface.material;

	mesh->surfaces = (Mesh::Surface **)memrealloc(mesh->surfaces, sizeof(Mesh::Surface *) * (mesh->surface_count + 1));
	mesh->surfaces[mesh->surface_count] = s;
	mesh->surface_count++;

	for (MeshInstance *mi : mesh->instances) {
		_mesh_instance_add_surface(mi, mesh, mesh->surface_count - 1);
	}

	mesh->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_MESH);

	for (Mesh *E : mesh->shadow_owners) {
		Mesh *shadow_owner = E;
		shadow_owner->shadow_mesh = RID();
		shadow_owner->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_MESH);
	}

	mesh->material_cache.clear();
}

int MeshStorage::mesh_get_blend_shape_count(RID p_mesh) const {
	const Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND_V(!mesh, -1);
	return mesh->blend_shape_count;
}

void MeshStorage::mesh_set_blend_shape_mode(RID p_mesh, RS::BlendShapeMode p_mode) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND(!mesh);
	ERR_FAIL_INDEX((int)p_mode, 2);

	mesh->blend_shape_mode = p_mode;
}

RS::BlendShapeMode MeshStorage::mesh_get_blend_shape_mode(RID p_mesh) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND_V(!mesh, RS::BLEND_SHAPE_MODE_NORMALIZED);
	return mesh->blend_shape_mode;
}

void MeshStorage::mesh_surface_update_vertex_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) {
}

void MeshStorage::mesh_surface_update_attribute_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) {
}

void MeshStorage::mesh_surface_update_skin_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) {
}

void MeshStorage::mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND(!mesh);
	ERR_FAIL_UNSIGNED_INDEX((uint32_t)p_surface, mesh->surface_count);
	mesh->surfaces[p_surface]->material = p_material;

	mesh->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_MATERIAL);
	mesh->material_cache.clear();
}

RID MeshStorage::mesh_surface_get_material(RID p_mesh, int p_surface) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND_V(!mesh, RID());
	ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)p_surface, mesh->surface_count, RID());

	return mesh->surfaces[p_surface]->material;
}

RS::SurfaceData MeshStorage::mesh_get_surface(RID p_mesh, int p_surface) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND_V(!mesh, RS::SurfaceData());
	ERR_FAIL_UNSIGNED_INDEX_V((uint32_t)p_surface, mesh->surface_count, RS::SurfaceData());

	Mesh::Surface &s = *mesh->surfaces[p_surface];

	RS::SurfaceData sd;
	sd.format = s.format;
	sd.vertex_data = RasterizerStorageGLES3::buffer_get_data(GL_ARRAY_BUFFER, s.vertex_buffer, s.vertex_buffer_size);

	if (s.attribute_buffer != 0) {
		sd.attribute_data = RasterizerStorageGLES3::buffer_get_data(GL_ARRAY_BUFFER, s.attribute_buffer, s.attribute_buffer_size);
	}

	sd.vertex_count = s.vertex_count;
	sd.index_count = s.index_count;
	sd.primitive = s.primitive;

	if (sd.index_count) {
		sd.index_data = RasterizerStorageGLES3::buffer_get_data(GL_ELEMENT_ARRAY_BUFFER, s.index_buffer, s.index_buffer_size);
	}

	sd.aabb = s.aabb;
	for (uint32_t i = 0; i < s.lod_count; i++) {
		RS::SurfaceData::LOD lod;
		lod.edge_length = s.lods[i].edge_length;
		lod.index_data = RasterizerStorageGLES3::buffer_get_data(GL_ELEMENT_ARRAY_BUFFER, s.lods[i].index_buffer, s.lods[i].index_buffer_size);
		sd.lods.push_back(lod);
	}

	sd.bone_aabbs = s.bone_aabbs;
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	return sd;
}

int MeshStorage::mesh_get_surface_count(RID p_mesh) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND_V(!mesh, 0);
	return mesh->surface_count;
}

void MeshStorage::mesh_set_custom_aabb(RID p_mesh, const AABB &p_aabb) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND(!mesh);
	mesh->custom_aabb = p_aabb;
}

AABB MeshStorage::mesh_get_custom_aabb(RID p_mesh) const {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND_V(!mesh, AABB());
	return mesh->custom_aabb;
}

AABB MeshStorage::mesh_get_aabb(RID p_mesh, RID p_skeleton) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND_V(!mesh, AABB());

	if (mesh->custom_aabb != AABB()) {
		return mesh->custom_aabb;
	}

	Skeleton *skeleton = skeleton_owner.get_or_null(p_skeleton);

	if (!skeleton || skeleton->size == 0) {
		return mesh->aabb;
	}

	// Calculate AABB based on Skeleton

	AABB aabb;

	for (uint32_t i = 0; i < mesh->surface_count; i++) {
		AABB laabb;
		if ((mesh->surfaces[i]->format & RS::ARRAY_FORMAT_BONES) && mesh->surfaces[i]->bone_aabbs.size()) {
			int bs = mesh->surfaces[i]->bone_aabbs.size();
			const AABB *skbones = mesh->surfaces[i]->bone_aabbs.ptr();

			int sbs = skeleton->size;
			ERR_CONTINUE(bs > sbs);
			const float *baseptr = skeleton->data.ptr();

			bool first = true;

			if (skeleton->use_2d) {
				for (int j = 0; j < bs; j++) {
					if (skbones[0].size == Vector3()) {
						continue; //bone is unused
					}

					const float *dataptr = baseptr + j * 8;

					Transform3D mtx;

					mtx.basis.rows[0].x = dataptr[0];
					mtx.basis.rows[1].x = dataptr[1];
					mtx.origin.x = dataptr[3];

					mtx.basis.rows[0].y = dataptr[4];
					mtx.basis.rows[1].y = dataptr[5];
					mtx.origin.y = dataptr[7];

					AABB baabb = mtx.xform(skbones[j]);

					if (first) {
						laabb = baabb;
						first = false;
					} else {
						laabb.merge_with(baabb);
					}
				}
			} else {
				for (int j = 0; j < bs; j++) {
					if (skbones[0].size == Vector3()) {
						continue; //bone is unused
					}

					const float *dataptr = baseptr + j * 12;

					Transform3D mtx;

					mtx.basis.rows[0][0] = dataptr[0];
					mtx.basis.rows[0][1] = dataptr[1];
					mtx.basis.rows[0][2] = dataptr[2];
					mtx.origin.x = dataptr[3];
					mtx.basis.rows[1][0] = dataptr[4];
					mtx.basis.rows[1][1] = dataptr[5];
					mtx.basis.rows[1][2] = dataptr[6];
					mtx.origin.y = dataptr[7];
					mtx.basis.rows[2][0] = dataptr[8];
					mtx.basis.rows[2][1] = dataptr[9];
					mtx.basis.rows[2][2] = dataptr[10];
					mtx.origin.z = dataptr[11];

					AABB baabb = mtx.xform(skbones[j]);
					if (first) {
						laabb = baabb;
						first = false;
					} else {
						laabb.merge_with(baabb);
					}
				}
			}

			if (laabb.size == Vector3()) {
				laabb = mesh->surfaces[i]->aabb;
			}
		} else {
			laabb = mesh->surfaces[i]->aabb;
		}

		if (i == 0) {
			aabb = laabb;
		} else {
			aabb.merge_with(laabb);
		}
	}

	return aabb;
}

void MeshStorage::mesh_set_shadow_mesh(RID p_mesh, RID p_shadow_mesh) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND(!mesh);

	Mesh *shadow_mesh = mesh_owner.get_or_null(mesh->shadow_mesh);
	if (shadow_mesh) {
		shadow_mesh->shadow_owners.erase(mesh);
	}
	mesh->shadow_mesh = p_shadow_mesh;

	shadow_mesh = mesh_owner.get_or_null(mesh->shadow_mesh);

	if (shadow_mesh) {
		shadow_mesh->shadow_owners.insert(mesh);
	}

	mesh->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_MESH);
}

void MeshStorage::mesh_clear(RID p_mesh) {
	Mesh *mesh = mesh_owner.get_or_null(p_mesh);
	ERR_FAIL_COND(!mesh);
	for (uint32_t i = 0; i < mesh->surface_count; i++) {
		Mesh::Surface &s = *mesh->surfaces[i];

		if (s.vertex_buffer != 0) {
			glDeleteBuffers(1, &s.vertex_buffer);
			s.vertex_buffer = 0;
		}

		if (s.version_count != 0) {
			for (uint32_t j = 0; j < s.version_count; j++) {
				glDeleteVertexArrays(1, &s.versions[j].vertex_array);
				s.versions[j].vertex_array = 0;
			}
		}

		if (s.attribute_buffer != 0) {
			glDeleteBuffers(1, &s.attribute_buffer);
			s.attribute_buffer = 0;
		}

		if (s.skin_buffer != 0) {
			glDeleteBuffers(1, &s.skin_buffer);
			s.skin_buffer = 0;
		}

		if (s.index_buffer != 0) {
			glDeleteBuffers(1, &s.index_buffer);
			s.index_buffer = 0;
		}
		memdelete(mesh->surfaces[i]);
	}
	if (mesh->surfaces) {
		memfree(mesh->surfaces);
	}

	mesh->surfaces = nullptr;
	mesh->surface_count = 0;
	mesh->material_cache.clear();
	//clear instance data
	for (MeshInstance *mi : mesh->instances) {
		_mesh_instance_clear(mi);
	}
	mesh->has_bone_weights = false;
	mesh->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_MESH);

	for (Mesh *E : mesh->shadow_owners) {
		Mesh *shadow_owner = E;
		shadow_owner->shadow_mesh = RID();
		shadow_owner->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_MESH);
	}
}

void MeshStorage::_mesh_surface_generate_version_for_input_mask(Mesh::Surface::Version &v, Mesh::Surface *s, uint32_t p_input_mask, MeshInstance::Surface *mis) {
	Mesh::Surface::Attrib attribs[RS::ARRAY_MAX];

	int attributes_stride = 0;
	int vertex_stride = 0;
	int skin_stride = 0;

	for (int i = 0; i < RS::ARRAY_INDEX; i++) {
		if (!(s->format & (1 << i))) {
			attribs[i].enabled = false;
			attribs[i].integer = false;
			continue;
		}

		attribs[i].enabled = true;
		attribs[i].integer = false;

		switch (i) {
			case RS::ARRAY_VERTEX: {
				attribs[i].offset = vertex_stride;
				if (s->format & RS::ARRAY_FLAG_USE_2D_VERTICES) {
					attribs[i].size = 2;
				} else {
					attribs[i].size = 3;
				}
				attribs[i].type = GL_FLOAT;
				vertex_stride += attribs[i].size * sizeof(float);
				attribs[i].normalized = GL_FALSE;
			} break;
			case RS::ARRAY_NORMAL: {
				attribs[i].offset = vertex_stride;
				// Will need to change to accommodate octahedral compression
				attribs[i].size = 4;
				attribs[i].type = GL_UNSIGNED_INT_2_10_10_10_REV;
				vertex_stride += sizeof(float);
				attribs[i].normalized = GL_TRUE;
			} break;
			case RS::ARRAY_TANGENT: {
				attribs[i].offset = vertex_stride;
				attribs[i].size = 4;
				attribs[i].type = GL_UNSIGNED_INT_2_10_10_10_REV;
				vertex_stride += sizeof(float);
				attribs[i].normalized = GL_TRUE;
			} break;
			case RS::ARRAY_COLOR: {
				attribs[i].offset = attributes_stride;
				attribs[i].size = 4;
				attribs[i].type = GL_UNSIGNED_BYTE;
				attributes_stride += 4;
				attribs[i].normalized = GL_TRUE;
			} break;
			case RS::ARRAY_TEX_UV: {
				attribs[i].offset = attributes_stride;
				attribs[i].size = 2;
				attribs[i].type = GL_FLOAT;
				attributes_stride += 2 * sizeof(float);
				attribs[i].normalized = GL_FALSE;
			} break;
			case RS::ARRAY_TEX_UV2: {
				attribs[i].offset = attributes_stride;
				attribs[i].size = 2;
				attribs[i].type = GL_FLOAT;
				attributes_stride += 2 * sizeof(float);
				attribs[i].normalized = GL_FALSE;
			} break;
			case RS::ARRAY_CUSTOM0:
			case RS::ARRAY_CUSTOM1:
			case RS::ARRAY_CUSTOM2:
			case RS::ARRAY_CUSTOM3: {
				attribs[i].offset = attributes_stride;

				int idx = i - RS::ARRAY_CUSTOM0;
				uint32_t fmt_shift[RS::ARRAY_CUSTOM_COUNT] = { RS::ARRAY_FORMAT_CUSTOM0_SHIFT, RS::ARRAY_FORMAT_CUSTOM1_SHIFT, RS::ARRAY_FORMAT_CUSTOM2_SHIFT, RS::ARRAY_FORMAT_CUSTOM3_SHIFT };
				uint32_t fmt = (s->format >> fmt_shift[idx]) & RS::ARRAY_FORMAT_CUSTOM_MASK;
				uint32_t fmtsize[RS::ARRAY_CUSTOM_MAX] = { 4, 4, 4, 8, 4, 8, 12, 16 };
				GLenum gl_type[RS::ARRAY_CUSTOM_MAX] = { GL_UNSIGNED_BYTE, GL_BYTE, GL_HALF_FLOAT, GL_HALF_FLOAT, GL_FLOAT, GL_FLOAT, GL_FLOAT, GL_FLOAT };
				GLboolean norm[RS::ARRAY_CUSTOM_MAX] = { GL_TRUE, GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE };
				attribs[i].type = gl_type[fmt];
				attributes_stride += fmtsize[fmt];
				attribs[i].size = fmtsize[fmt] / sizeof(float);
				attribs[i].normalized = norm[fmt];
			} break;
			case RS::ARRAY_BONES: {
				attribs[i].offset = skin_stride;
				attribs[i].size = 4;
				attribs[i].type = GL_UNSIGNED_SHORT;
				attributes_stride += 4 * sizeof(uint16_t);
				attribs[i].normalized = GL_FALSE;
				attribs[i].integer = true;
			} break;
			case RS::ARRAY_WEIGHTS: {
				attribs[i].offset = skin_stride;
				attribs[i].size = 4;
				attribs[i].type = GL_UNSIGNED_SHORT;
				attributes_stride += 4 * sizeof(uint16_t);
				attribs[i].normalized = GL_TRUE;
			} break;
		}
	}

	glGenVertexArrays(1, &v.vertex_array);
	glBindVertexArray(v.vertex_array);

	for (int i = 0; i < RS::ARRAY_INDEX; i++) {
		if (!attribs[i].enabled) {
			glDisableVertexAttribArray(i);
			continue;
		}
		if (i <= RS::ARRAY_TANGENT) {
			attribs[i].stride = vertex_stride;
			if (mis) {
				glBindBuffer(GL_ARRAY_BUFFER, mis->vertex_buffer);
			} else {
				glBindBuffer(GL_ARRAY_BUFFER, s->vertex_buffer);
			}
		} else if (i <= RS::ARRAY_CUSTOM3) {
			attribs[i].stride = attributes_stride;
			glBindBuffer(GL_ARRAY_BUFFER, s->attribute_buffer);
		} else {
			attribs[i].stride = skin_stride;
			glBindBuffer(GL_ARRAY_BUFFER, s->skin_buffer);
		}

		if (attribs[i].integer) {
			glVertexAttribIPointer(i, attribs[i].size, attribs[i].type, attribs[i].stride, CAST_INT_TO_UCHAR_PTR(attribs[i].offset));
		} else {
			glVertexAttribPointer(i, attribs[i].size, attribs[i].type, attribs[i].normalized, attribs[i].stride, CAST_INT_TO_UCHAR_PTR(attribs[i].offset));
		}
		glEnableVertexAttribArray(i);
	}

	// Do not bind index here as we want to switch between index buffers for LOD

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	v.input_mask = p_input_mask;
}

/* MESH INSTANCE API */

RID MeshStorage::mesh_instance_create(RID p_base) {
	Mesh *mesh = mesh_owner.get_or_null(p_base);
	ERR_FAIL_COND_V(!mesh, RID());

	RID rid = mesh_instance_owner.make_rid();
	MeshInstance *mi = mesh_instance_owner.get_or_null(rid);

	mi->mesh = mesh;

	for (uint32_t i = 0; i < mesh->surface_count; i++) {
		_mesh_instance_add_surface(mi, mesh, i);
	}

	mi->I = mesh->instances.push_back(mi);

	mi->dirty = true;

	return rid;
}

void MeshStorage::mesh_instance_free(RID p_rid) {
	MeshInstance *mi = mesh_instance_owner.get_or_null(p_rid);
	_mesh_instance_clear(mi);
	mi->mesh->instances.erase(mi->I);
	mi->I = nullptr;

	mesh_instance_owner.free(p_rid);
}

void MeshStorage::mesh_instance_set_skeleton(RID p_mesh_instance, RID p_skeleton) {
	MeshInstance *mi = mesh_instance_owner.get_or_null(p_mesh_instance);
	if (mi->skeleton == p_skeleton) {
		return;
	}
	mi->skeleton = p_skeleton;
	mi->skeleton_version = 0;
	mi->dirty = true;
}

void MeshStorage::mesh_instance_set_blend_shape_weight(RID p_mesh_instance, int p_shape, float p_weight) {
	MeshInstance *mi = mesh_instance_owner.get_or_null(p_mesh_instance);
	ERR_FAIL_COND(!mi);
	ERR_FAIL_INDEX(p_shape, (int)mi->blend_weights.size());
	mi->blend_weights[p_shape] = p_weight;
	mi->weights_dirty = true;
}

void MeshStorage::_mesh_instance_clear(MeshInstance *mi) {
	for (uint32_t i = 0; i < mi->surfaces.size(); i++) {
		if (mi->surfaces[i].version_count != 0) {
			for (uint32_t j = 0; j < mi->surfaces[i].version_count; j++) {
				glDeleteVertexArrays(1, &mi->surfaces[i].versions[j].vertex_array);
				mi->surfaces[i].versions[j].vertex_array = 0;
			}
			memfree(mi->surfaces[i].versions);
		}
		if (mi->surfaces[i].vertex_buffer != 0) {
			glDeleteBuffers(1, &mi->surfaces[i].vertex_buffer);
			mi->surfaces[i].vertex_buffer = 0;
		}
	}
	mi->surfaces.clear();

	if (mi->blend_weights_buffer != 0) {
		glDeleteBuffers(1, &mi->blend_weights_buffer);
		mi->blend_weights_buffer = 0;
	}
	mi->blend_weights.clear();
	mi->weights_dirty = false;
	mi->skeleton_version = 0;
}

void MeshStorage::_mesh_instance_add_surface(MeshInstance *mi, Mesh *mesh, uint32_t p_surface) {
	if (mesh->blend_shape_count > 0 && mi->blend_weights_buffer == 0) {
		mi->blend_weights.resize(mesh->blend_shape_count);
		for (uint32_t i = 0; i < mi->blend_weights.size(); i++) {
			mi->blend_weights[i] = 0;
		}
		// Todo allocate buffer for blend_weights and copy data to it
		//mi->blend_weights_buffer = RD::get_singleton()->storage_buffer_create(sizeof(float) * mi->blend_weights.size(), mi->blend_weights.to_byte_array());

		mi->weights_dirty = true;
	}

	MeshInstance::Surface s;
	if (mesh->blend_shape_count > 0 || (mesh->surfaces[p_surface]->format & RS::ARRAY_FORMAT_BONES)) {
		//surface warrants transform
		//s.vertex_buffer = RD::get_singleton()->vertex_buffer_create(mesh->surfaces[p_surface]->vertex_buffer_size, Vector<uint8_t>(), true);
	}

	mi->surfaces.push_back(s);
	mi->dirty = true;
}

void MeshStorage::mesh_instance_check_for_update(RID p_mesh_instance) {
	MeshInstance *mi = mesh_instance_owner.get_or_null(p_mesh_instance);

	bool needs_update = mi->dirty;

	if (mi->weights_dirty && !mi->weight_update_list.in_list()) {
		dirty_mesh_instance_weights.add(&mi->weight_update_list);
		needs_update = true;
	}

	if (mi->array_update_list.in_list()) {
		return;
	}

	if (!needs_update && mi->skeleton.is_valid()) {
		Skeleton *sk = skeleton_owner.get_or_null(mi->skeleton);
		if (sk && sk->version != mi->skeleton_version) {
			needs_update = true;
		}
	}

	if (needs_update) {
		dirty_mesh_instance_arrays.add(&mi->array_update_list);
	}
}

void MeshStorage::update_mesh_instances() {
	while (dirty_mesh_instance_weights.first()) {
		MeshInstance *mi = dirty_mesh_instance_weights.first()->self();

		if (mi->blend_weights_buffer != 0) {
			//RD::get_singleton()->buffer_update(mi->blend_weights_buffer, 0, mi->blend_weights.size() * sizeof(float), mi->blend_weights.ptr());
		}
		dirty_mesh_instance_weights.remove(&mi->weight_update_list);
		mi->weights_dirty = false;
	}
	if (dirty_mesh_instance_arrays.first() == nullptr) {
		return; //nothing to do
	}

	// Process skeletons and blend shapes using transform feedback
	// TODO: Implement when working on skeletons and blend shapes
}

/* MULTIMESH API */

RID MeshStorage::multimesh_allocate() {
	return multimesh_owner.allocate_rid();
}

void MeshStorage::multimesh_initialize(RID p_rid) {
	multimesh_owner.initialize_rid(p_rid, MultiMesh());
}

void MeshStorage::multimesh_free(RID p_rid) {
	_update_dirty_multimeshes();
	multimesh_allocate_data(p_rid, 0, RS::MULTIMESH_TRANSFORM_2D);
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_rid);
	multimesh->dependency.deleted_notify(p_rid);
	multimesh_owner.free(p_rid);
}

void MeshStorage::multimesh_allocate_data(RID p_multimesh, int p_instances, RS::MultimeshTransformFormat p_transform_format, bool p_use_colors, bool p_use_custom_data) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND(!multimesh);

	if (multimesh->instances == p_instances && multimesh->xform_format == p_transform_format && multimesh->uses_colors == p_use_colors && multimesh->uses_custom_data == p_use_custom_data) {
		return;
	}

	if (multimesh->buffer) {
		glDeleteBuffers(1, &multimesh->buffer);
		multimesh->buffer = 0;
	}

	if (multimesh->data_cache_dirty_regions) {
		memdelete_arr(multimesh->data_cache_dirty_regions);
		multimesh->data_cache_dirty_regions = nullptr;
		multimesh->data_cache_used_dirty_regions = 0;
	}

	multimesh->instances = p_instances;
	multimesh->xform_format = p_transform_format;
	multimesh->uses_colors = p_use_colors;
	multimesh->color_offset_cache = p_transform_format == RS::MULTIMESH_TRANSFORM_2D ? 8 : 12;
	multimesh->uses_custom_data = p_use_custom_data;
	multimesh->custom_data_offset_cache = multimesh->color_offset_cache + (p_use_colors ? 2 : 0);
	multimesh->stride_cache = multimesh->custom_data_offset_cache + (p_use_custom_data ? 2 : 0);
	multimesh->buffer_set = false;

	multimesh->data_cache = Vector<float>();
	multimesh->aabb = AABB();
	multimesh->aabb_dirty = false;
	multimesh->visible_instances = MIN(multimesh->visible_instances, multimesh->instances);

	if (multimesh->instances) {
		glGenBuffers(1, &multimesh->buffer);
		glBindBuffer(GL_ARRAY_BUFFER, multimesh->buffer);
		glBufferData(GL_ARRAY_BUFFER, multimesh->instances * multimesh->stride_cache * sizeof(float), nullptr, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	multimesh->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_MULTIMESH);
}

int MeshStorage::multimesh_get_instance_count(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, 0);
	return multimesh->instances;
}

void MeshStorage::multimesh_set_mesh(RID p_multimesh, RID p_mesh) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	if (multimesh->mesh == p_mesh || p_mesh.is_null()) {
		return;
	}
	multimesh->mesh = p_mesh;

	if (multimesh->instances == 0) {
		return;
	}

	if (multimesh->data_cache.size()) {
		//we have a data cache, just mark it dirty
		_multimesh_mark_all_dirty(multimesh, false, true);
	} else if (multimesh->instances) {
		// Need to re-create AABB. Unfortunately, calling this has a penalty.
		if (multimesh->buffer_set) {
			Vector<uint8_t> buffer = RasterizerStorageGLES3::buffer_get_data(GL_ARRAY_BUFFER, multimesh->buffer, multimesh->instances * multimesh->stride_cache * sizeof(float));
			const uint8_t *r = buffer.ptr();
			const float *data = (const float *)r;
			_multimesh_re_create_aabb(multimesh, data, multimesh->instances);
		}
	}

	multimesh->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_MESH);
}

#define MULTIMESH_DIRTY_REGION_SIZE 512

void MeshStorage::_multimesh_make_local(MultiMesh *multimesh) const {
	if (multimesh->data_cache.size() > 0) {
		return; //already local
	}
	ERR_FAIL_COND(multimesh->data_cache.size() > 0);
	// this means that the user wants to load/save individual elements,
	// for this, the data must reside on CPU, so just copy it there.
	multimesh->data_cache.resize(multimesh->instances * multimesh->stride_cache);
	{
		float *w = multimesh->data_cache.ptrw();

		if (multimesh->buffer_set) {
			Vector<uint8_t> buffer = RasterizerStorageGLES3::buffer_get_data(GL_ARRAY_BUFFER, multimesh->buffer, multimesh->instances * multimesh->stride_cache * sizeof(float));

			{
				const uint8_t *r = buffer.ptr();
				memcpy(w, r, buffer.size());
			}
		} else {
			memset(w, 0, (size_t)multimesh->instances * multimesh->stride_cache * sizeof(float));
		}
	}
	uint32_t data_cache_dirty_region_count = (multimesh->instances - 1) / MULTIMESH_DIRTY_REGION_SIZE + 1;
	multimesh->data_cache_dirty_regions = memnew_arr(bool, data_cache_dirty_region_count);
	for (uint32_t i = 0; i < data_cache_dirty_region_count; i++) {
		multimesh->data_cache_dirty_regions[i] = false;
	}
	multimesh->data_cache_used_dirty_regions = 0;
}

void MeshStorage::_multimesh_mark_dirty(MultiMesh *multimesh, int p_index, bool p_aabb) {
	uint32_t region_index = p_index / MULTIMESH_DIRTY_REGION_SIZE;
#ifdef DEBUG_ENABLED
	uint32_t data_cache_dirty_region_count = (multimesh->instances - 1) / MULTIMESH_DIRTY_REGION_SIZE + 1;
	ERR_FAIL_UNSIGNED_INDEX(region_index, data_cache_dirty_region_count); //bug
#endif
	if (!multimesh->data_cache_dirty_regions[region_index]) {
		multimesh->data_cache_dirty_regions[region_index] = true;
		multimesh->data_cache_used_dirty_regions++;
	}

	if (p_aabb) {
		multimesh->aabb_dirty = true;
	}

	if (!multimesh->dirty) {
		multimesh->dirty_list = multimesh_dirty_list;
		multimesh_dirty_list = multimesh;
		multimesh->dirty = true;
	}
}

void MeshStorage::_multimesh_mark_all_dirty(MultiMesh *multimesh, bool p_data, bool p_aabb) {
	if (p_data) {
		uint32_t data_cache_dirty_region_count = (multimesh->instances - 1) / MULTIMESH_DIRTY_REGION_SIZE + 1;

		for (uint32_t i = 0; i < data_cache_dirty_region_count; i++) {
			if (!multimesh->data_cache_dirty_regions[i]) {
				multimesh->data_cache_dirty_regions[i] = true;
				multimesh->data_cache_used_dirty_regions++;
			}
		}
	}

	if (p_aabb) {
		multimesh->aabb_dirty = true;
	}

	if (!multimesh->dirty) {
		multimesh->dirty_list = multimesh_dirty_list;
		multimesh_dirty_list = multimesh;
		multimesh->dirty = true;
	}
}

void MeshStorage::_multimesh_re_create_aabb(MultiMesh *multimesh, const float *p_data, int p_instances) {
	ERR_FAIL_COND(multimesh->mesh.is_null());
	AABB aabb;
	AABB mesh_aabb = mesh_get_aabb(multimesh->mesh);
	for (int i = 0; i < p_instances; i++) {
		const float *data = p_data + multimesh->stride_cache * i;
		Transform3D t;

		if (multimesh->xform_format == RS::MULTIMESH_TRANSFORM_3D) {
			t.basis.rows[0][0] = data[0];
			t.basis.rows[0][1] = data[1];
			t.basis.rows[0][2] = data[2];
			t.origin.x = data[3];
			t.basis.rows[1][0] = data[4];
			t.basis.rows[1][1] = data[5];
			t.basis.rows[1][2] = data[6];
			t.origin.y = data[7];
			t.basis.rows[2][0] = data[8];
			t.basis.rows[2][1] = data[9];
			t.basis.rows[2][2] = data[10];
			t.origin.z = data[11];

		} else {
			t.basis.rows[0].x = data[0];
			t.basis.rows[1].x = data[1];
			t.origin.x = data[3];

			t.basis.rows[0].y = data[4];
			t.basis.rows[1].y = data[5];
			t.origin.y = data[7];
		}

		if (i == 0) {
			aabb = t.xform(mesh_aabb);
		} else {
			aabb.merge_with(t.xform(mesh_aabb));
		}
	}

	multimesh->aabb = aabb;
}

void MeshStorage::multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform3D &p_transform) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	ERR_FAIL_INDEX(p_index, multimesh->instances);
	ERR_FAIL_COND(multimesh->xform_format != RS::MULTIMESH_TRANSFORM_3D);

	_multimesh_make_local(multimesh);

	{
		float *w = multimesh->data_cache.ptrw();

		float *dataptr = w + p_index * multimesh->stride_cache;

		dataptr[0] = p_transform.basis.rows[0][0];
		dataptr[1] = p_transform.basis.rows[0][1];
		dataptr[2] = p_transform.basis.rows[0][2];
		dataptr[3] = p_transform.origin.x;
		dataptr[4] = p_transform.basis.rows[1][0];
		dataptr[5] = p_transform.basis.rows[1][1];
		dataptr[6] = p_transform.basis.rows[1][2];
		dataptr[7] = p_transform.origin.y;
		dataptr[8] = p_transform.basis.rows[2][0];
		dataptr[9] = p_transform.basis.rows[2][1];
		dataptr[10] = p_transform.basis.rows[2][2];
		dataptr[11] = p_transform.origin.z;
	}

	_multimesh_mark_dirty(multimesh, p_index, true);
}

void MeshStorage::multimesh_instance_set_transform_2d(RID p_multimesh, int p_index, const Transform2D &p_transform) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	ERR_FAIL_INDEX(p_index, multimesh->instances);
	ERR_FAIL_COND(multimesh->xform_format != RS::MULTIMESH_TRANSFORM_2D);

	_multimesh_make_local(multimesh);

	{
		float *w = multimesh->data_cache.ptrw();

		float *dataptr = w + p_index * multimesh->stride_cache;

		dataptr[0] = p_transform.columns[0][0];
		dataptr[1] = p_transform.columns[1][0];
		dataptr[2] = 0;
		dataptr[3] = p_transform.columns[2][0];
		dataptr[4] = p_transform.columns[0][1];
		dataptr[5] = p_transform.columns[1][1];
		dataptr[6] = 0;
		dataptr[7] = p_transform.columns[2][1];
	}

	_multimesh_mark_dirty(multimesh, p_index, true);
}

void MeshStorage::multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	ERR_FAIL_INDEX(p_index, multimesh->instances);
	ERR_FAIL_COND(!multimesh->uses_colors);

	_multimesh_make_local(multimesh);

	{
		// Colors are packed into 2 floats.
		float *w = multimesh->data_cache.ptrw();

		float *dataptr = w + p_index * multimesh->stride_cache + multimesh->color_offset_cache;
		uint16_t val[4] = { Math::make_half_float(p_color.r), Math::make_half_float(p_color.g), Math::make_half_float(p_color.b), Math::make_half_float(p_color.a) };
		memcpy(dataptr, val, 2 * 4);
	}

	_multimesh_mark_dirty(multimesh, p_index, false);
}

void MeshStorage::multimesh_instance_set_custom_data(RID p_multimesh, int p_index, const Color &p_color) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	ERR_FAIL_INDEX(p_index, multimesh->instances);
	ERR_FAIL_COND(!multimesh->uses_custom_data);

	_multimesh_make_local(multimesh);

	{
		float *w = multimesh->data_cache.ptrw();

		float *dataptr = w + p_index * multimesh->stride_cache + multimesh->custom_data_offset_cache;
		uint16_t val[4] = { Math::make_half_float(p_color.r), Math::make_half_float(p_color.g), Math::make_half_float(p_color.b), Math::make_half_float(p_color.a) };
		memcpy(dataptr, val, 2 * 4);
	}

	_multimesh_mark_dirty(multimesh, p_index, false);
}

RID MeshStorage::multimesh_get_mesh(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, RID());

	return multimesh->mesh;
}

AABB MeshStorage::multimesh_get_aabb(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, AABB());
	if (multimesh->aabb_dirty) {
		const_cast<MeshStorage *>(this)->_update_dirty_multimeshes();
	}
	return multimesh->aabb;
}

Transform3D MeshStorage::multimesh_instance_get_transform(RID p_multimesh, int p_index) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, Transform3D());
	ERR_FAIL_INDEX_V(p_index, multimesh->instances, Transform3D());
	ERR_FAIL_COND_V(multimesh->xform_format != RS::MULTIMESH_TRANSFORM_3D, Transform3D());

	_multimesh_make_local(multimesh);

	Transform3D t;
	{
		const float *r = multimesh->data_cache.ptr();

		const float *dataptr = r + p_index * multimesh->stride_cache;

		t.basis.rows[0][0] = dataptr[0];
		t.basis.rows[0][1] = dataptr[1];
		t.basis.rows[0][2] = dataptr[2];
		t.origin.x = dataptr[3];
		t.basis.rows[1][0] = dataptr[4];
		t.basis.rows[1][1] = dataptr[5];
		t.basis.rows[1][2] = dataptr[6];
		t.origin.y = dataptr[7];
		t.basis.rows[2][0] = dataptr[8];
		t.basis.rows[2][1] = dataptr[9];
		t.basis.rows[2][2] = dataptr[10];
		t.origin.z = dataptr[11];
	}

	return t;
}

Transform2D MeshStorage::multimesh_instance_get_transform_2d(RID p_multimesh, int p_index) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, Transform2D());
	ERR_FAIL_INDEX_V(p_index, multimesh->instances, Transform2D());
	ERR_FAIL_COND_V(multimesh->xform_format != RS::MULTIMESH_TRANSFORM_2D, Transform2D());

	_multimesh_make_local(multimesh);

	Transform2D t;
	{
		const float *r = multimesh->data_cache.ptr();

		const float *dataptr = r + p_index * multimesh->stride_cache;

		t.columns[0][0] = dataptr[0];
		t.columns[1][0] = dataptr[1];
		t.columns[2][0] = dataptr[3];
		t.columns[0][1] = dataptr[4];
		t.columns[1][1] = dataptr[5];
		t.columns[2][1] = dataptr[7];
	}

	return t;
}

Color MeshStorage::multimesh_instance_get_color(RID p_multimesh, int p_index) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, Color());
	ERR_FAIL_INDEX_V(p_index, multimesh->instances, Color());
	ERR_FAIL_COND_V(!multimesh->uses_colors, Color());

	_multimesh_make_local(multimesh);

	Color c;
	{
		const float *r = multimesh->data_cache.ptr();

		const float *dataptr = r + p_index * multimesh->stride_cache + multimesh->color_offset_cache;
		uint16_t raw_data[4];
		memcpy(raw_data, dataptr, 2 * 4);
		c.r = Math::half_to_float(raw_data[0]);
		c.g = Math::half_to_float(raw_data[1]);
		c.b = Math::half_to_float(raw_data[2]);
		c.a = Math::half_to_float(raw_data[3]);
	}

	return c;
}

Color MeshStorage::multimesh_instance_get_custom_data(RID p_multimesh, int p_index) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, Color());
	ERR_FAIL_INDEX_V(p_index, multimesh->instances, Color());
	ERR_FAIL_COND_V(!multimesh->uses_custom_data, Color());

	_multimesh_make_local(multimesh);

	Color c;
	{
		const float *r = multimesh->data_cache.ptr();

		const float *dataptr = r + p_index * multimesh->stride_cache + multimesh->custom_data_offset_cache;
		uint16_t raw_data[4];
		memcpy(raw_data, dataptr, 2 * 4);
		c.r = Math::half_to_float(raw_data[0]);
		c.g = Math::half_to_float(raw_data[1]);
		c.b = Math::half_to_float(raw_data[2]);
		c.a = Math::half_to_float(raw_data[3]);
	}

	return c;
}

void MeshStorage::multimesh_set_buffer(RID p_multimesh, const Vector<float> &p_buffer) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND(!multimesh);

	if (multimesh->uses_colors || multimesh->uses_custom_data) {
		// Color and custom need to be packed so copy buffer to data_cache and pack.

		_multimesh_make_local(multimesh);
		multimesh->data_cache = p_buffer;

		float *w = multimesh->data_cache.ptrw();
		uint32_t old_stride = multimesh->xform_format == RS::MULTIMESH_TRANSFORM_2D ? 8 : 12;
		old_stride += multimesh->uses_colors ? 4 : 0;
		old_stride += multimesh->uses_custom_data ? 4 : 0;
		for (int i = 0; i < multimesh->instances; i++) {
			{
				float *dataptr = w + i * old_stride;
				float *newptr = w + i * multimesh->stride_cache;
				float vals[8] = { dataptr[0], dataptr[1], dataptr[2], dataptr[3], dataptr[4], dataptr[5], dataptr[6], dataptr[7] };
				memcpy(newptr, vals, 8 * 4);
			}

			if (multimesh->xform_format == RS::MULTIMESH_TRANSFORM_3D) {
				float *dataptr = w + i * old_stride + 8;
				float *newptr = w + i * multimesh->stride_cache + 8;
				float vals[8] = { dataptr[0], dataptr[1], dataptr[2], dataptr[3] };
				memcpy(newptr, vals, 4 * 4);
			}

			if (multimesh->uses_colors) {
				float *dataptr = w + i * old_stride + (multimesh->xform_format == RS::MULTIMESH_TRANSFORM_2D ? 8 : 12);
				float *newptr = w + i * multimesh->stride_cache + multimesh->color_offset_cache;
				uint16_t val[4] = { Math::make_half_float(dataptr[0]), Math::make_half_float(dataptr[1]), Math::make_half_float(dataptr[2]), Math::make_half_float(dataptr[3]) };
				memcpy(newptr, val, 2 * 4);
			}
			if (multimesh->uses_custom_data) {
				float *dataptr = w + i * old_stride + (multimesh->xform_format == RS::MULTIMESH_TRANSFORM_2D ? 8 : 12) + (multimesh->uses_colors ? 4 : 0);
				float *newptr = w + i * multimesh->stride_cache + multimesh->custom_data_offset_cache;
				uint16_t val[4] = { Math::make_half_float(dataptr[0]), Math::make_half_float(dataptr[1]), Math::make_half_float(dataptr[2]), Math::make_half_float(dataptr[3]) };
				memcpy(newptr, val, 2 * 4);
			}
		}

		multimesh->data_cache.resize(multimesh->instances * (int)multimesh->stride_cache);
		const float *r = multimesh->data_cache.ptr();
		glBindBuffer(GL_ARRAY_BUFFER, multimesh->buffer);
		glBufferData(GL_ARRAY_BUFFER, multimesh->data_cache.size() * sizeof(float), r, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

	} else {
		// Only Transform is being used, so we can upload directly.
		ERR_FAIL_COND(p_buffer.size() != (multimesh->instances * (int)multimesh->stride_cache));
		const float *r = p_buffer.ptr();
		glBindBuffer(GL_ARRAY_BUFFER, multimesh->buffer);
		glBufferData(GL_ARRAY_BUFFER, p_buffer.size() * sizeof(float), r, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	multimesh->buffer_set = true;

	if (multimesh->data_cache.size() || multimesh->uses_colors || multimesh->uses_custom_data) {
		//if we have a data cache, just update it
		multimesh->data_cache = multimesh->data_cache;
		{
			//clear dirty since nothing will be dirty anymore
			uint32_t data_cache_dirty_region_count = (multimesh->instances - 1) / MULTIMESH_DIRTY_REGION_SIZE + 1;
			for (uint32_t i = 0; i < data_cache_dirty_region_count; i++) {
				multimesh->data_cache_dirty_regions[i] = false;
			}
			multimesh->data_cache_used_dirty_regions = 0;
		}

		_multimesh_mark_all_dirty(multimesh, false, true); //update AABB
	} else if (multimesh->mesh.is_valid()) {
		//if we have a mesh set, we need to re-generate the AABB from the new data
		const float *data = multimesh->data_cache.ptr();

		_multimesh_re_create_aabb(multimesh, data, multimesh->instances);
		multimesh->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_AABB);
	}
}

Vector<float> MeshStorage::multimesh_get_buffer(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, Vector<float>());
	Vector<float> ret;
	if (multimesh->buffer == 0) {
		return Vector<float>();
	} else if (multimesh->data_cache.size()) {
		ret = multimesh->data_cache;
	} else {
		// Buffer not cached, so fetch from GPU memory. This can be a stalling operation, avoid whenever possible.

		Vector<uint8_t> buffer = RasterizerStorageGLES3::buffer_get_data(GL_ARRAY_BUFFER, multimesh->buffer, multimesh->instances * multimesh->stride_cache * sizeof(float));
		ret.resize(multimesh->instances * multimesh->stride_cache);
		{
			float *w = ret.ptrw();
			const uint8_t *r = buffer.ptr();
			memcpy(w, r, buffer.size());
		}
	}
	if (multimesh->uses_colors || multimesh->uses_custom_data) {
		// Need to decompress buffer.
		uint32_t new_stride = multimesh->xform_format == RS::MULTIMESH_TRANSFORM_2D ? 8 : 12;
		new_stride += multimesh->uses_colors ? 4 : 0;
		new_stride += multimesh->uses_custom_data ? 4 : 0;

		Vector<float> decompressed;
		decompressed.resize(multimesh->instances * (int)new_stride);
		float *w = decompressed.ptrw();
		const float *r = ret.ptr();

		for (int i = 0; i < multimesh->instances; i++) {
			{
				float *newptr = w + i * new_stride;
				const float *oldptr = r + i * multimesh->stride_cache;
				float vals[8] = { oldptr[0], oldptr[1], oldptr[2], oldptr[3], oldptr[4], oldptr[5], oldptr[6], oldptr[7] };
				memcpy(newptr, vals, 8 * 4);
			}

			if (multimesh->xform_format == RS::MULTIMESH_TRANSFORM_3D) {
				float *newptr = w + i * new_stride + 8;
				const float *oldptr = r + i * multimesh->stride_cache + 8;
				float vals[8] = { oldptr[0], oldptr[1], oldptr[2], oldptr[3] };
				memcpy(newptr, vals, 4 * 4);
			}

			if (multimesh->uses_colors) {
				float *newptr = w + i * new_stride + (multimesh->xform_format == RS::MULTIMESH_TRANSFORM_2D ? 8 : 12);
				const float *oldptr = r + i * multimesh->stride_cache + multimesh->color_offset_cache;
				uint16_t raw_data[4];
				memcpy(raw_data, oldptr, 2 * 4);
				newptr[0] = Math::half_to_float(raw_data[0]);
				newptr[1] = Math::half_to_float(raw_data[1]);
				newptr[2] = Math::half_to_float(raw_data[2]);
				newptr[3] = Math::half_to_float(raw_data[3]);
			}
			if (multimesh->uses_custom_data) {
				float *newptr = w + i * new_stride + (multimesh->xform_format == RS::MULTIMESH_TRANSFORM_2D ? 8 : 12) + (multimesh->uses_colors ? 4 : 0);
				const float *oldptr = r + i * multimesh->stride_cache + multimesh->custom_data_offset_cache;
				uint16_t raw_data[4];
				memcpy(raw_data, oldptr, 2 * 4);
				newptr[0] = Math::half_to_float(raw_data[0]);
				newptr[1] = Math::half_to_float(raw_data[1]);
				newptr[2] = Math::half_to_float(raw_data[2]);
				newptr[3] = Math::half_to_float(raw_data[3]);
			}
		}
		return decompressed;
	} else {
		return ret;
	}
}

void MeshStorage::multimesh_set_visible_instances(RID p_multimesh, int p_visible) {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND(!multimesh);
	ERR_FAIL_COND(p_visible < -1 || p_visible > multimesh->instances);
	if (multimesh->visible_instances == p_visible) {
		return;
	}

	if (multimesh->data_cache.size()) {
		//there is a data cache..
		_multimesh_mark_all_dirty(multimesh, false, true);
	}

	multimesh->visible_instances = p_visible;

	multimesh->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_MULTIMESH_VISIBLE_INSTANCES);
}

int MeshStorage::multimesh_get_visible_instances(RID p_multimesh) const {
	MultiMesh *multimesh = multimesh_owner.get_or_null(p_multimesh);
	ERR_FAIL_COND_V(!multimesh, 0);
	return multimesh->visible_instances;
}

void MeshStorage::_update_dirty_multimeshes() {
	while (multimesh_dirty_list) {
		MultiMesh *multimesh = multimesh_dirty_list;

		if (multimesh->data_cache.size()) { //may have been cleared, so only process if it exists
			const float *data = multimesh->data_cache.ptr();

			uint32_t visible_instances = multimesh->visible_instances >= 0 ? multimesh->visible_instances : multimesh->instances;

			if (multimesh->data_cache_used_dirty_regions) {
				uint32_t data_cache_dirty_region_count = (multimesh->instances - 1) / MULTIMESH_DIRTY_REGION_SIZE + 1;
				uint32_t visible_region_count = visible_instances == 0 ? 0 : (visible_instances - 1) / MULTIMESH_DIRTY_REGION_SIZE + 1;

				GLint region_size = multimesh->stride_cache * MULTIMESH_DIRTY_REGION_SIZE * sizeof(float);

				if (multimesh->data_cache_used_dirty_regions > 32 || multimesh->data_cache_used_dirty_regions > visible_region_count / 2) {
					// If there too many dirty regions, or represent the majority of regions, just copy all, else transfer cost piles up too much
					glBindBuffer(GL_ARRAY_BUFFER, multimesh->buffer);
					glBufferData(GL_ARRAY_BUFFER, MIN(visible_region_count * region_size, multimesh->instances * multimesh->stride_cache * sizeof(float)), data, GL_STATIC_DRAW);
					glBindBuffer(GL_ARRAY_BUFFER, 0);
				} else {
					// Not that many regions? update them all
					// TODO: profile the performance cost on low end
					glBindBuffer(GL_ARRAY_BUFFER, multimesh->buffer);
					for (uint32_t i = 0; i < visible_region_count; i++) {
						if (multimesh->data_cache_dirty_regions[i]) {
							GLint offset = i * region_size;
							GLint size = multimesh->stride_cache * (uint32_t)multimesh->instances * (uint32_t)sizeof(float);
							uint32_t region_start_index = multimesh->stride_cache * MULTIMESH_DIRTY_REGION_SIZE * i;
							glBufferSubData(GL_ARRAY_BUFFER, offset, MIN(region_size, size - offset), &data[region_start_index]);
						}
					}
					glBindBuffer(GL_ARRAY_BUFFER, 0);
				}

				for (uint32_t i = 0; i < data_cache_dirty_region_count; i++) {
					multimesh->data_cache_dirty_regions[i] = false;
				}

				multimesh->data_cache_used_dirty_regions = 0;
			}

			if (multimesh->aabb_dirty && multimesh->mesh.is_valid()) {
				_multimesh_re_create_aabb(multimesh, data, visible_instances);
				multimesh->aabb_dirty = false;
				multimesh->dependency.changed_notify(RendererStorage::DEPENDENCY_CHANGED_AABB);
			}
		}

		multimesh_dirty_list = multimesh->dirty_list;

		multimesh->dirty_list = nullptr;
		multimesh->dirty = false;
	}

	multimesh_dirty_list = nullptr;
}

/* SKELETON API */

RID MeshStorage::skeleton_allocate() {
	return RID();
}

void MeshStorage::skeleton_initialize(RID p_rid) {
}

void MeshStorage::skeleton_free(RID p_rid) {
}

void MeshStorage::skeleton_allocate_data(RID p_skeleton, int p_bones, bool p_2d_skeleton) {
}

void MeshStorage::skeleton_set_base_transform_2d(RID p_skeleton, const Transform2D &p_base_transform) {
}

int MeshStorage::skeleton_get_bone_count(RID p_skeleton) const {
	return 0;
}

void MeshStorage::skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform3D &p_transform) {
}

Transform3D MeshStorage::skeleton_bone_get_transform(RID p_skeleton, int p_bone) const {
	return Transform3D();
}

void MeshStorage::skeleton_bone_set_transform_2d(RID p_skeleton, int p_bone, const Transform2D &p_transform) {
}

Transform2D MeshStorage::skeleton_bone_get_transform_2d(RID p_skeleton, int p_bone) const {
	return Transform2D();
}

void MeshStorage::skeleton_update_dependency(RID p_base, RendererStorage::DependencyTracker *p_instance) {
}

#endif // GLES3_ENABLED
