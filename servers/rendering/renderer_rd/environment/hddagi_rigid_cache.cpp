/**************************************************************************/
/*  hddagi_rigid_cache.cpp                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "hddagi_rigid_cache.h"

#include "servers/rendering/renderer_rd/renderer_scene_render_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/mesh_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"
#include "servers/rendering/rendering_server_globals.h"

using namespace RendererRD;

namespace {

constexpr uint32_t SURFACE_CACHE_HEADER_WORDS = 8;
constexpr uint32_t SURFACE_CACHE_BLOCK_CELLS = 8;
// Two words per cell and five shared material words per block in hddagi_rigid_cache.glsl.
constexpr uint32_t SURFACE_CACHE_WORDS_PER_BLOCK = 21;

constexpr float SURFACE_MAX_AXIS_CELLS = 92.0f;
constexpr int SURFACE_PADDING_CELLS = 5;
constexpr int PATCH_BLOCK_SIZE = 16;
constexpr float PROJECT_MAX_SCALE = 2.0f;

} // namespace

void HDDAGIRigidCache::Textures::create(const Vector3i &p_size) {
	RD *rd = RD::get_singleton();
	RD::TextureFormat format;
	format.texture_type = RD::TEXTURE_TYPE_3D;
	format.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;
	auto create_clear = [&]() {
		RID texture = rd->texture_create(format, RD::TextureView());
		rd->texture_clear(texture, Color(0, 0, 0, 0), 0, 1, 0, 1);
		return texture;
	};
	format.width = p_size.x / 2;
	format.height = p_size.y / 2;
	format.depth = p_size.z / 2;
	format.format = RD::DATA_FORMAT_R32_UINT;
	format.usage_bits |= RD::TEXTURE_USAGE_STORAGE_ATOMIC_BIT;
	emission = create_clear();
	emission_aniso = create_clear();
	format.usage_bits &= ~RD::TEXTURE_USAGE_STORAGE_ATOMIC_BIT;
	format.depth *= 6;
	format.format = RD::DATA_FORMAT_R16_UINT;
	albedo = create_clear();
	format.width = p_size.x;
	format.height = p_size.y;
	format.depth = p_size.z;
	format.format = RD::DATA_FORMAT_R32_UINT;
	format.usage_bits |= RD::TEXTURE_USAGE_STORAGE_ATOMIC_BIT;
	normals = create_clear();
}

void HDDAGIRigidCache::Textures::clear() const {
	RD *rd = RD::get_singleton();
	for (RID texture : { albedo, emission, emission_aniso, normals }) {
		if (texture.is_valid()) {
			rd->texture_clear(texture, Color(0, 0, 0, 0), 0, 1, 0, 1);
		}
	}
}

void HDDAGIRigidCache::Textures::free() const {
	if (normals.is_valid()) {
		RD *rd = RD::get_singleton();
		rd->free_rid(albedo);
		rd->free_rid(emission);
		rd->free_rid(emission_aniso);
		rd->free_rid(normals);
	}
}

void HDDAGIRigidCache::Surface::free() {
	if (samples.is_valid()) {
		RD::get_singleton()->free_rid(samples);
		RD::get_singleton()->free_rid(dispatch);
		samples = RID();
		dispatch = RID();
	}
}

uint64_t HDDAGIRigidCache::get_material_revision(const RenderGeometryInstanceBase *p_instance) const {
	const auto *data = p_instance->data;
	MaterialStorage *materials = MaterialStorage::get_singleton();
	uint64_t revision = 0;
	auto include_material = [&](RID p_material) {
		revision = MAX(revision, materials->material_get_revision(p_material));
	};
	include_material(data->material_overlay);
	if (data->material_override.is_valid()) {
		include_material(data->material_override);
	} else if (data->base_type == RSE::INSTANCE_MESH) {
		MeshStorage *meshes = MeshStorage::get_singleton();
		int count = meshes->mesh_get_surface_count(data->base);
		for (int i = 0; i < count; i++) {
			RID override = i < data->surface_materials.size() ? data->surface_materials[i] : RID();
			include_material(override.is_valid() ? override : meshes->mesh_surface_get_material(data->base, i));
		}
	}
	return revision;
}

bool HDDAGIRigidCache::track(RenderGeometryInstance *p_instance) {
	auto *geometry = static_cast<RenderGeometryInstanceBase *>(p_instance);
	auto it = surfaces.find(p_instance);
	const bool inserted = !it;
	if (!it) {
		it = surfaces.insert(p_instance, Surface());
	}
	Surface &surface = it->value;
	const bool moving = !inserted && surface.transform != geometry->transform;
	bool changed = inserted;
	if (moving != surface.moving) {
		moving_count += moving ? 1 : -1;
		changed = true;
	}
	surface.transform = geometry->transform;
	surface.moving = moving;
	const uint64_t epoch = MaterialStorage::get_singleton()->get_material_revision();
	if (inserted || surface.revision != geometry->surface_revision || surface.material_epoch != epoch) {
		const uint64_t revision = get_material_revision(geometry);
		const bool changed_surface = surface.revision != geometry->surface_revision || surface.material_revision != revision;
		changed |= changed_surface;
		surface.dirty |= changed_surface;
		surface.revision = geometry->surface_revision;
		surface.material_revision = revision;
		surface.material_epoch = epoch;
	}
	return changed;
}

void HDDAGIRigidCache::remove(RenderGeometryInstance *p_instance) {
	auto entry = surfaces.find(p_instance);
	if (entry) {
		moving_count -= entry->value.moving;
		entry->value.free();
		surfaces.remove(entry);
	}
}

HDDAGIRigidCache::~HDDAGIRigidCache() {
	for (KeyValue<RenderGeometryInstance *, Surface> &entry : surfaces) {
		entry.value.free();
	}
	if (version.is_valid()) {
		shader.version_free(version);
	}
}

void HDDAGIRigidCache::ensure_shader() {
	if (version.is_valid()) {
		return;
	}
	Vector<String> modes = { "\n#define MODE_CAPTURE\n", "\n#define MODE_RESTORE\n", "\n#define MODE_CLEAR\n", "\n#define MODE_PROJECT\n" };
	shader.initialize(modes);
	version = shader.version_create();
	for (int i = 0; i < MODE_MAX; i++) {
		shaders[i] = shader.version_get_shader(version, i);
		pipelines[i] = RD::get_singleton()->compute_pipeline_create(shaders[i]);
	}
}

void HDDAGIRigidCache::capture(Surface &r_surface, RenderGeometryInstanceBase *p_instance, const Ref<RenderSceneBuffersRD> &p_buffers, float p_cell) {
	RD *rd = RD::get_singleton();
	r_surface.free();
	Transform3D frame(p_instance->transform.basis.orthonormalized(), p_instance->transform.origin);
	if (frame.basis.determinant() < 0.0f) {
		frame.basis.set_column(0, -frame.basis.get_column(0));
	}
	Transform3D to_object = p_instance->transform.affine_inverse() * frame;
	AABB bounds = to_object.affine_inverse().xform(p_instance->data->aabb);
	float pitch = MAX(p_cell * 0.5f, float(bounds.size[bounds.size.max_axis_index()]) / SURFACE_MAX_AXIS_CELLS);
	Vector3i size = (Vector3i((bounds.size / pitch).ceil()) + Vector3i(SURFACE_PADDING_CELLS, SURFACE_PADDING_CELLS, SURFACE_PADDING_CELLS)) / 2 * 2;
	AABB volume(bounds.get_center() - Vector3(size) * pitch * 0.5f, Vector3(size) * pitch);
	r_surface.grid_to_object = to_object * Transform3D(Basis::from_scale(Vector3(pitch, pitch, pitch)), volume.position);
	r_surface.dirty = false;

	RENDER_TIMESTAMP("> HDDAGI Capture Rigid Surface");
	Textures textures;
	textures.create(size);
	PagedArray<RenderGeometryInstance *> instances;
	instances.set_page_pool(&instance_pool);
	instances.push_back(p_instance);
	RendererSceneRenderRD::get_singleton()->_render_hddagi(p_buffers, Vector3i(), size, volume, instances, textures.albedo, textures.emission, textures.emission_aniso, textures.normals, 1.0f, &frame);

	uint32_t header[4] = { 0, 1, 1, 0 };
	const uint32_t cell_count = size.x * size.y * size.z;
	const uint32_t cache_bytes = SURFACE_CACHE_HEADER_WORDS * sizeof(uint32_t) + cell_count / SURFACE_CACHE_BLOCK_CELLS * SURFACE_CACHE_WORDS_PER_BLOCK * sizeof(uint32_t);
	r_surface.samples = rd->storage_buffer_create(cache_bytes);
	r_surface.dispatch = rd->storage_buffer_create(sizeof(header), Vector<uint8_t>(), RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
	rd->buffer_update(r_surface.samples, 0, sizeof(header), header);
	RID uniforms = UniformSetCacheRD::get_singleton()->get_cache(shaders[CAPTURE], 0,
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 0, textures.albedo),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, textures.emission),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, textures.emission_aniso),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, textures.normals),
			RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 8, r_surface.samples));
	RD::ComputeListID list = rd->compute_list_begin();
	rd->compute_list_bind_compute_pipeline(list, pipelines[CAPTURE]);
	rd->compute_list_bind_uniform_set(list, uniforms, 0);
	rd->compute_list_dispatch_threads(list, size.x, size.y, size.z);
	rd->compute_list_end();
	rd->buffer_copy(r_surface.samples, r_surface.dispatch, 0, 0, sizeof(header));
	textures.free();
	RENDER_TIMESTAMP("< HDDAGI Capture Rigid Surface");
}

void HDDAGIRigidCache::project(const Surface &p_surface, const Transform3D &p_transform, const AABB &p_world_bounds, const Vector3i &p_from, const Vector3i &p_size, const Textures &p_output, float p_exposure) {
	RD *rd = RD::get_singleton();
	Vector3 to_grid = Vector3(p_size) / p_world_bounds.size;
	Transform3D to_world = p_transform * p_surface.grid_to_object;
	Transform3D to_cell(Basis::from_scale(to_grid), Vector3(p_from) - to_grid * p_world_bounds.position);
	struct {
		float to_cell[12];
		float normal[12];
		int32_t from[3];
		uint32_t subsamples;
		int32_t end[3];
		float exposure;
	} params;
	MaterialStorage::store_transform_transposed_3x4(to_cell * to_world, params.to_cell);
	MaterialStorage::store_basis_3x4(to_world.basis.inverse().transposed(), params.normal);
	Vector3 scale = (to_cell * to_world).basis.get_scale_abs();
	params.exposure = p_exposure;
	params.subsamples = MAX(1, int(Math::ceil(scale[scale.max_axis_index()] - 0.001f)));
	for (int i = 0; i < 3; i++) {
		params.from[i] = p_from[i];
		params.end[i] = p_from[i] + p_size[i];
	}

	RID uniforms = UniformSetCacheRD::get_singleton()->get_cache(shaders[PROJECT], 0,
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 4, p_output.albedo),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, p_output.emission),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 6, p_output.emission_aniso),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 7, p_output.normals),
			RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 8, p_surface.samples));
	RD::ComputeListID list = rd->compute_list_begin();
	rd->compute_list_bind_compute_pipeline(list, pipelines[PROJECT]);
	rd->compute_list_bind_uniform_set(list, uniforms, 0);
	rd->compute_list_set_push_constant(list, &params, sizeof(params));
	rd->compute_list_dispatch_indirect(list, p_surface.dispatch, 0);
	rd->compute_list_end();
}

void HDDAGIRigidCache::get_patch_extent(const Vector3i &p_from, const Vector3i &p_size, const Vector3i &p_grid, Vector3i &r_from, Vector3i &r_end) {
	r_from = p_from / PATCH_BLOCK_SIZE * PATCH_BLOCK_SIZE;
	r_end = ((p_from + p_size + Vector3i(PATCH_BLOCK_SIZE - 1, PATCH_BLOCK_SIZE - 1, PATCH_BLOCK_SIZE - 1)) / PATCH_BLOCK_SIZE * PATCH_BLOCK_SIZE).min(p_grid);
}

void HDDAGIRigidCache::restore(const Textures *p_static, const Vector3i &p_offset, const Vector3i &p_from, const Vector3i &p_size, const Textures &p_output, float p_exposure) {
	RD *rd = RD::get_singleton();
	struct {
		int32_t from[4] = {};
		int32_t offset[4] = {};
		int32_t size[3] = {};
		float exposure;
	} params;
	params.exposure = p_exposure;
	for (int axis = 0; axis < 3; axis++) {
		params.from[axis] = p_from[axis];
		params.offset[axis] = p_offset[axis];
		params.size[axis] = p_size[axis];
	}
	const Mode mode = p_static ? RESTORE : CLEAR;
	LocalVector<RD::Uniform> bindings;
	if (p_static) {
		bindings.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 0, p_static->albedo));
		bindings.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 1, p_static->emission));
		bindings.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 2, p_static->emission_aniso));
		bindings.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 3, p_static->normals));
	}
	bindings.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 4, p_output.albedo));
	bindings.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 5, p_output.emission));
	bindings.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 6, p_output.emission_aniso));
	bindings.push_back(RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 7, p_output.normals));
	RID uniforms = UniformSetCacheRD::get_singleton()->get_cache_vec(shaders[mode], 0, bindings);
	RD::ComputeListID list = rd->compute_list_begin();
	rd->compute_list_bind_compute_pipeline(list, pipelines[mode]);
	rd->compute_list_bind_uniform_set(list, uniforms, 0);
	rd->compute_list_set_push_constant(list, &params, sizeof(params));
	rd->compute_list_dispatch_threads(list, p_size.x, p_size.y, p_size.z);
	rd->compute_list_end();
}

void HDDAGIRigidCache::render(const Ref<RenderSceneBuffersRD> &p_buffers, StaticPatch (&p_patches)[PATCH_COUNT], const Vector3i &p_from, const Vector3i &p_size, const AABB &p_bounds, const Vector3i &p_grid, float p_cell, const PagedArray<RenderGeometryInstance *> &p_instances, const Textures &p_output, float p_exposure) {
	ensure_shader();
	const Vector3 cell = p_bounds.size / Vector3(p_size);
	Vector3i from, end;
	get_patch_extent(p_from, p_size, p_grid, from, end);
	Vector3i size = end - from;
	if (!is_moving() || size.x * size.y * size.z > p_grid.x * p_grid.y * p_grid.z / (2 * PATCH_COUNT)) {
		restore(nullptr, Vector3i(), p_from, p_size, p_output, p_exposure);
		render_cascade(p_buffers, p_from, p_size, p_bounds, p_cell, p_instances, p_output, p_exposure);
		return;
	}
	StaticPatch *patch = &p_patches[0];
	bool fits = false;
	for (StaticPatch &candidate : p_patches) {
		if (candidate.valid && from == candidate.from && end == candidate.from + candidate.size) {
			patch = &candidate;
			fits = true;
			break;
		}
		if (candidate.last_use < patch->last_use) {
			patch = &candidate;
		}
	}
	StaticPatch &r_patch = *patch;
	r_patch.last_use = ++patch_serial;
	uint64_t epoch = MaterialStorage::get_singleton()->get_material_revision();
	AABB bounds(p_bounds.position + Vector3(from - p_from) * cell, Vector3(size) * cell);
	PagedArray<RenderGeometryInstance *> statistics;
	statistics.set_page_pool(&instance_pool);
	LocalVector<RenderGeometryInstanceBase *> rigid;
	bool rebuild = !fits;
	for (uint32_t i = 0; i < p_instances.size(); i++) {
		auto *instance = static_cast<RenderGeometryInstanceBase *>(p_instances[i]);
		const Surface *surface = surfaces.getptr(instance);
		if (surface && surface->moving) {
			if (p_bounds.intersects_inclusive(instance->transformed_aabb)) {
				rigid.push_back(instance);
			}
		} else if (bounds.intersects_inclusive(instance->transformed_aabb)) {
			statistics.push_back(instance);
			const StaticStamp *stamp = r_patch.instances.getptr(instance);
			rebuild |= !stamp || stamp->transform != instance->transform || stamp->revision != instance->surface_revision || stamp->mesh != instance->data->base || (epoch != r_patch.material_epoch && stamp->material_revision != get_material_revision(instance));
		}
	}
	r_patch.material_epoch = epoch;
	if (rebuild || r_patch.instances.size() != statistics.size()) {
		r_patch.textures.free();
		r_patch.textures.create(size);
		r_patch.from = from;
		r_patch.size = size;
		r_patch.valid = true;
		r_patch.instances.clear();
		for (uint32_t i = 0; i < statistics.size(); i++) {
			auto *instance = static_cast<RenderGeometryInstanceBase *>(statistics[i]);
			r_patch.instances.insert(instance, { instance->transform, instance->data->base, instance->surface_revision, get_material_revision(instance) });
		}
		if (statistics.size()) {
			RendererSceneRenderRD::get_singleton()->_render_hddagi(p_buffers, Vector3i(), size, bounds, statistics, r_patch.textures.albedo, r_patch.textures.emission, r_patch.textures.emission_aniso, r_patch.textures.normals, 1.0f);
		}
	}

	RENDER_TIMESTAMP("> HDDAGI Project Rigid Surface");
	restore(&r_patch.textures, p_from - from, p_from, p_size, p_output, p_exposure);
	for (RenderGeometryInstanceBase *instance : rigid) {
		render_object(instance, p_buffers, p_from, p_size, p_bounds, p_cell, p_output, p_exposure);
	}
	RENDER_TIMESTAMP("< HDDAGI Project Rigid Surface");
}

void HDDAGIRigidCache::render_object(RenderGeometryInstanceBase *p_instance, const Ref<RenderSceneBuffersRD> &p_buffers, const Vector3i &p_from, const Vector3i &p_size, const AABB &p_bounds, float p_cell, const Textures &p_output, float p_exposure) {
	Surface &surface = surfaces.get(p_instance);
	if (surface.samples.is_null() || surface.dirty) {
		capture(surface, p_instance, p_buffers, p_cell);
	}
	Transform3D to_grid(Basis::from_scale(Vector3(p_size) / p_bounds.size), Vector3());
	Vector3 scale = (to_grid * p_instance->transform * surface.grid_to_object).basis.get_scale_abs();
	if (scale[scale.max_axis_index()] > PROJECT_MAX_SCALE) {
		PagedArray<RenderGeometryInstance *> instances;
		instances.set_page_pool(&instance_pool);
		instances.push_back(p_instance);
		RendererSceneRenderRD::get_singleton()->_render_hddagi(p_buffers, p_from, p_size, p_bounds, instances, p_output.albedo, p_output.emission, p_output.emission_aniso, p_output.normals, p_exposure);
	} else {
		project(surface, p_instance->transform, p_bounds, p_from, p_size, p_output, p_exposure);
	}
}

void HDDAGIRigidCache::render_cascade(const Ref<RenderSceneBuffersRD> &p_buffers, const Vector3i &p_from, const Vector3i &p_size, const AABB &p_bounds, float p_cell, const PagedArray<RenderGeometryInstance *> &p_instances, const Textures &p_output, float p_exposure) {
	ensure_shader();
	PagedArray<RenderGeometryInstance *> statistics;
	statistics.set_page_pool(&instance_pool);
	LocalVector<RenderGeometryInstanceBase *> rigid;
	for (uint32_t i = 0; i < p_instances.size(); i++) {
		const Surface *surface = surfaces.getptr(p_instances[i]);
		if (surface && surface->moving) {
			rigid.push_back(static_cast<RenderGeometryInstanceBase *>(p_instances[i]));
		} else {
			statistics.push_back(p_instances[i]);
		}
	}
	if (statistics.size() != 0) {
		RendererSceneRenderRD::get_singleton()->_render_hddagi(p_buffers, p_from, p_size, p_bounds, statistics, p_output.albedo, p_output.emission, p_output.emission_aniso, p_output.normals, p_exposure);
	}
	for (RenderGeometryInstanceBase *instance : rigid) {
		render_object(instance, p_buffers, p_from, p_size, p_bounds, p_cell, p_output, p_exposure);
	}
}
