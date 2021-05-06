/*************************************************************************/
/*  raycast_occlusion_cull.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "raycast_occlusion_cull.h"
#include "core/config/project_settings.h"
#include "core/templates/local_vector.h"

#ifdef __SSE2__
#include <pmmintrin.h>
#endif

RaycastOcclusionCull *RaycastOcclusionCull::raycast_singleton = nullptr;

void RaycastOcclusionCull::RaycastHZBuffer::clear() {
	HZBuffer::clear();

	camera_rays.clear();
	camera_ray_masks.clear();
	packs_size = Size2i();
}

void RaycastOcclusionCull::RaycastHZBuffer::resize(const Size2i &p_size) {
	if (p_size == Size2i()) {
		clear();
		return;
	}

	if (!sizes.is_empty() && p_size == sizes[0]) {
		return; // Size didn't change
	}

	HZBuffer::resize(p_size);

	packs_size = Size2i(Math::ceil(p_size.x / (float)TILE_SIZE), Math::ceil(p_size.y / (float)TILE_SIZE));
	int ray_packets_count = packs_size.x * packs_size.y;
	camera_rays.resize(ray_packets_count);
	camera_ray_masks.resize(ray_packets_count * TILE_SIZE * TILE_SIZE);
}

void RaycastOcclusionCull::RaycastHZBuffer::update_camera_rays(const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_orthogonal, ThreadWorkPool &p_thread_work_pool) {
	CameraRayThreadData td;
	td.camera_matrix = p_cam_projection;
	td.camera_transform = p_cam_transform;
	td.camera_orthogonal = p_cam_orthogonal;
	td.thread_count = p_thread_work_pool.get_thread_count();

	p_thread_work_pool.do_work(td.thread_count, this, &RaycastHZBuffer::_camera_rays_threaded, &td);
}

void RaycastOcclusionCull::RaycastHZBuffer::_camera_rays_threaded(uint32_t p_thread, RaycastOcclusionCull::RaycastHZBuffer::CameraRayThreadData *p_data) {
	uint32_t packs_total = camera_rays.size();
	uint32_t total_threads = p_data->thread_count;
	uint32_t from = p_thread * packs_total / total_threads;
	uint32_t to = (p_thread + 1 == total_threads) ? packs_total : ((p_thread + 1) * packs_total / total_threads);
	_generate_camera_rays(p_data->camera_transform, p_data->camera_matrix, p_data->camera_orthogonal, from, to);
}

void RaycastOcclusionCull::RaycastHZBuffer::_generate_camera_rays(const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_orthogonal, int p_from, int p_to) {
	Size2i buffer_size = sizes[0];

	CameraMatrix inv_camera_matrix = p_cam_projection.inverse();
	float z_far = p_cam_projection.get_z_far() * 1.05f;
	debug_tex_range = z_far;

	RayPacket *ray_packets = camera_rays.ptr();
	uint32_t *ray_masks = camera_ray_masks.ptr();

	for (int i = p_from; i < p_to; i++) {
		RayPacket &packet = ray_packets[i];
		int tile_x = (i % packs_size.x) * TILE_SIZE;
		int tile_y = (i / packs_size.x) * TILE_SIZE;

		for (int j = 0; j < TILE_RAYS; j++) {
			float x = tile_x + j % TILE_SIZE;
			float y = tile_y + j / TILE_SIZE;

			ray_masks[i * TILE_RAYS + j] = ~0U;

			if (x >= buffer_size.x || y >= buffer_size.y) {
				ray_masks[i * TILE_RAYS + j] = 0U;
			} else {
				float u = x / (buffer_size.x - 1);
				float v = y / (buffer_size.y - 1);
				u = u * 2.0f - 1.0f;
				v = v * 2.0f - 1.0f;

				Plane pixel_proj = Plane(u, v, -1.0, 1.0);
				Plane pixel_view = inv_camera_matrix.xform4(pixel_proj);
				Vector3 pixel_world = p_cam_transform.xform(pixel_view.normal);

				Vector3 dir;
				if (p_cam_orthogonal) {
					dir = -p_cam_transform.basis.get_axis(2);
				} else {
					dir = (pixel_world - p_cam_transform.origin).normalized();
				}

				packet.ray.org_x[j] = pixel_world.x;
				packet.ray.org_y[j] = pixel_world.y;
				packet.ray.org_z[j] = pixel_world.z;

				packet.ray.dir_x[j] = dir.x;
				packet.ray.dir_y[j] = dir.y;
				packet.ray.dir_z[j] = dir.z;

				packet.ray.tnear[j] = 0.0f;

				packet.ray.time[j] = 0.0f;

				packet.ray.flags[j] = 0;
				packet.ray.mask[j] = -1;
				packet.hit.geomID[j] = RTC_INVALID_GEOMETRY_ID;
			}

			packet.ray.tfar[j] = z_far;
		}
	}
}

void RaycastOcclusionCull::RaycastHZBuffer::sort_rays() {
	if (is_empty()) {
		return;
	}

	Size2i buffer_size = sizes[0];
	for (int i = 0; i < packs_size.y; i++) {
		for (int j = 0; j < packs_size.x; j++) {
			for (int tile_i = 0; tile_i < TILE_SIZE; tile_i++) {
				for (int tile_j = 0; tile_j < TILE_SIZE; tile_j++) {
					int x = j * TILE_SIZE + tile_j;
					int y = i * TILE_SIZE + tile_i;
					if (x >= buffer_size.x || y >= buffer_size.y) {
						continue;
					}
					int k = tile_i * TILE_SIZE + tile_j;
					int packet_index = i * packs_size.x + j;
					mips[0][y * buffer_size.x + x] = camera_rays[packet_index].ray.tfar[k];
				}
			}
		}
	}
}

////////////////////////////////////////////////////////

bool RaycastOcclusionCull::is_occluder(RID p_rid) {
	return occluder_owner.owns(p_rid);
}

RID RaycastOcclusionCull::occluder_allocate() {
	return occluder_owner.allocate_rid();
}

void RaycastOcclusionCull::occluder_initialize(RID p_occluder) {
	Occluder *occluder = memnew(Occluder);
	occluder_owner.initialize_rid(p_occluder, occluder);
}

void RaycastOcclusionCull::occluder_set_mesh(RID p_occluder, const PackedVector3Array &p_vertices, const PackedInt32Array &p_indices) {
	Occluder *occluder = occluder_owner.getornull(p_occluder);
	ERR_FAIL_COND(!occluder);

	occluder->vertices = p_vertices;
	occluder->indices = p_indices;

	for (Set<InstanceID>::Element *E = occluder->users.front(); E; E = E->next()) {
		RID scenario_rid = E->get().scenario;
		RID instance_rid = E->get().instance;
		ERR_CONTINUE(!scenarios.has(scenario_rid));
		Scenario &scenario = scenarios[scenario_rid];
		ERR_CONTINUE(!scenario.instances.has(instance_rid));

		if (!scenario.dirty_instances.has(instance_rid)) {
			scenario.dirty_instances.insert(instance_rid);
			scenario.dirty_instances_array.push_back(instance_rid);
		}
	}
}

void RaycastOcclusionCull::free_occluder(RID p_occluder) {
	Occluder *occluder = occluder_owner.getornull(p_occluder);
	ERR_FAIL_COND(!occluder);
	memdelete(occluder);
	occluder_owner.free(p_occluder);
}

////////////////////////////////////////////////////////

void RaycastOcclusionCull::add_scenario(RID p_scenario) {
	if (scenarios.has(p_scenario)) {
		scenarios[p_scenario].removed = false;
	} else {
		scenarios[p_scenario] = Scenario();
	}
}

void RaycastOcclusionCull::remove_scenario(RID p_scenario) {
	ERR_FAIL_COND(!scenarios.has(p_scenario));
	Scenario &scenario = scenarios[p_scenario];
	scenario.removed = true;
}

void RaycastOcclusionCull::scenario_set_instance(RID p_scenario, RID p_instance, RID p_occluder, const Transform &p_xform, bool p_enabled) {
	ERR_FAIL_COND(!scenarios.has(p_scenario));
	Scenario &scenario = scenarios[p_scenario];

	if (!scenario.instances.has(p_instance)) {
		scenario.instances[p_instance] = OccluderInstance();
	}

	OccluderInstance &instance = scenario.instances[p_instance];

	if (instance.removed) {
		instance.removed = false;
		scenario.removed_instances.erase(p_instance);
	}

	bool changed = false;

	if (instance.occluder != p_occluder) {
		Occluder *old_occluder = occluder_owner.getornull(instance.occluder);
		if (old_occluder) {
			old_occluder->users.erase(InstanceID(p_scenario, p_instance));
		}

		instance.occluder = p_occluder;

		if (p_occluder.is_valid()) {
			Occluder *occluder = occluder_owner.getornull(p_occluder);
			ERR_FAIL_COND(!occluder);
			occluder->users.insert(InstanceID(p_scenario, p_instance));
		}
		changed = true;
	}

	if (instance.xform != p_xform) {
		scenario.instances[p_instance].xform = p_xform;
		changed = true;
	}

	if (instance.enabled != p_enabled) {
		instance.enabled = p_enabled;
		scenario.dirty = true; // The scenario needs a scene re-build, but the instance doesn't need update
	}

	if (changed && !scenario.dirty_instances.has(p_instance)) {
		scenario.dirty_instances.insert(p_instance);
		scenario.dirty_instances_array.push_back(p_instance);
		scenario.dirty = true;
	}
}

void RaycastOcclusionCull::scenario_remove_instance(RID p_scenario, RID p_instance) {
	ERR_FAIL_COND(!scenarios.has(p_scenario));
	Scenario &scenario = scenarios[p_scenario];

	if (scenario.instances.has(p_instance)) {
		OccluderInstance &instance = scenario.instances[p_instance];

		if (!instance.removed) {
			Occluder *occluder = occluder_owner.getornull(instance.occluder);
			if (occluder) {
				occluder->users.erase(InstanceID(p_scenario, p_instance));
			}

			scenario.removed_instances.push_back(p_instance);
			instance.removed = true;
		}
	}
}

void RaycastOcclusionCull::Scenario::_update_dirty_instance_thread(int p_idx, RID *p_instances) {
	_update_dirty_instance(p_idx, p_instances, nullptr);
}

void RaycastOcclusionCull::Scenario::_update_dirty_instance(int p_idx, RID *p_instances, ThreadWorkPool *p_thread_pool) {
	OccluderInstance *occ_inst = instances.getptr(p_instances[p_idx]);

	if (!occ_inst) {
		return;
	}

	Occluder *occ = raycast_singleton->occluder_owner.getornull(occ_inst->occluder);

	if (!occ) {
		return;
	}

	int vertices_size = occ->vertices.size();

	// Embree requires the last element to be readable by a 16-byte SSE load instruction, so we add padding to be safe.
	occ_inst->xformed_vertices.resize(vertices_size + 1);

	const Vector3 *read_ptr = occ->vertices.ptr();
	Vector3 *write_ptr = occ_inst->xformed_vertices.ptr();

	if (p_thread_pool && vertices_size > 1024) {
		TransformThreadData td;
		td.xform = occ_inst->xform;
		td.read = read_ptr;
		td.write = write_ptr;
		td.vertex_count = vertices_size;
		td.thread_count = p_thread_pool->get_thread_count();
		p_thread_pool->do_work(td.thread_count, this, &Scenario::_transform_vertices_thread, &td);
	} else {
		_transform_vertices_range(read_ptr, write_ptr, occ_inst->xform, 0, vertices_size);
	}

	occ_inst->indices.resize(occ->indices.size());
	memcpy(occ_inst->indices.ptr(), occ->indices.ptr(), occ->indices.size() * sizeof(int32_t));
}

void RaycastOcclusionCull::Scenario::_transform_vertices_thread(uint32_t p_thread, TransformThreadData *p_data) {
	uint32_t vertex_total = p_data->vertex_count;
	uint32_t total_threads = p_data->thread_count;
	uint32_t from = p_thread * vertex_total / total_threads;
	uint32_t to = (p_thread + 1 == total_threads) ? vertex_total : ((p_thread + 1) * vertex_total / total_threads);
	_transform_vertices_range(p_data->read, p_data->write, p_data->xform, from, to);
}

void RaycastOcclusionCull::Scenario::_transform_vertices_range(const Vector3 *p_read, Vector3 *p_write, const Transform &p_xform, int p_from, int p_to) {
	for (int i = p_from; i < p_to; i++) {
		p_write[i] = p_xform.xform(p_read[i]);
	}
}

void RaycastOcclusionCull::Scenario::_commit_scene(void *p_ud) {
	Scenario *scenario = (Scenario *)p_ud;
	int commit_idx = 1 - (scenario->current_scene_idx);
	rtcCommitScene(scenario->ebr_scene[commit_idx]);
	scenario->commit_done = true;
}

bool RaycastOcclusionCull::Scenario::update(ThreadWorkPool &p_thread_pool) {
	ERR_FAIL_COND_V(singleton == nullptr, false);

	if (commit_thread == nullptr) {
		commit_thread = memnew(Thread);
	}

	if (commit_thread->is_started()) {
		if (commit_done) {
			commit_thread->wait_to_finish();
			current_scene_idx = 1 - current_scene_idx;
		} else {
			return false;
		}
	}

	if (removed) {
		if (ebr_scene[0]) {
			rtcReleaseScene(ebr_scene[0]);
		}
		if (ebr_scene[1]) {
			rtcReleaseScene(ebr_scene[1]);
		}
		return true;
	}

	if (!dirty && removed_instances.is_empty() && dirty_instances_array.is_empty()) {
		return false;
	}

	for (unsigned int i = 0; i < removed_instances.size(); i++) {
		instances.erase(removed_instances[i]);
	}

	if (dirty_instances_array.size() / p_thread_pool.get_thread_count() > 128) {
		// Lots of instances, use per-instance threading
		p_thread_pool.do_work(dirty_instances_array.size(), this, &Scenario::_update_dirty_instance_thread, dirty_instances_array.ptr());
	} else {
		// Few instances, use threading on the vertex transforms
		for (unsigned int i = 0; i < dirty_instances_array.size(); i++) {
			_update_dirty_instance(i, dirty_instances_array.ptr(), &p_thread_pool);
		}
	}

	dirty_instances.clear();
	dirty_instances_array.clear();
	removed_instances.clear();

	if (raycast_singleton->ebr_device == nullptr) {
		raycast_singleton->_init_embree();
	}

	int next_scene_idx = 1 - current_scene_idx;
	RTCScene &next_scene = ebr_scene[next_scene_idx];

	if (next_scene) {
		rtcReleaseScene(next_scene);
	}

	next_scene = rtcNewScene(raycast_singleton->ebr_device);
	rtcSetSceneBuildQuality(next_scene, RTCBuildQuality(raycast_singleton->build_quality));

	const RID *inst_rid = nullptr;
	while ((inst_rid = instances.next(inst_rid))) {
		OccluderInstance *occ_inst = instances.getptr(*inst_rid);
		Occluder *occ = raycast_singleton->occluder_owner.getornull(occ_inst->occluder);

		if (!occ || !occ_inst->enabled) {
			continue;
		}

		RTCGeometry geom = rtcNewGeometry(raycast_singleton->ebr_device, RTC_GEOMETRY_TYPE_TRIANGLE);
		rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, occ_inst->xformed_vertices.ptr(), 0, sizeof(Vector3), occ_inst->xformed_vertices.size());
		rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, occ_inst->indices.ptr(), 0, sizeof(uint32_t) * 3, occ_inst->indices.size() / 3);
		rtcCommitGeometry(geom);
		rtcAttachGeometry(next_scene, geom);
		rtcReleaseGeometry(geom);
	}

	dirty = false;
	commit_done = false;
	commit_thread->start(&Scenario::_commit_scene, this);
	return false;
}

void RaycastOcclusionCull::Scenario::_raycast(uint32_t p_idx, const RaycastThreadData *p_raycast_data) const {
	RTCIntersectContext ctx;
	rtcInitIntersectContext(&ctx);
	ctx.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;

	rtcIntersect16((const int *)&p_raycast_data->masks[p_idx * TILE_RAYS], ebr_scene[current_scene_idx], &ctx, &p_raycast_data->rays[p_idx]);
}

void RaycastOcclusionCull::Scenario::raycast(LocalVector<RayPacket> &r_rays, const LocalVector<uint32_t> p_valid_masks, ThreadWorkPool &p_thread_pool) const {
	ERR_FAIL_COND(singleton == nullptr);
	if (raycast_singleton->ebr_device == nullptr) {
		return; // Embree is initialized on demand when there is some scenario with occluders in it.
	}

	if (ebr_scene[current_scene_idx] == nullptr) {
		return;
	}

	RaycastThreadData td;
	td.rays = r_rays.ptr();
	td.masks = p_valid_masks.ptr();

	p_thread_pool.do_work(r_rays.size(), this, &Scenario::_raycast, &td);
}

////////////////////////////////////////////////////////

void RaycastOcclusionCull::add_buffer(RID p_buffer) {
	ERR_FAIL_COND(buffers.has(p_buffer));
	buffers[p_buffer] = RaycastHZBuffer();
}

void RaycastOcclusionCull::remove_buffer(RID p_buffer) {
	ERR_FAIL_COND(!buffers.has(p_buffer));
	buffers.erase(p_buffer);
}

void RaycastOcclusionCull::buffer_set_scenario(RID p_buffer, RID p_scenario) {
	ERR_FAIL_COND(!buffers.has(p_buffer));
	ERR_FAIL_COND(p_scenario.is_valid() && !scenarios.has(p_scenario));
	buffers[p_buffer].scenario_rid = p_scenario;
}

void RaycastOcclusionCull::buffer_set_size(RID p_buffer, const Vector2i &p_size) {
	ERR_FAIL_COND(!buffers.has(p_buffer));
	buffers[p_buffer].resize(p_size);
}

void RaycastOcclusionCull::buffer_update(RID p_buffer, const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_orthogonal, ThreadWorkPool &p_thread_pool) {
	if (!buffers.has(p_buffer)) {
		return;
	}

	RaycastHZBuffer &buffer = buffers[p_buffer];

	if (buffer.is_empty() || !scenarios.has(buffer.scenario_rid)) {
		return;
	}

	Scenario &scenario = scenarios[buffer.scenario_rid];

	bool removed = scenario.update(p_thread_pool);

	if (removed) {
		scenarios.erase(buffer.scenario_rid);
		return;
	}

	buffer.update_camera_rays(p_cam_transform, p_cam_projection, p_cam_orthogonal, p_thread_pool);

	scenario.raycast(buffer.camera_rays, buffer.camera_ray_masks, p_thread_pool);
	buffer.sort_rays();
	buffer.update_mips();
}

RaycastOcclusionCull::HZBuffer *RaycastOcclusionCull::buffer_get_ptr(RID p_buffer) {
	if (!buffers.has(p_buffer)) {
		return nullptr;
	}
	return &buffers[p_buffer];
}

RID RaycastOcclusionCull::buffer_get_debug_texture(RID p_buffer) {
	ERR_FAIL_COND_V(!buffers.has(p_buffer), RID());
	return buffers[p_buffer].get_debug_texture();
}

////////////////////////////////////////////////////////

void RaycastOcclusionCull::set_build_quality(RS::ViewportOcclusionCullingBuildQuality p_quality) {
	if (build_quality == p_quality) {
		return;
	}

	build_quality = p_quality;

	const RID *scenario_rid = nullptr;
	while ((scenario_rid = scenarios.next(scenario_rid))) {
		scenarios[*scenario_rid].dirty = true;
	}
}

void RaycastOcclusionCull::_init_embree() {
#ifdef __SSE2__
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	String settings = vformat("threads=%d", MAX(1, OS::get_singleton()->get_processor_count() - 2));
	ebr_device = rtcNewDevice(settings.utf8().ptr());
}

RaycastOcclusionCull::RaycastOcclusionCull() {
	raycast_singleton = this;
	int default_quality = GLOBAL_GET("rendering/occlusion_culling/bvh_build_quality");
	build_quality = RS::ViewportOcclusionCullingBuildQuality(default_quality);
}

RaycastOcclusionCull::~RaycastOcclusionCull() {
	const RID *scenario_rid = nullptr;
	while ((scenario_rid = scenarios.next(scenario_rid))) {
		Scenario &scenario = scenarios[*scenario_rid];
		if (scenario.commit_thread) {
			scenario.commit_thread->wait_to_finish();
			memdelete(scenario.commit_thread);
		}
	}

	if (ebr_device != nullptr) {
#ifdef __SSE2__
		_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
		_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
#endif
		rtcReleaseDevice(ebr_device);
	}

	raycast_singleton = nullptr;
}
