/**************************************************************************/
/*  hddagi_rigid_cache.h                                                  */
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

#pragma once

#include "core/templates/hash_map.h"
#include "core/templates/paged_array.h"
#include "servers/rendering/renderer_geometry_instance.h"
#include "servers/rendering/renderer_rd/shaders/environment/hddagi_rigid_cache.glsl.gen.h"
#include "servers/rendering/rendering_device.h"

class RenderSceneBuffersRD;

namespace RendererRD {

class HDDAGIRigidCache {
public:
	struct Textures {
		RID albedo;
		RID emission;
		RID emission_aniso;
		RID normals;

		void create(const Vector3i &p_size);
		void free() const;
		void clear() const;
	};

	struct StaticStamp {
		Transform3D transform;
		RID mesh;
		uint64_t revision;
		uint64_t material_revision;
	};
	static constexpr int PATCH_COUNT = 4;
	struct StaticPatch {
		Textures textures;
		Vector3i from;
		Vector3i size;
		HashMap<RenderGeometryInstance *, StaticStamp> instances;
		bool valid = false;
		uint64_t last_use = 0;
		uint64_t material_epoch = 0;
	};

private:
	struct Surface {
		Transform3D grid_to_object;
		Transform3D transform;
		bool moving = false;
		uint64_t revision = 0;
		uint64_t material_revision = 0;
		uint64_t material_epoch = 0;
		bool dirty = true;
		RID samples;
		RID dispatch;

		void free();
	};

	int32_t moving_count = 0;
	uint64_t patch_serial = 0;
	HashMap<RenderGeometryInstance *, Surface> surfaces;
	uint64_t get_material_revision(const RenderGeometryInstanceBase *p_instance) const;
	PagedArrayPool<RenderGeometryInstance *> instance_pool{ 64 };
	enum Mode {
		CAPTURE,
		RESTORE,
		CLEAR,
		PROJECT,
		MODE_MAX
	};
	HddagiRigidCacheShaderRD shader;
	RID version;
	RID shaders[MODE_MAX];
	RID pipelines[MODE_MAX];

	void ensure_shader();
	void restore(const Textures *p_static, const Vector3i &p_offset, const Vector3i &p_from, const Vector3i &p_size, const Textures &p_output, float p_exposure);
	void capture(Surface &r_surface, RenderGeometryInstanceBase *p_instance, const Ref<RenderSceneBuffersRD> &p_buffers, float p_cell);
	void project(const Surface &p_surface, const Transform3D &p_transform, const AABB &p_world_bounds, const Vector3i &p_from, const Vector3i &p_size, const Textures &p_output, float p_exposure);

	void render_object(RenderGeometryInstanceBase *p_instance, const Ref<RenderSceneBuffersRD> &p_buffers, const Vector3i &p_from, const Vector3i &p_size, const AABB &p_bounds, float p_cell, const Textures &p_output, float p_exposure);

public:
	static void get_patch_extent(const Vector3i &p_from, const Vector3i &p_size, const Vector3i &p_grid, Vector3i &r_from, Vector3i &r_end);
	bool is_moving() const { return moving_count > 0; }
	bool track(RenderGeometryInstance *p_instance);
	void remove(RenderGeometryInstance *p_instance);
	void render(const Ref<RenderSceneBuffersRD> &p_buffers, StaticPatch (&p_patches)[PATCH_COUNT], const Vector3i &p_from, const Vector3i &p_size, const AABB &p_bounds, const Vector3i &p_grid, float p_cell, const PagedArray<RenderGeometryInstance *> &p_instances, const Textures &p_output, float p_exposure);
	void render_cascade(const Ref<RenderSceneBuffersRD> &p_buffers, const Vector3i &p_from, const Vector3i &p_size, const AABB &p_bounds, float p_cell, const PagedArray<RenderGeometryInstance *> &p_instances, const Textures &p_output, float p_exposure);
	~HDDAGIRigidCache();
};

} // namespace RendererRD
