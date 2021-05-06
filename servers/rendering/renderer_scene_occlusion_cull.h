/*************************************************************************/
/*  renderer_scene_occlusion_cull.h                                      */
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

#ifndef RENDERER_SCENE_OCCLUSION_CULL_H
#define RENDERER_SCENE_OCCLUSION_CULL_H

#include "core/math/camera_matrix.h"
#include "core/templates/local_vector.h"
#include "servers/rendering_server.h"

class RendererSceneOcclusionCull {
protected:
	static RendererSceneOcclusionCull *singleton;

public:
	class HZBuffer {
	protected:
		static const Vector3 corners[8];

		LocalVector<float> data;
		LocalVector<Size2i> sizes;
		LocalVector<float *> mips;

		RID debug_texture;
		Ref<Image> debug_image;
		PackedByteArray debug_data;
		float debug_tex_range = 0.0f;

	public:
		bool is_empty() const;
		virtual void clear();
		virtual void resize(const Size2i &p_size);

		void update_mips();

		_FORCE_INLINE_ bool is_occluded(const float p_bounds[6], const Vector3 &p_cam_position, const Transform &p_cam_inv_transform, const CameraMatrix &p_cam_projection, float p_near) const {
			if (is_empty()) {
				return false;
			}

			Vector3 closest_point = Vector3(CLAMP(p_cam_position.x, p_bounds[0], p_bounds[3]), CLAMP(p_cam_position.y, p_bounds[1], p_bounds[4]), CLAMP(p_cam_position.z, p_bounds[2], p_bounds[5]));

			if (closest_point == p_cam_position) {
				return false;
			}

			Vector3 closest_point_view = p_cam_inv_transform.xform(closest_point);
			if (closest_point_view.z > -p_near) {
				return false;
			}

			float min_depth;
			if (p_cam_projection.is_orthogonal()) {
				min_depth = (-closest_point_view.z) - p_near;
			} else {
				float r = -p_near / closest_point_view.z;
				Vector3 closest_point_proj = Vector3(closest_point_view.x * r, closest_point_view.y * r, -p_near);
				min_depth = closest_point_proj.distance_to(closest_point_view);
			}

			Vector2 rect_min = Vector2(FLT_MAX, FLT_MAX);
			Vector2 rect_max = Vector2(FLT_MIN, FLT_MIN);

			for (int j = 0; j < 8; j++) {
				Vector3 c = RendererSceneOcclusionCull::HZBuffer::corners[j];
				Vector3 nc = Vector3(1, 1, 1) - c;
				Vector3 corner = Vector3(p_bounds[0] * c.x + p_bounds[3] * nc.x, p_bounds[1] * c.y + p_bounds[4] * nc.y, p_bounds[2] * c.z + p_bounds[5] * nc.z);
				Vector3 view = p_cam_inv_transform.xform(corner);

				Vector3 projected = p_cam_projection.xform(view);
				Vector2 normalized = Vector2(projected.x * 0.5f + 0.5f, projected.y * 0.5f + 0.5f);
				rect_min = rect_min.min(normalized);
				rect_max = rect_max.max(normalized);
			}

			rect_max = rect_max.min(Vector2(1, 1));
			rect_min = rect_min.max(Vector2(0, 0));

			int mip_count = mips.size();

			Vector2 screen_diagonal = (rect_max - rect_min) * sizes[0];
			float size = MAX(screen_diagonal.x, screen_diagonal.y);
			float l = Math::ceil(Math::log2(size));
			int lod = CLAMP(l, 0, mip_count - 1);

			const int max_samples = 512;
			int sample_count = 0;
			bool visible = true;

			for (; lod >= 0; lod--) {
				int w = sizes[lod].x;
				int h = sizes[lod].y;

				int minx = CLAMP(rect_min.x * w - 1, 0, w - 1);
				int maxx = CLAMP(rect_max.x * w + 1, 0, w - 1);

				int miny = CLAMP(rect_min.y * h - 1, 0, h - 1);
				int maxy = CLAMP(rect_max.y * h + 1, 0, h - 1);

				sample_count += (maxx - minx + 1) * (maxy - miny + 1);

				if (sample_count > max_samples) {
					return false;
				}

				visible = false;
				for (int y = miny; y <= maxy; y++) {
					for (int x = minx; x <= maxx; x++) {
						float depth = mips[lod][y * w + x];
						if (depth > min_depth) {
							visible = true;
							break;
						}
					}
					if (visible) {
						break;
					}
				}

				if (!visible) {
					return true;
				}
			}

			return !visible;
		}

		RID get_debug_texture();

		virtual ~HZBuffer(){};
	};

	static RendererSceneOcclusionCull *get_singleton() { return singleton; }

	void _print_warining() {
		WARN_PRINT_ONCE("Occlusion culling is disabled at build time.");
	}

	virtual bool is_occluder(RID p_rid) { return false; }
	virtual RID occluder_allocate() { return RID(); }
	virtual void occluder_initialize(RID p_occluder) {}
	virtual void free_occluder(RID p_occluder) { _print_warining(); }
	virtual void occluder_set_mesh(RID p_occluder, const PackedVector3Array &p_vertices, const PackedInt32Array &p_indices) { _print_warining(); }

	virtual void add_scenario(RID p_scenario) {}
	virtual void remove_scenario(RID p_scenario) {}
	virtual void scenario_set_instance(RID p_scenario, RID p_instance, RID p_occluder, const Transform &p_xform, bool p_enabled) { _print_warining(); }
	virtual void scenario_remove_instance(RID p_scenario, RID p_instance) { _print_warining(); }

	virtual void add_buffer(RID p_buffer) { _print_warining(); }
	virtual void remove_buffer(RID p_buffer) { _print_warining(); }
	virtual HZBuffer *buffer_get_ptr(RID p_buffer) {
		return nullptr;
	}
	virtual void buffer_set_scenario(RID p_buffer, RID p_scenario) { _print_warining(); }
	virtual void buffer_set_size(RID p_buffer, const Vector2i &p_size) { _print_warining(); }
	virtual void buffer_update(RID p_buffer, const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_orthogonal, ThreadWorkPool &p_thread_pool) {}
	virtual RID buffer_get_debug_texture(RID p_buffer) {
		_print_warining();
		return RID();
	}

	virtual void set_build_quality(RS::ViewportOcclusionCullingBuildQuality p_quality) {}

	RendererSceneOcclusionCull() {
		singleton = this;
	};

	virtual ~RendererSceneOcclusionCull() {
		singleton = nullptr;
	};
};

#endif //RENDERER_SCENE_OCCLUSION_CULL_H
