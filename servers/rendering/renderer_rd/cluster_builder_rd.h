/**************************************************************************/
/*  cluster_builder_rd.h                                                  */
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

#include "core/templates/local_vector.h"
#include "servers/rendering/renderer_rd/shaders/cluster_debug.glsl.gen.h"
#include "servers/rendering/renderer_rd/shaders/cluster_render.glsl.gen.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"

class ClusterBuilderSharedDataRD {
	friend class ClusterBuilderRD;

	RID sphere_vertex_buffer;
	RID sphere_vertex_array;
	RID sphere_index_buffer;
	RID sphere_index_array;
	float sphere_overfit = 0.0; // Because an icosphere is not a perfect sphere, we need to enlarge it to cover the sphere area.

	RID cone_vertex_buffer;
	RID cone_vertex_array;
	RID cone_index_buffer;
	RID cone_index_array;
	float cone_overfit = 0.0; // Because an cone mesh is not a perfect cone, we need to enlarge it to cover the actual cone area.

	RID box_vertex_buffer;
	RID box_vertex_array;
	RID box_index_buffer;
	RID box_index_array;

	enum Divisor {
		DIVISOR_1,
		DIVISOR_2,
		DIVISOR_4,
	};

	struct ClusterRender {
		struct PushConstant {
			uint32_t base_index;
			uint32_t pad0;
			uint32_t pad1;
			uint32_t pad2;
		};

		ClusterRenderShaderRD cluster_render_shader;
		RID shader_version;
		RID shader;

		enum ShaderVariant {
			SHADER_NORMAL,
			SHADER_USE_ATTACHMENT,
		};

		enum PipelineVersion {
			PIPELINE_NORMAL,
			PIPELINE_MSAA,
			PIPELINE_MAX
		};

		RID shader_pipelines[PIPELINE_MAX];
	} cluster_render;

	struct ClusterDebug {
		struct PushConstant {
			uint32_t screen_size[2];
			uint32_t screen_size_in_tiles[2];

			uint32_t tile_shift;
			uint32_t cluster_type;
			float z_near;
			float z_far;

			uint32_t orthogonal;
			uint32_t bitmasks_per_tile_count;
			uint32_t z_bin_offset;
			uint32_t z_bin_count;

			float z_bin_scale;
			uint32_t pad0;
			uint32_t pad1;
			uint32_t pad2;
		};

		ClusterDebugShaderRD cluster_debug_shader;
		RID shader_version;
		RID shader;
		RID shader_pipeline;
	} cluster_debug;

public:
	ClusterBuilderSharedDataRD();
	~ClusterBuilderSharedDataRD();
};

class ClusterBuilderRD {
public:
	static constexpr float WIDE_SPOT_ANGLE_THRESHOLD_DEG = 60.0f;
	static constexpr float Z_BIN_DEPTH = 0.5f;
	static constexpr uint32_t MAX_Z_BINS = 8192; // Upper limit on number of bins, once we reach it the bins grow deeper instead

	enum LightType {
		LIGHT_TYPE_OMNI,
		LIGHT_TYPE_SPOT,
		LIGHT_TYPE_AREA,
	};

	enum BoxType {
		BOX_TYPE_REFLECTION_PROBE,
		BOX_TYPE_DECAL,
	};

	enum ElementType {
		ELEMENT_TYPE_OMNI_LIGHT,
		ELEMENT_TYPE_SPOT_LIGHT,
		ELEMENT_TYPE_AREA_LIGHT,
		ELEMENT_TYPE_DECAL,
		ELEMENT_TYPE_REFLECTION_PROBE,
		ELEMENT_TYPE_MAX,
	};

private:
	ClusterBuilderSharedDataRD *shared = nullptr;

	struct RenderElementData {
		uint32_t type; // 0-4
		uint32_t original_index;
		uint32_t pad[2];
		float transform_inv[12]; // Transposed transform for less space.
		float scale[3];
		uint32_t has_wide_spot_angle;
	}; // Keep aligned to 32 bytes.

	uint32_t cluster_count_by_type[ELEMENT_TYPE_MAX] = {};
	uint32_t max_elements_by_type = 0;

	RenderElementData *render_elements = nullptr;
	uint32_t render_element_count = 0;
	uint32_t render_element_max = 0;

	Transform3D view_xform;
	Projection adjusted_projection;
	Projection projection;
	float z_far = 0;
	float z_near = 0;
	bool camera_orthogonal = false;

	enum Divisor {
		DIVISOR_1,
		DIVISOR_2,
		DIVISOR_4,
	};

	uint32_t tile_size = 32;
	bool use_msaa = true;
	Divisor divisor = DIVISOR_4;

	Size2i screen_size;
	Size2i screen_size_in_tiles;

	RID framebuffer;
	RID cluster_buffer; // SS tile bitmasks of every element type, followed by the Z bins of every element type.
	RID element_buffer; // Used for rendering the element proxies.
	uint32_t cluster_buffer_size = 0;
	uint32_t bitmasks_per_type_count = 0; // number of bitmasks of all tiles for one element type

	LocalVector<uint32_t> z_bins; // Each bin packs the lowest element index and the highest one plus one (zero is empty)
	uint32_t z_bin_offset = 0;
	uint32_t z_bin_count = 1;
	float z_bin_scale = 0;

	RID cluster_render_uniform_set;

	// Persistent data.

	void _clear();
	void _add_to_z_bins(ElementType p_type, uint32_t p_index, const Vector2 &p_depth_range);

	static _FORCE_INLINE_ Vector3 _get_area_light_half_size(float p_radius, const Vector2 &p_area_size) {
		return Vector3(p_area_size.x / 2.0 + p_radius, p_area_size.y / 2.0 + p_radius, p_radius / 2.0);
	}

	struct StateUniform {
		float projection[16];
		uint32_t screen_to_tiles_shift; // Shift to obtain coordinates in block indices.
		uint32_t tiles_per_row;
		uint32_t bitmasks_per_tile_count; // Number of mask words for a single tile
		uint32_t bitmasks_per_type_count; // Number of mask words all tiles of one element type take
	};

	RID state_uniform;

	RID debug_uniform_set;

public:
	static uint32_t get_z_bin_count(float p_z_near, float p_z_far) {
		return (uint32_t)CLAMP(Math::ceil((p_z_far - p_z_near) / Z_BIN_DEPTH), 1.0f, (float)MAX_Z_BINS);
	}

	static float get_z_bin_scale(float p_z_near, float p_z_far) {
		return MIN(1.0f / Z_BIN_DEPTH, get_z_bin_count(p_z_near, p_z_far) / (p_z_far - p_z_near));
	}

	// in view space, returned as (near, far)
	static Vector2 get_box_depth_range(const Transform3D &p_view_space_transform, const Vector3 &p_half_size) {
		const real_t depth = -p_view_space_transform.origin.z;
		const real_t half_depth = p_view_space_transform.basis.rows[Vector3::AXIS_Z].abs().dot(p_half_size);
		return Vector2(depth - half_depth, depth + half_depth);
	}

	static Vector2 get_light_depth_range(LightType p_type, const Transform3D &p_view_space_transform, float p_radius, float p_spot_aperture, const Vector2 &p_area_size);

	void setup(Size2i p_screen_size, uint32_t p_max_elements, RID p_depth_buffer, RID p_depth_buffer_sampler, RID p_color_buffer);

	void begin(const Transform3D &p_view_transform, const Projection &p_cam_projection, bool p_flip_y);

	_FORCE_INLINE_ void add_light(LightType p_type, const Transform3D &p_transform, float p_radius, float p_spot_aperture, const Vector2 &p_area_size) {
		if (p_type == LIGHT_TYPE_OMNI && cluster_count_by_type[ELEMENT_TYPE_OMNI_LIGHT] == max_elements_by_type) {
			return; // Max number elements reached.
		}
		if (p_type == LIGHT_TYPE_SPOT && cluster_count_by_type[ELEMENT_TYPE_SPOT_LIGHT] == max_elements_by_type) {
			return; // Max number elements reached.
		}
		if (p_type == LIGHT_TYPE_AREA && cluster_count_by_type[ELEMENT_TYPE_AREA_LIGHT] == max_elements_by_type) {
			return; // Max number elements reached.
		}

		RenderElementData &e = render_elements[render_element_count];

		Transform3D xform = view_xform * p_transform;

		float radius = xform.basis.get_uniform_scale();
		if (radius < 0.98 || radius > 1.02) {
			xform.basis.orthonormalize();
		}

		radius *= p_radius;

		const Vector2 depth_range = get_light_depth_range(p_type, xform, radius, p_spot_aperture, p_area_size);

		// Spotlights with wide angle are trated as Omni lights.
		// If the spot angle is above the threshold, we need a sphere instead of a cone for building the clusters
		// since the cone gets too flat/large (spot angle close to 90 degrees) or
		// can't even cover the affected area of the light (spot angle above 90 degrees).
		if (p_type == LIGHT_TYPE_OMNI || (p_type == LIGHT_TYPE_SPOT && p_spot_aperture > WIDE_SPOT_ANGLE_THRESHOLD_DEG)) {
			radius *= shared->sphere_overfit; // Overfit icosphere.

			e.scale[0] = radius;
			e.scale[1] = radius;
			e.scale[2] = radius;
			if (p_type == LIGHT_TYPE_OMNI) {
				e.type = ELEMENT_TYPE_OMNI_LIGHT;
				e.original_index = cluster_count_by_type[ELEMENT_TYPE_OMNI_LIGHT];
				cluster_count_by_type[ELEMENT_TYPE_OMNI_LIGHT]++;
			} else { // LIGHT_TYPE_SPOT with wide angle.
				e.type = ELEMENT_TYPE_SPOT_LIGHT;
				e.has_wide_spot_angle = true;
				e.original_index = cluster_count_by_type[ELEMENT_TYPE_SPOT_LIGHT];
				cluster_count_by_type[ELEMENT_TYPE_SPOT_LIGHT]++;
			}

			RendererRD::MaterialStorage::store_transform_transposed_3x4(xform, e.transform_inv);

		} else if (p_type == LIGHT_TYPE_SPOT) { /*LIGHT_TYPE_SPOT with no wide angle*/
			radius *= shared->cone_overfit; // Overfit cone.

			real_t len = Math::tan(Math::deg_to_rad(p_spot_aperture)) * radius;

			e.scale[0] = len * shared->cone_overfit;
			e.scale[1] = len * shared->cone_overfit;
			e.scale[2] = radius;
			e.has_wide_spot_angle = false;
			e.type = ELEMENT_TYPE_SPOT_LIGHT;
			e.original_index = cluster_count_by_type[ELEMENT_TYPE_SPOT_LIGHT];

			RendererRD::MaterialStorage::store_transform_transposed_3x4(xform, e.transform_inv);

			cluster_count_by_type[ELEMENT_TYPE_SPOT_LIGHT]++;
		} else { /* LIGHT_TYPE_AREA */
			Vector3 scale = _get_area_light_half_size(radius, p_area_size);

			for (uint32_t i = 0; i < 3; i++) {
				float s = xform.basis.rows[i].length();
				//scale[i] *= s; // lights ignore scale
				xform.basis.rows[i] /= s;
			}
			xform.origin -= xform.basis.get_column(Vector3::AXIS_Z) * scale.z; // translate center to center of box

			e.scale[0] = scale.x;
			e.scale[1] = scale.y;
			e.scale[2] = scale.z;

			e.type = ELEMENT_TYPE_AREA_LIGHT;
			e.original_index = cluster_count_by_type[ELEMENT_TYPE_AREA_LIGHT];

			RendererRD::MaterialStorage::store_transform_transposed_3x4(xform, e.transform_inv);

			cluster_count_by_type[ELEMENT_TYPE_AREA_LIGHT]++;
		}

		_add_to_z_bins((ElementType)e.type, e.original_index, depth_range);

		render_element_count++;
	}

	_FORCE_INLINE_ void add_box(BoxType p_box_type, const Transform3D &p_transform, const Vector3 &p_half_size) {
		if (p_box_type == BOX_TYPE_DECAL && cluster_count_by_type[ELEMENT_TYPE_DECAL] == max_elements_by_type) {
			return; // Max number elements reached.
		}
		if (p_box_type == BOX_TYPE_REFLECTION_PROBE && cluster_count_by_type[ELEMENT_TYPE_REFLECTION_PROBE] == max_elements_by_type) {
			return; // Max number elements reached.
		}

		RenderElementData &e = render_elements[render_element_count];
		Transform3D xform = view_xform * p_transform;

		const Vector2 depth_range = get_box_depth_range(xform, p_half_size);

		// Extract scale and scale the matrix by it, makes things simpler.
		Vector3 scale = p_half_size;
		for (uint32_t i = 0; i < 3; i++) {
			float s = xform.basis.rows[i].length();
			scale[i] *= s;
			xform.basis.rows[i] /= s;
		};

		e.scale[0] = scale.x;
		e.scale[1] = scale.y;
		e.scale[2] = scale.z;

		e.type = (p_box_type == BOX_TYPE_DECAL) ? ELEMENT_TYPE_DECAL : ELEMENT_TYPE_REFLECTION_PROBE;
		e.original_index = cluster_count_by_type[e.type];

		RendererRD::MaterialStorage::store_transform_transposed_3x4(xform, e.transform_inv);

		_add_to_z_bins((ElementType)e.type, e.original_index, depth_range);

		cluster_count_by_type[e.type]++;
		render_element_count++;
	}

	_FORCE_INLINE_ uint32_t get_cluster_count_by_type(ElementType p_element_type) const {
		DEV_ASSERT(p_element_type < ELEMENT_TYPE_MAX);
		return cluster_count_by_type[p_element_type];
	}

	void bake_cluster();
	void debug(ElementType p_element);

	RID get_cluster_buffer() const;
	uint32_t get_tile_size() const;
	uint32_t get_max_cluster_elements() const;

	void set_shared(ClusterBuilderSharedDataRD *p_shared);

	ClusterBuilderRD();
	~ClusterBuilderRD();
};
