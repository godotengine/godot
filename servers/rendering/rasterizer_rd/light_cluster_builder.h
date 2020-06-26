/*************************************************************************/
/*  light_cluster_builder.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef LIGHT_CLUSTER_BUILDER_H
#define LIGHT_CLUSTER_BUILDER_H

#include "servers/rendering/rasterizer_rd/rasterizer_storage_rd.h"

class LightClusterBuilder {
public:
	enum LightType {
		LIGHT_TYPE_OMNI,
		LIGHT_TYPE_SPOT
	};

	enum ItemType {
		ITEM_TYPE_OMNI_LIGHT,
		ITEM_TYPE_SPOT_LIGHT,
		ITEM_TYPE_REFLECTION_PROBE,
		ITEM_TYPE_DECAL,
		ITEM_TYPE_MAX //should always be 4
	};

	enum {
		COUNTER_SHIFT = 20, //one million total ids
		POINTER_MASK = (1 << COUNTER_SHIFT) - 1,
		COUNTER_MASK = 0xfff // 4096 items per cell
	};

private:
	struct LightData {
		float position[3];
		uint32_t type;
		float radius;
		float spot_aperture;
		uint32_t pad[2];
	};

	uint32_t light_count = 0;
	uint32_t light_max = 0;
	LightData *lights = nullptr;

	struct OrientedBoxData {
		float position[3];
		uint32_t pad;
		float x_axis[3];
		uint32_t pad2;
		float y_axis[3];
		uint32_t pad3;
		float z_axis[3];
		uint32_t pad4;
	};

	uint32_t refprobe_count = 0;
	uint32_t refprobe_max = 0;
	OrientedBoxData *refprobes = nullptr;

	uint32_t decal_count = 0;
	uint32_t decal_max = 0;
	OrientedBoxData *decals = nullptr;

	struct Item {
		AABB aabb;
		ItemType type;
		uint32_t index;
	};

	Item *items = nullptr;
	uint32_t item_count = 0;
	uint32_t item_max = 0;

	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t depth = 0;

	struct Cell {
		uint32_t item_pointers[ITEM_TYPE_MAX];
	};

	Vector<uint8_t> cluster_data;
	RID cluster_texture;

	struct SortID {
		uint32_t cell_index;
		uint32_t item_index;
		ItemType item_type;
	};

	SortID *sort_ids = nullptr;
	Vector<uint32_t> ids;
	uint32_t sort_id_count = 0;
	uint32_t sort_id_max = 0;
	RID items_buffer;

	Transform view_xform;
	CameraMatrix projection;
	float z_far = 0;
	float z_near = 0;

	_FORCE_INLINE_ void _add_item(const AABB &p_aabb, ItemType p_type, uint32_t p_index) {
		if (unlikely(item_count == item_max)) {
			item_max = nearest_power_of_2_templated(item_max + 1);
			items = (Item *)memrealloc(items, sizeof(Item) * item_max);
		}

		Item &item = items[item_count];
		item.aabb = p_aabb;
		item.index = p_index;
		item.type = p_type;
		item_count++;
	}

public:
	void begin(const Transform &p_view_transform, const CameraMatrix &p_cam_projection);

	_FORCE_INLINE_ void add_light(LightType p_type, const Transform &p_transform, float p_radius, float p_spot_aperture) {
		if (unlikely(light_count == light_max)) {
			light_max = nearest_power_of_2_templated(light_max + 1);
			lights = (LightData *)memrealloc(lights, sizeof(LightData) * light_max);
		}

		LightData &ld = lights[light_count];
		ld.type = p_type;
		ld.position[0] = p_transform.origin.x;
		ld.position[1] = p_transform.origin.y;
		ld.position[2] = p_transform.origin.z;
		ld.radius = p_radius;
		ld.spot_aperture = p_spot_aperture;

		Transform xform = view_xform * p_transform;

		ld.radius *= xform.basis.get_uniform_scale();

		AABB aabb;

		switch (p_type) {
			case LIGHT_TYPE_OMNI: {
				aabb.position = xform.origin;
				aabb.size = Vector3(ld.radius, ld.radius, ld.radius);
				aabb.position -= aabb.size;
				aabb.size *= 2.0;

				_add_item(aabb, ITEM_TYPE_OMNI_LIGHT, light_count);
			} break;
			case LIGHT_TYPE_SPOT: {
				float r = ld.radius;
				real_t len = Math::tan(Math::deg2rad(ld.spot_aperture)) * r;

				aabb.position = xform.origin;
				aabb.expand_to(xform.xform(Vector3(len, len, -r)));
				aabb.expand_to(xform.xform(Vector3(-len, len, -r)));
				aabb.expand_to(xform.xform(Vector3(-len, -len, -r)));
				aabb.expand_to(xform.xform(Vector3(len, -len, -r)));
				_add_item(aabb, ITEM_TYPE_SPOT_LIGHT, light_count);
			} break;
		}

		light_count++;
	}

	_FORCE_INLINE_ void add_reflection_probe(const Transform &p_transform, const Vector3 &p_half_extents) {
		if (unlikely(refprobe_count == refprobe_max)) {
			refprobe_max = nearest_power_of_2_templated(refprobe_max + 1);
			refprobes = (OrientedBoxData *)memrealloc(refprobes, sizeof(OrientedBoxData) * refprobe_max);
		}

		Transform xform = view_xform * p_transform;

		OrientedBoxData &rp = refprobes[refprobe_count];
		Vector3 origin = xform.origin;
		rp.position[0] = origin.x;
		rp.position[1] = origin.y;
		rp.position[2] = origin.z;

		Vector3 x_axis = xform.basis.get_axis(0) * p_half_extents.x;
		rp.x_axis[0] = x_axis.x;
		rp.x_axis[1] = x_axis.y;
		rp.x_axis[2] = x_axis.z;

		Vector3 y_axis = xform.basis.get_axis(1) * p_half_extents.y;
		rp.y_axis[0] = y_axis.x;
		rp.y_axis[1] = y_axis.y;
		rp.y_axis[2] = y_axis.z;

		Vector3 z_axis = xform.basis.get_axis(2) * p_half_extents.z;
		rp.z_axis[0] = z_axis.x;
		rp.z_axis[1] = z_axis.y;
		rp.z_axis[2] = z_axis.z;

		AABB aabb;

		aabb.position = origin + x_axis + y_axis + z_axis;
		aabb.expand_to(origin + x_axis + y_axis - z_axis);
		aabb.expand_to(origin + x_axis - y_axis + z_axis);
		aabb.expand_to(origin + x_axis - y_axis - z_axis);
		aabb.expand_to(origin - x_axis + y_axis + z_axis);
		aabb.expand_to(origin - x_axis + y_axis - z_axis);
		aabb.expand_to(origin - x_axis - y_axis + z_axis);
		aabb.expand_to(origin - x_axis - y_axis - z_axis);

		_add_item(aabb, ITEM_TYPE_REFLECTION_PROBE, refprobe_count);

		refprobe_count++;
	}

	_FORCE_INLINE_ void add_decal(const Transform &p_transform, const Vector3 &p_half_extents) {
		if (unlikely(decal_count == decal_max)) {
			decal_max = nearest_power_of_2_templated(decal_max + 1);
			decals = (OrientedBoxData *)memrealloc(decals, sizeof(OrientedBoxData) * decal_max);
		}

		Transform xform = view_xform * p_transform;

		OrientedBoxData &dc = decals[decal_count];

		Vector3 origin = xform.origin;
		dc.position[0] = origin.x;
		dc.position[1] = origin.y;
		dc.position[2] = origin.z;

		Vector3 x_axis = xform.basis.get_axis(0) * p_half_extents.x;
		dc.x_axis[0] = x_axis.x;
		dc.x_axis[1] = x_axis.y;
		dc.x_axis[2] = x_axis.z;

		Vector3 y_axis = xform.basis.get_axis(1) * p_half_extents.y;
		dc.y_axis[0] = y_axis.x;
		dc.y_axis[1] = y_axis.y;
		dc.y_axis[2] = y_axis.z;

		Vector3 z_axis = xform.basis.get_axis(2) * p_half_extents.z;
		dc.z_axis[0] = z_axis.x;
		dc.z_axis[1] = z_axis.y;
		dc.z_axis[2] = z_axis.z;

		AABB aabb;

		aabb.position = origin + x_axis + y_axis + z_axis;
		aabb.expand_to(origin + x_axis + y_axis - z_axis);
		aabb.expand_to(origin + x_axis - y_axis + z_axis);
		aabb.expand_to(origin + x_axis - y_axis - z_axis);
		aabb.expand_to(origin - x_axis + y_axis + z_axis);
		aabb.expand_to(origin - x_axis + y_axis - z_axis);
		aabb.expand_to(origin - x_axis - y_axis + z_axis);
		aabb.expand_to(origin - x_axis - y_axis - z_axis);

		_add_item(aabb, ITEM_TYPE_DECAL, decal_count);

		decal_count++;
	}

	void bake_cluster();

	void setup(uint32_t p_width, uint32_t p_height, uint32_t p_depth);

	RID get_cluster_texture() const;
	RID get_cluster_indices_buffer() const;

	LightClusterBuilder();
	~LightClusterBuilder();
};

#endif // LIGHT_CLUSTER_BUILDER_H
