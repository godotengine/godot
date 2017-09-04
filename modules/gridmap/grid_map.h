/*************************************************************************/
/*  grid_map.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef GRID_MAP_H
#define GRID_MAP_H

#include "scene/3d/navigation.h"
#include "scene/3d/spatial.h"
#include "scene/resources/mesh_library.h"
#include "scene/resources/multimesh.h"

//heh heh, godotsphir!! this shares no code and the design is completely different with previous projects i've done..
//should scale better with hardware that supports instancing

class GridMap : public Spatial {

	GDCLASS(GridMap, Spatial);

	enum {
		MAP_DIRTY_TRANSFORMS = 1,
		MAP_DIRTY_INSTANCES = 2,
	};

	union IndexKey {

		struct {
			int16_t x;
			int16_t y;
			int16_t z;
		};
		uint64_t key;

		_FORCE_INLINE_ bool operator<(const IndexKey &p_key) const {

			return key < p_key.key;
		}

		IndexKey() { key = 0; }
	};

	/**
	 * @brief A Cell is a single cell in the cube map space; it is defined by its coordinates and the populating Item, identified by int id.
	 */
	union Cell {

		struct {
			unsigned int item : 16;
			unsigned int rot : 5;
			unsigned int layer : 8;
		};
		uint32_t cell;

		Cell() {
			item = 0;
			rot = 0;
			layer = 0;
		}
	};

	/**
	 * @brief An Octant is a prism containing Cells, and possibly belonging to an Area.
	 * A GridMap can have multiple Octants.
	 */
	struct Octant {

		struct NavMesh {
			int id;
			Transform xform;
		};

		struct MultimeshInstance {
			RID instance;
			RID multimesh;
			struct Item {
				int index;
				Transform transform;
				IndexKey key;
			};

			Vector<Item> items; //tools only, for changing visibility
		};

		Vector<MultimeshInstance> multimesh_instances;
		Set<IndexKey> cells;
		RID collision_debug;
		RID collision_debug_instance;

		bool dirty;
		RID static_body;
		Map<IndexKey, NavMesh> navmesh_ids;
	};

	union OctantKey {

		struct {
			int16_t x;
			int16_t y;
			int16_t z;
			int16_t empty;
		};

		uint64_t key;

		_FORCE_INLINE_ bool operator<(const OctantKey &p_key) const {

			return key < p_key.key;
		}

		//OctantKey(const IndexKey& p_k, int p_item) { indexkey=p_k.key; item=p_item; }
		OctantKey() { key = 0; }
	};

	Transform last_transform;

	bool _in_tree;
	Vector3 cell_size;
	int octant_size;
	bool center_x, center_y, center_z;
	float cell_scale;
	Navigation *navigation;

	bool clip;
	bool clip_above;
	int clip_floor;
	Vector3::Axis clip_axis;

	Ref<MeshLibrary> theme;

	Map<OctantKey, Octant *> octant_map;
	Map<IndexKey, Cell> cell_map;

	void _recreate_octant_data();

	struct BakeLight {

		VS::LightType type;
		Vector3 pos;
		Vector3 dir;
		float param[VS::LIGHT_PARAM_MAX];
	};

	_FORCE_INLINE_ Vector3 _octant_get_offset(const OctantKey &p_key) const {

		return Vector3(p_key.x, p_key.y, p_key.z) * cell_size * octant_size;
	}

	void _octant_enter_world(const OctantKey &p_key);
	void _octant_exit_world(const OctantKey &p_key);
	bool _octant_update(const OctantKey &p_key);
	void _octant_clean_up(const OctantKey &p_key);
	void _octant_transform(const OctantKey &p_key);
	bool awaiting_update;

	void _queue_octants_dirty();
	void _update_octants_callback();

	void resource_changed(const RES &p_res);

	void _clear_internal();

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	void _notification(int p_what);
	void _update_visibility();
	static void _bind_methods();

public:
	enum {
		INVALID_CELL_ITEM = -1
	};

	void set_theme(const Ref<MeshLibrary> &p_theme);
	Ref<MeshLibrary> get_theme() const;

	void set_cell_size(const Vector3 &p_size);
	Vector3 get_cell_size() const;

	void set_octant_size(int p_size);
	int get_octant_size() const;

	void set_center_x(bool p_enable);
	bool get_center_x() const;
	void set_center_y(bool p_enable);
	bool get_center_y() const;
	void set_center_z(bool p_enable);
	bool get_center_z() const;

	void set_cell_item(int p_x, int p_y, int p_z, int p_item, int p_rot = 0);
	int get_cell_item(int p_x, int p_y, int p_z) const;
	int get_cell_item_orientation(int p_x, int p_y, int p_z) const;

	void set_clip(bool p_enabled, bool p_clip_above = true, int p_floor = 0, Vector3::Axis p_axis = Vector3::AXIS_X);

	void set_cell_scale(float p_scale);
	float get_cell_scale() const;

	Array get_meshes();

	void clear();

	GridMap();
	~GridMap();
};

#endif // GRID_MAP_H
