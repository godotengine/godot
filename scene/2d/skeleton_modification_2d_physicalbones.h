/*************************************************************************/
/*  skeleton_modification_2d_physicalbones.h                             */
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

#ifndef SKELETON_MODIFICATION_2D_PHYSICALBONES_H
#define SKELETON_MODIFICATION_2D_PHYSICALBONES_H

#include "scene/2d/physical_bone_2d.h"
#include "scene/2d/skeleton_2d.h"
#include "scene/2d/skeleton_modification_2d.h"

///////////////////////////////////////
// SkeletonModification2DJIGGLE
///////////////////////////////////////

class SkeletonModification2DPhysicalBones : public SkeletonModification2D {
	GDCLASS(SkeletonModification2DPhysicalBones, SkeletonModification2D);

private:
	struct PhysicalBone_Data2D {
		NodePath physical_bone_node;
		ObjectID physical_bone_node_cache;
	};
	Vector<PhysicalBone_Data2D> physical_bone_chain;

	void _physical_bone_update_cache(int p_joint_idx);

	bool _simulation_state_dirty = false;
	TypedArray<StringName> _simulation_state_dirty_names;
	bool _simulation_state_dirty_process = false;
	void _update_simulation_state();

protected:
	static void _bind_methods();
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void _notification(int p_what) {
		switch (p_what) {
			case NOTIFICATION_ENTER_TREE: {
				set_process_internal(false);
				set_physics_process_internal(false);
				if (get_execution_mode() == 0) {
					set_process_internal(true);
				} else if (get_execution_mode() == 1) {
					set_physics_process_internal(true);
				}
				is_setup = true;
				skeleton = cast_to<Skeleton2D>(get_node_or_null(get_skeleton_path()));
				if (skeleton) {
					for (int i = 0; i < physical_bone_chain.size(); i++) {
						_physical_bone_update_cache(i);
					}
				}
			} break;
			case NOTIFICATION_INTERNAL_PHYSICS_PROCESS:
				[[fallthrough]];
			case NOTIFICATION_INTERNAL_PROCESS: {
				ERR_FAIL_COND_MSG(!is_setup || skeleton == nullptr,
						"Modification is not setup and therefore cannot execute!");
				if (!enabled) {
					return;
				}

				if (_simulation_state_dirty) {
					_update_simulation_state();
				}

				for (int i = 0; i < physical_bone_chain.size(); i++) {
					PhysicalBone_Data2D bone_data = physical_bone_chain[i];
					if (bone_data.physical_bone_node_cache.is_null()) {
						WARN_PRINT_ONCE("PhysicalBone2D cache " + itos(i) + " is out of date. Attempting to update...");
						_physical_bone_update_cache(i);
						continue;
					}

					PhysicalBone2D *physical_bone = Object::cast_to<PhysicalBone2D>(ObjectDB::get_instance(bone_data.physical_bone_node_cache));
					if (!physical_bone) {
						ERR_PRINT_ONCE("PhysicalBone2D not found at index " + itos(i) + "!");
						return;
					}
					if (physical_bone->get_bone2d_index() < 0 || physical_bone->get_bone2d_index() > skeleton->get_bone_count()) {
						ERR_PRINT_ONCE("PhysicalBone2D at index " + itos(i) + " has invalid Bone2D!");
						return;
					}
					Bone2D *bone_2d = skeleton->get_bone(physical_bone->get_bone2d_index());

					if (physical_bone->get_simulate_physics() && !physical_bone->get_follow_bone_when_simulating()) {
						bone_2d->set_global_transform(physical_bone->get_global_transform());
						skeleton->set_bone_local_pose_override(physical_bone->get_bone2d_index(), bone_2d->get_transform(), 1.0, true);
					}
				}
			} break;
		}
	}

	int get_physical_bone_chain_length();
	void set_physical_bone_chain_length(int p_new_length);

	void set_physical_bone_node(int p_joint_idx, const NodePath &p_path);
	NodePath get_physical_bone_node(int p_joint_idx) const;

	void fetch_physical_bones();
	void start_simulation(const TypedArray<StringName> &p_bones);
	void stop_simulation(const TypedArray<StringName> &p_bones);

	SkeletonModification2DPhysicalBones();
	~SkeletonModification2DPhysicalBones();
};

#endif // SKELETON_MODIFICATION_2D_PHYSICALBONES_H
