/*************************************************************************/
/*  skeleton_modification_2d.h                                           */
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

#ifndef SKELETON_MODIFICATION_2D_H
#define SKELETON_MODIFICATION_2D_H

#include "scene/2d/skeleton_2d.h"

///////////////////////////////////////
// SkeletonModification2D
///////////////////////////////////////

class Bone2D;

class SkeletonModification2D : public Node {
	GDCLASS(SkeletonModification2D, Node);

private:
	static void _bind_methods();

	bool editor_gizmo_dirty = false;
	bool enabled = true;
	bool run_in_editor = true;
	bool skeleton_change_queued = true;
	mutable Variant cached_skeleton;
	NodePath skeleton_path = NodePath("..");

	void _do_gizmo_draw();

protected:
	bool _cache_node(Variant &cache, const NodePath &target_node_path) const;
	Bone2D *_cache_bone(Variant &cache, const NodePath &target_node_path) const;
	PackedStringArray get_configuration_warnings() const override;

public:
	enum { UNCACHED_BONE_IDX = -2 };

	void set_enabled(bool p_enabled);
	bool get_enabled() const;
	void set_run_in_editor(bool p_enabled_in_editor);
	bool get_run_in_editor() const;

	NodePath get_skeleton_path() const;
	void set_skeleton_path(NodePath p_path);
	Skeleton2D *get_skeleton() const;

	void _validate_property(PropertyInfo &p_property) const;
	void _notification(int32_t p_what);

	virtual void execute(real_t delta);
	GDVIRTUAL1(_execute, real_t);
	virtual void draw_editor_gizmo();
	GDVIRTUAL0(_draw_editor_gizmo);
	virtual bool is_property_hidden(String property_name) const;
	GDVIRTUAL1R(bool, _is_property_hidden, String);
	void set_editor_gizmo_dirty(bool p_dirty);

	Variant resolve_node(const NodePath &target_node_path) const;
	Variant resolve_bone(const NodePath &target_node_path) const;
	Transform2D get_target_transform(Variant resolved_target) const;
	real_t get_target_rotation(Variant resolved_target) const;
	Vector2 get_target_position(Variant resolved_target) const;

	float clamp_angle(float p_angle, float p_min_bound, float p_max_bound, bool p_invert);
	void editor_draw_angle_constraints(Bone2D *p_operation_bone, float p_min_bound, float p_max_bound, bool p_constraint_enabled, bool p_constraint_in_localspace, bool p_constraint_inverted);

	SkeletonModification2D() {}
};

#endif // SKELETON_MODIFICATION_2D_H
