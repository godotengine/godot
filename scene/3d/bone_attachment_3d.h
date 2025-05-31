/**************************************************************************/
/*  bone_attachment_3d.h                                                  */
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

#include "scene/3d/skeleton_3d.h"

class BoneAttachment3D : public Node3D {
	GDCLASS(BoneAttachment3D, Node3D);

public:
	enum ExtraUpdate {
		EXTRA_UPDATE_NONE,
		EXTRA_UPDATE_BEFORE_SKELETON_UPDATE,
		EXTRA_UPDATE_SUBSCRIBE_MODIFIER,
	};

private:
	bool bound_skeleton = false;
	bool bound_subscribe = false;

	String bone_name;
	int bone_idx = -1;

	bool override_pose = false;
	bool _override_dirty = false;
	bool overriding = false;

	bool use_external_skeleton = false;
	NodePath external_skeleton_node;
	ObjectID external_skeleton_node_cache;

	ExtraUpdate extra_update = EXTRA_UPDATE_NONE;
	NodePath subscribe_modifier;
	ObjectID subscribe_node_cache;

	void _validate_bind_states();

	void _check_bind();
	void _check_bind_skeleton();
	void _check_bind_subscribe();

	void _check_unbind();
	void _check_unbind_skeleton();
	void _check_unbind_subscribe();

	bool updating = false;
	void _transform_changed();

	void _update_node_cache();
	void _update_external_skeleton_cache();
	void _update_subscribe_node_cache();

protected:
	void _validate_property(PropertyInfo &p_property) const;
	void _notification(int p_what);

	static void _bind_methods();
#ifndef DISABLE_DEPRECATED
	virtual void _on_bone_pose_update_bind_compat_90575(int p_bone_index);
	static void _bind_compatibility_methods();
#endif

public:
#ifdef TOOLS_ENABLED
	virtual void notify_skeleton_bones_renamed(Node *p_base_scene, Skeleton3D *p_skeleton, Dictionary p_rename_map);
#endif // TOOLS_ENABLED

	virtual PackedStringArray get_configuration_warnings() const override;

	Skeleton3D *get_skeleton();

	void set_bone_name(const String &p_name);
	String get_bone_name() const;

	void set_bone_idx(const int &p_idx);
	int get_bone_idx() const;

	void set_override_pose(bool p_override);
	bool get_override_pose() const;

	void set_use_external_skeleton(bool p_use_external_skeleton);
	bool get_use_external_skeleton() const;
	void set_external_skeleton(NodePath p_external_skeleton);
	NodePath get_external_skeleton() const;

	void set_extra_update(ExtraUpdate p_extra_update);
	ExtraUpdate get_extra_update() const;
	void set_subscribe_modifier(NodePath p_subscribe_modifier);
	NodePath get_subscribe_modifier() const;

	virtual void on_skeleton_update();

#ifdef TOOLS_ENABLED
	virtual void notify_rebind_required();
#endif

	BoneAttachment3D();
};

VARIANT_ENUM_CAST(BoneAttachment3D::ExtraUpdate);
