/*************************************************************************/
/*  skeleton_retarget.h                                                  */
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

#ifndef SKELETON_RETARGET_H
#define SKELETON_RETARGET_H

#include "core/io/resource.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/resources/animation.h"

// Resources

class RetargetProfile : public Resource {
	GDCLASS(RetargetProfile, Resource);

protected:
	Vector<StringName> intermediate_bone_names;

	virtual bool _get(const StringName &p_path, Variant &r_ret) const;
	virtual bool _set(const StringName &p_path, const Variant &p_value);
	virtual void _get_property_list(List<PropertyInfo> *p_list) const;
	virtual void _validate_property(PropertyInfo &property) const override;
	static void _bind_methods();
#ifdef TOOLS_ENABLED
	void redraw();
#endif // TOOLS_ENABLED

public:
	virtual void add_intermediate_bone(const StringName &p_intermediate_bone_name, int to_index = -1);
	virtual void remove_intermediate_bone(int p_id);
	int find_intermediate_bone(const StringName &p_intermediate_bone_name);
	int get_intermediate_bones_size();

	void set_intermediate_bone_name(int p_id, const StringName &p_intermediate_bone_name);
	StringName get_intermediate_bone_name(int p_id) const;

	RetargetProfile();
	~RetargetProfile();
};

class RetargetRichProfile : public RetargetProfile {
	GDCLASS(RetargetRichProfile, RetargetProfile);

protected:
	Vector<StringName> group_names;
	Vector<Ref<Texture2D>> group_textures;

	Vector<Vector2> intermediate_bone_handle_offsets;
	Vector<int> intermediate_bone_group_ids;

	virtual bool _get(const StringName &p_path, Variant &r_ret) const override;
	virtual bool _set(const StringName &p_path, const Variant &p_value) override;
	virtual void _get_property_list(List<PropertyInfo> *p_list) const override;
	virtual void _validate_property(PropertyInfo &property) const override;
	static void _bind_methods();

public:
	// Group settings
	void add_group(const StringName &p_group_name, int to_index = -1);
	void remove_group(int p_id);
	int find_group(const StringName &p_group_name);
	int get_groups_size() const;

	void set_group_name(int p_id, const StringName &p_group_name);
	StringName get_group_name(int p_id) const;
	void set_group_texture(int p_id, const Ref<Texture2D> &p_group_texture);
	Ref<Texture2D> get_group_texture(int p_id) const;

	// Intermediate bones
	virtual void add_intermediate_bone(const StringName &p_intermediate_bone_name, int to_index = -1) override;
	virtual void remove_intermediate_bone(int p_id) override;

	void set_intermediate_bone_handle_offset(int p_id, Vector2 p_handle_offset);
	Vector2 get_intermediate_bone_handle_offset(int p_id) const;
	void set_intermediate_bone_group_id(int p_id, int p_group_id, bool p_emit_signal = true);
	int get_intermediate_bone_group_id(int p_id) const;

	RetargetRichProfile();
	~RetargetRichProfile();
};

class RetargetBoneOption : public Resource {
	GDCLASS(RetargetBoneOption, Resource);

public:
	struct RetargetBoneOptionParams {
		Animation::RetargetMode retarget_mode = Animation::RETARGET_MODE_GLOBAL;
	};

protected:
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;
	virtual void _validate_property(PropertyInfo &property) const override;
	static void _bind_methods();

private:
	Map<StringName, RetargetBoneOptionParams> retarget_options;

public:
	Vector<StringName> get_keys() const;

	bool has_key(const StringName &p_intermediate_bone_name);
	void add_key(const StringName &p_intermediate_bone_name);
	void remove_key(const StringName &p_intermediate_bone_name);

	void set_retarget_mode(const StringName &p_intermediate_bone_name, Animation::RetargetMode p_retarget_mode);
	Animation::RetargetMode get_retarget_mode(const StringName &p_intermediate_bone_name) const;
#ifdef TOOLS_ENABLED
	void redraw();
#endif // TOOLS_ENABLED

	RetargetBoneOption();
	~RetargetBoneOption();
};

class RetargetBoneMap : public Resource {
	GDCLASS(RetargetBoneMap, Resource);

protected:
	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;
	virtual void _validate_property(PropertyInfo &property) const override;
	static void _bind_methods();

private:
	Map<StringName, StringName> retarget_map;

public:
	Vector<StringName> get_keys() const;

	bool has_key(const StringName &p_intermediate_bone_name);
	void add_key(const StringName &p_intermediate_bone_name);
	void remove_key(const StringName &p_intermediate_bone_name);
	StringName find_key(const StringName &p_bone_name) const;

	void set_bone_name(const StringName &p_intermediate_bone_name, const StringName &p_bone_name);
	StringName get_bone_name(const StringName &p_intermediate_bone_name) const;
#ifdef TOOLS_ENABLED
	void redraw();
#endif // TOOLS_ENABLED

	RetargetBoneMap();
	~RetargetBoneMap();
};

// Retarget Node

class SkeletonRetarget : public Node {
	GDCLASS(SkeletonRetarget, Node);

	Ref<RetargetProfile> retarget_profile;
	Ref<RetargetBoneOption> retarget_option;
	NodePath source_skeleton_path;
	Ref<RetargetBoneMap> source_map;
	NodePath target_skeleton_path;
	Ref<RetargetBoneMap> target_map;

	bool retarget_position = false;
	bool retarget_rotation = true;
	bool retarget_scale = false;

public:
	void set_retarget_profile(const Ref<RetargetProfile> &p_retarget_profile);
	Ref<RetargetProfile> get_retarget_profile() const;
	void set_retarget_option(const Ref<RetargetBoneOption> &p_retarget_option);
	Ref<RetargetBoneOption> get_retarget_option() const;

	void set_source_skeleton(const NodePath &p_skeleton);
	NodePath get_source_skeleton() const;
	void set_source_map(const Ref<RetargetBoneMap> &p_source_map);
	Ref<RetargetBoneMap> get_source_map() const;

	void set_target_skeleton(const NodePath &p_skeleton);
	NodePath get_target_skeleton() const;
	void set_target_map(const Ref<RetargetBoneMap> &p_target_map);
	Ref<RetargetBoneMap> get_target_map() const;

	void set_retarget_position(bool p_enabled);
	bool is_retarget_position() const;
	void set_retarget_rotation(bool p_enabled);
	bool is_retarget_rotation() const;
	void set_retarget_scale(bool p_enabled);
	bool is_retarget_scale() const;

protected:
	void _notification(int p_what);
	static void _bind_methods();

private:
	Skeleton3D *source_skeleton = nullptr;

	void _transpote_pose();
	void _clear_override();

#ifdef TOOLS_ENABLED
	void _redraw();
#endif // TOOLS_ENABLED

	SkeletonRetarget();
	~SkeletonRetarget();
};

#endif // SKELETON_RETARGET_H
