/*************************************************************************/
/*  retarget_profile.h                                                   */
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

#ifndef RETARGET_PROFILE_H
#define RETARGET_PROFILE_H

#include "core/io/resource.h"

class RetargetProfile : public Resource {
	GDCLASS(RetargetProfile, Resource);

protected:
	// For preset, don't make it public.
	bool is_read_only = false;

	// Knowing animation meta data is hard since animation resource doesn't have inspector.
	// For knowing RealtimeRetargetPreset type, import plugin add this label to animation name as prefix.
	String label_for_animation_name;

	Vector<StringName> global_transform_targets;
	Vector<StringName> local_transform_targets;
	Vector<StringName> absolute_transform_targets;

	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _validate_property(PropertyInfo &p_property) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	static void _bind_methods();

public:
	void set_label_for_animation_name(const String p_label_for_animation_name);
	String get_label_for_animation_name() const;

	void set_global_transform_target_size(int p_size);
	int get_global_transform_target_size() const;
	void add_global_transform_target(const StringName p_bone_name);
	void set_global_transform_target(int p_idx, const StringName p_bone_name);
	StringName get_global_transform_target(int p_idx) const;
	bool has_global_transform_target(const StringName p_bone_name);

	void set_local_transform_target_size(int p_size);
	int get_local_transform_target_size() const;
	void add_local_transform_target(const StringName p_bone_name);
	void set_local_transform_target(int p_idx, const StringName p_bone_name);
	StringName get_local_transform_target(int p_idx) const;
	bool has_local_transform_target(const StringName p_bone_name);

	void set_absolute_transform_target_size(int p_size);
	int get_absolute_transform_target_size() const;
	void add_absolute_transform_target(const StringName p_bone_name);
	void set_absolute_transform_target(int p_idx, const StringName p_bone_name);
	StringName get_absolute_transform_target(int p_idx) const;
	bool has_absolute_transform_target(const StringName p_bone_name);

	RetargetProfile();
	~RetargetProfile();
};

class RetargetProfileGlobalAll : public RetargetProfile {
	GDCLASS(RetargetProfileGlobalAll, RetargetProfile);

public:
	RetargetProfileGlobalAll();
	~RetargetProfileGlobalAll();
};

class RetargetProfileLocalAll : public RetargetProfile {
	GDCLASS(RetargetProfileLocalAll, RetargetProfile);

public:
	RetargetProfileLocalAll();
	~RetargetProfileLocalAll();
};

class RetargetProfileAbsoluteAll : public RetargetProfile {
	GDCLASS(RetargetProfileAbsoluteAll, RetargetProfile);

public:
	RetargetProfileAbsoluteAll();
	~RetargetProfileAbsoluteAll();
};

class RetargetProfileLocalFingersGlobalOthers : public RetargetProfile {
	GDCLASS(RetargetProfileLocalFingersGlobalOthers, RetargetProfile);

public:
	RetargetProfileLocalFingersGlobalOthers();
	~RetargetProfileLocalFingersGlobalOthers();
};

class RetargetProfileLocalLimbsGlobalOthers : public RetargetProfile {
	GDCLASS(RetargetProfileLocalLimbsGlobalOthers, RetargetProfile);

public:
	RetargetProfileLocalLimbsGlobalOthers();
	~RetargetProfileLocalLimbsGlobalOthers();
};

class RetargetProfileAbsoluteFingersGlobalOthers : public RetargetProfile {
	GDCLASS(RetargetProfileAbsoluteFingersGlobalOthers, RetargetProfile);

public:
	RetargetProfileAbsoluteFingersGlobalOthers();
	~RetargetProfileAbsoluteFingersGlobalOthers();
};

class RetargetProfileAbsoluteLimbsGlobalOthers : public RetargetProfile {
	GDCLASS(RetargetProfileAbsoluteLimbsGlobalOthers, RetargetProfile);

public:
	RetargetProfileAbsoluteLimbsGlobalOthers();
	~RetargetProfileAbsoluteLimbsGlobalOthers();
};

class RetargetProfileAbsoluteFingersLocalLimbsGlobalOthers : public RetargetProfile {
	GDCLASS(RetargetProfileAbsoluteFingersLocalLimbsGlobalOthers, RetargetProfile);

public:
	RetargetProfileAbsoluteFingersLocalLimbsGlobalOthers();
	~RetargetProfileAbsoluteFingersLocalLimbsGlobalOthers();
};

#endif // RETARGET_PROFILE_H
