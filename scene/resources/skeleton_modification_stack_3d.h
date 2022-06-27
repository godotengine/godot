/*************************************************************************/
/*  skeleton_modification_stack_3d.h                                     */
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

#ifndef SKELETONMODIFICATIONSTACK3D_H
#define SKELETONMODIFICATIONSTACK3D_H

#include "core/templates/local_vector.h"
#include "scene/3d/skeleton_3d.h"

class Skeleton3D;
class SkeletonModification3D;

class SkeletonModificationStack3D : public Resource {
	GDCLASS(SkeletonModificationStack3D, Resource);
	friend class Skeleton3D;
	friend class SkeletonModification3D;

protected:
	static void _bind_methods();
	virtual void _get_property_list(List<PropertyInfo> *p_list) const;
	virtual bool _set(const StringName &p_path, const Variant &p_value);
	virtual bool _get(const StringName &p_path, Variant &r_ret) const;

public:
	Skeleton3D *skeleton = nullptr;
	bool is_setup = false;
	bool enabled = false;
	real_t strength = 1.0;

	enum EXECUTION_MODE {
		execution_mode_process,
		execution_mode_physics_process,
	};

	LocalVector<Ref<SkeletonModification3D>> modifications = LocalVector<Ref<SkeletonModification3D>>();
	int modifications_count = 0;

	virtual void setup();
	virtual void execute(real_t p_delta, int p_execution_mode);

	void enable_all_modifications(bool p_enable);
	Ref<SkeletonModification3D> get_modification(int p_mod_idx) const;
	void add_modification(Ref<SkeletonModification3D> p_mod);
	void delete_modification(int p_mod_idx);
	void set_modification(int p_mod_idx, Ref<SkeletonModification3D> p_mod);

	void set_modification_count(int p_count);
	int get_modification_count() const;

	void set_skeleton(Skeleton3D *p_skeleton);
	Skeleton3D *get_skeleton() const;

	bool get_is_setup() const;

	void set_enabled(bool p_enabled);
	bool get_enabled() const;

	void set_strength(real_t p_strength);
	real_t get_strength() const;

	SkeletonModificationStack3D();
};

#endif // SKELETONMODIFICATIONSTACK3D_H
