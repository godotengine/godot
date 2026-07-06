/**************************************************************************/
/*  skeleton_modifier_2d.h                                                */
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

#include "scene/2d/skeleton_2d.h"

class SkeletonModifier2D : public Node2D {
	GDCLASS(SkeletonModifier2D, Node2D);

protected:
	bool active = true;
	real_t influence = 1.0;

	ObjectID skeleton_id;

	void _update_skeleton();
	void _update_skeleton_path();
	void _force_update_skeleton();

	virtual void _skeleton_changed(Skeleton2D *p_old, Skeleton2D *p_new);
	virtual void _validate_bone_names();
	GDVIRTUAL2(_skeleton_changed, Skeleton2D *, Skeleton2D *);
	GDVIRTUAL0(_validate_bone_names);

	void _notification(int p_what);
	static void _bind_methods();

	virtual void _set_active(bool p_active);

	virtual void _process_modification(double p_delta);
	GDVIRTUAL1(_process_modification, double);

public:
	virtual PackedStringArray get_configuration_warnings() const override;
	virtual bool has_process() const { return false; }

	void set_active(bool p_active);
	bool is_active() const;

	void set_influence(real_t p_influence);
	real_t get_influence() const;

	Skeleton2D *get_skeleton() const;
	void process_modification(double p_delta);

	SkeletonModifier2D();
};
