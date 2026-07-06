/**************************************************************************/
/*  fabrik_2d.h                                                           */
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

#include "scene/2d/iterate_ik_2d.h"

class FABRIK2D : public IterateIK2D {
	GDCLASS(FABRIK2D, IterateIK2D);

	NodePath target_node;
	int root_bone = -1;
	int tip_bone = -1;
	int max_iterations = 10;
	real_t tolerance = 1.0;

protected:
	static void _bind_methods();
	virtual PackedStringArray get_configuration_warnings() const override;
	virtual void _process_modification(double p_delta) override;

public:
	void set_target_node(const NodePath &p_target_node);
	NodePath get_target_node() const;
	void set_root_bone(int p_bone);
	int get_root_bone() const;
	void set_tip_bone(int p_bone);
	int get_tip_bone() const;
	void set_max_iterations(int p_iterations);
	int get_max_iterations() const;
	void set_tolerance(real_t p_tolerance);
	real_t get_tolerance() const;
};
