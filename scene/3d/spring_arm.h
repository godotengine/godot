/**************************************************************************/
/*  spring_arm.h                                                          */
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

#ifndef SPRING_ARM_H
#define SPRING_ARM_H

#include "scene/3d/spatial.h"

class SpringArm : public Spatial {
	GDCLASS(SpringArm, Spatial);

	Ref<Shape> shape;
	Set<RID> excluded_objects;
	float spring_length;
	float current_spring_length;
	bool keep_child_basis;
	uint32_t mask;
	float margin;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_length(float p_length);
	float get_length() const;
	void set_shape(Ref<Shape> p_shape);
	Ref<Shape> get_shape() const;
	void set_mask(uint32_t p_mask);
	uint32_t get_mask();
	void add_excluded_object(RID p_rid);
	bool remove_excluded_object(RID p_rid);
	void clear_excluded_objects();
	float get_hit_length();
	void set_margin(float p_margin);
	float get_margin();

	SpringArm();

private:
	void process_spring();
};

#endif // SPRING_ARM_H
