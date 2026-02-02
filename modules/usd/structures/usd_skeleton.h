/**************************************************************************/
/*  usd_skeleton.h                                                        */
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

#include "core/io/resource.h"
#include "core/variant/typed_array.h"

class USDSkeleton : public Resource {
	GDCLASS(USDSkeleton, Resource);

private:
	Vector<String> joint_paths;
	Vector<int> joint_parents;
	Vector<Transform3D> bind_transforms;
	Vector<Transform3D> rest_transforms;

protected:
	static void _bind_methods();

public:
	Vector<String> get_joint_paths() const;
	void set_joint_paths(const Vector<String> &p_joint_paths);

	Vector<int> get_joint_parents() const;
	void set_joint_parents(const Vector<int> &p_joint_parents);

	TypedArray<Transform3D> get_bind_transforms() const;
	void set_bind_transforms(const TypedArray<Transform3D> &p_bind_transforms);

	TypedArray<Transform3D> get_rest_transforms() const;
	void set_rest_transforms(const TypedArray<Transform3D> &p_rest_transforms);
};
