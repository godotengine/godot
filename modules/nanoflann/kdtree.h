/*************************************************************************/
/*  kdtree.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef KDTREE_H
#define KDTREE_H
#include "resource.h"

class KDTree2D : public Reference {
	GDCLASS(KDTree2D, Reference)
	class KDTreeData2D;
	KDTreeData2D *data;

protected:
	static void _bind_methods();

public:
	KDTree2D();
	~KDTree2D();
	void rebuild();
	Vector<Vector2> search(const Vector2 &coord, real_t radius);
	void add_point(const Vector2 &point);
	void add_points(const Vector<Vector2> &points);
};
class KDTree3D : public Reference {
	GDCLASS(KDTree3D, Reference)
	class KDTreeData3D;
	KDTreeData3D *data;

protected:
	static void _bind_methods();

public:
	KDTree3D();
	~KDTree3D();
	void rebuild();
	Vector<Vector3> search(const Vector3 &coord, real_t radius);
	void add_point(const Vector3 &point);
	void add_points(const Vector<Vector3> &points);
};
#endif
