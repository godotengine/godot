/**************************************************************************/
/*  shape_owner_bullet.h                                                  */
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

#ifndef SHAPE_OWNER_BULLET_H
#define SHAPE_OWNER_BULLET_H

#include "rid_bullet.h"

/**
	@author AndreaCatania
*/

class ShapeBullet;
class btCollisionShape;
class CollisionObjectBullet;

/// Each class that want to use Shapes must inherit this class
/// E.G. BodyShape is a child of this
class ShapeOwnerBullet {
public:
	virtual int find_shape(ShapeBullet *p_shape) const = 0;
	virtual void shape_changed(int p_shape_index) = 0;
	virtual void reload_shapes() = 0;
	virtual void remove_shape_full(class ShapeBullet *p_shape) = 0;
	virtual ~ShapeOwnerBullet() {}
};

#endif // SHAPE_OWNER_BULLET_H
