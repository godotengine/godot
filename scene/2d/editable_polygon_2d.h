/*************************************************************************/
/*  editable_polygon_2d.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef EDITABLEPOLYGON2D_H
#define EDITABLEPOLYGON2D_H

#include "core/undo_redo.h"
#include "scene/resources/texture.h"

class EditablePolygon2D {
public:
	virtual int edit_get_polygon_count() const = 0;
	virtual Vector<Vector2> edit_get_polygon(int p_polygon) const = 0;
	virtual void edit_set_polygon(int p_polygon, const Vector<Vector2> &p_points) = 0;

	virtual PoolVector<Vector2> edit_get_uv() const;
	virtual void edit_set_uv(const PoolVector<Vector2> &p_uv);
	virtual Ref<Texture> edit_get_texture() const;
	virtual Vector2 edit_get_offset();

	virtual bool edit_is_wip_destructive() const = 0;
	virtual Color edit_get_previous_outline_color() const = 0;

	virtual void edit_create_wip_close_action(UndoRedo *undo_redo, const Vector<Vector2> &p_wip) = 0;
	virtual void edit_create_edit_poly_action(UndoRedo *undo_redo, int p_polygon, const Vector<Vector2> &p_before, const Vector<Vector2> &p_after) = 0;
	virtual void edit_create_remove_point_action(UndoRedo *undo_redo, int p_polygon, int p_point) = 0;
};

#endif // EDITABLEPOLYGON2D_H
