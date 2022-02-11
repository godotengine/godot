/*************************************************************************/
/*  collision_shape_2d_editor_plugin.cpp                                 */
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

#include "collision_shape_2d_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "scene/resources/capsule_shape_2d.h"
#include "scene/resources/circle_shape_2d.h"
#include "scene/resources/concave_polygon_shape_2d.h"
#include "scene/resources/convex_polygon_shape_2d.h"
#include "scene/resources/line_shape_2d.h"
#include "scene/resources/rectangle_shape_2d.h"
#include "scene/resources/segment_shape_2d.h"

void CollisionShape2DEditor::_node_removed(Node *p_node) {
	if (p_node == node) {
		node = nullptr;
	}
}

Variant CollisionShape2DEditor::get_handle_value(int idx) const {
	switch (shape_type) {
		case CAPSULE_SHAPE: {
			Ref<CapsuleShape2D> capsule = node->get_shape();

			if (idx == 0) {
				return capsule->get_radius();
			} else if (idx == 1) {
				return capsule->get_height();
			}

		} break;

		case CIRCLE_SHAPE: {
			Ref<CircleShape2D> circle = node->get_shape();

			if (idx == 0) {
				return circle->get_radius();
			}

		} break;

		case CONCAVE_POLYGON_SHAPE: {
		} break;

		case CONVEX_POLYGON_SHAPE: {
		} break;

		case LINE_SHAPE: {
			Ref<LineShape2D> line = node->get_shape();

			if (idx == 0) {
				return line->get_d();
			} else {
				return line->get_normal();
			}

		} break;

		case RAY_SHAPE: {
			Ref<RayShape2D> ray = node->get_shape();

			if (idx == 0) {
				return ray->get_length();
			}

		} break;

		case RECTANGLE_SHAPE: {
			Ref<RectangleShape2D> rect = node->get_shape();

			if (idx < 8) {
				return rect->get_extents().abs();
			}

		} break;

		case SEGMENT_SHAPE: {
			Ref<SegmentShape2D> seg = node->get_shape();

			if (idx == 0) {
				return seg->get_a();
			} else if (idx == 1) {
				return seg->get_b();
			}

		} break;
	}

	return Variant();
}

void CollisionShape2DEditor::set_handle(int idx, Point2 &p_point) {
	switch (shape_type) {
		case CAPSULE_SHAPE: {
			if (idx < 2) {
				Ref<CapsuleShape2D> capsule = node->get_shape();

				real_t parameter = Math::abs(p_point[idx]);

				if (idx == 0) {
					capsule->set_radius(parameter);
				} else if (idx == 1) {
					capsule->set_height(parameter * 2 - capsule->get_radius() * 2);
				}

				canvas_item_editor->update_viewport();
			}

		} break;

		case CIRCLE_SHAPE: {
			Ref<CircleShape2D> circle = node->get_shape();
			circle->set_radius(p_point.length());

			canvas_item_editor->update_viewport();

		} break;

		case CONCAVE_POLYGON_SHAPE: {
		} break;

		case CONVEX_POLYGON_SHAPE: {
		} break;

		case LINE_SHAPE: {
			if (idx < 2) {
				Ref<LineShape2D> line = node->get_shape();

				if (idx == 0) {
					line->set_d(p_point.length());
				} else {
					line->set_normal(p_point.normalized());
				}

				canvas_item_editor->update_viewport();
			}

		} break;

		case RAY_SHAPE: {
			Ref<RayShape2D> ray = node->get_shape();

			ray->set_length(Math::abs(p_point.y));

			canvas_item_editor->update_viewport();

		} break;

		case RECTANGLE_SHAPE: {
			if (idx < 8) {
				Ref<RectangleShape2D> rect = node->get_shape();
				Vector2 extents = (Point2)original;

				if (RECT_HANDLES[idx].x != 0) {
					extents.x = p_point.x * RECT_HANDLES[idx].x;
				}
				if (RECT_HANDLES[idx].y != 0) {
					extents.y = p_point.y * RECT_HANDLES[idx].y;
				}

				if (Input::get_singleton()->is_key_pressed(KEY_ALT)) {
					rect->set_extents(extents.abs());
					node->set_global_position(original_transform.get_origin());
				} else {
					rect->set_extents(((Point2)original + (extents - (Point2)original) * 0.5).abs());
					Point2 pos = original_transform.affine_inverse().xform(original_transform.get_origin());
					pos += (extents - (Point2)original) * RECT_HANDLES[idx] * 0.5;
					node->set_global_position(original_transform.xform(pos));
				}

				canvas_item_editor->update_viewport();
			}

		} break;

		case SEGMENT_SHAPE: {
			if (edit_handle < 2) {
				Ref<SegmentShape2D> seg = node->get_shape();

				if (idx == 0) {
					seg->set_a(p_point);
				} else if (idx == 1) {
					seg->set_b(p_point);
				}

				canvas_item_editor->update_viewport();
			}

		} break;
	}
	node->get_shape()->_change_notify();
}

void CollisionShape2DEditor::commit_handle(int idx, Variant &p_org) {
	undo_redo->create_action(TTR("Set Handle"));

	switch (shape_type) {
		case CAPSULE_SHAPE: {
			Ref<CapsuleShape2D> capsule = node->get_shape();

			if (idx == 0) {
				undo_redo->add_do_method(capsule.ptr(), "set_radius", capsule->get_radius());
				undo_redo->add_do_method(canvas_item_editor, "update_viewport");
				undo_redo->add_undo_method(capsule.ptr(), "set_radius", p_org);
				undo_redo->add_do_method(canvas_item_editor, "update_viewport");
			} else if (idx == 1) {
				undo_redo->add_do_method(capsule.ptr(), "set_height", capsule->get_height());
				undo_redo->add_do_method(canvas_item_editor, "update_viewport");
				undo_redo->add_undo_method(capsule.ptr(), "set_height", p_org);
				undo_redo->add_undo_method(canvas_item_editor, "update_viewport");
			}

		} break;

		case CIRCLE_SHAPE: {
			Ref<CircleShape2D> circle = node->get_shape();

			undo_redo->add_do_method(circle.ptr(), "set_radius", circle->get_radius());
			undo_redo->add_do_method(canvas_item_editor, "update_viewport");
			undo_redo->add_undo_method(circle.ptr(), "set_radius", p_org);
			undo_redo->add_undo_method(canvas_item_editor, "update_viewport");

		} break;

		case CONCAVE_POLYGON_SHAPE: {
			// Cannot be edited directly, use CollisionPolygon2D instead.
		} break;

		case CONVEX_POLYGON_SHAPE: {
			// Cannot be edited directly, use CollisionPolygon2D instead.
		} break;

		case LINE_SHAPE: {
			Ref<LineShape2D> line = node->get_shape();

			if (idx == 0) {
				undo_redo->add_do_method(line.ptr(), "set_d", line->get_d());
				undo_redo->add_do_method(canvas_item_editor, "update_viewport");
				undo_redo->add_undo_method(line.ptr(), "set_d", p_org);
				undo_redo->add_undo_method(canvas_item_editor, "update_viewport");
			} else {
				undo_redo->add_do_method(line.ptr(), "set_normal", line->get_normal());
				undo_redo->add_do_method(canvas_item_editor, "update_viewport");
				undo_redo->add_undo_method(line.ptr(), "set_normal", p_org);
				undo_redo->add_undo_method(canvas_item_editor, "update_viewport");
			}

		} break;

		case RAY_SHAPE: {
			Ref<RayShape2D> ray = node->get_shape();

			undo_redo->add_do_method(ray.ptr(), "set_length", ray->get_length());
			undo_redo->add_do_method(canvas_item_editor, "update_viewport");
			undo_redo->add_undo_method(ray.ptr(), "set_length", p_org);
			undo_redo->add_undo_method(canvas_item_editor, "update_viewport");

		} break;

		case RECTANGLE_SHAPE: {
			Ref<RectangleShape2D> rect = node->get_shape();

			undo_redo->add_do_method(rect.ptr(), "set_extents", rect->get_extents());
			undo_redo->add_do_method(node, "set_global_transform", node->get_global_transform());
			undo_redo->add_do_method(canvas_item_editor, "update_viewport");
			undo_redo->add_undo_method(rect.ptr(), "set_extents", p_org);
			undo_redo->add_undo_method(node, "set_global_transform", original_transform);
			undo_redo->add_undo_method(canvas_item_editor, "update_viewport");

		} break;

		case SEGMENT_SHAPE: {
			Ref<SegmentShape2D> seg = node->get_shape();
			if (idx == 0) {
				undo_redo->add_do_method(seg.ptr(), "set_a", seg->get_a());
				undo_redo->add_do_method(canvas_item_editor, "update_viewport");
				undo_redo->add_undo_method(seg.ptr(), "set_a", p_org);
				undo_redo->add_undo_method(canvas_item_editor, "update_viewport");
			} else if (idx == 1) {
				undo_redo->add_do_method(seg.ptr(), "set_b", seg->get_b());
				undo_redo->add_do_method(canvas_item_editor, "update_viewport");
				undo_redo->add_undo_method(seg.ptr(), "set_b", p_org);
				undo_redo->add_undo_method(canvas_item_editor, "update_viewport");
			}

		} break;
	}

	undo_redo->commit_action();
}

bool CollisionShape2DEditor::forward_canvas_gui_input(const Ref<InputEvent> &p_event) {
	if (!node) {
		return false;
	}

	if (!node->get_shape().is_valid()) {
		return false;
	}

	if (!node->is_visible_in_tree()) {
		return false;
	}

	if (shape_type == -1) {
		return false;
	}

	Ref<InputEventMouseButton> mb = p_event;
	Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_global_transform();

	if (mb.is_valid()) {
		Vector2 gpoint = mb->get_position();

		if (mb->get_button_index() == BUTTON_LEFT) {
			if (mb->is_pressed()) {
				for (int i = 0; i < handles.size(); i++) {
					if (xform.xform(handles[i]).distance_to(gpoint) < 8) {
						edit_handle = i;

						break;
					}
				}

				if (edit_handle == -1) {
					pressed = false;

					return false;
				}

				original = get_handle_value(edit_handle);
				original_transform = node->get_global_transform();
				last_point = original;
				pressed = true;

				return true;

			} else {
				if (pressed) {
					commit_handle(edit_handle, original);

					edit_handle = -1;
					pressed = false;

					return true;
				}
			}
		}

		return false;
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		if (edit_handle == -1 || !pressed) {
			return false;
		}

		Vector2 cpoint = canvas_item_editor->snap_point(canvas_item_editor->get_canvas_transform().affine_inverse().xform(mm->get_position()));
		cpoint = original_transform.affine_inverse().xform(cpoint);
		last_point = cpoint;

		set_handle(edit_handle, cpoint);

		return true;
	}

	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {
		if (edit_handle == -1 || !pressed || k->is_echo()) {
			return false;
		}

		if (shape_type == RECTANGLE_SHAPE && k->get_scancode() == KEY_ALT) {
			set_handle(edit_handle, last_point); // Update handle when Alt key is toggled.
		}
	}

	return false;
}

void CollisionShape2DEditor::_get_current_shape_type() {
	if (!node) {
		return;
	}

	Ref<Shape2D> s = node->get_shape();

	if (!s.is_valid()) {
		return;
	}

	if (Object::cast_to<CapsuleShape2D>(*s)) {
		shape_type = CAPSULE_SHAPE;
	} else if (Object::cast_to<CircleShape2D>(*s)) {
		shape_type = CIRCLE_SHAPE;
	} else if (Object::cast_to<ConcavePolygonShape2D>(*s)) {
		shape_type = CONCAVE_POLYGON_SHAPE;
	} else if (Object::cast_to<ConvexPolygonShape2D>(*s)) {
		shape_type = CONVEX_POLYGON_SHAPE;
	} else if (Object::cast_to<LineShape2D>(*s)) {
		shape_type = LINE_SHAPE;
	} else if (Object::cast_to<RayShape2D>(*s)) {
		shape_type = RAY_SHAPE;
	} else if (Object::cast_to<RectangleShape2D>(*s)) {
		shape_type = RECTANGLE_SHAPE;
	} else if (Object::cast_to<SegmentShape2D>(*s)) {
		shape_type = SEGMENT_SHAPE;
	} else {
		shape_type = -1;
	}

	canvas_item_editor->update_viewport();
}

void CollisionShape2DEditor::forward_canvas_draw_over_viewport(Control *p_overlay) {
	if (!node) {
		return;
	}

	if (!node->get_shape().is_valid()) {
		return;
	}

	if (!node->is_visible_in_tree()) {
		return;
	}

	_get_current_shape_type();

	if (shape_type == -1) {
		return;
	}

	Transform2D gt = canvas_item_editor->get_canvas_transform() * node->get_global_transform();

	Ref<Texture> h = get_icon("EditorHandle", "EditorIcons");
	Vector2 size = h->get_size() * 0.5;

	handles.clear();

	switch (shape_type) {
		case CAPSULE_SHAPE: {
			Ref<CapsuleShape2D> shape = node->get_shape();

			handles.resize(2);
			float radius = shape->get_radius();
			float height = shape->get_height() / 2;

			handles.write[0] = Point2(radius, -height);
			handles.write[1] = Point2(0, -(height + radius));

			p_overlay->draw_texture(h, gt.xform(handles[0]) - size);
			p_overlay->draw_texture(h, gt.xform(handles[1]) - size);

		} break;

		case CIRCLE_SHAPE: {
			Ref<CircleShape2D> shape = node->get_shape();

			handles.resize(1);
			handles.write[0] = Point2(shape->get_radius(), 0);

			p_overlay->draw_texture(h, gt.xform(handles[0]) - size);

		} break;

		case CONCAVE_POLYGON_SHAPE: {
		} break;

		case CONVEX_POLYGON_SHAPE: {
		} break;

		case LINE_SHAPE: {
			Ref<LineShape2D> shape = node->get_shape();

			handles.resize(2);
			handles.write[0] = shape->get_normal() * shape->get_d();
			handles.write[1] = shape->get_normal() * (shape->get_d() + 30.0);

			p_overlay->draw_texture(h, gt.xform(handles[0]) - size);
			p_overlay->draw_texture(h, gt.xform(handles[1]) - size);

		} break;

		case RAY_SHAPE: {
			Ref<RayShape2D> shape = node->get_shape();

			handles.resize(1);
			handles.write[0] = Point2(0, shape->get_length());

			p_overlay->draw_texture(h, gt.xform(handles[0]) - size);

		} break;

		case RECTANGLE_SHAPE: {
			Ref<RectangleShape2D> shape = node->get_shape();

			handles.resize(8);
			Vector2 ext = shape->get_extents();
			for (int i = 0; i < handles.size(); i++) {
				handles.write[i] = RECT_HANDLES[i] * ext;
				p_overlay->draw_texture(h, gt.xform(handles[i]) - size);
			}
		} break;

		case SEGMENT_SHAPE: {
			Ref<SegmentShape2D> shape = node->get_shape();

			handles.resize(2);
			handles.write[0] = shape->get_a();
			handles.write[1] = shape->get_b();

			p_overlay->draw_texture(h, gt.xform(handles[0]) - size);
			p_overlay->draw_texture(h, gt.xform(handles[1]) - size);

		} break;
	}
}

void CollisionShape2DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			get_tree()->connect("node_removed", this, "_node_removed");
		} break;

		case NOTIFICATION_EXIT_TREE: {
			get_tree()->disconnect("node_removed", this, "_node_removed");
		} break;
	}
}

void CollisionShape2DEditor::edit(Node *p_node) {
	if (!canvas_item_editor) {
		canvas_item_editor = CanvasItemEditor::get_singleton();
	}

	if (p_node) {
		node = Object::cast_to<CollisionShape2D>(p_node);

		_get_current_shape_type();

	} else {
		edit_handle = -1;
		shape_type = -1;

		node = nullptr;
	}

	canvas_item_editor->update_viewport();
}

void CollisionShape2DEditor::_bind_methods() {
	ClassDB::bind_method("_get_current_shape_type", &CollisionShape2DEditor::_get_current_shape_type);
	ClassDB::bind_method(D_METHOD("_node_removed"), &CollisionShape2DEditor::_node_removed);
}

CollisionShape2DEditor::CollisionShape2DEditor(EditorNode *p_editor) {
	node = nullptr;
	canvas_item_editor = nullptr;
	editor = p_editor;

	undo_redo = p_editor->get_undo_redo();

	edit_handle = -1;
	pressed = false;
}

void CollisionShape2DEditorPlugin::edit(Object *p_obj) {
	collision_shape_2d_editor->edit(Object::cast_to<Node>(p_obj));
}

bool CollisionShape2DEditorPlugin::handles(Object *p_obj) const {
	return p_obj->is_class("CollisionShape2D");
}

void CollisionShape2DEditorPlugin::make_visible(bool visible) {
	if (!visible) {
		edit(nullptr);
	}
}

CollisionShape2DEditorPlugin::CollisionShape2DEditorPlugin(EditorNode *p_editor) {
	editor = p_editor;

	collision_shape_2d_editor = memnew(CollisionShape2DEditor(p_editor));
	p_editor->get_gui_base()->add_child(collision_shape_2d_editor);
}

CollisionShape2DEditorPlugin::~CollisionShape2DEditorPlugin() {
}
