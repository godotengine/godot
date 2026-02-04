/**************************************************************************/
/*  virtual_joystick_editor_plugin.cpp                                    */
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

#include "virtual_joystick_editor_plugin.h"

#include "editor/scene/canvas_item_editor_plugin.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/virtual_joystick.h"

void VirtualJoystickEditorPlugin::edit(Object *p_object) {
	if (virtual_joystick) {
		virtual_joystick->disconnect(SceneStringName(draw), callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
	}

	virtual_joystick = Object::cast_to<VirtualJoystick>(p_object);

	if (virtual_joystick) {
		virtual_joystick->connect(SceneStringName(draw), callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
	}
	CanvasItemEditor::get_singleton()->update_viewport();
}

bool VirtualJoystickEditorPlugin::handles(Object *p_object) const {
	return Object::cast_to<VirtualJoystick>(p_object) != nullptr;
}

void VirtualJoystickEditorPlugin::forward_canvas_draw_over_viewport(Control *p_viewport_control) {
	if (!virtual_joystick || !virtual_joystick->is_visible_in_tree()) {
		return;
	}

	Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * virtual_joystick->get_screen_transform();

	Vector2 center = virtual_joystick->get_joystick_position();
	float base_radius = virtual_joystick->get_joystick_size() * 0.5f;

	float clampzone_radius = base_radius * virtual_joystick->get_clampzone_ratio();
	float deadzone_radius = clampzone_radius * virtual_joystick->get_deadzone_ratio();

	// NOTE: This color is copied from Camera2DEditor::forward_canvas_draw_over_viewport.
	// We may want to unify them somehow in the future.
	Color clampzone_color = Color(1, 1, 0.25, 0.63);
	Color deadzone_color = Color(1, 1, 0.25, 0.63);

	int width = Math::round(1 * EDSCALE);

	p_viewport_control->draw_circle(xform.xform(center), clampzone_radius * xform.get_scale().x, clampzone_color, false, width);

	p_viewport_control->draw_circle(xform.xform(center), deadzone_radius * xform.get_scale().x, deadzone_color, false, width);
}
