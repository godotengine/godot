/**************************************************************************/
/*  particles_editor_plugin.cpp                                           */
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

#include "particles_editor_plugin.h"

#include "core/object/callable_mp.h"
#include "editor/docks/scene_tree_dock.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/settings/editor_settings.h"
#include "scene/gui/box_container.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/spin_box.h"
#include "scene/resources/gradient_texture.h"
#include "scene/resources/particle_process_material.h"

void ParticlesEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (handled_type.ends_with("2D")) {
				add_control_to_container(CONTAINER_CANVAS_EDITOR_MENU, toolbar);
			} else if (handled_type.ends_with("3D")) {
				add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, toolbar);
			} else {
				DEV_ASSERT(false);
			}

			menu->set_button_icon(menu->get_editor_theme_icon(handled_type));
			menu->set_text(handled_type);

			PopupMenu *popup = menu->get_popup();
			popup->add_shortcut(ED_SHORTCUT("particles/restart_emission", TTRC("Restart Emission"), KeyModifierMask::CTRL | Key::R), MENU_RESTART);
			_add_menu_options(popup);
			popup->add_item(conversion_option_name, MENU_OPTION_CONVERT);
		} break;
	}
}

Ref<ParticleProcessMaterial> ParticlesEditorPlugin::get_material_for_object_created_in_editor() const {
	Ref<ParticleProcessMaterial> particle_process_material;
	particle_process_material.instantiate();

	// Reduce particle spread to make the particle node's rotation easier to notice.
	particle_process_material->set_spread(5.0);

	// Randomize initial particle rotation.
	particle_process_material->set_param_min(ParticleProcessMaterial::PARAM_ANGLE, 0.0);
	particle_process_material->set_param_max(ParticleProcessMaterial::PARAM_ANGLE, 360.0);

	// Scale particles up and down as the lifetime progresses.
	Ref<CurveTexture> curve_texture = memnew(CurveTexture);
	Ref<Curve> curve = memnew(Curve);
	// The wind-up occurs significantly faster than the fade-out.
	curve->add_point(Vector2(0.0, 0.0));
	curve->add_point(Vector2(0.1, 1.0));
	curve->add_point(Vector2(1.0, 0.0));
	curve_texture->set_curve(curve);
	particle_process_material->set_param_texture(ParticleProcessMaterial::PARAM_SCALE, curve_texture);

	// Fade particles with transparency as the lifetime progresses.
	Ref<GradientTexture1D> gradient_texture = memnew(GradientTexture1D);
	Ref<Gradient> gradient = memnew(Gradient);
	// The fade-in occurs significantly faster than the fade-out.
	gradient->set_color(0, Color(1, 1, 1, 0));
	gradient->set_color(1, Color(1, 1, 1, 0));
	gradient->add_point(0.1, Color(1, 1, 1, 1));
	gradient_texture->set_gradient(gradient);
	particle_process_material->set_color_ramp(gradient_texture);

	return particle_process_material;
}

Ref<Texture2D> ParticlesEditorPlugin::get_texture_for_object_created_in_editor() const {
	// Create a solid square texture.
	// We use a square instead of a circle, so that billboarded particle rotation is visible.
	Ref<GradientTexture2D> texture = memnew(GradientTexture2D);
	// Use a low resolution, so that it displays at a comparable size between 2D and 3D.
	// Texture size defines each particle's size in 2D, but not in 3D.
	texture->set_width(12);
	texture->set_height(12);
	texture->set_fill(GradientTexture2D::FILL_SQUARE);
	texture->set_fill_from(Vector2(0.5, 0.5));
	texture->set_fill_to(Vector2(0.5, 0.01));
	Ref<Gradient> gradient = memnew(Gradient);
	gradient->set_color(0, Color(1, 1, 1));
	gradient->set_color(1, Color(1, 1, 1, 0));
	// Harden the gradient so the texture has a sharper (but still somewhat soft) falloff.
	gradient->set_offset(0, 0.75);
	texture->set_gradient(gradient);

	return texture;
}

bool ParticlesEditorPlugin::need_show_lifetime_dialog(SpinBox *p_seconds) {
	// Add one second to the default generation lifetime, since the progress is updated every second.
	p_seconds->set_value(MAX(1.0, std::trunc(edited_node->get("lifetime").operator double()) + 1.0));

	if (p_seconds->get_value() >= 11.0 + CMP_EPSILON) {
		// Only pop up the time dialog if the particle's lifetime is long enough to warrant shortening it.
		return true;
	} else {
		// Generate the visibility rect/AABB immediately.
		return false;
	}
}

void ParticlesEditorPlugin::_menu_callback(int p_idx) {
	switch (p_idx) {
		case MENU_OPTION_CONVERT: {
			Node *converted_node = _convert_particles();

			EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
			ur->create_action(conversion_option_name, UndoRedo::MERGE_DISABLE, edited_node);
			SceneTreeDock::get_singleton()->replace_node(edited_node, converted_node);
			ur->commit_action(false);
		} break;

		case MENU_RESTART: {
			edited_node->call("restart");
		}
	}
}

void ParticlesEditorPlugin::edit(Object *p_object) {
	edited_node = Object::cast_to<Node>(p_object);
}

bool ParticlesEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class(handled_type);
}

void ParticlesEditorPlugin::make_visible(bool p_visible) {
	toolbar->set_visible(p_visible);
}

ParticlesEditorPlugin::ParticlesEditorPlugin() {
	toolbar = memnew(HBoxContainer);
	toolbar->hide();

	menu = memnew(MenuButton);
	menu->set_switch_on_hover(true);
	menu->set_flat(false);
	menu->set_theme_type_variation("FlatMenuButtonNoIconTint");
	toolbar->add_child(menu);
	menu->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &ParticlesEditorPlugin::_menu_callback));
}
