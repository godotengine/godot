/**************************************************************************/
/*  cpu_particles_3d_editor_plugin.cpp                                    */
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

#include "cpu_particles_3d_editor_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/gui/scene_tree_editor.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "editor/scene_tree_dock.h"
#include "scene/gui/menu_button.h"

void CPUParticles3DEditor::_node_removed(Node *p_node) {
	if (p_node == node) {
		node = nullptr;
		hide();
	}
}

void CPUParticles3DEditor::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_ENTER_TREE: {
			options->set_icon(get_editor_theme_icon(SNAME("CPUParticles3D")));
		} break;
	}
}

void CPUParticles3DEditor::_menu_option(int p_option) {
	switch (p_option) {
		case MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_NODE: {
			emission_tree_dialog->popup_scenetree_dialog();
		} break;

		case MENU_OPTION_RESTART: {
			node->restart();
		} break;

		case MENU_OPTION_CONVERT_TO_GPU_PARTICLES: {
			GPUParticles3D *gpu_particles = memnew(GPUParticles3D);
			gpu_particles->convert_from_particles(node);
			gpu_particles->set_name(node->get_name());
			gpu_particles->set_transform(node->get_transform());
			gpu_particles->set_visible(node->is_visible());
			gpu_particles->set_process_mode(node->get_process_mode());

			EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
			ur->create_action(TTR("Convert to GPUParticles3D"), UndoRedo::MERGE_DISABLE, node);
			SceneTreeDock::get_singleton()->replace_node(node, gpu_particles);
			ur->commit_action(false);

		} break;
		case MENU_OPTION_GENERATE_AABB: {
			// Add one second to the default generation lifetime, since the progress is updated every second.
			generate_seconds->set_value(MAX(1.0, trunc(node->get_lifetime()) + 1.0));

			if (generate_seconds->get_value() >= 11.0 + CMP_EPSILON) {
				// Only pop up the time dialog if the particle's lifetime is long enough to warrant shortening it.
				generate_aabb->popup_centered();
			} else {
				// Generate the visibility AABB immediately.
				_generate_aabb();
			}
		} break;
	}
}

void CPUParticles3DEditor::_generate_aabb() {
	double time = generate_seconds->get_value();

	double running = 0.0;

	EditorProgress ep("gen_aabb", TTR("Generating Visibility AABB (Waiting for Particle Simulation)"), int(time));

	bool was_emitting = node->is_emitting();
	if (!was_emitting) {
		node->set_emitting(true);
		OS::get_singleton()->delay_usec(1000);
	}

	AABB rect;

	while (running < time) {
		uint64_t ticks = OS::get_singleton()->get_ticks_usec();
		ep.step(TTR("Generating..."), int(running), true);
		OS::get_singleton()->delay_usec(1000);

		AABB capture = node->capture_aabb();
		if (rect == AABB()) {
			rect = capture;
		} else {
			rect.merge_with(capture);
		}

		running += (OS::get_singleton()->get_ticks_usec() - ticks) / 1000000.0;
	}

	if (!was_emitting) {
		node->set_emitting(false);
	}

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(TTR("Generate Visibility AABB"));
	ur->add_do_method(node, "set_visibility_aabb", rect);
	ur->add_undo_method(node, "set_visibility_aabb", node->get_visibility_aabb());
	ur->commit_action();
}

void CPUParticles3DEditor::edit(CPUParticles3D *p_particles) {
	base_node = p_particles;
	node = p_particles;
}

void CPUParticles3DEditor::_generate_emission_points() {
	/// hacer codigo aca
	Vector<Vector3> points;
	Vector<Vector3> normals;

	if (!_generate(points, normals)) {
		return;
	}

	if (normals.size() == 0) {
		node->set_emission_shape(CPUParticles3D::EMISSION_SHAPE_POINTS);
		node->set_emission_points(points);
	} else {
		node->set_emission_shape(CPUParticles3D::EMISSION_SHAPE_DIRECTED_POINTS);
		node->set_emission_points(points);
		node->set_emission_normals(normals);
	}
}

CPUParticles3DEditor::CPUParticles3DEditor() {
	particles_editor_hb = memnew(HBoxContainer);
	Node3DEditor::get_singleton()->add_control_to_menu_panel(particles_editor_hb);
	options = memnew(MenuButton);
	options->set_switch_on_hover(true);
	particles_editor_hb->add_child(options);
	particles_editor_hb->hide();

	options->set_text(TTR("CPUParticles3D"));
	options->get_popup()->add_shortcut(ED_GET_SHORTCUT("particles/restart_emission"), MENU_OPTION_RESTART);
	options->get_popup()->add_item(TTR("Generate AABB"), MENU_OPTION_GENERATE_AABB);
	options->get_popup()->add_item(TTR("Create Emission Points From Node"), MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_NODE);
	options->get_popup()->add_item(TTR("Convert to GPUParticles3D"), MENU_OPTION_CONVERT_TO_GPU_PARTICLES);
	options->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &CPUParticles3DEditor::_menu_option));

	generate_aabb = memnew(ConfirmationDialog);
	generate_aabb->set_title(TTR("Generate Visibility AABB"));
	VBoxContainer *genvb = memnew(VBoxContainer);
	generate_aabb->add_child(genvb);
	generate_seconds = memnew(SpinBox);
	genvb->add_margin_child(TTR("Generation Time (sec):"), generate_seconds);
	generate_seconds->set_min(0.1);
	generate_seconds->set_max(25);
	generate_seconds->set_value(2);

	add_child(generate_aabb);

	generate_aabb->connect(SceneStringName(confirmed), callable_mp(this, &CPUParticles3DEditor::_generate_aabb));
}

void CPUParticles3DEditorPlugin::edit(Object *p_object) {
	particles_editor->edit(Object::cast_to<CPUParticles3D>(p_object));
}

bool CPUParticles3DEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("CPUParticles3D");
}

void CPUParticles3DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		particles_editor->show();
		particles_editor->particles_editor_hb->show();
	} else {
		particles_editor->particles_editor_hb->hide();
		particles_editor->hide();
		particles_editor->edit(nullptr);
	}
}

CPUParticles3DEditorPlugin::CPUParticles3DEditorPlugin() {
	particles_editor = memnew(CPUParticles3DEditor);
	EditorNode::get_singleton()->get_gui_base()->add_child(particles_editor);

	particles_editor->hide();
}

CPUParticles3DEditorPlugin::~CPUParticles3DEditorPlugin() {
}
