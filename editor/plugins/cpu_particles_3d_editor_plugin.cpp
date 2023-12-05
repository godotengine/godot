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
			ur->create_action(TTR("Convert to GPUParticles3D"));
			SceneTreeDock::get_singleton()->replace_node(node, gpu_particles);
			ur->commit_action(false);

		} break;
	}
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

void CPUParticles3DEditor::_bind_methods() {
}

CPUParticles3DEditor::CPUParticles3DEditor() {
	particles_editor_hb = memnew(HBoxContainer);
	Node3DEditor::get_singleton()->add_control_to_menu_panel(particles_editor_hb);
	options = memnew(MenuButton);
	options->set_switch_on_hover(true);
	particles_editor_hb->add_child(options);
	particles_editor_hb->hide();

	options->set_text(TTR("CPUParticles3D"));
	options->get_popup()->add_item(TTR("Restart"), MENU_OPTION_RESTART);
	options->get_popup()->add_item(TTR("Create Emission Points From Node"), MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_NODE);
	options->get_popup()->add_item(TTR("Convert to GPUParticles3D"), MENU_OPTION_CONVERT_TO_GPU_PARTICLES);
	options->get_popup()->connect("id_pressed", callable_mp(this, &CPUParticles3DEditor::_menu_option));
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
	EditorNode::get_singleton()->get_main_screen_control()->add_child(particles_editor);

	particles_editor->hide();
}

CPUParticles3DEditorPlugin::~CPUParticles3DEditorPlugin() {
}
