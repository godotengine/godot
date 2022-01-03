/*************************************************************************/
/*  gpu_particles_3d_editor_plugin.cpp                                   */
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

#include "gpu_particles_3d_editor_plugin.h"

#include "core/io/resource_loader.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/cpu_particles_3d.h"
#include "scene/resources/particles_material.h"

bool GPUParticles3DEditorBase::_generate(Vector<Vector3> &points, Vector<Vector3> &normals) {
	bool use_normals = emission_fill->get_selected() == 1;

	if (emission_fill->get_selected() < 2) {
		float area_accum = 0;
		Map<float, int> triangle_area_map;

		for (int i = 0; i < geometry.size(); i++) {
			float area = geometry[i].get_area();
			if (area < CMP_EPSILON) {
				continue;
			}
			triangle_area_map[area_accum] = i;
			area_accum += area;
		}

		if (!triangle_area_map.size() || area_accum == 0) {
			EditorNode::get_singleton()->show_warning(TTR("The geometry's faces don't contain any area."));
			return false;
		}

		int emissor_count = emission_amount->get_value();

		for (int i = 0; i < emissor_count; i++) {
			float areapos = Math::random(0.0f, area_accum);

			Map<float, int>::Element *E = triangle_area_map.find_closest(areapos);
			ERR_FAIL_COND_V(!E, false);
			int index = E->get();
			ERR_FAIL_INDEX_V(index, geometry.size(), false);

			// ok FINALLY get face
			Face3 face = geometry[index];
			//now compute some position inside the face...

			Vector3 pos = face.get_random_point_inside();

			points.push_back(pos);

			if (use_normals) {
				Vector3 normal = face.get_plane().normal;
				normals.push_back(normal);
			}
		}
	} else {
		int gcount = geometry.size();

		if (gcount == 0) {
			EditorNode::get_singleton()->show_warning(TTR("The geometry doesn't contain any faces."));
			return false;
		}

		const Face3 *r = geometry.ptr();

		AABB aabb;

		for (int i = 0; i < gcount; i++) {
			for (int j = 0; j < 3; j++) {
				if (i == 0 && j == 0) {
					aabb.position = r[i].vertex[j];
				} else {
					aabb.expand_to(r[i].vertex[j]);
				}
			}
		}

		int emissor_count = emission_amount->get_value();

		for (int i = 0; i < emissor_count; i++) {
			int attempts = 5;

			for (int j = 0; j < attempts; j++) {
				Vector3 dir;
				dir[Math::rand() % 3] = 1.0;
				Vector3 ofs = (Vector3(1, 1, 1) - dir) * Vector3(Math::randf(), Math::randf(), Math::randf()) * aabb.size + aabb.position;

				Vector3 ofsv = ofs + aabb.size * dir;

				//space it a little
				ofs -= dir;
				ofsv += dir;

				float max = -1e7, min = 1e7;

				for (int k = 0; k < gcount; k++) {
					const Face3 &f3 = r[k];

					Vector3 res;
					if (f3.intersects_segment(ofs, ofsv, &res)) {
						res -= ofs;
						float d = dir.dot(res);

						if (d < min) {
							min = d;
						}
						if (d > max) {
							max = d;
						}
					}
				}

				if (max < min) {
					continue; //lost attempt
				}

				float val = min + (max - min) * Math::randf();

				Vector3 point = ofs + dir * val;

				points.push_back(point);
				break;
			}
		}
	}

	return true;
}

void GPUParticles3DEditorBase::_node_selected(const NodePath &p_path) {
	Node *sel = get_node(p_path);
	if (!sel) {
		return;
	}

	if (!sel->is_class("Node3D")) {
		EditorNode::get_singleton()->show_warning(vformat(TTR("\"%s\" doesn't inherit from Node3D."), sel->get_name()));
		return;
	}

	VisualInstance3D *vi = Object::cast_to<VisualInstance3D>(sel);
	if (!vi) {
		EditorNode::get_singleton()->show_warning(vformat(TTR("\"%s\" doesn't contain geometry."), sel->get_name()));
		return;
	}

	geometry = vi->get_faces(VisualInstance3D::FACES_SOLID);

	if (geometry.size() == 0) {
		EditorNode::get_singleton()->show_warning(vformat(TTR("\"%s\" doesn't contain face geometry."), sel->get_name()));
		return;
	}

	Transform3D geom_xform = base_node->get_global_transform().affine_inverse() * vi->get_global_transform();

	int gc = geometry.size();
	Face3 *w = geometry.ptrw();

	for (int i = 0; i < gc; i++) {
		for (int j = 0; j < 3; j++) {
			w[i].vertex[j] = geom_xform.xform(w[i].vertex[j]);
		}
	}

	emission_dialog->popup_centered(Size2(300, 130));
}

void GPUParticles3DEditorBase::_bind_methods() {
}

GPUParticles3DEditorBase::GPUParticles3DEditorBase() {
	emission_dialog = memnew(ConfirmationDialog);
	emission_dialog->set_title(TTR("Create Emitter"));
	add_child(emission_dialog);
	VBoxContainer *emd_vb = memnew(VBoxContainer);
	emission_dialog->add_child(emd_vb);

	emission_amount = memnew(SpinBox);
	emission_amount->set_min(1);
	emission_amount->set_max(100000);
	emission_amount->set_value(512);
	emd_vb->add_margin_child(TTR("Emission Points:"), emission_amount);

	emission_fill = memnew(OptionButton);
	emission_fill->add_item(TTR("Surface Points"));
	emission_fill->add_item(TTR("Surface Points+Normal (Directed)"));
	emission_fill->add_item(TTR("Volume"));
	emd_vb->add_margin_child(TTR("Emission Source: "), emission_fill);

	emission_dialog->get_ok_button()->set_text(TTR("Create"));
	emission_dialog->connect("confirmed", callable_mp(this, &GPUParticles3DEditorBase::_generate_emission_points));

	emission_tree_dialog = memnew(SceneTreeDialog);
	add_child(emission_tree_dialog);
	emission_tree_dialog->connect("selected", callable_mp(this, &GPUParticles3DEditorBase::_node_selected));
}

void GPUParticles3DEditor::_node_removed(Node *p_node) {
	if (p_node == node) {
		node = nullptr;
		hide();
	}
}

void GPUParticles3DEditor::_notification(int p_notification) {
	if (p_notification == NOTIFICATION_ENTER_TREE) {
		options->set_icon(options->get_popup()->get_theme_icon(SNAME("GPUParticles3D"), SNAME("EditorIcons")));
		get_tree()->connect("node_removed", callable_mp(this, &GPUParticles3DEditor::_node_removed));
	}
}

void GPUParticles3DEditor::_menu_option(int p_option) {
	switch (p_option) {
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
		case MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_NODE: {
			Ref<ParticlesMaterial> material = node->get_process_material();
			if (material.is_null()) {
				EditorNode::get_singleton()->show_warning(TTR("A processor material of type 'ParticlesMaterial' is required."));
				return;
			}

			emission_tree_dialog->popup_scenetree_dialog();

		} break;
		case MENU_OPTION_CONVERT_TO_CPU_PARTICLES: {
			CPUParticles3D *cpu_particles = memnew(CPUParticles3D);
			cpu_particles->convert_from_particles(node);
			cpu_particles->set_name(node->get_name());
			cpu_particles->set_transform(node->get_transform());
			cpu_particles->set_visible(node->is_visible());
			cpu_particles->set_process_mode(node->get_process_mode());

			UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
			ur->create_action(TTR("Convert to CPUParticles3D"));
			ur->add_do_method(EditorNode::get_singleton()->get_scene_tree_dock(), "replace_node", node, cpu_particles, true, false);
			ur->add_do_reference(cpu_particles);
			ur->add_undo_method(EditorNode::get_singleton()->get_scene_tree_dock(), "replace_node", cpu_particles, node, false, false);
			ur->add_undo_reference(node);
			ur->commit_action();

		} break;
		case MENU_OPTION_RESTART: {
			node->restart();

		} break;
	}
}

void GPUParticles3DEditor::_generate_aabb() {
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
		ep.step("Generating...", int(running), true);
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

	UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
	ur->create_action(TTR("Generate Visibility AABB"));
	ur->add_do_method(node, "set_visibility_aabb", rect);
	ur->add_undo_method(node, "set_visibility_aabb", node->get_visibility_aabb());
	ur->commit_action();
}

void GPUParticles3DEditor::edit(GPUParticles3D *p_particles) {
	base_node = p_particles;
	node = p_particles;
}

void GPUParticles3DEditor::_generate_emission_points() {
	/// hacer codigo aca
	Vector<Vector3> points;
	Vector<Vector3> normals;

	if (!_generate(points, normals)) {
		return;
	}

	int point_count = points.size();

	int w = 2048;
	int h = (point_count / 2048) + 1;

	Vector<uint8_t> point_img;
	point_img.resize(w * h * 3 * sizeof(float));

	{
		uint8_t *iw = point_img.ptrw();
		memset(iw, 0, w * h * 3 * sizeof(float));
		const Vector3 *r = points.ptr();
		float *wf = (float *)iw;
		for (int i = 0; i < point_count; i++) {
			wf[i * 3 + 0] = r[i].x;
			wf[i * 3 + 1] = r[i].y;
			wf[i * 3 + 2] = r[i].z;
		}
	}

	Ref<Image> image = memnew(Image(w, h, false, Image::FORMAT_RGBF, point_img));

	Ref<ImageTexture> tex;
	tex.instantiate();
	tex->create_from_image(image);

	Ref<ParticlesMaterial> material = node->get_process_material();
	ERR_FAIL_COND(material.is_null());

	if (normals.size() > 0) {
		material->set_emission_shape(ParticlesMaterial::EMISSION_SHAPE_DIRECTED_POINTS);
		material->set_emission_point_count(point_count);
		material->set_emission_point_texture(tex);

		Vector<uint8_t> point_img2;
		point_img2.resize(w * h * 3 * sizeof(float));

		{
			uint8_t *iw = point_img2.ptrw();
			memset(iw, 0, w * h * 3 * sizeof(float));
			const Vector3 *r = normals.ptr();
			float *wf = (float *)iw;
			for (int i = 0; i < point_count; i++) {
				wf[i * 3 + 0] = r[i].x;
				wf[i * 3 + 1] = r[i].y;
				wf[i * 3 + 2] = r[i].z;
			}
		}

		Ref<Image> image2 = memnew(Image(w, h, false, Image::FORMAT_RGBF, point_img2));

		Ref<ImageTexture> tex2;
		tex2.instantiate();
		tex2->create_from_image(image2);

		material->set_emission_normal_texture(tex2);
	} else {
		material->set_emission_shape(ParticlesMaterial::EMISSION_SHAPE_POINTS);
		material->set_emission_point_count(point_count);
		material->set_emission_point_texture(tex);
	}
}

void GPUParticles3DEditor::_bind_methods() {
}

GPUParticles3DEditor::GPUParticles3DEditor() {
	node = nullptr;
	particles_editor_hb = memnew(HBoxContainer);
	Node3DEditor::get_singleton()->add_control_to_menu_panel(particles_editor_hb);
	options = memnew(MenuButton);
	options->set_switch_on_hover(true);
	particles_editor_hb->add_child(options);
	particles_editor_hb->hide();

	options->set_text(TTR("GPUParticles3D"));
	options->get_popup()->add_item(TTR("Restart"), MENU_OPTION_RESTART);
	options->get_popup()->add_item(TTR("Generate AABB"), MENU_OPTION_GENERATE_AABB);
	options->get_popup()->add_item(TTR("Create Emission Points From Node"), MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_NODE);
	options->get_popup()->add_item(TTR("Convert to CPUParticles3D"), MENU_OPTION_CONVERT_TO_CPU_PARTICLES);

	options->get_popup()->connect("id_pressed", callable_mp(this, &GPUParticles3DEditor::_menu_option));

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

	generate_aabb->connect("confirmed", callable_mp(this, &GPUParticles3DEditor::_generate_aabb));
}

void GPUParticles3DEditorPlugin::edit(Object *p_object) {
	particles_editor->edit(Object::cast_to<GPUParticles3D>(p_object));
}

bool GPUParticles3DEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("GPUParticles3D");
}

void GPUParticles3DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		particles_editor->show();
		particles_editor->particles_editor_hb->show();
	} else {
		particles_editor->particles_editor_hb->hide();
		particles_editor->hide();
		particles_editor->edit(nullptr);
	}
}

GPUParticles3DEditorPlugin::GPUParticles3DEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	particles_editor = memnew(GPUParticles3DEditor);
	editor->get_main_control()->add_child(particles_editor);

	particles_editor->hide();
}

GPUParticles3DEditorPlugin::~GPUParticles3DEditorPlugin() {
}
