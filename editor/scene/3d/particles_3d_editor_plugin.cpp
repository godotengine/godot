/**************************************************************************/
/*  particles_3d_editor_plugin.cpp                                        */
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

#include "particles_3d_editor_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/scene/scene_tree_editor.h"
#include "scene/3d/cpu_particles_3d.h"
#include "scene/3d/gpu_particles_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/gui/box_container.h"
#include "scene/gui/option_button.h"
#include "scene/gui/spin_box.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/particle_process_material.h"

void Particles3DEditorPlugin::_generate_aabb() {
	double time = generate_seconds->get_value();

	double running = 0.0;

	EditorProgress ep("gen_aabb", TTR("Generating Visibility AABB (Waiting for Particle Simulation)"), int(time));

	bool was_emitting = edited_node->get("emitting");
	if (!was_emitting) {
		edited_node->set("emitting", true);
		OS::get_singleton()->delay_usec(1000);
	}

	AABB rect;
	Callable capture_aabb = Callable(edited_node, "capture_aabb");

	while (running < time) {
		uint64_t ticks = OS::get_singleton()->get_ticks_usec();
		ep.step(TTR("Generating..."), int(running), true);
		OS::get_singleton()->delay_usec(1000);

		AABB capture = capture_aabb.call();
		if (rect == AABB()) {
			rect = capture;
		} else {
			rect.merge_with(capture);
		}

		running += (OS::get_singleton()->get_ticks_usec() - ticks) / 1000000.0;
	}

	if (!was_emitting) {
		edited_node->set("emitting", false);
	}

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(TTR("Generate Visibility AABB"));
	ur->add_do_property(edited_node, "visibility_aabb", rect);
	ur->add_undo_property(edited_node, "visibility_aabb", edited_node->get("visibility_aabb"));
	ur->commit_action();
}

void Particles3DEditorPlugin::_node_selected(const NodePath &p_path) {
	Node *sel = get_node(p_path);
	if (!sel) {
		return;
	}

	if (!sel->is_class("Node3D")) {
		EditorNode::get_singleton()->show_warning(vformat(TTR("\"%s\" doesn't inherit from Node3D."), sel->get_name()));
		return;
	}

	MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(sel);
	if (!mi || mi->get_mesh().is_null()) {
		EditorNode::get_singleton()->show_warning(vformat(TTR("\"%s\" doesn't contain geometry."), sel->get_name()));
		return;
	}

	geometry = mi->get_mesh()->get_faces();
	if (geometry.is_empty()) {
		EditorNode::get_singleton()->show_warning(vformat(TTR("\"%s\" doesn't contain face geometry."), sel->get_name()));
		return;
	}

	Transform3D geom_xform = edited_node->get("global_transform");
	geom_xform = geom_xform.affine_inverse() * mi->get_global_transform();
	int gc = geometry.size();
	Face3 *w = geometry.ptrw();

	for (int i = 0; i < gc; i++) {
		for (int j = 0; j < 3; j++) {
			w[i].vertex[j] = geom_xform.xform(w[i].vertex[j]);
		}
	}
	emission_dialog->popup_centered(Size2(300, 130));
}

void Particles3DEditorPlugin::_menu_callback(int p_idx) {
	switch (p_idx) {
		case MENU_OPTION_GENERATE_AABB: {
			if (need_show_lifetime_dialog(generate_seconds)) {
				generate_aabb->popup_centered();
			} else {
				_generate_aabb();
			}
		} break;

		case MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_NODE: {
			if (_can_generate_points()) {
				emission_tree_dialog->popup_scenetree_dialog();
			}
		} break;

		default: {
			ParticlesEditorPlugin::_menu_callback(p_idx);
		}
	}
}

void Particles3DEditorPlugin::_add_menu_options(PopupMenu *p_menu) {
	p_menu->add_item(TTR("Generate AABB"), MENU_OPTION_GENERATE_AABB);
	p_menu->add_item(TTR("Create Emission Points From Node"), MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_NODE);
}

bool Particles3DEditorPlugin::_generate(Vector<Vector3> &r_points, Vector<Vector3> &r_normals) {
	bool use_normals = emission_fill->get_selected() == 1;

	if (emission_fill->get_selected() < 2) {
		float area_accum = 0;
		RBMap<float, int> triangle_area_map;

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

			RBMap<float, int>::Iterator E = triangle_area_map.find_closest(areapos);
			ERR_FAIL_COND_V(!E, false);
			int index = E->value;
			ERR_FAIL_INDEX_V(index, geometry.size(), false);

			// ok FINALLY get face
			Face3 face = geometry[index];
			//now compute some position inside the face...

			Vector3 pos = face.get_random_point_inside();

			r_points.push_back(pos);

			if (use_normals) {
				Vector3 normal = face.get_plane().normal;
				r_normals.push_back(normal);
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

				r_points.push_back(point);
				break;
			}
		}
	}
	return true;
}

Particles3DEditorPlugin::Particles3DEditorPlugin() {
	generate_aabb = memnew(ConfirmationDialog);
	generate_aabb->set_title(TTR("Generate Visibility AABB"));

	VBoxContainer *genvb = memnew(VBoxContainer);
	generate_aabb->add_child(genvb);

	generate_seconds = memnew(SpinBox);
	generate_seconds->set_accessibility_name(TTRC("Generation Time (sec)"));
	generate_seconds->set_min(0.1);
	generate_seconds->set_max(25);
	generate_seconds->set_value(2);
	genvb->add_margin_child(TTR("Generation Time (sec):"), generate_seconds);

	EditorNode::get_singleton()->get_gui_base()->add_child(generate_aabb);

	generate_aabb->connect(SceneStringName(confirmed), callable_mp(this, &Particles3DEditorPlugin::_generate_aabb));

	emission_tree_dialog = memnew(SceneTreeDialog);
	Vector<StringName> valid_types;
	valid_types.push_back("MeshInstance3D");
	emission_tree_dialog->set_valid_types(valid_types);
	EditorNode::get_singleton()->get_gui_base()->add_child(emission_tree_dialog);
	emission_tree_dialog->connect("selected", callable_mp(this, &Particles3DEditorPlugin::_node_selected));

	emission_dialog = memnew(ConfirmationDialog);
	emission_dialog->set_title(TTR("Create Emitter"));
	EditorNode::get_singleton()->get_gui_base()->add_child(emission_dialog);

	VBoxContainer *emd_vb = memnew(VBoxContainer);
	emission_dialog->add_child(emd_vb);

	emission_amount = memnew(SpinBox);
	emission_amount->set_accessibility_name(TTRC("Emission Points:"));
	emission_amount->set_min(1);
	emission_amount->set_max(100000);
	emission_amount->set_value(512);
	emd_vb->add_margin_child(TTR("Emission Points:"), emission_amount);

	emission_fill = memnew(OptionButton);
	emission_fill->set_accessibility_name(TTRC("Emission Source:"));
	emission_fill->add_item(TTR("Surface Points"));
	emission_fill->add_item(TTR("Surface Points+Normal (Directed)"));
	emission_fill->add_item(TTR("Volume"));
	emd_vb->add_margin_child(TTR("Emission Source:"), emission_fill);

	emission_dialog->set_ok_button_text(TTR("Create"));
	emission_dialog->connect(SceneStringName(confirmed), callable_mp(this, &Particles3DEditorPlugin::_generate_emission_points));
}

Node *GPUParticles3DEditorPlugin::_convert_particles() {
	GPUParticles3D *particles = Object::cast_to<GPUParticles3D>(edited_node);

	CPUParticles3D *cpu_particles = memnew(CPUParticles3D);
	cpu_particles->convert_from_particles(particles);
	cpu_particles->set_name(particles->get_name());
	cpu_particles->set_transform(particles->get_transform());
	cpu_particles->set_visible(particles->is_visible());
	cpu_particles->set_process_mode(particles->get_process_mode());
	return cpu_particles;
}

bool GPUParticles3DEditorPlugin::_can_generate_points() const {
	GPUParticles3D *particles = Object::cast_to<GPUParticles3D>(edited_node);
	Ref<ParticleProcessMaterial> mat = particles->get_process_material();
	if (mat.is_null()) {
		EditorNode::get_singleton()->show_warning(TTR("A processor material of type 'ParticleProcessMaterial' is required."));
		return false;
	}
	return true;
}

void GPUParticles3DEditorPlugin::_generate_emission_points() {
	GPUParticles3D *particles = Object::cast_to<GPUParticles3D>(edited_node);

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
		float *wf = reinterpret_cast<float *>(iw);
		for (int i = 0; i < point_count; i++) {
			wf[i * 3 + 0] = r[i].x;
			wf[i * 3 + 1] = r[i].y;
			wf[i * 3 + 2] = r[i].z;
		}
	}

	Ref<Image> image = memnew(Image(w, h, false, Image::FORMAT_RGBF, point_img));
	Ref<ImageTexture> tex = ImageTexture::create_from_image(image);

	Ref<ParticleProcessMaterial> mat = particles->get_process_material();
	ERR_FAIL_COND(mat.is_null());

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Create Emission Points"));
	ParticleProcessMaterial *matptr = mat.ptr();

	if (!normals.is_empty()) {
		undo_redo->add_do_property(matptr, "emission_shape", ParticleProcessMaterial::EMISSION_SHAPE_DIRECTED_POINTS);
		undo_redo->add_undo_property(matptr, "emission_shape", matptr->get_emission_shape());

		Vector<uint8_t> point_img2;
		point_img2.resize(w * h * 3 * sizeof(float));

		{
			uint8_t *iw = point_img2.ptrw();
			memset(iw, 0, w * h * 3 * sizeof(float));
			const Vector3 *r = normals.ptr();
			float *wf = reinterpret_cast<float *>(iw);
			for (int i = 0; i < point_count; i++) {
				wf[i * 3 + 0] = r[i].x;
				wf[i * 3 + 1] = r[i].y;
				wf[i * 3 + 2] = r[i].z;
			}
		}

		Ref<Image> image2 = memnew(Image(w, h, false, Image::FORMAT_RGBF, point_img2));
		undo_redo->add_do_property(matptr, "emission_normal_texture", ImageTexture::create_from_image(image2));
		undo_redo->add_undo_property(matptr, "emission_normal_texture", matptr->get_emission_normal_texture());
	} else {
		undo_redo->add_do_property(matptr, "emission_shape", ParticleProcessMaterial::EMISSION_SHAPE_POINTS);
		undo_redo->add_undo_property(matptr, "emission_shape", matptr->get_emission_shape());
	}
	undo_redo->add_do_property(matptr, "emission_point_count", point_count);
	undo_redo->add_undo_property(matptr, "emission_point_count", matptr->get_emission_point_count());
	undo_redo->add_do_property(matptr, "emission_point_texture", tex);
	undo_redo->add_undo_property(matptr, "emission_point_texture", matptr->get_emission_point_texture());
	undo_redo->commit_action();
}

GPUParticles3DEditorPlugin::GPUParticles3DEditorPlugin() {
	handled_type = TTRC("GPUParticles3D");
	conversion_option_name = TTR("Convert to CPUParticles3D");
}

Node *CPUParticles3DEditorPlugin::_convert_particles() {
	CPUParticles3D *particles = Object::cast_to<CPUParticles3D>(edited_node);

	GPUParticles3D *gpu_particles = memnew(GPUParticles3D);
	gpu_particles->convert_from_particles(particles);
	gpu_particles->set_name(particles->get_name());
	gpu_particles->set_transform(particles->get_transform());
	gpu_particles->set_visible(particles->is_visible());
	gpu_particles->set_process_mode(particles->get_process_mode());
	return gpu_particles;
}

void CPUParticles3DEditorPlugin::_generate_emission_points() {
	CPUParticles3D *particles = Object::cast_to<CPUParticles3D>(edited_node);

	/// hacer codigo aca
	Vector<Vector3> points;
	Vector<Vector3> normals;

	if (!_generate(points, normals)) {
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Create Emission Points"));

	if (normals.is_empty()) {
		undo_redo->add_do_property(particles, "emission_shape", ParticleProcessMaterial::EMISSION_SHAPE_POINTS);
		undo_redo->add_undo_property(particles, "emission_shape", particles->get_emission_shape());
	} else {
		undo_redo->add_do_property(particles, "emission_shape", ParticleProcessMaterial::EMISSION_SHAPE_DIRECTED_POINTS);
		undo_redo->add_undo_property(particles, "emission_shape", particles->get_emission_shape());
		undo_redo->add_do_property(particles, "emission_normals", normals);
		undo_redo->add_undo_property(particles, "emission_normals", particles->get_emission_normals());
	}
	undo_redo->add_do_property(particles, "emission_points", points);
	undo_redo->add_undo_property(particles, "emission_points", particles->get_emission_points());
	undo_redo->commit_action();
}

CPUParticles3DEditorPlugin::CPUParticles3DEditorPlugin() {
	handled_type = TTRC("CPUParticles3D");
	conversion_option_name = TTR("Convert to GPUParticles3D");
}
