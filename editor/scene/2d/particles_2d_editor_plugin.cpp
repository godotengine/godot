/**************************************************************************/
/*  particles_2d_editor_plugin.cpp                                        */
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

#include "particles_2d_editor_plugin.h"

#include "core/io/image_loader.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/gui/editor_file_dialog.h"
#include "scene/2d/cpu_particles_2d.h"
#include "scene/2d/gpu_particles_2d.h"
#include "scene/gui/box_container.h"
#include "scene/gui/check_box.h"
#include "scene/gui/option_button.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/spin_box.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/particle_process_material.h"

void GPUParticles2DEditorPlugin::_menu_callback(int p_idx) {
	if (p_idx == MENU_GENERATE_VISIBILITY_RECT) {
		if (need_show_lifetime_dialog(generate_seconds)) {
			generate_visibility_rect->popup_centered();
		} else {
			_generate_visibility_rect();
		}
	} else {
		Particles2DEditorPlugin::_menu_callback(p_idx);
	}
}

void GPUParticles2DEditorPlugin::_add_menu_options(PopupMenu *p_menu) {
	Particles2DEditorPlugin::_add_menu_options(p_menu);
	p_menu->add_item(TTR("Generate Visibility Rect"), MENU_GENERATE_VISIBILITY_RECT);
}

void Particles2DEditorPlugin::_file_selected(const String &p_file) {
	source_emission_file = p_file;
	emission_mask->popup_centered();
}

void Particles2DEditorPlugin::_get_base_emission_mask(PackedVector2Array &r_valid_positions, PackedVector2Array &r_valid_normals, PackedByteArray &r_valid_colors, Vector2i &r_image_size) {
	Ref<Image> img;
	img.instantiate();
	Error err = ImageLoader::load_image(source_emission_file, img);
	ERR_FAIL_COND_MSG(err != OK, "Error loading image '" + source_emission_file + "'.");

	if (img->is_compressed()) {
		img->decompress();
	}
	img->convert(Image::FORMAT_RGBA8);
	ERR_FAIL_COND(img->get_format() != Image::FORMAT_RGBA8);
	Size2i s = img->get_size();
	ERR_FAIL_COND(s.width == 0 || s.height == 0);

	r_image_size = s;

	r_valid_positions.resize(s.width * s.height);

	EmissionMode emode = (EmissionMode)emission_mask_mode->get_selected();

	if (emode == EMISSION_MODE_BORDER_DIRECTED) {
		r_valid_normals.resize(s.width * s.height);
	}

	bool capture_colors = emission_colors->is_pressed();

	if (capture_colors) {
		r_valid_colors.resize(s.width * s.height * 4);
	}

	int vpc = 0;

	{
		Vector<uint8_t> img_data = img->get_data();
		const uint8_t *r = img_data.ptr();

		for (int i = 0; i < s.width; i++) {
			for (int j = 0; j < s.height; j++) {
				uint8_t a = r[(j * s.width + i) * 4 + 3];

				if (a > 128) {
					if (emode == EMISSION_MODE_SOLID) {
						if (capture_colors) {
							r_valid_colors.write[vpc * 4 + 0] = r[(j * s.width + i) * 4 + 0];
							r_valid_colors.write[vpc * 4 + 1] = r[(j * s.width + i) * 4 + 1];
							r_valid_colors.write[vpc * 4 + 2] = r[(j * s.width + i) * 4 + 2];
							r_valid_colors.write[vpc * 4 + 3] = r[(j * s.width + i) * 4 + 3];
						}
						r_valid_positions.write[vpc++] = Point2(i, j);

					} else {
						bool on_border = false;
						for (int x = i - 1; x <= i + 1; x++) {
							for (int y = j - 1; y <= j + 1; y++) {
								if (x < 0 || y < 0 || x >= s.width || y >= s.height || r[(y * s.width + x) * 4 + 3] <= 128) {
									on_border = true;
									break;
								}
							}

							if (on_border) {
								break;
							}
						}

						if (on_border) {
							r_valid_positions.write[vpc] = Point2(i, j);

							if (emode == EMISSION_MODE_BORDER_DIRECTED) {
								Vector2 normal;
								for (int x = i - 2; x <= i + 2; x++) {
									for (int y = j - 2; y <= j + 2; y++) {
										if (x == i && y == j) {
											continue;
										}

										if (x < 0 || y < 0 || x >= s.width || y >= s.height || r[(y * s.width + x) * 4 + 3] <= 128) {
											normal += Vector2(x - i, y - j).normalized();
										}
									}
								}

								normal.normalize();
								r_valid_normals.write[vpc] = normal;
							}

							if (capture_colors) {
								r_valid_colors.write[vpc * 4 + 0] = r[(j * s.width + i) * 4 + 0];
								r_valid_colors.write[vpc * 4 + 1] = r[(j * s.width + i) * 4 + 1];
								r_valid_colors.write[vpc * 4 + 2] = r[(j * s.width + i) * 4 + 2];
								r_valid_colors.write[vpc * 4 + 3] = r[(j * s.width + i) * 4 + 3];
							}

							vpc++;
						}
					}
				}
			}
		}
	}

	r_valid_positions.resize(vpc);
	if (!r_valid_normals.is_empty()) {
		r_valid_normals.resize(vpc);
	}
}

Particles2DEditorPlugin::Particles2DEditorPlugin() {
	file = memnew(EditorFileDialog);

	List<String> ext;
	ImageLoader::get_recognized_extensions(&ext);
	for (const String &E : ext) {
		file->add_filter("*." + E, E.to_upper());
	}

	file->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	EditorNode::get_singleton()->get_gui_base()->add_child(file);
	file->connect("file_selected", callable_mp(this, &Particles2DEditorPlugin::_file_selected));

	emission_mask = memnew(ConfirmationDialog);
	emission_mask->set_title(TTR("Load Emission Mask"));

	VBoxContainer *emvb = memnew(VBoxContainer);
	emission_mask->add_child(emvb);

	emission_mask_mode = memnew(OptionButton);
	emission_mask_mode->add_item(TTR("Solid Pixels"), EMISSION_MODE_SOLID);
	emission_mask_mode->add_item(TTR("Border Pixels"), EMISSION_MODE_BORDER);
	emission_mask_mode->add_item(TTR("Directed Border Pixels"), EMISSION_MODE_BORDER_DIRECTED);
	emvb->add_margin_child(TTR("Emission Mask"), emission_mask_mode);

	VBoxContainer *optionsvb = memnew(VBoxContainer);
	emvb->add_margin_child(TTR("Options"), optionsvb);

	emission_mask_centered = memnew(CheckBox(TTR("Centered")));
	optionsvb->add_child(emission_mask_centered);
	emission_colors = memnew(CheckBox(TTR("Capture Colors from Pixel")));
	optionsvb->add_child(emission_colors);

	EditorNode::get_singleton()->get_gui_base()->add_child(emission_mask);

	emission_mask->connect(SceneStringName(confirmed), callable_mp(this, &Particles2DEditorPlugin::_generate_emission_mask));
}

void Particles2DEditorPlugin::_set_show_gizmos(Node *p_node, bool p_show) {
	GPUParticles2D *gpu_particles = Object::cast_to<GPUParticles2D>(p_node);
	if (gpu_particles) {
		gpu_particles->set_show_gizmos(p_show);
	}
	CPUParticles2D *cpu_particles = Object::cast_to<CPUParticles2D>(p_node);
	if (cpu_particles) {
		cpu_particles->set_show_gizmos(p_show);
	}

	// The `selection_changed` signal is deferred. A node could be deleted before the signal is emitted.
	if (p_show) {
		p_node->connect(SceneStringName(tree_exiting), callable_mp(this, &Particles2DEditorPlugin::_node_removed).bind(p_node));
	} else {
		p_node->disconnect(SceneStringName(tree_exiting), callable_mp(this, &Particles2DEditorPlugin::_node_removed));
	}
}

void Particles2DEditorPlugin::_selection_changed() {
	List<Node *> current_selection = EditorNode::get_singleton()->get_editor_selection()->get_top_selected_node_list();
	if (selected_particles.is_empty() && current_selection.is_empty()) {
		return;
	}

	// Turn gizmos off for nodes that are no longer selected.
	for (List<Node *>::Element *E = selected_particles.front(); E;) {
		Node *node = E->get();
		List<Node *>::Element *N = E->next();
		if (current_selection.find(node) == nullptr) {
			_set_show_gizmos(node, false);
			selected_particles.erase(E);
		}
		E = N;
	}

	// Turn gizmos on for nodes that are newly selected.
	for (Node *node : current_selection) {
		if (selected_particles.find(node) == nullptr) {
			_set_show_gizmos(node, true);
			selected_particles.push_back(node);
		}
	}
}

void Particles2DEditorPlugin::_node_removed(Node *p_node) {
	List<Node *>::Element *E = selected_particles.find(p_node);
	if (E) {
		_set_show_gizmos(E->get(), false);
		selected_particles.erase(E);
	}
}

void GPUParticles2DEditorPlugin::_generate_visibility_rect() {
	GPUParticles2D *particles = Object::cast_to<GPUParticles2D>(edited_node);

	double time = generate_seconds->get_value();

	float running = 0.0;

	EditorProgress ep("gen_vrect", TTR("Generating Visibility Rect (Waiting for Particle Simulation)"), int(time));

	bool was_emitting = particles->is_emitting();
	if (!was_emitting) {
		particles->set_emitting(true);
		OS::get_singleton()->delay_usec(1000);
	}

	Rect2 rect;
	while (running < time) {
		uint64_t ticks = OS::get_singleton()->get_ticks_usec();
		ep.step(TTR("Generating..."), int(running), true);
		OS::get_singleton()->delay_usec(1000);

		Rect2 capture = particles->capture_rect();
		if (rect == Rect2()) {
			rect = capture;
		} else {
			rect = rect.merge(capture);
		}

		running += (OS::get_singleton()->get_ticks_usec() - ticks) / 1000000.0;
	}

	if (!was_emitting) {
		particles->set_emitting(false);
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Generate Visibility Rect"));
	undo_redo->add_do_method(particles, "set_visibility_rect", rect);
	undo_redo->add_undo_method(particles, "set_visibility_rect", particles->get_visibility_rect());
	undo_redo->commit_action();
}

void Particles2DEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			EditorNode::get_singleton()->get_editor_selection()->connect("selection_changed", callable_mp(this, &Particles2DEditorPlugin::_selection_changed));
		} break;
	}
}

void Particles2DEditorPlugin::_menu_callback(int p_idx) {
	if (p_idx == MENU_LOAD_EMISSION_MASK) {
		GPUParticles2D *particles = Object::cast_to<GPUParticles2D>(edited_node);
		if (particles && particles->get_process_material().is_null()) {
			EditorNode::get_singleton()->show_warning(TTR("Loading emission mask requires ParticleProcessMaterial."));
			return;
		}

		file->popup_file_dialog();
	} else {
		ParticlesEditorPlugin::_menu_callback(p_idx);
	}
}

void Particles2DEditorPlugin::_add_menu_options(PopupMenu *p_menu) {
	p_menu->add_item(TTR("Load Emission Mask"), MENU_LOAD_EMISSION_MASK);
}

Node *GPUParticles2DEditorPlugin::_convert_particles() {
	GPUParticles2D *particles = Object::cast_to<GPUParticles2D>(edited_node);

	CPUParticles2D *cpu_particles = memnew(CPUParticles2D);
	cpu_particles->convert_from_particles(particles);
	cpu_particles->set_name(particles->get_name());
	cpu_particles->set_transform(particles->get_transform());
	cpu_particles->set_visible(particles->is_visible());
	cpu_particles->set_process_mode(particles->get_process_mode());
	cpu_particles->set_z_index(particles->get_z_index());
	return cpu_particles;
}

void GPUParticles2DEditorPlugin::_generate_emission_mask() {
	GPUParticles2D *particles = Object::cast_to<GPUParticles2D>(edited_node);
	Ref<ParticleProcessMaterial> pm = particles->get_process_material();
	ERR_FAIL_COND(pm.is_null());

	PackedVector2Array valid_positions;
	PackedVector2Array valid_normals;
	PackedByteArray valid_colors;
	Vector2i image_size;
	_get_base_emission_mask(valid_positions, valid_normals, valid_colors, image_size);

	ERR_FAIL_COND_MSG(valid_positions.is_empty(), "No pixels with transparency > 128 in image...");

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Load Emission Mask"));
	ParticleProcessMaterial *pmptr = pm.ptr();

	Vector<uint8_t> texdata;

	int vpc = valid_positions.size();
	int w = 2048;
	int h = (vpc / 2048) + 1;

	texdata.resize(w * h * 2 * sizeof(float));

	{
		Vector2 offset;
		if (emission_mask_centered->is_pressed()) {
			offset = Vector2(-image_size.width * 0.5, -image_size.height * 0.5);
		}

		uint8_t *tw = texdata.ptrw();
		float *twf = reinterpret_cast<float *>(tw);
		for (int i = 0; i < vpc; i++) {
			twf[i * 2 + 0] = valid_positions[i].x + offset.x;
			twf[i * 2 + 1] = valid_positions[i].y + offset.y;
		}
	}

	Ref<Image> img;
	img.instantiate();
	img->set_data(w, h, false, Image::FORMAT_RGF, texdata);
	undo_redo->add_do_property(pmptr, "emission_point_texture", ImageTexture::create_from_image(img));
	undo_redo->add_undo_property(pmptr, "emission_point_texture", pm->get_emission_point_texture());
	undo_redo->add_do_property(pmptr, "emission_point_count", vpc);
	undo_redo->add_undo_property(pmptr, "emission_point_count", pm->get_emission_point_count());

	if (emission_colors->is_pressed()) {
		Vector<uint8_t> colordata;
		colordata.resize(w * h * 4); //use RG texture

		{
			uint8_t *tw = colordata.ptrw();
			for (int i = 0; i < vpc * 4; i++) {
				tw[i] = valid_colors[i];
			}
		}

		img.instantiate();
		img->set_data(w, h, false, Image::FORMAT_RGBA8, colordata);
		undo_redo->add_do_property(pmptr, "emission_color_texture", ImageTexture::create_from_image(img));
		undo_redo->add_undo_property(pmptr, "emission_color_texture", pm->get_emission_color_texture());
	}

	if (!valid_normals.is_empty()) {
		undo_redo->add_do_property(pmptr, "emission_shape", ParticleProcessMaterial::EMISSION_SHAPE_DIRECTED_POINTS);
		undo_redo->add_undo_property(pmptr, "emission_shape", pm->get_emission_shape());
		pm->set_emission_shape(ParticleProcessMaterial::EMISSION_SHAPE_DIRECTED_POINTS);

		Vector<uint8_t> normdata;
		normdata.resize(w * h * 2 * sizeof(float)); //use RG texture

		{
			uint8_t *tw = normdata.ptrw();
			float *twf = reinterpret_cast<float *>(tw);
			for (int i = 0; i < vpc; i++) {
				twf[i * 2 + 0] = valid_normals[i].x;
				twf[i * 2 + 1] = valid_normals[i].y;
			}
		}

		img.instantiate();
		img->set_data(w, h, false, Image::FORMAT_RGF, normdata);
		undo_redo->add_do_property(pmptr, "emission_normal_texture", ImageTexture::create_from_image(img));
		undo_redo->add_undo_property(pmptr, "emission_normal_texture", pm->get_emission_normal_texture());
	} else {
		undo_redo->add_do_property(pmptr, "emission_shape", ParticleProcessMaterial::EMISSION_SHAPE_POINTS);
		undo_redo->add_undo_property(pmptr, "emission_shape", pm->get_emission_shape());
	}
	undo_redo->commit_action();
}

GPUParticles2DEditorPlugin::GPUParticles2DEditorPlugin() {
	handled_type = TTRC("GPUParticles2D");
	conversion_option_name = TTR("Convert to CPUParticles2D");

	generate_visibility_rect = memnew(ConfirmationDialog);
	generate_visibility_rect->set_title(TTR("Generate Visibility Rect"));

	VBoxContainer *genvb = memnew(VBoxContainer);
	generate_visibility_rect->add_child(genvb);

	generate_seconds = memnew(SpinBox);
	generate_seconds->set_min(0.1);
	generate_seconds->set_max(25);
	generate_seconds->set_value(2);
	genvb->add_margin_child(TTR("Generation Time (sec):"), generate_seconds);

	EditorNode::get_singleton()->get_gui_base()->add_child(generate_visibility_rect);

	generate_visibility_rect->connect(SceneStringName(confirmed), callable_mp(this, &GPUParticles2DEditorPlugin::_generate_visibility_rect));
}

Node *CPUParticles2DEditorPlugin::_convert_particles() {
	CPUParticles2D *particles = Object::cast_to<CPUParticles2D>(edited_node);

	GPUParticles2D *gpu_particles = memnew(GPUParticles2D);
	gpu_particles->convert_from_particles(particles);
	gpu_particles->set_name(particles->get_name());
	gpu_particles->set_transform(particles->get_transform());
	gpu_particles->set_visible(particles->is_visible());
	gpu_particles->set_process_mode(particles->get_process_mode());
	return gpu_particles;
}

void CPUParticles2DEditorPlugin::_generate_emission_mask() {
	CPUParticles2D *particles = Object::cast_to<CPUParticles2D>(edited_node);

	PackedVector2Array valid_positions;
	PackedVector2Array valid_normals;
	PackedByteArray valid_colors;
	Vector2i image_size;
	_get_base_emission_mask(valid_positions, valid_normals, valid_colors, image_size);

	ERR_FAIL_COND_MSG(valid_positions.is_empty(), "No pixels with transparency > 128 in image...");

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Load Emission Mask"));

	int vpc = valid_positions.size();
	if (emission_colors->is_pressed()) {
		PackedColorArray pca;
		pca.resize(vpc);
		Color *pcaw = pca.ptrw();
		for (int i = 0; i < vpc; i += 1) {
			Color color;
			color.r = valid_colors[i * 4 + 0] / 255.0f;
			color.g = valid_colors[i * 4 + 1] / 255.0f;
			color.b = valid_colors[i * 4 + 2] / 255.0f;
			color.a = valid_colors[i * 4 + 3] / 255.0f;
			pcaw[i] = color;
		}
		undo_redo->add_do_property(particles, "emission_colors", pca);
		undo_redo->add_undo_property(particles, "emission_colors", particles->get_emission_colors());
	}

	if (!valid_normals.is_empty()) {
		undo_redo->add_do_property(particles, "emission_shape", CPUParticles2D::EMISSION_SHAPE_DIRECTED_POINTS);
		undo_redo->add_undo_property(particles, "emission_shape", particles->get_emission_shape());
		PackedVector2Array norms;
		norms.resize(valid_normals.size());
		Vector2 *normsw = norms.ptrw();
		for (int i = 0; i < valid_normals.size(); i += 1) {
			normsw[i] = valid_normals[i];
		}
		undo_redo->add_do_property(particles, "emission_normals", norms);
		undo_redo->add_undo_property(particles, "emission_normals", particles->get_emission_normals());
	} else {
		undo_redo->add_do_property(particles, "emission_shape", CPUParticles2D::EMISSION_SHAPE_POINTS);
		undo_redo->add_undo_property(particles, "emission_shape", particles->get_emission_shape());
	}

	{
		Vector2 offset;
		if (emission_mask_centered->is_pressed()) {
			offset = Vector2(-image_size.width * 0.5, -image_size.height * 0.5);
		}

		PackedVector2Array points;
		points.resize(valid_positions.size());
		Vector2 *pointsw = points.ptrw();
		for (int i = 0; i < valid_positions.size(); i += 1) {
			pointsw[i] = valid_positions[i] + offset;
		}
		undo_redo->add_do_property(particles, "emission_points", points);
		undo_redo->add_undo_property(particles, "emission_shape", particles->get_emission_points());
	}
	undo_redo->commit_action();
}

CPUParticles2DEditorPlugin::CPUParticles2DEditorPlugin() {
	handled_type = TTRC("CPUParticles2D");
	conversion_option_name = TTR("Convert to GPUParticles2D");
}
