/**************************************************************************/
/*  cpu_particles_2d_editor_plugin.cpp                                    */
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

#include "cpu_particles_2d_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "core/io/image_loader.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/scene_tree_dock.h"
#include "scene/2d/cpu_particles_2d.h"
#include "scene/2d/gpu_particles_2d.h"
#include "scene/gui/check_box.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/option_button.h"
#include "scene/gui/separator.h"
#include "scene/gui/spin_box.h"
#include "scene/resources/particle_process_material.h"

void CPUParticles2DEditorPlugin::edit(Object *p_object) {
	particles = Object::cast_to<CPUParticles2D>(p_object);
}

bool CPUParticles2DEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("CPUParticles2D");
}

void CPUParticles2DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		toolbar->show();
	} else {
		toolbar->hide();
	}
}

void CPUParticles2DEditorPlugin::_file_selected(const String &p_file) {
	source_emission_file = p_file;
	emission_mask->popup_centered();
}

void CPUParticles2DEditorPlugin::_menu_callback(int p_idx) {
	switch (p_idx) {
		case MENU_LOAD_EMISSION_MASK: {
			file->popup_file_dialog();
		} break;
		case MENU_CLEAR_EMISSION_MASK: {
			emission_mask->popup_centered();
		} break;
		case MENU_RESTART: {
			particles->restart();
		} break;
		case MENU_CONVERT_TO_GPU_PARTICLES: {
			GPUParticles2D *gpu_particles = memnew(GPUParticles2D);
			gpu_particles->convert_from_particles(particles);
			gpu_particles->set_name(particles->get_name());
			gpu_particles->set_transform(particles->get_transform());
			gpu_particles->set_visible(particles->is_visible());
			gpu_particles->set_process_mode(particles->get_process_mode());

			EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
			ur->create_action(TTR("Convert to GPUParticles3D"));
			SceneTreeDock::get_singleton()->replace_node(particles, gpu_particles);
			ur->commit_action(false);
		} break;
	}
}

void CPUParticles2DEditorPlugin::_generate_emission_mask() {
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

	Vector<Point2> valid_positions;
	Vector<Point2> valid_normals;
	Vector<uint8_t> valid_colors;

	valid_positions.resize(s.width * s.height);

	EmissionMode emode = (EmissionMode)emission_mask_mode->get_selected();

	if (emode == EMISSION_MODE_BORDER_DIRECTED) {
		valid_normals.resize(s.width * s.height);
	}

	bool capture_colors = emission_colors->is_pressed();

	if (capture_colors) {
		valid_colors.resize(s.width * s.height * 4);
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
							valid_colors.write[vpc * 4 + 0] = r[(j * s.width + i) * 4 + 0];
							valid_colors.write[vpc * 4 + 1] = r[(j * s.width + i) * 4 + 1];
							valid_colors.write[vpc * 4 + 2] = r[(j * s.width + i) * 4 + 2];
							valid_colors.write[vpc * 4 + 3] = r[(j * s.width + i) * 4 + 3];
						}
						valid_positions.write[vpc++] = Point2(i, j);

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
							valid_positions.write[vpc] = Point2(i, j);

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
								valid_normals.write[vpc] = normal;
							}

							if (capture_colors) {
								valid_colors.write[vpc * 4 + 0] = r[(j * s.width + i) * 4 + 0];
								valid_colors.write[vpc * 4 + 1] = r[(j * s.width + i) * 4 + 1];
								valid_colors.write[vpc * 4 + 2] = r[(j * s.width + i) * 4 + 2];
								valid_colors.write[vpc * 4 + 3] = r[(j * s.width + i) * 4 + 3];
							}

							vpc++;
						}
					}
				}
			}
		}
	}

	valid_positions.resize(vpc);
	if (valid_normals.size()) {
		valid_normals.resize(vpc);
	}

	ERR_FAIL_COND_MSG(valid_positions.size() == 0, "No pixels with transparency > 128 in image...");

	if (capture_colors) {
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
		particles->set_emission_colors(pca);
	}

	if (valid_normals.size()) {
		particles->set_emission_shape(CPUParticles2D::EMISSION_SHAPE_DIRECTED_POINTS);
		PackedVector2Array norms;
		norms.resize(valid_normals.size());
		Vector2 *normsw = norms.ptrw();
		for (int i = 0; i < valid_normals.size(); i += 1) {
			normsw[i] = valid_normals[i];
		}
		particles->set_emission_normals(norms);
	} else {
		particles->set_emission_shape(CPUParticles2D::EMISSION_SHAPE_POINTS);
	}

	{
		Vector2 offset;
		if (emission_mask_centered->is_pressed()) {
			offset = Vector2(-s.width * 0.5, -s.height * 0.5);
		}

		PackedVector2Array points;
		points.resize(valid_positions.size());
		Vector2 *pointsw = points.ptrw();
		for (int i = 0; i < valid_positions.size(); i += 1) {
			pointsw[i] = valid_positions[i] + offset;
		}
		particles->set_emission_points(points);
	}
}

void CPUParticles2DEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			menu->get_popup()->connect("id_pressed", callable_mp(this, &CPUParticles2DEditorPlugin::_menu_callback));
			menu->set_icon(epoints->get_editor_theme_icon(SNAME("CPUParticles2D")));
			file->connect("file_selected", callable_mp(this, &CPUParticles2DEditorPlugin::_file_selected));
		} break;
	}
}

void CPUParticles2DEditorPlugin::_bind_methods() {
}

CPUParticles2DEditorPlugin::CPUParticles2DEditorPlugin() {
	particles = nullptr;

	toolbar = memnew(HBoxContainer);
	add_control_to_container(CONTAINER_CANVAS_EDITOR_MENU, toolbar);
	toolbar->hide();

	menu = memnew(MenuButton);
	menu->get_popup()->add_item(TTR("Restart"), MENU_RESTART);
	menu->get_popup()->add_item(TTR("Load Emission Mask"), MENU_LOAD_EMISSION_MASK);
	menu->get_popup()->add_item(TTR("Convert to GPUParticles2D"), MENU_CONVERT_TO_GPU_PARTICLES);
	menu->set_text(TTR("CPUParticles2D"));
	menu->set_switch_on_hover(true);
	toolbar->add_child(menu);

	file = memnew(EditorFileDialog);
	List<String> ext;
	ImageLoader::get_recognized_extensions(&ext);
	for (const String &E : ext) {
		file->add_filter("*." + E, E.to_upper());
	}
	file->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	toolbar->add_child(file);

	epoints = memnew(SpinBox);
	epoints->set_min(1);
	epoints->set_max(8192);
	epoints->set_step(1);
	epoints->set_value(512);
	file->get_vbox()->add_margin_child(TTR("Generated Point Count:"), epoints);

	emission_mask = memnew(ConfirmationDialog);
	emission_mask->set_title(TTR("Load Emission Mask"));
	VBoxContainer *emvb = memnew(VBoxContainer);
	emission_mask->add_child(emvb);
	emission_mask_mode = memnew(OptionButton);
	emvb->add_margin_child(TTR("Emission Mask"), emission_mask_mode);
	emission_mask_mode->add_item(TTR("Solid Pixels"), EMISSION_MODE_SOLID);
	emission_mask_mode->add_item(TTR("Border Pixels"), EMISSION_MODE_BORDER);
	emission_mask_mode->add_item(TTR("Directed Border Pixels"), EMISSION_MODE_BORDER_DIRECTED);
	VBoxContainer *optionsvb = memnew(VBoxContainer);
	emvb->add_margin_child(TTR("Options"), optionsvb);
	emission_mask_centered = memnew(CheckBox);
	emission_mask_centered->set_text(TTR("Centered"));
	optionsvb->add_child(emission_mask_centered);
	emission_colors = memnew(CheckBox);
	emission_colors->set_text(TTR("Capture Colors from Pixel"));
	optionsvb->add_child(emission_colors);

	toolbar->add_child(emission_mask);

	emission_mask->connect("confirmed", callable_mp(this, &CPUParticles2DEditorPlugin::_generate_emission_mask));
}

CPUParticles2DEditorPlugin::~CPUParticles2DEditorPlugin() {
}
