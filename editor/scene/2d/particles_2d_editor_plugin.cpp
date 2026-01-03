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
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/gui/editor_file_dialog.h"
#include "scene/2d/cpu_particles_2d.h"
#include "scene/2d/gpu_particles_2d.h"
#include "scene/gui/box_container.h"
#include "scene/gui/check_box.h"
#include "scene/gui/margin_container.h"
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
	p_menu->set_item_metadata(-1, "disable_on_multiselect");
}

void Particles2DEditorPlugin::_browse_mask_texture_pressed() {
	browsing_texture_type = TEXTURE_TYPE_MASK;
	file_dialog->popup_file_dialog();
}

void Particles2DEditorPlugin::_browse_direction_texture_pressed() {
	browsing_texture_type = TEXTURE_TYPE_DIRECTION;
	file_dialog->popup_centered();
}

void Particles2DEditorPlugin::_file_selected(const String &p_file) {
	switch (browsing_texture_type) {
		case TEXTURE_TYPE_MASK: {
			mask_img_path_line_edit->set_text(p_file);
			break;
		}
		case TEXTURE_TYPE_DIRECTION: {
			direction_img_path_line_edit->set_text(p_file);
			break;
		}
	}

	_validate_textures();
}

void Particles2DEditorPlugin::_process_emission_masks(PackedVector2Array &r_valid_positions, PackedVector2Array &r_valid_normals, PackedByteArray &r_valid_colors, Vector2i &r_image_size) {
	Ref<Image> mask_img;
	mask_img.instantiate();
	Error err = ImageLoader::load_image(mask_img_path_line_edit->get_text(), mask_img);
	ERR_FAIL_COND_MSG(err != OK, vformat("Error loading image '%s'.", mask_img_path_line_edit->get_text()));

	if (mask_img->is_compressed()) {
		mask_img->decompress();
	}
	mask_img->convert(Image::FORMAT_RGBA8);
	ERR_FAIL_COND(mask_img->get_format() != Image::FORMAT_RGBA8);
	Size2i mask_img_size = mask_img->get_size();
	ERR_FAIL_COND(mask_img_size.width == 0 || mask_img_size.height == 0);

	r_image_size = mask_img_size;

	r_valid_positions.resize(mask_img_size.width * mask_img_size.height);

	MaskMode emission_mode = static_cast<MaskMode>(emission_mask_mode->get_selected());
	DirectionMode direction_mode = static_cast<DirectionMode>(emission_direction_mode->get_selected());

	if (direction_mode != DIRECTION_MODE_NONE) {
		r_valid_normals.resize(mask_img_size.width * mask_img_size.height);
	}

	bool capture_colors = emission_mask_colors->is_pressed();

	if (capture_colors) {
		r_valid_colors.resize(mask_img_size.width * mask_img_size.height * 4);
	}

	int valid_point_count = 0;

	{
		Vector<uint8_t> mask_img_data = mask_img->get_data();
		const uint8_t *mask_img_ptr = mask_img_data.ptr();

		for (int mask_img_x = 0; mask_img_x < mask_img_size.width; mask_img_x++) {
			for (int mask_img_y = 0; mask_img_y < mask_img_size.height; mask_img_y++) {
				uint8_t mask_alpha = mask_img_ptr[(mask_img_y * mask_img_size.width + mask_img_x) * 4 + 3];

				if (mask_alpha <= 128) {
					continue;
				}

				if (emission_mode == MASK_MODE_SOLID) {
					r_valid_positions.write[valid_point_count++] = Point2(mask_img_x, mask_img_y);
				} else {
					bool pixel_is_on_border = false;
					for (int x = mask_img_x - 1; x <= mask_img_x + 1; x++) {
						for (int y = mask_img_y - 1; y <= mask_img_y + 1; y++) {
							if (x < 0 || y < 0 || x >= mask_img_size.width || y >= mask_img_size.height || mask_img_ptr[(y * mask_img_size.width + x) * 4 + 3] <= 128) {
								pixel_is_on_border = true;
								break;
							}
						}

						if (pixel_is_on_border) {
							break;
						}
					}

					if (!pixel_is_on_border) {
						continue;
					}

					r_valid_positions.write[valid_point_count] = Point2(mask_img_x, mask_img_y);

					if (direction_mode == DIRECTION_MODE_GENERATE) {
						Vector2 normal;
						for (int x = mask_img_x - 2; x <= mask_img_x + 2; x++) {
							for (int y = mask_img_y - 2; y <= mask_img_y + 2; y++) {
								if (x == mask_img_x && y == mask_img_y) {
									continue;
								}

								if (x < 0 || y < 0 || x >= mask_img_size.width || y >= mask_img_size.height || mask_img_ptr[(y * mask_img_size.width + x) * 4 + 3] <= 128) {
									normal += Vector2(x - mask_img_x, y - mask_img_y).normalized();
								}
							}
						}

						normal.normalize();
						r_valid_normals.write[valid_point_count] = normal;
					}

					valid_point_count++;
				}
			}
		}

		if (capture_colors) {
			for (int i = 0; i < valid_point_count; ++i) {
				const Point2i point = r_valid_positions.get(i);
				r_valid_colors.write[i * 4 + 0] = mask_img_ptr[(point.y * mask_img_size.width + point.x) * 4 + 0];
				r_valid_colors.write[i * 4 + 1] = mask_img_ptr[(point.y * mask_img_size.width + point.x) * 4 + 1];
				r_valid_colors.write[i * 4 + 2] = mask_img_ptr[(point.y * mask_img_size.width + point.x) * 4 + 2];
				r_valid_colors.write[i * 4 + 3] = mask_img_ptr[(point.y * mask_img_size.width + point.x) * 4 + 3];
			}
		}
	}

	if (direction_mode == DIRECTION_MODE_TEXTURE) {
		Ref<Image> normal_img;
		normal_img.instantiate();
		err = ImageLoader::load_image(direction_img_path_line_edit->get_text(), normal_img);
		ERR_FAIL_COND_MSG(err != OK, vformat("Error loading image '%s'.", direction_img_path_line_edit->get_text()));

		if (normal_img->is_compressed()) {
			normal_img->decompress();
		}
		normal_img->convert(Image::FORMAT_RGB8);
		ERR_FAIL_COND(normal_img->get_format() != Image::FORMAT_RGB8);
		Size2i normal_img_size = normal_img->get_size();
		ERR_FAIL_COND(normal_img_size.width == 0 || normal_img_size.height == 0);
		ERR_FAIL_COND_MSG(normal_img_size != mask_img_size, "Mask and Normal texture must have the same size.");

		Vector<uint8_t> normal_img_data = normal_img->get_data();
		const uint8_t *normal_img_ptr = normal_img_data.ptr();

		for (int i = 0; i < valid_point_count; ++i) {
			const Point2i point = r_valid_positions.get(i);
			const uint8_t normal_r = normal_img_ptr[(point.y * normal_img_size.width + point.x) * 3 + 0];
			const uint8_t normal_g = normal_img_ptr[(point.y * normal_img_size.width + point.x) * 3 + 1];

			Vector2 normal;
			normal.x = static_cast<float>(normal_r) / 255.0f - 0.5f;
			normal.y = static_cast<float>(normal_g) / 255.0f - 0.5f;

			normal.normalize();

			r_valid_normals.write[i] = normal;
		}
	}

	r_valid_positions.resize(valid_point_count);
	if (!r_valid_normals.is_empty()) {
		r_valid_normals.resize(valid_point_count);
	}
}

Particles2DEditorPlugin::Particles2DEditorPlugin() {
	file_dialog = memnew(EditorFileDialog);

	List<String> ext;
	ImageLoader::get_recognized_extensions(&ext);
	for (const String &E : ext) {
		file_dialog->add_filter("*." + E, E.to_upper());
	}

	file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	file_dialog->connect("file_selected", callable_mp(this, &Particles2DEditorPlugin::_file_selected));

	emission_mask_dialog = memnew(ConfirmationDialog);
	emission_mask_dialog->set_title(TTRC("Load Emission Mask"));
	emission_mask_dialog->add_child(file_dialog);
	emission_mask_dialog->get_ok_button()->set_disabled(true);

	VBoxContainer *emvb = memnew(VBoxContainer);
	emission_mask_dialog->add_child(emvb);

	HBoxContainer *mask_img_hbox = memnew(HBoxContainer);

	mask_img_path_line_edit = memnew(LineEdit);
	mask_img_hbox->add_child(mask_img_path_line_edit);
	mask_img_path_line_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	mask_img_path_line_edit->set_editable(false);
	mask_img_path_line_edit->connect(SceneStringName(text_changed), callable_mp(this, &Particles2DEditorPlugin::_validate_textures).unbind(1));

	mask_browse_button = memnew(Button);
	mask_img_hbox->add_child(mask_browse_button);
	mask_browse_button->connect(SceneStringName(pressed), callable_mp(this, &Particles2DEditorPlugin::_browse_mask_texture_pressed));
	emvb->add_margin_child(TTRC("Mask Texture"), mask_img_hbox);

	emission_mask_mode = memnew(OptionButton);
	emission_mask_mode->add_item(TTRC("Solid Pixels"), MASK_MODE_SOLID);
	emission_mask_mode->add_item(TTRC("Border Pixels"), MASK_MODE_BORDER);
	emission_mask_mode->connect(SceneStringName(item_selected), callable_mp(this, &Particles2DEditorPlugin::_emission_mask_mode_item_changed));
	emvb->add_margin_child(TTRC("Mask Mode"), emission_mask_mode);

	emission_direction_mode = memnew(OptionButton);
	emission_direction_mode->add_item(TTRC("None"), DIRECTION_MODE_NONE);
	emission_direction_mode->add_item(TTRC("Generate"), DIRECTION_MODE_GENERATE);
	emission_direction_mode->add_item(TTRC("Texture"), DIRECTION_MODE_TEXTURE);
	emission_direction_mode->connect(SceneStringName(item_selected), callable_mp(this, &Particles2DEditorPlugin::_validate_textures).unbind(1));
	emission_direction_mode->set_item_disabled(DIRECTION_MODE_GENERATE, true);
	emvb->add_margin_child(TTRC("Direction Mode"), emission_direction_mode);

	direction_img_label = memnew(Label);
	direction_img_label->set_text(TTRC("Direction Texture"));
	direction_img_label->set_theme_type_variation("HeaderSmall");
	emvb->add_child(direction_img_label);
	direction_img_label->hide();

	direction_img_hbox = memnew(HBoxContainer);
	direction_img_path_line_edit = memnew(LineEdit);
	direction_img_hbox->add_child(direction_img_path_line_edit);
	direction_img_path_line_edit->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	direction_img_path_line_edit->set_editable(false);
	direction_img_path_line_edit->connect(SceneStringName(text_changed), callable_mp(this, &Particles2DEditorPlugin::_validate_textures).unbind(1));

	direction_browse_button = memnew(Button);
	direction_img_hbox->add_child(direction_browse_button);
	direction_browse_button->connect(SceneStringName(pressed), callable_mp(this, &Particles2DEditorPlugin::_browse_direction_texture_pressed));
	emvb->add_child(direction_img_hbox);
	direction_img_hbox->hide();

	VBoxContainer *optionsvb = memnew(VBoxContainer);
	emvb->add_margin_child(TTRC("Options"), optionsvb);

	emission_mask_centered = memnew(CheckBox(TTRC("Centered")));
	emission_mask_centered->set_pressed(true);
	optionsvb->add_child(emission_mask_centered);
	emission_mask_colors = memnew(CheckBox(TTRC("Copy Color from Mask Texture")));
	optionsvb->add_child(emission_mask_colors);

	error_message = memnew(Label);
	error_message->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	error_message->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	error_message->add_theme_color_override(SceneStringName(font_color), EditorNode::get_singleton()->get_editor_theme()->get_color(SNAME("error_color"), EditorStringName(Editor)));
	emvb->add_child(error_message);

	EditorNode::get_singleton()->get_gui_base()->add_child(emission_mask_dialog);

	emission_mask_dialog->connect(SceneStringName(confirmed), callable_mp(this, &Particles2DEditorPlugin::_generate_emission_mask));
	emission_mask_dialog->connect(SceneStringName(theme_changed), callable_mp(this, &Particles2DEditorPlugin::_theme_changed));
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
	const List<Node *> &current_selection = EditorNode::get_singleton()->get_editor_selection()->get_top_selected_node_list();
	if (selected_particles.is_empty() && current_selection.is_empty()) {
		return;
	}

	// Turn gizmos on for nodes that are newly selected.
	HashSet<ObjectID> nodes_in_current_selection;
	for (Node *node : current_selection) {
		ObjectID nid = node->get_instance_id();
		nodes_in_current_selection.insert(nid);
		if (!selected_particles.has(nid)) {
			_set_show_gizmos(node, true);
			selected_particles.insert(nid);
		}
	}

	mask_img_path_line_edit->set_text("");
	emission_mask_mode->select(MASK_MODE_SOLID);
	emission_direction_mode->select(DIRECTION_MODE_NONE);
	emission_mask_centered->set_pressed(true);
	emission_mask_colors->set_pressed(false);
	direction_img_path_line_edit->set_text("");

	// Turn gizmos off for nodes that are no longer selected.
	LocalVector<ObjectID> to_erase;
	for (const ObjectID &nid : selected_particles) {
		if (!nodes_in_current_selection.has(nid)) {
			Node *node = ObjectDB::get_instance<Node>(nid);
			if (node) {
				_set_show_gizmos(node, false);
			}
			to_erase.push_back(nid);
		}
	}

	for (const ObjectID &nid : to_erase) {
		selected_particles.erase(nid);
	}
}

void Particles2DEditorPlugin::_node_removed(Node *p_node) {
	if (p_node && selected_particles.erase(p_node->get_instance_id())) {
		_set_show_gizmos(p_node, false);
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

void Particles2DEditorPlugin::_theme_changed() {
	mask_browse_button->set_button_icon(mask_browse_button->get_editor_theme_icon(SNAME("FileBrowse")));
	direction_browse_button->set_button_icon(direction_browse_button->get_editor_theme_icon(SNAME("FileBrowse")));
}

void Particles2DEditorPlugin::_menu_callback(int p_idx) {
	if (p_idx == MENU_LOAD_EMISSION_MASK) {
		GPUParticles2D *particles = Object::cast_to<GPUParticles2D>(edited_node);
		if (particles && particles->get_process_material().is_null()) {
			EditorNode::get_singleton()->show_warning(TTR("Loading emission mask requires ParticleProcessMaterial."));
			return;
		}

		emission_mask_dialog->popup_centered();
	} else {
		ParticlesEditorPlugin::_menu_callback(p_idx);
	}
}

void Particles2DEditorPlugin::_add_menu_options(PopupMenu *p_menu) {
	p_menu->add_item(TTR("Load Emission Mask"), MENU_LOAD_EMISSION_MASK);
}

void Particles2DEditorPlugin::_validate_textures() {
	DirectionMode direction_mode = static_cast<DirectionMode>(emission_direction_mode->get_selected());
	direction_img_label->set_visible(direction_mode == DIRECTION_MODE_TEXTURE);
	direction_img_hbox->set_visible(direction_mode == DIRECTION_MODE_TEXTURE);

	error_message->hide();
	emission_mask_dialog->get_ok_button()->set_disabled(true);

	if (mask_img_path_line_edit->get_text().is_empty()) {
		emission_mask_dialog->reset_size();
		return;
	}

	Ref<Image> mask_img;
	mask_img.instantiate();
	Error err = ImageLoader::load_image(mask_img_path_line_edit->get_text(), mask_img);
	if (err != OK) {
		error_message->show();
		error_message->set_text(TTRC("Failed to load mask texture."));
		emission_mask_dialog->reset_size();
		return;
	}

	if (mask_img->is_compressed()) {
		mask_img->decompress();
	}
	mask_img->convert(Image::FORMAT_RGBA8);

	if (mask_img->get_format() != Image::FORMAT_RGBA8) {
		error_message->show();
		error_message->set_text(TTRC("Failed to convert mask texture to RGBA8."));
		emission_mask_dialog->reset_size();
		return;
	}

	Size2i mask_img_size = mask_img->get_size();
	if (mask_img_size.width == 0 || mask_img_size.height == 0) {
		error_message->show();
		error_message->set_text(TTRC("Mask texture has an invalid size."));
		emission_mask_dialog->reset_size();
		return;
	}

	if (direction_mode == DIRECTION_MODE_TEXTURE) {
		if (direction_img_path_line_edit->get_text().is_empty()) {
			return;
		}

		Ref<Image> direction_img;
		direction_img.instantiate();
		err = ImageLoader::load_image(direction_img_path_line_edit->get_text(), direction_img);

		if (err != OK) {
			error_message->show();
			error_message->set_text(TTRC("Failed to load direction texture."));
			emission_mask_dialog->reset_size();
			return;
		}

		if (direction_img->is_compressed()) {
			direction_img->decompress();
		}
		direction_img->convert(Image::FORMAT_RGBA8);

		if (direction_img->get_format() != Image::FORMAT_RGBA8) {
			error_message->show();
			error_message->set_text(TTRC("Failed to convert direction texture to RGBA8."));
			emission_mask_dialog->reset_size();
			return;
		}

		Size2i direction_img_size = direction_img->get_size();

		if (direction_img_size.width == 0 || direction_img_size.height == 0 || direction_img_size != mask_img_size) {
			error_message->show();
			error_message->set_text(TTRC("Direction texture has an invalid size. It must have the same size as the mask texture."));
			emission_mask_dialog->reset_size();
			return;
		}
	}

	emission_mask_dialog->get_ok_button()->set_disabled(false);
	emission_mask_dialog->reset_size();
}

void Particles2DEditorPlugin::_emission_mask_mode_item_changed(int p_idx) const {
	emission_direction_mode->set_item_disabled(DIRECTION_MODE_GENERATE, p_idx == static_cast<int>(MASK_MODE_SOLID));

	if (emission_direction_mode->get_selected() == DIRECTION_MODE_GENERATE) {
		emission_direction_mode->select(DIRECTION_MODE_NONE);
	}
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

	PackedVector2Array emission_positions;
	PackedVector2Array emission_normals;
	PackedByteArray emission_colors;
	Vector2i texture_size;
	_process_emission_masks(emission_positions, emission_normals, emission_colors, texture_size);

	ERR_FAIL_COND_MSG(emission_positions.is_empty(), "No pixels with transparency > 128 in image...");

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Load Emission Mask"));
	ParticleProcessMaterial *pmptr = pm.ptr();

	Vector<uint8_t> mask_texture_data;

	int valid_positions_count = emission_positions.size();
	int w = 2048;
	int h = (valid_positions_count / 2048) + 1;

	mask_texture_data.resize_initialized(w * h * 2 * sizeof(float));

	{
		Vector2 offset;
		if (emission_mask_centered->is_pressed()) {
			offset = Vector2(-texture_size.width * 0.5, -texture_size.height * 0.5);
		}

		uint8_t *tw = mask_texture_data.ptrw();
		float *twf = reinterpret_cast<float *>(tw);
		for (int i = 0; i < valid_positions_count; i++) {
			twf[i * 2 + 0] = emission_positions[i].x + offset.x;
			twf[i * 2 + 1] = emission_positions[i].y + offset.y;
		}
	}

	Ref<Image> img;
	img.instantiate();
	img->set_data(w, h, false, Image::FORMAT_RGF, mask_texture_data);
	undo_redo->add_do_property(pmptr, "emission_point_texture", ImageTexture::create_from_image(img));
	undo_redo->add_undo_property(pmptr, "emission_point_texture", pm->get_emission_point_texture());
	undo_redo->add_do_property(pmptr, "emission_point_count", valid_positions_count);
	undo_redo->add_undo_property(pmptr, "emission_point_count", pm->get_emission_point_count());

	if (emission_mask_colors->is_pressed()) {
		Vector<uint8_t> color_texture_data;
		color_texture_data.resize_initialized(w * h * 4);

		{
			uint8_t *tw = color_texture_data.ptrw();
			for (int i = 0; i < valid_positions_count * 4; i++) {
				tw[i] = emission_colors[i];
			}
		}

		img.instantiate();
		img->set_data(w, h, false, Image::FORMAT_RGBA8, color_texture_data);
		undo_redo->add_do_property(pmptr, "emission_color_texture", ImageTexture::create_from_image(img));
		undo_redo->add_undo_property(pmptr, "emission_color_texture", pm->get_emission_color_texture());
	}

	if (emission_normals.size()) {
		undo_redo->add_do_property(pmptr, "emission_shape", ParticleProcessMaterial::EMISSION_SHAPE_DIRECTED_POINTS);
		undo_redo->add_undo_property(pmptr, "emission_shape", pm->get_emission_shape());
		pm->set_emission_shape(ParticleProcessMaterial::EMISSION_SHAPE_DIRECTED_POINTS);

		Vector<uint8_t> normal_texture_data;
		normal_texture_data.resize_initialized(w * h * 2 * sizeof(float));

		{
			uint8_t *tw = normal_texture_data.ptrw();
			float *twf = reinterpret_cast<float *>(tw);
			for (int i = 0; i < valid_positions_count; i++) {
				twf[i * 2 + 0] = emission_normals[i].x;
				twf[i * 2 + 1] = emission_normals[i].y;
			}
		}

		img.instantiate();
		img->set_data(w, h, false, Image::FORMAT_RGF, normal_texture_data);
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
	_process_emission_masks(valid_positions, valid_normals, valid_colors, image_size);

	ERR_FAIL_COND_MSG(valid_positions.is_empty(), "No pixels with transparency > 128 in image...");

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Load Emission Mask"));

	int valid_point_count = valid_positions.size();
	if (emission_mask_colors->is_pressed()) {
		PackedColorArray pca;
		pca.resize(valid_point_count);
		Color *pcaw = pca.ptrw();
		for (int i = 0; i < valid_point_count; i += 1) {
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
