/*************************************************************************/
/*  particles_2d_editor_plugin.cpp                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "particles_2d_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "io/image_loader.h"
#include "scene/3d/particles.h"
#include "scene/gui/separator.h"

void Particles2DEditorPlugin::edit(Object *p_object) {

	particles = Object::cast_to<Particles2D>(p_object);
}

bool Particles2DEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("Particles2D");
}

void Particles2DEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {

		toolbar->show();
	} else {

		toolbar->hide();
	}
}

void Particles2DEditorPlugin::_file_selected(const String &p_file) {

	print_line("file: " + p_file);

	source_emission_file = p_file;
	emission_mask->popup_centered_minsize();
}

void Particles2DEditorPlugin::_menu_callback(int p_idx) {

	switch (p_idx) {
		case MENU_GENERATE_VISIBILITY_RECT: {
			generate_aabb->popup_centered_minsize();
		} break;
		case MENU_LOAD_EMISSION_MASK: {

			file->popup_centered_ratio();

		} break;
		case MENU_CLEAR_EMISSION_MASK: {

			emission_mask->popup_centered_minsize();
		} break;
	}
}

void Particles2DEditorPlugin::_generate_visibility_rect() {

	float time = generate_seconds->get_value();

	float running = 0.0;

	EditorProgress ep("gen_aabb", TTR("Generating AABB"), int(time));

	Rect2 rect;
	while (running < time) {

		uint64_t ticks = OS::get_singleton()->get_ticks_usec();
		ep.step("Generating..", int(running), true);
		OS::get_singleton()->delay_usec(1000);

		Rect2 capture = particles->capture_rect();
		if (rect == Rect2())
			rect = capture;
		else
			rect = rect.merge(capture);

		running += (OS::get_singleton()->get_ticks_usec() - ticks) / 1000000.0;
	}

	particles->set_visibility_rect(rect);
}

void Particles2DEditorPlugin::_generate_emission_mask() {

	Ref<ParticlesMaterial> pm = particles->get_process_material();
	if (!pm.is_valid()) {
		EditorNode::get_singleton()->show_warning(TTR("Can only set point into a ParticlesMaterial process material"));
		return;
	}

	Ref<Image> img;
	img.instance();
	Error err = ImageLoader::load_image(source_emission_file, img);
	ERR_EXPLAIN(TTR("Error loading image:") + " " + source_emission_file);
	ERR_FAIL_COND(err != OK);

	if (img->is_compressed()) {
		img->decompress();
	}
	img->convert(Image::FORMAT_RGBA8);
	ERR_FAIL_COND(img->get_format() != Image::FORMAT_RGBA8);
	Size2i s = Size2(img->get_width(), img->get_height());
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
		PoolVector<uint8_t> data = img->get_data();
		PoolVector<uint8_t>::Read r = data.read();

		for (int i = 0; i < s.width; i++) {
			for (int j = 0; j < s.height; j++) {

				uint8_t a = r[(j * s.width + i) * 4 + 3];

				if (a > 128) {

					if (emode == EMISSION_MODE_SOLID) {

						if (capture_colors) {
							valid_colors[vpc * 4 + 0] = r[(j * s.width + i) * 4 + 0];
							valid_colors[vpc * 4 + 1] = r[(j * s.width + i) * 4 + 1];
							valid_colors[vpc * 4 + 2] = r[(j * s.width + i) * 4 + 2];
							valid_colors[vpc * 4 + 3] = r[(j * s.width + i) * 4 + 3];
						}
						valid_positions[vpc++] = Point2(i, j);

					} else {

						bool on_border = false;
						for (int x = i - 1; x <= i + 1; x++) {
							for (int y = j - 1; y <= j + 1; y++) {

								if (x < 0 || y < 0 || x >= s.width || y >= s.height || r[(y * s.width + x) * 4 + 3] <= 128) {
									on_border = true;
									break;
								}
							}

							if (on_border)
								break;
						}

						if (on_border) {
							valid_positions[vpc] = Point2(i, j);

							if (emode == EMISSION_MODE_BORDER_DIRECTED) {
								Vector2 normal;
								for (int x = i - 2; x <= i + 2; x++) {
									for (int y = j - 2; y <= j + 2; y++) {

										if (x == i && y == j)
											continue;

										if (x < 0 || y < 0 || x >= s.width || y >= s.height || r[(y * s.width + x) * 4 + 3] <= 128) {
											normal += Vector2(x - i, y - j).normalized();
										}
									}
								}

								normal.normalize();
								valid_normals[vpc] = normal;
							}

							if (capture_colors) {
								valid_colors[vpc * 4 + 0] = r[(j * s.width + i) * 4 + 0];
								valid_colors[vpc * 4 + 1] = r[(j * s.width + i) * 4 + 1];
								valid_colors[vpc * 4 + 2] = r[(j * s.width + i) * 4 + 2];
								valid_colors[vpc * 4 + 3] = r[(j * s.width + i) * 4 + 3];
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

	ERR_EXPLAIN(TTR("No pixels with transparency > 128 in image.."));
	ERR_FAIL_COND(valid_positions.size() == 0);

	PoolVector<uint8_t> texdata;

	int w = 2048;
	int h = (vpc / 2048) + 1;

	texdata.resize(w * h * 2 * sizeof(float));

	{
		PoolVector<uint8_t>::Write tw = texdata.write();
		float *twf = (float *)tw.ptr();
		for (int i = 0; i < vpc; i++) {

			twf[i * 2 + 0] = valid_positions[i].x;
			twf[i * 2 + 1] = valid_positions[i].y;
		}
	}

	img.instance();
	img->create(w, h, false, Image::FORMAT_RGF, texdata);

	Ref<ImageTexture> imgt;
	imgt.instance();
	imgt->create_from_image(img, 0);

	pm->set_emission_point_texture(imgt);
	pm->set_emission_point_count(vpc);

	if (capture_colors) {

		PoolVector<uint8_t> colordata;
		colordata.resize(w * h * 4); //use RG texture

		{
			PoolVector<uint8_t>::Write tw = colordata.write();
			for (int i = 0; i < vpc * 4; i++) {

				tw[i] = valid_colors[i];
			}
		}

		img.instance();
		img->create(w, h, false, Image::FORMAT_RGBA8, colordata);

		imgt.instance();
		imgt->create_from_image(img, 0);
		pm->set_emission_color_texture(imgt);
	}

	if (valid_normals.size()) {
		pm->set_emission_shape(ParticlesMaterial::EMISSION_SHAPE_DIRECTED_POINTS);

		PoolVector<uint8_t> normdata;
		normdata.resize(w * h * 2 * sizeof(float)); //use RG texture

		{
			PoolVector<uint8_t>::Write tw = normdata.write();
			float *twf = (float *)tw.ptr();
			for (int i = 0; i < vpc; i++) {
				twf[i * 2 + 0] = valid_normals[i].x;
				twf[i * 2 + 1] = valid_normals[i].y;
			}
		}

		img.instance();
		img->create(w, h, false, Image::FORMAT_RGF, normdata);

		imgt.instance();
		imgt->create_from_image(img, 0);
		pm->set_emission_normal_texture(imgt);

	} else {
		pm->set_emission_shape(ParticlesMaterial::EMISSION_SHAPE_POINTS);
	}
}

void Particles2DEditorPlugin::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		menu->get_popup()->connect("id_pressed", this, "_menu_callback");
		menu->set_icon(menu->get_popup()->get_icon("Particles2D", "EditorIcons"));
		file->connect("file_selected", this, "_file_selected");
	}
}

void Particles2DEditorPlugin::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_menu_callback"), &Particles2DEditorPlugin::_menu_callback);
	ClassDB::bind_method(D_METHOD("_file_selected"), &Particles2DEditorPlugin::_file_selected);
	ClassDB::bind_method(D_METHOD("_generate_visibility_rect"), &Particles2DEditorPlugin::_generate_visibility_rect);
	ClassDB::bind_method(D_METHOD("_generate_emission_mask"), &Particles2DEditorPlugin::_generate_emission_mask);
}

Particles2DEditorPlugin::Particles2DEditorPlugin(EditorNode *p_node) {

	particles = NULL;
	editor = p_node;
	undo_redo = editor->get_undo_redo();

	toolbar = memnew(HBoxContainer);
	add_control_to_container(CONTAINER_CANVAS_EDITOR_MENU, toolbar);
	toolbar->hide();

	toolbar->add_child(memnew(VSeparator));

	menu = memnew(MenuButton);
	menu->get_popup()->add_item(TTR("Generate Visibility Rect"), MENU_GENERATE_VISIBILITY_RECT);
	menu->get_popup()->add_separator();
	menu->get_popup()->add_item(TTR("Load Emission Mask"), MENU_LOAD_EMISSION_MASK);
	//	menu->get_popup()->add_item(TTR("Clear Emission Mask"), MENU_CLEAR_EMISSION_MASK);
	menu->set_text(TTR("Particles"));
	toolbar->add_child(menu);

	file = memnew(EditorFileDialog);
	List<String> ext;
	ImageLoader::get_recognized_extensions(&ext);
	for (List<String>::Element *E = ext.front(); E; E = E->next()) {
		file->add_filter("*." + E->get() + "; " + E->get().to_upper());
	}
	file->set_mode(EditorFileDialog::MODE_OPEN_FILE);
	toolbar->add_child(file);

	epoints = memnew(SpinBox);
	epoints->set_min(1);
	epoints->set_max(8192);
	epoints->set_step(1);
	epoints->set_value(512);
	file->get_vbox()->add_margin_child(TTR("Generated Point Count:"), epoints);

	generate_aabb = memnew(ConfirmationDialog);
	generate_aabb->set_title(TTR("Generate Visibility Rect"));
	VBoxContainer *genvb = memnew(VBoxContainer);
	generate_aabb->add_child(genvb);
	generate_seconds = memnew(SpinBox);
	genvb->add_margin_child(TTR("Generation Time (sec):"), generate_seconds);
	generate_seconds->set_min(0.1);
	generate_seconds->set_max(25);
	generate_seconds->set_value(2);

	toolbar->add_child(generate_aabb);

	generate_aabb->connect("confirmed", this, "_generate_visibility_rect");

	emission_mask = memnew(ConfirmationDialog);
	emission_mask->set_title(TTR("Generate Visibility Rect"));
	VBoxContainer *emvb = memnew(VBoxContainer);
	emission_mask->add_child(emvb);
	emission_mask_mode = memnew(OptionButton);
	emvb->add_margin_child(TTR("Emission Mask"), emission_mask_mode);
	emission_mask_mode->add_item("Solid Pixels", EMISSION_MODE_SOLID);
	emission_mask_mode->add_item("Border Pixels", EMISSION_MODE_BORDER);
	emission_mask_mode->add_item("Directed Border Pixels", EMISSION_MODE_BORDER_DIRECTED);
	emission_colors = memnew(CheckBox);
	emission_colors->set_text(TTR("Capture from Pixel"));
	emvb->add_margin_child(TTR("Emission Colors"), emission_colors);

	toolbar->add_child(emission_mask);

	emission_mask->connect("confirmed", this, "_generate_emission_mask");
}

Particles2DEditorPlugin::~Particles2DEditorPlugin() {
}
