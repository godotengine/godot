/*************************************************************************/
/*  particles_2d_editor_plugin.cpp                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "scene/gui/separator.h"

void Particles2DEditorPlugin::edit(Object *p_object) {

	if (p_object) {
		particles = p_object->cast_to<Particles2D>();
	} else {
		particles = NULL;
	}
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

	int epc = epoints->get_value();

	Image img;
	Error err = ImageLoader::load_image(p_file, &img);
	ERR_EXPLAIN(TTR("Error loading image:") + " " + p_file);
	ERR_FAIL_COND(err != OK);

	img.convert(Image::FORMAT_LA8);
	ERR_FAIL_COND(img.get_format() != Image::FORMAT_LA8);
	Size2i s = Size2(img.get_width(), img.get_height());
	ERR_FAIL_COND(s.width == 0 || s.height == 0);

	PoolVector<uint8_t> data = img.get_data();
	PoolVector<uint8_t>::Read r = data.read();

	Vector<Point2i> valid_positions;
	valid_positions.resize(s.width * s.height);
	int vpc = 0;

	for (int i = 0; i < s.width * s.height; i++) {

		uint8_t a = r[i * 2 + 1];
		if (a > 128) {
			valid_positions[vpc++] = Point2i(i % s.width, i / s.width);
		}
	}

	valid_positions.resize(vpc);

	ERR_EXPLAIN(TTR("No pixels with transparency > 128 in image.."));
	ERR_FAIL_COND(valid_positions.size() == 0);

	PoolVector<Point2> epoints;
	epoints.resize(epc);
	PoolVector<Point2>::Write w = epoints.write();

	Size2 extents = Size2(img.get_width() * 0.5, img.get_height() * 0.5);

	for (int i = 0; i < epc; i++) {

		Point2 p = valid_positions[Math::rand() % vpc];
		p -= s / 2;
		w[i] = p / extents;
	}

	w = PoolVector<Point2>::Write();

	undo_redo->create_action(TTR("Set Emission Mask"));
	undo_redo->add_do_method(particles, "set_emission_points", epoints);
	undo_redo->add_do_method(particles, "set_emission_half_extents", extents);
	undo_redo->add_undo_method(particles, "set_emission_points", particles->get_emission_points());
	undo_redo->add_undo_method(particles, "set_emission_half_extents", particles->get_emission_half_extents());
	undo_redo->commit_action();
}

void Particles2DEditorPlugin::_menu_callback(int p_idx) {

	switch (p_idx) {
		case MENU_LOAD_EMISSION_MASK: {

			file->popup_centered_ratio();

		} break;
		case MENU_CLEAR_EMISSION_MASK: {

			undo_redo->create_action(TTR("Clear Emission Mask"));
			undo_redo->add_do_method(particles, "set_emission_points", PoolVector<Vector2>());
			undo_redo->add_undo_method(particles, "set_emission_points", particles->get_emission_points());
			undo_redo->commit_action();
		} break;
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
	menu->get_popup()->add_item(TTR("Load Emission Mask"), MENU_LOAD_EMISSION_MASK);
	menu->get_popup()->add_item(TTR("Clear Emission Mask"), MENU_CLEAR_EMISSION_MASK);
	menu->set_text("Particles");
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
}

Particles2DEditorPlugin::~Particles2DEditorPlugin() {
}
