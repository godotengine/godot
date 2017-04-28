/*************************************************************************/
/*  particles_editor_plugin.cpp                                          */
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

#include "particles_editor_plugin.h"
#include "editor/plugins/spatial_editor_plugin.h"
#include "io/resource_loader.h"

void ParticlesEditor::_node_removed(Node *p_node) {

	if (p_node == node) {
		node = NULL;
		hide();
	}
}

void ParticlesEditor::_resource_seleted(const String &p_res) {

	//print_line("selected resource path: "+p_res);
}

void ParticlesEditor::_node_selected(const NodePath &p_path) {

	Node *sel = get_node(p_path);
	if (!sel)
		return;

	VisualInstance *vi = sel->cast_to<VisualInstance>();
	if (!vi) {

		err_dialog->set_text(TTR("Node does not contain geometry."));
		err_dialog->popup_centered_minsize();
		return;
	}

	geometry = vi->get_faces(VisualInstance::FACES_SOLID);

	if (geometry.size() == 0) {

		err_dialog->set_text(TTR("Node does not contain geometry (faces)."));
		err_dialog->popup_centered_minsize();
		return;
	}

	Transform geom_xform = node->get_global_transform().affine_inverse() * vi->get_global_transform();

	int gc = geometry.size();
	PoolVector<Face3>::Write w = geometry.write();

	for (int i = 0; i < gc; i++) {
		for (int j = 0; j < 3; j++) {
			w[i].vertex[j] = geom_xform.xform(w[i].vertex[j]);
		}
	}

	w = PoolVector<Face3>::Write();

	emission_dialog->popup_centered(Size2(300, 130));
}

/*

void ParticlesEditor::_populate() {

	if(!node)
		return;


	if (node->get_particles().is_null())
		return;

	node->get_particles()->set_instance_count(populate_amount->get_text().to_int());
	node->populate_parent(populate_rotate_random->get_val(),populate_tilt_random->get_val(),populate_scale_random->get_text().to_double(),populate_scale->get_text().to_double());

}
*/

void ParticlesEditor::_notification(int p_notification) {

	if (p_notification == NOTIFICATION_ENTER_TREE) {
		options->set_icon(options->get_popup()->get_icon("Particles", "EditorIcons"));
	}
}

void ParticlesEditor::_menu_option(int p_option) {

	switch (p_option) {

		case MENU_OPTION_GENERATE_AABB: {
			generate_aabb->popup_centered_minsize();
		} break;
		case MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_MESH: {

			Ref<ParticlesMaterial> material = node->get_process_material();
			if (material.is_null()) {
				EditorNode::get_singleton()->show_warning(TTR("A processor material of type 'ParticlesMaterial' is required."));
				return;
			}
			emission_file_dialog->popup_centered_ratio();

		} break;

		case MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_NODE: {
			Ref<ParticlesMaterial> material = node->get_process_material();
			if (material.is_null()) {
				EditorNode::get_singleton()->show_warning(TTR("A processor material of type 'ParticlesMaterial' is required."));
				return;
			}
			/*
			Node *root = get_scene()->get_root_node();
			ERR_FAIL_COND(!root);
			EditorNode *en = root->cast_to<EditorNode>();
			ERR_FAIL_COND(!en);
			Node * node = en->get_edited_scene();
*/
			emission_tree_dialog->popup_centered_ratio();

		} break;
	}
}

void ParticlesEditor::_generate_aabb() {

	float time = generate_seconds->get_value();

	float running = 0.0;

	EditorProgress ep("gen_aabb", TTR("Generating AABB"), int(time));

	Rect3 rect;
	while (running < time) {

		uint64_t ticks = OS::get_singleton()->get_ticks_usec();
		ep.step("Generating..", int(running), true);
		OS::get_singleton()->delay_usec(1000);

		Rect3 capture = node->capture_aabb();
		if (rect == Rect3())
			rect = capture;
		else
			rect.merge_with(capture);

		running += (OS::get_singleton()->get_ticks_usec() - ticks) / 1000000.0;
	}

	node->set_visibility_aabb(rect);
}

void ParticlesEditor::edit(Particles *p_particles) {

	node = p_particles;
}

void ParticlesEditor::_generate_emission_points() {

	/// hacer codigo aca
	PoolVector<float> points;
	bool use_normals = emission_fill->get_selected() == 1;
	PoolVector<float> normals;

	if (emission_fill->get_selected() < 2) {

		float area_accum = 0;
		Map<float, int> triangle_area_map;
		print_line("geometry size: " + itos(geometry.size()));

		for (int i = 0; i < geometry.size(); i++) {

			float area = geometry[i].get_area();
			if (area < CMP_EPSILON)
				continue;
			triangle_area_map[area_accum] = i;
			area_accum += area;
		}

		if (!triangle_area_map.size() || area_accum == 0) {

			err_dialog->set_text(TTR("Faces contain no area!"));
			err_dialog->popup_centered_minsize();
			return;
		}

		int emissor_count = emission_amount->get_value();

		for (int i = 0; i < emissor_count; i++) {

			float areapos = Math::random(0.0f, area_accum);

			Map<float, int>::Element *E = triangle_area_map.find_closest(areapos);
			ERR_FAIL_COND(!E)
			int index = E->get();
			ERR_FAIL_INDEX(index, geometry.size());

			// ok FINALLY get face
			Face3 face = geometry[index];
			//now compute some position inside the face...

			Vector3 pos = face.get_random_point_inside();

			points.push_back(pos.x);
			points.push_back(pos.y);
			points.push_back(pos.z);

			if (use_normals) {
				Vector3 normal = face.get_plane().normal;
				normals.push_back(normal.x);
				normals.push_back(normal.y);
				normals.push_back(normal.z);
			}
		}
	} else {

		int gcount = geometry.size();

		if (gcount == 0) {

			err_dialog->set_text(TTR("No faces!"));
			err_dialog->popup_centered_minsize();
			return;
		}

		PoolVector<Face3>::Read r = geometry.read();

		Rect3 aabb;

		for (int i = 0; i < gcount; i++) {

			for (int j = 0; j < 3; j++) {

				if (i == 0 && j == 0)
					aabb.pos = r[i].vertex[j];
				else
					aabb.expand_to(r[i].vertex[j]);
			}
		}

		int emissor_count = emission_amount->get_value();

		for (int i = 0; i < emissor_count; i++) {

			int attempts = 5;

			for (int j = 0; j < attempts; j++) {

				Vector3 dir;
				dir[Math::rand() % 3] = 1.0;
				Vector3 ofs = Vector3(1, 1, 1) - dir;
				ofs = (Vector3(1, 1, 1) - dir) * Vector3(Math::randf(), Math::randf(), Math::randf()) * aabb.size;
				ofs += aabb.pos;

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

						if (d < min)
							min = d;
						if (d > max)
							max = d;
					}
				}

				if (max < min)
					continue; //lost attempt

				float val = min + (max - min) * Math::randf();

				Vector3 point = ofs + dir * val;

				points.push_back(point.x);
				points.push_back(point.y);
				points.push_back(point.z);
				break;
			}
		}
	}

	int point_count = points.size() / 3;

	int w = 2048;
	int h = (point_count / 2048) + 1;

	PoolVector<uint8_t> point_img;
	point_img.resize(w * h * 3 * sizeof(float));

	{
		PoolVector<uint8_t>::Write iw = point_img.write();
		zeromem(iw.ptr(), w * h * 3 * sizeof(float));
		PoolVector<float>::Read r = points.read();
		copymem(iw.ptr(), r.ptr(), point_count * sizeof(float) * 3);
	}

	Image image(w, h, false, Image::FORMAT_RGBF, point_img);

	Ref<ImageTexture> tex;
	tex.instance();
	tex->create_from_image(image, Texture::FLAG_FILTER);

	Ref<ParticlesMaterial> material = node->get_process_material();
	ERR_FAIL_COND(material.is_null());

	if (use_normals) {

		material->set_emission_shape(ParticlesMaterial::EMISSION_SHAPE_DIRECTED_POINTS);
		material->set_emission_point_count(point_count);
		material->set_emission_point_texture(tex);

		PoolVector<uint8_t> point_img2;
		point_img2.resize(w * h * 3 * sizeof(float));

		{
			PoolVector<uint8_t>::Write iw = point_img2.write();
			zeromem(iw.ptr(), w * h * 3 * sizeof(float));
			PoolVector<float>::Read r = normals.read();
			copymem(iw.ptr(), r.ptr(), point_count * sizeof(float) * 3);
		}

		Image image2(w, h, false, Image::FORMAT_RGBF, point_img2);

		Ref<ImageTexture> tex2;
		tex2.instance();
		tex2->create_from_image(image2, Texture::FLAG_FILTER);

		material->set_emission_normal_texture(tex2);
	} else {

		material->set_emission_shape(ParticlesMaterial::EMISSION_SHAPE_POINTS);
		material->set_emission_point_count(point_count);
		material->set_emission_point_texture(tex);
	}

	//print_line("point count: "+itos(points.size()));
	//node->set_emission_points(points);
}

void ParticlesEditor::_bind_methods() {

	ClassDB::bind_method("_menu_option", &ParticlesEditor::_menu_option);
	ClassDB::bind_method("_resource_seleted", &ParticlesEditor::_resource_seleted);
	ClassDB::bind_method("_node_selected", &ParticlesEditor::_node_selected);
	ClassDB::bind_method("_generate_emission_points", &ParticlesEditor::_generate_emission_points);
	ClassDB::bind_method("_generate_aabb", &ParticlesEditor::_generate_aabb);

	//ClassDB::bind_method("_populate",&ParticlesEditor::_populate);
}

ParticlesEditor::ParticlesEditor() {

	particles_editor_hb = memnew(HBoxContainer);
	SpatialEditor::get_singleton()->add_control_to_menu_panel(particles_editor_hb);
	options = memnew(MenuButton);
	particles_editor_hb->add_child(options);
	particles_editor_hb->hide();

	options->set_text("Particles");
	options->get_popup()->add_item(TTR("Generate AABB"), MENU_OPTION_GENERATE_AABB);
	options->get_popup()->add_separator();
	options->get_popup()->add_item(TTR("Create Emission Points From Mesh"), MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_MESH);
	options->get_popup()->add_item(TTR("Create Emission Points From Node"), MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_NODE);
	//	options->get_popup()->add_item(TTR("Clear Emitter"), MENU_OPTION_CLEAR_EMISSION_VOLUME);

	options->get_popup()->connect("id_pressed", this, "_menu_option");

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

	emission_dialog->get_ok()->set_text(TTR("Create"));
	emission_dialog->connect("confirmed", this, "_generate_emission_points");

	err_dialog = memnew(ConfirmationDialog);
	//err_dialog->get_cancel()->hide();
	add_child(err_dialog);

	emission_file_dialog = memnew(EditorFileDialog);
	add_child(emission_file_dialog);
	emission_file_dialog->connect("file_selected", this, "_resource_seleted");
	emission_tree_dialog = memnew(SceneTreeDialog);
	add_child(emission_tree_dialog);
	emission_tree_dialog->connect("selected", this, "_node_selected");

	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type("Mesh", &extensions);

	emission_file_dialog->clear_filters();
	for (int i = 0; i < extensions.size(); i++) {

		emission_file_dialog->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
	}

	emission_file_dialog->set_mode(EditorFileDialog::MODE_OPEN_FILE);

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

	generate_aabb->connect("confirmed", this, "_generate_aabb");

	//options->set_anchor(MARGIN_LEFT,Control::ANCHOR_END);
	//options->set_anchor(MARGIN_RIGHT,Control::ANCHOR_END);
}

void ParticlesEditorPlugin::edit(Object *p_object) {

	particles_editor->edit(p_object->cast_to<Particles>());
}

bool ParticlesEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("Particles");
}

void ParticlesEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		particles_editor->show();
		particles_editor->particles_editor_hb->show();
	} else {
		particles_editor->particles_editor_hb->hide();
		particles_editor->hide();
		particles_editor->edit(NULL);
	}
}

ParticlesEditorPlugin::ParticlesEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	particles_editor = memnew(ParticlesEditor);
	editor->get_viewport()->add_child(particles_editor);

	particles_editor->hide();
}

ParticlesEditorPlugin::~ParticlesEditorPlugin() {
}
