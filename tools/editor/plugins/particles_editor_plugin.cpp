/*************************************************************************/
/*  particles_editor_plugin.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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

#if 0
#include "particles_editor_plugin.h"
#include "io/resource_loader.h"
#include "servers/visual/particle_system_sw.h"
#include "tools/editor/plugins/spatial_editor_plugin.h"


void ParticlesEditor::_node_removed(Node *p_node) {

	if(p_node==node) {
		node=NULL;
		hide();
	}

}


void ParticlesEditor::_resource_seleted(const String& p_res) {

	//print_line("selected resource path: "+p_res);
}

void ParticlesEditor::_node_selected(const NodePath& p_path){


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

	if (geometry.size()==0) {

		err_dialog->set_text(TTR("Node does not contain geometry (faces)."));
		err_dialog->popup_centered_minsize();
		return;

	}

	Transform geom_xform = node->get_global_transform().affine_inverse() * vi->get_global_transform();

	int gc = geometry.size();
	PoolVector<Face3>::Write w = geometry.write();


	for(int i=0;i<gc;i++) {
		for(int j=0;j<3;j++) {
			w[i].vertex[j] = geom_xform.xform( w[i].vertex[j] );
		}
	}


	w = PoolVector<Face3>::Write();

	emission_dialog->popup_centered(Size2(300,130));
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

	if (p_notification==NOTIFICATION_ENTER_TREE) {
		options->set_icon(options->get_popup()->get_icon("Particles","EditorIcons"));

	}
}


void ParticlesEditor::_menu_option(int p_option) {


	switch(p_option) {

		case MENU_OPTION_GENERATE_AABB: {

			Transform globalizer = node->get_global_transform();
			ParticleSystemSW pssw;
			for(int i=0;i<VS::PARTICLE_VAR_MAX;i++) {

				pssw.particle_vars[i]=node->get_variable((Particles::Variable)i);
				pssw.particle_randomness[i]=node->get_randomness((Particles::Variable)i);
			}

			pssw.emission_half_extents=node->get_emission_half_extents();
			pssw.emission_points=node->get_emission_points();
			pssw.emission_base_velocity=node->get_emission_base_velocity();
			pssw.amount=node->get_amount();
			pssw.gravity_normal=node->get_gravity_normal();
			pssw.emitting=true;
			pssw.height_from_velocity=node->has_height_from_velocity();
			pssw.color_phase_count=1;


			ParticleSystemProcessSW pp;
			float delta=0.01;
			float lifetime=pssw.particle_vars[VS::PARTICLE_LIFETIME];


			Transform localizer = globalizer.affine_inverse();
			AABB aabb;
			for(float t=0;t<lifetime;t+=delta) {

				pp.process(&pssw,globalizer,delta);
				for(int i=0;i<pp.particle_data.size();i++) {

					Vector3 p = localizer.xform(pp.particle_data[i].pos);

					if (t==0 && i==0)
						aabb.pos=p;
					else
						aabb.expand_to(p);
				}
			}

			aabb.grow_by( aabb.get_longest_axis_size()*0.2);

			node->set_visibility_aabb(aabb);


		} break;
		case MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_MESH: {


			emission_file_dialog->popup_centered_ratio();

		} break;

		case MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_NODE: {
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


void ParticlesEditor::edit(Particles *p_particles) {

	node=p_particles;

}

void ParticlesEditor::_generate_emission_points() {

	/// hacer codigo aca
	PoolVector<Vector3> points;

	if (emission_fill->get_selected()==0) {

		float area_accum=0;
		Map<float,int> triangle_area_map;
		print_line("geometry size: "+itos(geometry.size()));

		for(int i=0;i<geometry.size();i++) {

			float area = geometry[i].get_area();;
			if (area<CMP_EPSILON)
				continue;
			triangle_area_map[area_accum]=i;
			area_accum+=area;
		}

		if (!triangle_area_map.size() || area_accum==0) {

			err_dialog->set_text(TTR("Faces contain no area!"));
			err_dialog->popup_centered_minsize();
			return;
		}

		int emissor_count=emission_amount->get_val();

		for(int i=0;i<emissor_count;i++) {

			float areapos = Math::random(0,area_accum);

			Map<float,int>::Element *E = triangle_area_map.find_closest(areapos);
			ERR_FAIL_COND(!E)
			int index = E->get();
			ERR_FAIL_INDEX(index,geometry.size());

			// ok FINALLY get face
			Face3 face = geometry[index];
			//now compute some position inside the face...

			Vector3 pos = face.get_random_point_inside();

			points.push_back(pos);
		}
	} else {

		int gcount = geometry.size();

		if (gcount==0) {

			err_dialog->set_text(TTR("No faces!"));
			err_dialog->popup_centered_minsize();
			return;
		}

		PoolVector<Face3>::Read r = geometry.read();

		AABB aabb;

		for(int i=0;i<gcount;i++) {

			for(int j=0;j<3;j++) {

				if (i==0 && j==0)
					aabb.pos=r[i].vertex[j];
				else
					aabb.expand_to(r[i].vertex[j]);
			}
		}

		int emissor_count=emission_amount->get_val();

		for(int i=0;i<emissor_count;i++) {

			int attempts=5;

			for(int j=0;j<attempts;j++) {

				Vector3 dir;
				dir[Math::rand()%3]=1.0;
				Vector3 ofs = Vector3(1,1,1)-dir;
				ofs=(Vector3(1,1,1)-dir)*Vector3(Math::randf(),Math::randf(),Math::randf())*aabb.size;
				ofs+=aabb.pos;

				Vector3 ofsv = ofs + aabb.size * dir;

				//space it a little
				ofs -= dir;
				ofsv += dir;

				float max=-1e7,min=1e7;

				for(int k=0;k<gcount;k++) {

					const Face3& f3 = r[k];

					Vector3 res;
					if (f3.intersects_segment(ofs,ofsv,&res)) {

						res-=ofs;
						float d = dir.dot(res);

						if (d<min)
							min=d;
						if (d>max)
							max=d;

					}
				}


				if (max<min)
					continue; //lost attempt

				float val = min + (max-min)*Math::randf();

				Vector3 point = ofs + dir * val;

				points.push_back(point);
				break;
			}
		}
	}

	//print_line("point count: "+itos(points.size()));
	node->set_emission_points(points);

}

void ParticlesEditor::_bind_methods() {

	ClassDB::bind_method("_menu_option",&ParticlesEditor::_menu_option);
	ClassDB::bind_method("_resource_seleted",&ParticlesEditor::_resource_seleted);
	ClassDB::bind_method("_node_selected",&ParticlesEditor::_node_selected);
	ClassDB::bind_method("_generate_emission_points",&ParticlesEditor::_generate_emission_points);

	//ClassDB::bind_method("_populate",&ParticlesEditor::_populate);

}

ParticlesEditor::ParticlesEditor() {

	particles_editor_hb = memnew ( HBoxContainer );
	SpatialEditor::get_singleton()->add_control_to_menu_panel(particles_editor_hb);
	options = memnew( MenuButton );
	particles_editor_hb->add_child(options);
	particles_editor_hb->hide();

	options->set_text("Particles");
	options->get_popup()->add_item(TTR("Generate AABB"),MENU_OPTION_GENERATE_AABB);
	options->get_popup()->add_separator();
	options->get_popup()->add_item(TTR("Create Emitter From Mesh"),MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_MESH);
	options->get_popup()->add_item(TTR("Create Emitter From Node"),MENU_OPTION_CREATE_EMISSION_VOLUME_FROM_NODE);
	options->get_popup()->add_item(TTR("Clear Emitter"),MENU_OPTION_CLEAR_EMISSION_VOLUME);

	options->get_popup()->connect("id_pressed", this,"_menu_option");

	emission_dialog = memnew( ConfirmationDialog );
	emission_dialog->set_title(TTR("Create Emitter"));
	add_child(emission_dialog);
	Label *l = memnew(Label);
	l->set_pos(Point2(5,5));
	l->set_text(TTR("Emission Positions:"));
	emission_dialog->add_child(l);


	emission_amount = memnew( SpinBox );
	emission_amount->set_anchor(MARGIN_RIGHT,ANCHOR_END);
	emission_amount->set_begin( Point2(20,23));
	emission_amount->set_end( Point2(5,25));
	emission_amount->set_min(1);
	emission_amount->set_max(65536);
	emission_amount->set_val(512);
	emission_dialog->add_child(emission_amount);
	emission_dialog->get_ok()->set_text(TTR("Create"));
	emission_dialog->connect("confirmed",this,"_generate_emission_points");

	l = memnew(Label);
	l->set_pos(Point2(5,50));
	l->set_text(TTR("Emission Fill:"));
	emission_dialog->add_child(l);

	emission_fill = memnew( OptionButton );
	emission_fill->set_anchor(MARGIN_RIGHT,ANCHOR_END);
	emission_fill->set_begin( Point2(20,70));
	emission_fill->set_end( Point2(5,75));
	emission_fill->add_item(TTR("Surface"));
	emission_fill->add_item(TTR("Volume"));
	emission_dialog->add_child(emission_fill);

	err_dialog = memnew( ConfirmationDialog );
	//err_dialog->get_cancel()->hide();
	add_child(err_dialog);


	emission_file_dialog = memnew( EditorFileDialog );
	add_child(emission_file_dialog);
	emission_file_dialog->connect("file_selected",this,"_resource_seleted");
	emission_tree_dialog = memnew( SceneTreeDialog );
	add_child(emission_tree_dialog);
	emission_tree_dialog->connect("selected",this,"_node_selected");

	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type("Mesh",&extensions);

	emission_file_dialog->clear_filters();
	for(int i=0;i<extensions.size();i++) {

		emission_file_dialog->add_filter("*."+extensions[i]+" ; "+extensions[i].to_upper());
	}

	emission_file_dialog->set_mode(EditorFileDialog::MODE_OPEN_FILE);

	//options->set_anchor(MARGIN_LEFT,Control::ANCHOR_END);
	//options->set_anchor(MARGIN_RIGHT,Control::ANCHOR_END);

}


void ParticlesEditorPlugin::edit(Object *p_object) {

	particles_editor->edit(p_object->cast_to<Particles>());
}

bool ParticlesEditorPlugin::handles(Object *p_object) const {

	return p_object->is_type("Particles");
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

	editor=p_node;
	particles_editor = memnew( ParticlesEditor );
	editor->get_viewport()->add_child(particles_editor);

	particles_editor->hide();
}


ParticlesEditorPlugin::~ParticlesEditorPlugin()
{
}


#endif
