/*************************************************************************/
/*  baked_light_editor_plugin.cpp                                        */
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
#include "baked_light_editor_plugin.h"

#include "io/marshalls.h"
#include "io/resource_saver.h"
#include "scene/3d/mesh_instance.h"
#include "scene/gui/box_container.h"

#if 0


void BakedLightEditor::_end_baking() {

	baker->clear();
	set_process(false);
	button_bake->set_pressed(false);
	bake_info->set_text("");
}

void BakedLightEditor::_node_removed(Node *p_node) {

	if(p_node==node) {
		_end_baking();
		node=NULL;

		hide();
	}

}





void BakedLightEditor::_notification(int p_option) {


	if (p_option==NOTIFICATION_ENTER_TREE) {

		button_bake->set_icon(get_icon("Bake","EditorIcons"));
		button_reset->set_icon(get_icon("Reload","EditorIcons"));
		button_make_lightmaps->set_icon(get_icon("LightMap","EditorIcons"));
	}

	if (p_option==NOTIFICATION_PROCESS) {

		if (baker->is_baking() && !baker->is_paused()) {

			update_timeout-=get_process_delta_time();
			if (update_timeout<0) {

				if (baker->get_baked_light()!=node->get_baked_light()) {
					_end_baking();
					return;
				}

				uint64_t t = OS::get_singleton()->get_ticks_msec();

#ifdef DEBUG_CUBES
				double norm =  baker->get_normalization();
				float max_lum=0;

				{
					PoolVector<Color>::Write cw=colors.write();
					BakedLightBaker::Octant *octants=baker->octant_pool.ptr();
					BakedLightBaker::Octant *oct = &octants[baker->leaf_list];
					int vert_idx=0;

					while(oct) {



						Color colors[8];
						for(int i=0;i<8;i++) {

							colors[i].r=oct->light_accum[i][0]/norm;
							colors[i].g=oct->light_accum[i][1]/norm;
							colors[i].b=oct->light_accum[i][2]/norm;

							float lum = colors[i].get_v();
							/*
							if (lum<0.05)
								color.a=0;
							*/
							if (lum>max_lum)
								max_lum=lum;

						}
						static const int vert2cub[36]={7,3,1,1,5,7,7,6,2,2,3,7,7,5,4,4,6,7,2,6,4,4,0,2,4,5,1,1,0,4,1,3,2,2,0,1};
						for (int i=0;i<36;i++) {


							cw[vert_idx++]=colors[vert2cub[i]];
						}

						if (oct->next_leaf)
							oct=&octants[oct->next_leaf];
						else
							oct=NULL;

					}
				}
				print_line("MSCOL: "+itos(OS::get_singleton()->get_ticks_msec()-t));
				t = OS::get_singleton()->get_ticks_msec();

				Array a;
				a.resize(Mesh::ARRAY_MAX);
				a[Mesh::ARRAY_VERTEX]=vertices;
				a[Mesh::ARRAY_COLOR]=colors;
				while(mesh->get_surface_count())
					mesh->surface_remove(0);
				mesh->add_surface(Mesh::PRIMITIVE_TRIANGLES,a);
				mesh->surface_set_material(0,material);
#endif
				ERR_FAIL_COND(node->get_baked_light().is_null());

				baker->update_octree_images(octree_texture,light_texture);
				baker->update_octree_sampler(octree_sampler);
				//print_line("sampler size: "+itos(octree_sampler.size()*4));

#if 1
//debug
				Image img(baker->baked_octree_texture_w,baker->baked_octree_texture_h,0,Image::FORMAT_RGBA8,octree_texture);
				Ref<ImageTexture> it = memnew( ImageTexture );
				it->create_from_image(img);
				ResourceSaver::save("baked_octree.png",it);

#endif


				uint64_t rays_snap = baker->get_rays_thrown();
				int rays_sec = (rays_snap-last_rays_time)*1.0-(update_timeout);
				last_rays_time=rays_snap;

				bake_info->set_text("rays/s: "+itos(rays_sec));
				update_timeout=1;
				print_line("MSUPDATE: "+itos(OS::get_singleton()->get_ticks_msec()-t));
				t=OS::get_singleton()->get_ticks_msec();
				node->get_baked_light()->set_octree(octree_texture);
				node->get_baked_light()->set_light(light_texture);
				node->get_baked_light()->set_sampler_octree(octree_sampler);
				node->get_baked_light()->set_edited(true);

				print_line("MSSET: "+itos(OS::get_singleton()->get_ticks_msec()-t));



			}
		}
	}
}


void BakedLightEditor::_menu_option(int p_option) {


	switch(p_option) {


		case MENU_OPTION_BAKE: {

			ERR_FAIL_COND(!node);
			ERR_FAIL_COND(node->get_baked_light().is_null());
			baker->bake(node->get_baked_light(),node);
			node->get_baked_light()->set_mode(BakedLight::MODE_OCTREE);
			update_timeout=0;
			set_process(true);


		} break;
		case MENU_OPTION_CLEAR: {



		} break;

	}
}

void BakedLightEditor::_bake_pressed() {

	ERR_FAIL_COND(!node);
	const String conf_warning = node->get_configuration_warning();
	if (!conf_warning.empty()) {
		err_dialog->set_text(conf_warning);
		err_dialog->popup_centered_minsize();
		button_bake->set_pressed(false);
		return;
	}

	if (baker->is_baking()) {

		baker->set_pause(!button_bake->is_pressed());
		if (baker->is_paused()) {

			set_process(false);
			bake_info->set_text("");
			button_reset->show();
			button_make_lightmaps->show();

		} else {

			update_timeout=0;
			set_process(true);
			button_make_lightmaps->hide();
			button_reset->hide();
		}
	} else {
		baker->bake(node->get_baked_light(),node);
		node->get_baked_light()->set_mode(BakedLight::MODE_OCTREE);
		update_timeout=0;

		last_rays_time=0;
		button_bake->set_pressed(false);

		set_process(true);
	}

}

void BakedLightEditor::_clear_pressed(){

	baker->clear();
	button_bake->set_pressed(false);
	bake_info->set_text("");

}

void BakedLightEditor::edit(BakedLightInstance *p_baked_light) {

	if (p_baked_light==NULL || node==p_baked_light) {
		return;
	}
	if (node && node!=p_baked_light)
		_end_baking();


	node=p_baked_light;
	//_end_baking();

}

void BakedLightEditor::_bake_lightmaps() {

	Error err = baker->transfer_to_lightmaps();
	if (err) {

		err_dialog->set_text("Error baking to lightmaps!\nMake sure that a bake has just\n happened and that lightmaps are\n configured. ");
		err_dialog->popup_centered_minsize();
		return;
	}

	node->get_baked_light()->set_mode(BakedLight::MODE_LIGHTMAPS);


}

void BakedLightEditor::_bind_methods() {

	ClassDB::bind_method("_menu_option",&BakedLightEditor::_menu_option);
	ClassDB::bind_method("_bake_pressed",&BakedLightEditor::_bake_pressed);
	ClassDB::bind_method("_clear_pressed",&BakedLightEditor::_clear_pressed);
	ClassDB::bind_method("_bake_lightmaps",&BakedLightEditor::_bake_lightmaps);
}

BakedLightEditor::BakedLightEditor() {


	bake_hbox = memnew( HBoxContainer );
	button_bake = memnew( ToolButton );
	button_bake->set_text(TTR("Bake!"));
	button_bake->set_toggle_mode(true);
	button_reset = memnew( Button );
	button_make_lightmaps = memnew( Button );
	button_bake->set_tooltip("Start/Unpause the baking process.\nThis bakes lighting into the lightmap octree.");
	button_make_lightmaps ->set_tooltip("Convert the lightmap octree to lightmap textures\n(must have set up UV/Lightmaps properly before!).");


	bake_info = memnew( Label );
	bake_hbox->add_child( button_bake );
	bake_hbox->add_child( button_reset );
	bake_hbox->add_child( bake_info );

	err_dialog = memnew( AcceptDialog );
	add_child(err_dialog);
	node=NULL;
	baker = memnew( BakedLightBaker );

	bake_hbox->add_child(button_make_lightmaps);
	button_make_lightmaps->hide();

	button_bake->connect("pressed",this,"_bake_pressed");
	button_reset->connect("pressed",this,"_clear_pressed");
	button_make_lightmaps->connect("pressed",this,"_bake_lightmaps");
	button_reset->hide();
	button_reset->set_tooltip(TTR("Reset the lightmap octree baking process (start over)."));


	update_timeout=0;



}

BakedLightEditor::~BakedLightEditor() {

	memdelete(baker);
}

void BakedLightEditorPlugin::edit(Object *p_object) {

	baked_light_editor->edit(p_object->cast_to<BakedLightInstance>());
}

bool BakedLightEditorPlugin::handles(Object *p_object) const {

	return p_object->is_type("BakedLightInstance");
}

void BakedLightEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		baked_light_editor->show();
		baked_light_editor->bake_hbox->show();
	} else {

		baked_light_editor->hide();
		baked_light_editor->bake_hbox->hide();
		baked_light_editor->edit(NULL);
	}

}

BakedLightEditorPlugin::BakedLightEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	baked_light_editor = memnew( BakedLightEditor );
	editor->get_viewport()->add_child(baked_light_editor);
	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU,baked_light_editor->bake_hbox);
	baked_light_editor->hide();
	baked_light_editor->bake_hbox->hide();
}


BakedLightEditorPlugin::~BakedLightEditorPlugin()
{
}

#endif
