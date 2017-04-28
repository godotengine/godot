/*************************************************************************/
/*  mesh_editor_plugin.cpp                                               */
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
#include "mesh_editor_plugin.h"

#if 0
void MeshEditor::_gui_input(InputEvent p_event) {


	if (p_event.type==InputEvent::MOUSE_MOTION && p_event.mouse_motion.button_mask&BUTTON_MASK_LEFT) {

		rot_x-=p_event.mouse_motion.relative_y*0.01;
		rot_y-=p_event.mouse_motion.relative_x*0.01;
		if (rot_x<-Math_PI/2)
			rot_x=-Math_PI/2;
		else if (rot_x>Math_PI/2) {
			rot_x=Math_PI/2;
		}
		_update_rotation();
	}
}

void MeshEditor::_notification(int p_what) {

	if (p_what==NOTIFICATION_FIXED_PROCESS) {

	}


	if (p_what==NOTIFICATION_READY) {

		//get_scene()->connect("node_removed",this,"_node_removed");

		if (first_enter) {
			//it's in propertyeditor so.. could be moved around

			light_1_switch->set_normal_texture(get_icon("MaterialPreviewLight1","EditorIcons"));
			light_1_switch->set_pressed_texture(get_icon("MaterialPreviewLight1Off","EditorIcons"));
			light_2_switch->set_normal_texture(get_icon("MaterialPreviewLight2","EditorIcons"));
			light_2_switch->set_pressed_texture(get_icon("MaterialPreviewLight2Off","EditorIcons"));
			first_enter=false;
		}

	}

	if (p_what==NOTIFICATION_DRAW) {


		Ref<Texture> checkerboard = get_icon("Checkerboard","EditorIcons");
		Size2 size = get_size();

		draw_texture_rect(checkerboard,Rect2(Point2(),size),true);

	}
}

void MeshEditor::_update_rotation() {

	Transform t;
	t.basis.rotate(Vector3(0, 1, 0), -rot_y);
	t.basis.rotate(Vector3(1, 0, 0), -rot_x);
	mesh_instance->set_transform(t);

}

void MeshEditor::edit(Ref<Mesh> p_mesh) {

	mesh=p_mesh;
	mesh_instance->set_mesh(mesh);

	if (mesh.is_null()) {

		hide();
	} else {
		rot_x=0;
		rot_y=0;
		_update_rotation();

		AABB aabb= mesh->get_aabb();
		Vector3 ofs = aabb.pos + aabb.size*0.5;
		aabb.pos-=ofs;
		float m = MAX(aabb.size.x,aabb.size.y)*0.5;
		if (m!=0) {
			m=1.0/m;
			m*=0.5;
			//print_line("scale: "+rtos(m));
			Transform xform;
			xform.basis.scale(Vector3(m,m,m));
			xform.origin=-xform.basis.xform(ofs); //-ofs*m;
			xform.origin.z-=aabb.size.z*2;
			mesh_instance->set_transform(xform);
		}

	}

}


void MeshEditor::_button_pressed(Node* p_button) {

	if (p_button==light_1_switch) {
		light1->set_enabled(!light_1_switch->is_pressed());
	}

	if (p_button==light_2_switch) {
		light2->set_enabled(!light_2_switch->is_pressed());
	}


}

void MeshEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_gui_input"),&MeshEditor::_gui_input);
	ClassDB::bind_method(D_METHOD("_button_pressed"),&MeshEditor::_button_pressed);

}

MeshEditor::MeshEditor() {

	viewport = memnew( Viewport );
	Ref<World> world;
	world.instance();
	viewport->set_world(world); //use own world
	add_child(viewport);
	viewport->set_disable_input(true);

	camera = memnew( Camera );
	camera->set_transform(Transform(Matrix3(),Vector3(0,0,3)));
	camera->set_perspective(45,0.1,10);
	viewport->add_child(camera);

	light1 = memnew( DirectionalLight );
	light1->set_transform(Transform().looking_at(Vector3(-1,-1,-1),Vector3(0,1,0)));
	viewport->add_child(light1);

	light2 = memnew( DirectionalLight );
	light2->set_transform(Transform().looking_at(Vector3(0,1,0),Vector3(0,0,1)));
	light2->set_color(Light::COLOR_DIFFUSE,Color(0.7,0.7,0.7));
	light2->set_color(Light::COLOR_SPECULAR,Color(0.7,0.7,0.7));
	viewport->add_child(light2);

	mesh_instance = memnew( MeshInstance );
	viewport->add_child(mesh_instance);



	set_custom_minimum_size(Size2(1,150)*EDSCALE);

	HBoxContainer *hb = memnew( HBoxContainer );
	add_child(hb);
	hb->set_area_as_parent_rect(2);

	hb->add_spacer();

	VBoxContainer *vb_light = memnew( VBoxContainer );
	hb->add_child(vb_light);

	light_1_switch = memnew( TextureButton );
	light_1_switch->set_toggle_mode(true);
	vb_light->add_child(light_1_switch);
	light_1_switch->connect("pressed",this,"_button_pressed",varray(light_1_switch));

	light_2_switch = memnew( TextureButton );
	light_2_switch->set_toggle_mode(true);
	vb_light->add_child(light_2_switch);
	light_2_switch->connect("pressed",this,"_button_pressed",varray(light_2_switch));

	first_enter=true;

	rot_x=0;
	rot_y=0;


}


void MeshEditorPlugin::edit(Object *p_object) {

	Mesh * s = p_object->cast_to<Mesh>();
	if (!s)
		return;

	mesh_editor->edit(Ref<Mesh>(s));
}

bool MeshEditorPlugin::handles(Object *p_object) const {

	return p_object->is_type("Mesh");
}

void MeshEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		mesh_editor->show();
		//mesh_editor->set_process(true);
	} else {

		mesh_editor->hide();
		//mesh_editor->set_process(false);
	}

}

MeshEditorPlugin::MeshEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	mesh_editor = memnew( MeshEditor );
	add_control_to_container(CONTAINER_PROPERTY_EDITOR_BOTTOM,mesh_editor);
	mesh_editor->hide();



}


MeshEditorPlugin::~MeshEditorPlugin()
{
}
#endif
