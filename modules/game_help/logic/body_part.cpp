#include "body_part.h"

void CharacterBodyPart::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_skin","p_skin"), &CharacterBodyPart::set_skin);
    ClassDB::bind_method(D_METHOD("get_skin"), &CharacterBodyPart::get_skin);

    ClassDB::bind_method(D_METHOD("set_mesh","p_mesh"), &CharacterBodyPart::set_mesh);
    ClassDB::bind_method(D_METHOD("get_mesh"), &CharacterBodyPart::get_mesh);

    ClassDB::bind_method(D_METHOD("set_material","p_material"), &CharacterBodyPart::set_material);
    ClassDB::bind_method(D_METHOD("get_material"), &CharacterBodyPart::get_material);

    ClassDB::bind_method(D_METHOD("init_form_mesh_instance","p_mesh_instance","bone_mapping"), &CharacterBodyPart::init_form_mesh_instance,DEFVAL(Dictionary()));

    ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,"skin"), "set_skin", "get_skin");
    ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,"mesh"), "set_mesh", "get_mesh");
    ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH,"material"), "set_material", "get_material");
    
}