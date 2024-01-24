#include "game_gui_compoent.h"

void GUIComponent::_bind_methods()
{

    ClassDB::bind_method(D_METHOD("set_horizontal_mode", "mode"), &GUIComponent::set_horizontal_mode);
    ClassDB::bind_method(D_METHOD("get_horizontal_mode"), &GUIComponent::get_horizontal_mode);

    ClassDB::bind_method(D_METHOD("set_vertical_mode", "mode"), &GUIComponent::set_vertical_mode);
    ClassDB::bind_method(D_METHOD("get_vertical_mode"), &GUIComponent::get_vertical_mode);

    ClassDB::bind_method(D_METHOD("set_layout_size", "value"), &GUIComponent::set_layout_size);
    ClassDB::bind_method(D_METHOD("get_layout_size"), &GUIComponent::get_layout_size);


    ClassDB::bind_method(D_METHOD("set_reference_node", "node"), &GUIComponent::set_reference_node);
    ClassDB::bind_method(D_METHOD("get_reference_node"), &GUIComponent::get_reference_node);

    
    ClassDB::bind_method(D_METHOD("set_width_parameter", "value"), &GUIComponent::set_width_parameter);
    ClassDB::bind_method(D_METHOD("get_width_parameter"), &GUIComponent::get_width_parameter);

    
    
    ClassDB::bind_method(D_METHOD("set_height_parameter", "value"), &GUIComponent::set_height_parameter);
    ClassDB::bind_method(D_METHOD("get_height_parameter"), &GUIComponent::get_height_parameter);

    
    ClassDB::bind_method(D_METHOD("set_parameters", "parameters"), &GUIComponent::set_parameters);
    ClassDB::bind_method(D_METHOD("get_parameters"), &GUIComponent::get_parameters);

	ADD_GROUP("Component Layout", "component_");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "component_horizontal_mode", PROPERTY_HINT_ENUM, "EXPAND_TO_FILL,ASPECT_FIT,ASPECT_FIT,PROPORTIONAL,SHRINK_TO_FIT,FIXED,PARAMETER"), "set_horizontal_mode", "get_horizontal_mode");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "component_vertical_mode", PROPERTY_HINT_ENUM, "EXPAND_TO_FILL,ASPECT_FIT,ASPECT_FIT,PROPORTIONAL,SHRINK_TO_FIT,FIXED,PARAMETER"), "set_vertical_mode", "get_vertical_mode");
    ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "component_layout_size"), "set_layout_size", "get_layout_size");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "component_reference_node",PROPERTY_HINT_NODE_TYPE, "Control"), "set_reference_node", "get_reference_node");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "component_width_parameter"), "set_width_parameter", "get_width_parameter");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "component_height_parameter"), "set_height_parameter", "get_height_parameter");
    ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "component_parameters",  PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_INTERNAL), "set_parameters", "get_parameters");



	BIND_ENUM_CONSTANT(S_EXPAND_TO_FILL);
	BIND_ENUM_CONSTANT(S_ASPECT_FIT);
	BIND_ENUM_CONSTANT(S_ASPECT_FILL);
	BIND_ENUM_CONSTANT(S_PROPORTIONAL);
	BIND_ENUM_CONSTANT(S_SHRINK_TO_FIT);
	BIND_ENUM_CONSTANT(S_FIXED);
	BIND_ENUM_CONSTANT(S_PARAMETER);

    
	BIND_ENUM_CONSTANT(DEFAULT);
	BIND_ENUM_CONSTANT(SCALE);
	BIND_ENUM_CONSTANT(PARAMETER);


	BIND_ENUM_CONSTANT(STRETCH);
	BIND_ENUM_CONSTANT(TILE);
	BIND_ENUM_CONSTANT(TILE_FIT);

    ADD_SIGNAL(MethodInfo("begin_layout"));
    ADD_SIGNAL(MethodInfo("end_layout"));

}
