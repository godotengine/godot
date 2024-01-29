#include "game_gui_space.h"

void GUISpace::_bind_methods() 
{
    ClassDB::bind_method(D_METHOD("set_margin_type", "type"), &GUISpace::set_margin_type);
    ClassDB::bind_method(D_METHOD("get_margin_type"), &GUISpace::get_margin_type);

    ClassDB::bind_method(D_METHOD("set_left_margin", "size"), &GUISpace::set_left_margin);
    ClassDB::bind_method(D_METHOD("get_left_margin"), &GUISpace::get_left_margin);

    ClassDB::bind_method(D_METHOD("set_right_margin", "size"), &GUISpace::set_right_margin);
    ClassDB::bind_method(D_METHOD("get_right_margin"), &GUISpace::get_right_margin);

    ClassDB::bind_method(D_METHOD("set_top_margin", "size"), &GUISpace::set_top_margin);
    ClassDB::bind_method(D_METHOD("get_top_margin"), &GUISpace::get_top_margin);

    ClassDB::bind_method(D_METHOD("set_bottom_margin", "size"), &GUISpace::set_bottom_margin);
    ClassDB::bind_method(D_METHOD("get_bottom_margin"), &GUISpace::get_bottom_margin);

    ClassDB::bind_method(D_METHOD("set_left_parameter", "value"), &GUISpace::set_left_parameter);
    ClassDB::bind_method(D_METHOD("get_left_parameter"), &GUISpace::get_left_parameter);

    ClassDB::bind_method(D_METHOD("set_right_parameter", "value"), &GUISpace::set_right_parameter);
    ClassDB::bind_method(D_METHOD("get_right_parameter"), &GUISpace::get_right_parameter);

    ClassDB::bind_method(D_METHOD("set_top_parameter", "value"), &GUISpace::set_top_parameter);
    ClassDB::bind_method(D_METHOD("get_top_parameter"), &GUISpace::get_top_parameter);

    ClassDB::bind_method(D_METHOD("set_bottom_parameter", "value"), &GUISpace::set_bottom_parameter);
    ClassDB::bind_method(D_METHOD("get_bottom_parameter"), &GUISpace::get_bottom_parameter);


    ADD_GROUP("Space", "space_");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "space_margin_type", PROPERTY_HINT_ENUM, "PROPORTIONAL,FIXED,PARAMETER"), "set_margin_type", "get_margin_type");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "space_left", PROPERTY_HINT_RANGE, "0.0,1,0.0001,or_less,or_greater"), "set_left_margin", "get_left_margin");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "space_right", PROPERTY_HINT_RANGE, "0.0,1,0.0001,or_less,or_greater"), "set_right_margin", "get_right_margin");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "space_top", PROPERTY_HINT_RANGE, "0.0,1,0.0001,or_less,or_greater"), "set_top_margin", "get_top_margin");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "space_bottom", PROPERTY_HINT_RANGE, "0.0,1,0.0001,or_less,or_greater"), "set_bottom_margin", "get_bottom_margin");

    ADD_GROUP("Space Parameter", "space_parameter_");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "space_left_parameter"), "set_left_parameter", "get_left_parameter");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "space_right_parameter"), "set_right_parameter", "get_right_parameter");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "space_top_parameter"), "set_top_parameter", "get_top_parameter");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "space_bottom_parameter"), "set_bottom_parameter", "get_bottom_parameter");


	BIND_ENUM_CONSTANT(PROPORTIONAL);
	BIND_ENUM_CONSTANT(FIXED);
	BIND_ENUM_CONSTANT(PARAMETER);

}