#include "game_gui_arbitrary_coord.h"


    void GUIArbitraryCoord::_bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_scale_font", "value"), &GUIArbitraryCoord::set_scale_font);
        ClassDB::bind_method(D_METHOD("get_scale_font"), &GUIArbitraryCoord::get_scale_font);

        ClassDB::bind_method(D_METHOD("set_reference_height", "value"), &GUIArbitraryCoord::set_reference_height);
        ClassDB::bind_method(D_METHOD("get_reference_height"), &GUIArbitraryCoord::get_reference_height);

        ClassDB::bind_method(D_METHOD("set_reference_font_size", "value"), &GUIArbitraryCoord::set_reference_font_size);
        ClassDB::bind_method(D_METHOD("get_reference_font_size"), &GUIArbitraryCoord::get_reference_font_size);

        ClassDB::bind_method(D_METHOD("set_positioning_mode", "value"), &GUIArbitraryCoord::set_positioning_mode);
        ClassDB::bind_method(D_METHOD("get_positioning_mode"), &GUIArbitraryCoord::get_positioning_mode);


        ClassDB::bind_method(D_METHOD("set_child_x", "value"), &GUIArbitraryCoord::set_child_x);
        ClassDB::bind_method(D_METHOD("get_child_x"), &GUIArbitraryCoord::get_child_x);

        ClassDB::bind_method(D_METHOD("set_child_y", "value"), &GUIArbitraryCoord::set_child_y);
        ClassDB::bind_method(D_METHOD("get_child_y"), &GUIArbitraryCoord::get_child_y);

        ClassDB::bind_method(D_METHOD("set_child_x_parameter", "value"), &GUIArbitraryCoord::set_child_x_parameter);
        ClassDB::bind_method(D_METHOD("get_child_x_parameter"), &GUIArbitraryCoord::get_child_x_parameter);

        ClassDB::bind_method(D_METHOD("set_child_y_parameter", "value"), &GUIArbitraryCoord::set_child_y_parameter);
        ClassDB::bind_method(D_METHOD("get_child_y_parameter"), &GUIArbitraryCoord::get_child_y_parameter);

        
        ClassDB::bind_method(D_METHOD("set_h_scale_factor", "value"), &GUIArbitraryCoord::set_h_scale_factor);
        ClassDB::bind_method(D_METHOD("get_h_scale_factor"), &GUIArbitraryCoord::get_h_scale_factor);

        ClassDB::bind_method(D_METHOD("set_v_scale_factor", "value"), &GUIArbitraryCoord::set_v_scale_factor);
        ClassDB::bind_method(D_METHOD("get_v_scale_factor"), &GUIArbitraryCoord::get_v_scale_factor);

        
        ClassDB::bind_method(D_METHOD("set_h_scale_constant"), &GUIArbitraryCoord::set_h_scale_constant);
        ClassDB::bind_method(D_METHOD("get_h_scale_constant"), &GUIArbitraryCoord::get_h_scale_constant);

        ClassDB::bind_method(D_METHOD("set_v_scale_constant"), &GUIArbitraryCoord::set_v_scale_constant);
        ClassDB::bind_method(D_METHOD("get_v_scale_constant"), &GUIArbitraryCoord::get_v_scale_constant);

         
        ClassDB::bind_method(D_METHOD("set_h_scale_parameter"), &GUIArbitraryCoord::set_h_scale_parameter);
        ClassDB::bind_method(D_METHOD("get_h_scale_parameter"), &GUIArbitraryCoord::get_h_scale_parameter);

        ClassDB::bind_method(D_METHOD("set_v_scale_parameter"), &GUIArbitraryCoord::set_v_scale_parameter);
        ClassDB::bind_method(D_METHOD("get_v_scale_parameter"), &GUIArbitraryCoord::get_v_scale_parameter);




        ADD_GROUP("ArbitraryCoord Pos", "arbitrary_pos_");
        ADD_PROPERTY(PropertyInfo(Variant::INT, "arbitrary_pos_positioning_mode",PROPERTY_HINT_ENUM,"PROPORTIONAL,FIXED,PARAMETER"), "set_positioning_mode", "get_positioning_mode");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "arbitrary_pos_child_x", PROPERTY_HINT_RANGE, "0,1,0.0001,or_greater"), "set_child_x", "get_child_x");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "arbitrary_pos_child_y", PROPERTY_HINT_RANGE, "0,1,0.0001,or_greater"), "set_child_y", "get_child_y");
        ADD_PROPERTY(PropertyInfo(Variant::STRING, "arbitrary_pos_child_x_parameter"), "set_child_x_parameter", "get_child_x_parameter");
        ADD_PROPERTY(PropertyInfo(Variant::STRING, "arbitrary_pos_child_y_parameter"), "set_child_y_parameter", "get_child_y_parameter");

        
        ADD_GROUP("ArbitraryCoord Scale", "arbitrary_scale_");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "arbitrary_scale_h_scale_factor", PROPERTY_HINT_ENUM,"CONSTANT,PARAMETER"), "set_h_scale_factor", "get_h_scale_factor");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "arbitrary_scale_v_scale_factor", PROPERTY_HINT_ENUM,"CONSTANT,PARAMETER"), "set_v_scale_factor", "get_v_scale_factor");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "arbitrary_scale_h_scale_constant", PROPERTY_HINT_RANGE, "0,1,0.0001,or_greater"), "set_h_scale_constant", "get_h_scale_constant");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "arbitrary_scale_v_scale_constant", PROPERTY_HINT_RANGE, "0,1,0.0001,or_greater"), "set_v_scale_constant", "get_v_scale_constant");
        ADD_PROPERTY(PropertyInfo(Variant::STRING, "arbitrary_scale_h_scale_parameter"), "set_h_scale_parameter", "get_h_scale_parameter");
        ADD_PROPERTY(PropertyInfo(Variant::STRING, "arbitrary_scale_v_scale_parameter"), "set_v_scale_parameter", "get_v_scale_parameter");



        ADD_GROUP("Font Scale", "font_scale_");
        ADD_PROPERTY(PropertyInfo(Variant::BOOL, "font_scale_scale_font"), "set_scale_font", "set_scale_font");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "font_scale_reference_height"), "set_reference_height", "get_reference_height");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "font_scale_reference_font_size"), "set_reference_font_size", "get_reference_font_size");


    }