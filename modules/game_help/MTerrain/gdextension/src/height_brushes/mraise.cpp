#include "mraise.h"
#include "core/variant/variant.h"
#include "core/variant/variant_utility.h"




MRaise::MRaise(){

}
MRaise::~MRaise(){

}
String MRaise::_get_name(){
    return "Raise";
}
//{"name":"name of props", type:Variant_type,hint:"Type hint",hint_string:"", default:default_value, min:min_value, max:max_value}
Array MRaise::_get_property_list(){
    Array props;
    // p1
    Dictionary p1;
    p1["name"] = "hardness";
    p1["type"] = Variant::FLOAT;
    p1["hint"] = "range";
    p1["hint_string"] = "0.001";
    p1["default_value"] = hardness;
    p1["min"] = 0.0;
    p1["max"] = 0.95;
    //p2
    Dictionary p2;
    p2["name"] = "amount";
    p2["type"] = Variant::FLOAT;
    p2["hint"] = "range";
    p2["hint_string"] = "0.01";
    p2["default_value"] = amount;
    p2["min"] = 0;
    p2["max"] = 1;
    //p3
    Dictionary p3;
    p3["name"] = "revers";
    p3["type"] = Variant::BOOL;
    p3["hint"] = "";
    p3["hint_string"] = "";
    p3["default_value"] = revers;
    p3["min"] = 0;
    p3["max"] = 1;
    props.append(p1);
    props.append(p2);
    props.append(p3);
    return props;
}
void MRaise::_set_property(String prop_name, Variant value){
    if (prop_name == "hardness"){
        hardness = value;
        return;
    } else if (prop_name == "amount")
    {
        amount = value;
        return;
    } else if (prop_name == "revers"){
        revers = value;
    }
}

bool MRaise::is_two_point_brush(){
    return false;
}

void MRaise::before_draw(){
    final_amount = amount*sqrt(grid->brush_radius/50.0);
    if(revers){
        final_amount *= -1.0;
    }
}
float MRaise::get_height(uint32_t x,uint32_t y){
    Vector3 world_pos = grid->get_pixel_world_pos(x,y);
    real_t dis = grid->brush_world_pos.distance_to(world_pos);
    dis = dis/grid->brush_radius;
    dis = VariantUtilityFunctions::smoothstep(1,hardness,dis)*final_amount*grid->get_brush_mask_value(x,y);

    return world_pos.y + dis;
}