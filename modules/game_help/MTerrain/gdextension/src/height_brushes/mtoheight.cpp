#include "mtoheight.h"

#include "core/variant/variant.h"
#include "core/variant/variant_utility.h"


MToHeight::MToHeight(){

}
MToHeight::~MToHeight(){

}
String MToHeight::_get_name(){
    return "To Height";
}
Array MToHeight::_get_property_list(){
    Array props;
    // p1
    Dictionary p1;
    p1["name"] = "weight";
    p1["type"] = Variant::FLOAT;
    p1["hint"] = "range";
    p1["hint_string"] = "0.01";
    p1["default_value"] = weight;
    p1["min"] = 0.1;
    p1["max"] = 1.0;
    //p2
    Dictionary p2;
    p2["name"] = "hardness";
    p2["type"] = Variant::FLOAT;
    p2["hint"] = "range";
    p2["hint_string"] = "0.01";
    p2["default_value"] = hardness;
    p2["min"] = 0.1;
    p2["max"] = 0.95;
    //p3
    Dictionary p3;
    p3["name"] = "offset";
    p3["type"] = Variant::FLOAT;
    p3["hint"] = "";
    p3["hint_string"] = "";
    p3["default_value"] = offset;
    p3["min"] = -1000;
    p3["max"] = 1000;
    // p5
    Dictionary p5;
    p5["name"] = "mode";
    p5["type"] = Variant::INT;
    p5["hint"] = "enum";
    p5["hint_string"] = "RELATIVE,START_POINT,ABSOLUTE";
    p5["default_value"] = mode;
    p5["min"] = 0;
    p5["max"] = 2;
    props.append(p1);
    props.append(p2);
    props.append(p3);
    props.append(p5);
    return props;
}

void MToHeight::_set_property(String prop_name, Variant value){
    if (prop_name == "hardness"){
        hardness = value;
        return;
    } else if (prop_name == "offset")
    {
        offset = value;
        return;
    }
    else if (prop_name == "weight")
    {
        weight = value;
        return;
    } else if (prop_name == "mode"){
        mode = value;
    }
}

bool MToHeight::is_two_point_brush(){
    return false;
}

void MToHeight::before_draw(){
    start_height = grid->get_height(grid->brush_world_pos_start);
}
float MToHeight::get_height(uint32_t x,uint32_t y){
    Vector3 world_pos = grid->get_pixel_world_pos(x,y);
    real_t dis = grid->brush_world_pos.distance_to(world_pos);
    dis = dis/grid->brush_radius;
    dis = VariantUtilityFunctions::smoothstep(1,hardness,dis);
    float toh;
    if(mode==2){ // absoulte
        toh=offset;
    } else if(mode==0) { // relative
        toh = grid->brush_world_pos.y + offset;
    } else if(mode==1) { // relative to start position
        toh = start_height + offset;
    }
    float h = grid->get_height_by_pixel(x,y);
    float mask = grid->get_brush_mask_value(x,y);
    mask = pow(mask,2);
    return (toh - h)*weight*dis*mask + h;
}