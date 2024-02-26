#include "mhole.h"



MHole::MHole(){

}

MHole::~MHole(){

}

String MHole::_get_name(){
    return "Hole";
}

Array MHole::_get_property_list(){
    Array props;
    // p1
    Dictionary p1;
    p1["name"] = "add";
    p1["type"] = Variant::BOOL;
    p1["hint"] = "";
    p1["hint_string"] = "0.";
    p1["default_value"] = add;
    p1["min"] = 0.;
    p1["max"] = 0.;
    props.push_back(p1);
    return props;
}

void MHole::_set_property(String prop_name, Variant value){
    if(prop_name=="add"){
        add = value;
    }
}

bool MHole::is_two_point_brush(){
    return false;
}

void MHole::before_draw(){

}
float MHole::get_height(uint32_t x,uint32_t y){
    uint32_t dx = ABS(x - grid->brush_px_pos_x);
    uint32_t dy = ABS(y - grid->brush_px_pos_y);
    float px_dis = (float)sqrt(dx*dx + dy*dy);
    px_dis /= (float)grid->brush_px_radius;
    float h = grid->get_height_by_pixel(x,y);
    if(px_dis<1.0){
        if(add){
            return std::numeric_limits<float>::quiet_NaN();
        } else {
            if(std::isnan(h)){
                return 0.0;
            }
        }
    }
    return h;
}