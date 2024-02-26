#include "mpaint_16.h"


String MPaint16::_get_name(){
    return "Paint 16";
}
void MPaint16::_set_property(String prop_name, Variant value){
    if(prop_name=="paint-layer"){
        int tmp = value;
        paint_layer = tmp;
    }
}
bool MPaint16::is_two_point_brush(){
    return false;
}
void MPaint16::before_draw(){
}
void MPaint16::set_color(uint32_t local_x,uint32_t local_y,uint32_t x,uint32_t y,MImage* img){
    //Calculating w
    float dx = (float)ABS(x-grid->brush_px_pos_x);
    float dy = (float)ABS(y-grid->brush_px_pos_y);
    float px_dis = sqrt(dx*dx + dy*dy);
    // setting color
    const uint8_t* ptr = grid->get_pixel_by_pointer(x,y,grid->current_paint_index);
    uint32_t ofs = (local_y*img->width + local_x);
    uint8_t* ptrw = img->data.ptrw() + ofs*img->pixel_size;
    memcpy(ptrw, ptr, img->pixel_size);
    if(px_dis<(float)grid->brush_px_radius && grid->get_brush_mask_value_bool(x,y)){
        uint16_t u = ((uint16_t *)img->data.ptrw())[ofs];
        u &= 0x0FFF;
        u |= paint_layer << 12;
        ((uint16_t *)img->data.ptrw())[ofs] = u;
    }
}