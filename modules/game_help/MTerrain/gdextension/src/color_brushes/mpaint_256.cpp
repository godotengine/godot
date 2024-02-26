#include "mpaint_256.h"
#include "core/variant/variant_utility.h"


String MPaint256::_get_name(){
    return "Paint 256";
}
void MPaint256::_set_property(String prop_name, Variant value){
    if(prop_name=="paint-layer"){
        int tmp = value;
        paint_layer = tmp;
    }
}
bool MPaint256::is_two_point_brush(){
    return false;
}
void MPaint256::before_draw(){

}
void MPaint256::set_color(uint32_t local_x,uint32_t local_y,uint32_t x,uint32_t y,MImage* img){
    //Calculating w
    float dx = (float)ABS(x - grid->brush_px_pos_x);
    float dy = (float)ABS(y - grid->brush_px_pos_y);
    float px_dis = sqrt(dx*dx + dy*dy);
    // setting color
    const uint8_t* ptr = grid->get_pixel_by_pointer(x,y,grid->current_paint_index);
    uint32_t ofs = (local_y*img->width + local_x)*img->pixel_size;
    uint8_t* ptrw = img->data.ptrw() + ofs;
    memcpy(ptrw, ptr, img->pixel_size);
    if(px_dis<(float)grid->brush_px_radius && grid->get_brush_mask_value_bool(x,y)){
        ptrw[0]=paint_layer;
    }
}