#include "mbitwise_brush.h"





String MBitwiseBrush::_get_name(){
    return "Bitwise Brush";
}
   //bool value=false;
    //uint8_t bit=0;
void MBitwiseBrush::_set_property(String prop_name, Variant _value){
    if(prop_name=="value"){
        value = _value;
    }
    else if(prop_name=="bit"){
        int tmp = _value;
        bit = (uint32_t)tmp;
    }
}
bool MBitwiseBrush::is_two_point_brush(){
    return false;
}
void MBitwiseBrush::before_draw(){
    uint32_t pixel_size = grid->regions[0].images[0]->pixel_size;
    ERR_FAIL_COND_MSG(bit>(pixel_size*8 - 1),"Bit is out of bound");
}
void MBitwiseBrush::set_color(uint32_t local_x,uint32_t local_y,uint32_t x,uint32_t y,MImage* img){
    //Calculating w
    uint32_t dx = ABS(x-grid->brush_px_pos_x);
    uint32_t dy = ABS(y-grid->brush_px_pos_y);
    float px_dis = (float)sqrt(dx*dx + dy*dy);
    px_dis /= (float)grid->brush_px_radius;
    // setting color
    const uint8_t* ptr = grid->get_pixel_by_pointer(x,y,grid->current_paint_index);
    uint32_t ofs = (local_y*img->width + local_x)*img->pixel_size;
    uint8_t* ptrw = img->data.ptrw() + ofs;
    memcpy(ptrw, ptr, img->pixel_size);
    if( bit>(img->pixel_size*8 - 1) ){
        return;
    }
    uint32_t ibyte = bit/8;
    uint32_t ibit = bit%8;
    uint8_t b = ptrw[ibyte];
    if(px_dis<1 && grid->get_brush_mask_value_bool(x,y)){
        if(value){
            b |= (1 << ibit);
        } else {
            b &= ~(1 << ibit);
        }
    }
    ptrw[ibyte] = b;
}