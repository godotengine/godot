#include "mchannel_painter.h"
#include "core/variant/variant_utility.h"



String MChannelPainter::_get_name(){
    return "Channel Painter";
}

void MChannelPainter::_set_property(String prop_name, Variant value){
    if(prop_name=="red"){
        red = value;
    }
    else if (prop_name=="green")
    {
        green = value;
    }
    else if (prop_name=="blue")
    {
        blue = value;
    }
    else if (prop_name=="alpha")
    {
        alpha = value;
    }
    else if (prop_name=="red-value")
    {
        red_value = value;
    }
    else if (prop_name=="green-value")
    {
        green_value = value;
    }
    else if (prop_name=="blue-value")
    {
        blue_value = value;
    }
    else if (prop_name=="alpha-value")
    {
        alpha_value = value;
    }
    else if (prop_name=="hardness")
    {
        float v = (float)value;
        hardness = CLAMP(v,0.0,0.99);
    }
}
bool MChannelPainter::is_two_point_brush(){
    return false;
}
void MChannelPainter::before_draw(){

}
void MChannelPainter::set_color(uint32_t local_x,uint32_t local_y,uint32_t x,uint32_t y,MImage* img){
    //Calculating w
    uint32_t dx = ABS(x - grid->brush_px_pos_x);
    uint32_t dy = ABS(y - grid->brush_px_pos_y);
    float px_dis = (float)sqrt(dx*dx + dy*dy);
    px_dis /= (float)grid->brush_px_radius;
    float w = VariantUtilityFunctions::smoothstep(1,hardness,px_dis);
    float mask = grid->get_brush_mask_value(x,y);
    mask = pow(mask,4.0);
    w = w * mask;
    // setting color
    const uint8_t* ptr = grid->get_pixel_by_pointer(x,y,grid->current_paint_index);
    uint32_t ofs = local_y*img->width + local_x;
    memcpy(img->data.ptrw() + ofs*img->pixel_size, ptr, img->pixel_size);
    if(red){
        float bg = img->get_pixel_in_channel(local_x,local_y,0);
        float val = (red_value - bg)*w + bg;
        img->set_pixel_in_channel(local_x,local_y,0,val);
    }
    if(green){
        float bg = img->get_pixel_in_channel(local_x,local_y,1);
        float val = (green_value - bg)*w + bg;
        img->set_pixel_in_channel(local_x,local_y,1,val);
    }
    if(blue){
        float bg = img->get_pixel_in_channel(local_x,local_y,2);
        float val = (blue_value - bg)*w + bg;
        img->set_pixel_in_channel(local_x,local_y,2,val);
    }
    if(alpha){
        float bg = img->get_pixel_in_channel(local_x,local_y,3);
        float val = (alpha_value - bg)*w + bg;
        img->set_pixel_in_channel(local_x,local_y,3,val);
    }
}