#ifndef MCHANNELPAINTER
#define MCHANNELPAINTER

#include "../mcolor_brush.h"


class MChannelPainter : public MColorBrush{
    bool red = false;
    bool green = false;
    bool blue = false;
    bool alpha = false;
    float red_value=0;
    float green_value=0;
    float blue_value=0;
    float alpha_value=0;
    float hardness = 0.9;
    String _get_name();
    void _set_property(String prop_name, Variant value);
    bool is_two_point_brush();
    void before_draw();
    void set_color(uint32_t local_x,uint32_t local_y,uint32_t x,uint32_t y,MImage* img);
};



#endif