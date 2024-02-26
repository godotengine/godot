#ifndef MREMOVELAYER
#define MREMOVELAYER

#include "../mheight_brush.h"

class MRemoveLayer : public MHeightBrush {
    public:
    float hardness=0.5;
    float amount=0.5;
    MRemoveLayer();
    ~MRemoveLayer();
    String _get_name();
    Array _get_property_list();
    void _set_property(String prop_name, Variant value);
    bool is_two_point_brush();
    void before_draw();
    float get_height(uint32_t x,uint32_t y);
};
#endif