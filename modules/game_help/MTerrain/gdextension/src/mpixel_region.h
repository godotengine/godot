#ifndef MPIXELREGION
#define MPIXELREGION

#include <stdint.h>
#include "core/templates/vector.h"




struct MPixelRegion {
    uint32_t left=0;
    uint32_t right=0;
    uint32_t top=0;
    uint32_t bottom=0;
    //uint32_t width=0;
    //uint32_t height=0;

    MPixelRegion();
    MPixelRegion(const uint32_t _left, const uint32_t _right, const uint32_t _top, const uint32_t _bottom);
    MPixelRegion(const uint32_t _width,const uint32_t _height);
    void grow_all_side(const MPixelRegion& limit);
    void grow_all_side(const MPixelRegion& limit,uint32_t amount);
    bool grow_positve(const uint32_t xamount,const uint32_t yamount,const MPixelRegion& limit);
    Vector<MPixelRegion> devide(uint32_t amount);
    //Recieve another pixel region and return that in this region local position
    MPixelRegion get_local(MPixelRegion region);
    uint32_t get_height();
    uint32_t get_width();
    uint32_t get_pixel_amount();
    void clear();
    void print_region(String prefix);
};

#endif