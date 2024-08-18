#ifndef MBOUND
#define MBOUND

#include "core/math/math_funcs.h"
#include "core/math/rect2i.h"
#include "core/templates/vector.h"
#include <math.h>

#include "mconfig.h"





struct MGridPos
{
    int32_t x = 0;
    int32_t y = 0;
    int32_t z = 0;
    MGridPos(){};
    MGridPos(const int32_t _x,const int32_t _y,const int32_t _z){
        x = _x;
        y= _y;
        z=_z;
    }

    int32_t get_distance(const MGridPos& other){
        if(Math::is_nan((float)y) || Math::is_nan((float)other.y)){ // in case there is a hole on ground, solution for now
            return sqrt(  pow((x - other.x),2) + pow((z-other.z),2) );
        }
        return sqrt(  pow((x - other.x),2) + pow((y-other.y),2) + pow((z-other.z),2) );
    }
};


struct  MBound
{
    int32_t left = 0;
    int32_t right = 0;
    int32_t top = 0;
    int32_t bottom = 0;
    MGridPos center;
    bool grow_left = true;
    bool grow_right = true;
    bool grow_top = true;
    bool grow_bottom = true;
    MGridPos cursor;

    MBound();
    MBound(const int32_t _left,const int32_t _right,const int32_t _top,const int32_t _bottom);
    MBound(const MGridPos& pos,const int32_t radius, const MGridPos& gird_size);
    MBound(const MGridPos& pos);
    MBound(const int32_t x,const int32_t z);
    Rect2i get_rect2i();
    
    void recalculate_center();
    void clear();
    bool has_point(const int32_t x, const int32_t y);
    bool has_point(const MGridPos& p);
    void intersect(const MBound& other);
    void merge(const MBound& other);

    bool operator==(const MBound& other);
    bool operator!=(const MBound& other);

    MGridPos closest_point_on_ground(const MGridPos& pos);
    // Make sure to have a correct grow even when we are outside of terrain
    void grow_when_outside(const real_t diff_x, const real_t diff_z,const MGridPos& _grid_pos, const MBound& limit_bound,const int32_t base_grid_size);
    // Grow only one unit untile reach the limit bound
    // When arrive at limit bound it is going to return false
    bool grow(const MBound& limit_bound,const int32_t amount_x,const int32_t amount_y);
    MGridPos get_edge_point();

    bool grow_positive(const int32_t amount, const MBound& limit_bound);
    //use to devide the image to different region
    bool get_next_region(const int32_t region_size, const MBound& limit_bound);
    bool get_next_shared_edge_region(const int32_t region_size, const MBound& limit_bound);

};







#endif