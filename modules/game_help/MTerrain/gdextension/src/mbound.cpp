#include "mbound.h"
#include "core/variant/variant.h"
#include "core/variant/variant_utility.h"

MBound::MBound(){};
MBound::MBound(const int32_t _left,const int32_t _right,const int32_t _top,const int32_t _bottom){
    left = _left;
    right = _right;
    top = _top;
    bottom = _bottom;
    center = MGridPos( (right - left)/2, 0, (bottom - top)/2 );
}
MBound::MBound(const MGridPos& pos,const int32_t radius, const MGridPos& gird_size){
    left = CLAMP(pos.x - radius, 0, gird_size.x - 1);
    right = CLAMP(pos.x + radius, 0, gird_size.x - 1);
    top = CLAMP(pos.z - radius, 0, gird_size.z - 1);
    bottom =CLAMP(pos.z + radius, 0, gird_size.z - 1);
    center = pos;
}
MBound::MBound(const MGridPos& pos){
    left = pos.x;
    right = pos.x;
    top = pos.z;
    bottom = pos.z;
    center = pos;
}

MBound::MBound(const int32_t x,const int32_t z){
    left = x;
    right = x;
    top = z;
    bottom = z;
    center.x = x;
    center.z = z;
}

Rect2i MBound::get_rect2i() {
    Rect2i rect(left, top, right - left, bottom - top);
    return rect;
}

void MBound::recalculate_center(){
    //In case we have large number this is safer than (left+right)/2 ...
    center.x = left + (right - left)/2;
    center.z = top + (bottom - top)/2;
}

void MBound::clear() {
    left = 0;
    right = 0;
    top = 0;
    bottom = 0;
}

bool MBound::has_point(const int32_t x, const int32_t y){
    if (x < left) return false;
    if (x > right) return false;
    if (y < top) return false;
    if (y > bottom) return false;
    return true;
}

bool MBound::has_point(const MGridPos& p) {
    if (p.x < left) return false;
    if (p.x > right) return false;
    if (p.z < top) return false;
    if (p.z > bottom) return false;
    return true;
}

void MBound::intersect(const MBound& other){
    left = std::max(other.left,left);
    right = std::min(other.right,right);
    top = std::max(other.top,top);
    bottom = std::min(other.bottom,bottom);
    if(left > right || top > bottom){
        left = center.x;
        right = center.x;
        top = center.z;
        bottom = center.z;
        return;
    }
    recalculate_center();
}

void MBound::merge(const MBound& other){
    left = std::min(other.left,left);
    right = std::max(other.right,right);
    top = std::min(other.top,top);
    bottom = std::max(other.bottom,bottom);
    recalculate_center();
}

bool MBound::operator==(const MBound& other) {
    return (left==other.left && right==other.right && top==other.top && bottom==other.bottom);
}

bool MBound::operator!=(const MBound& other) {
    return !(left==other.left && right==other.right && top==other.top && bottom==other.bottom);
}

MGridPos MBound::closest_point_on_ground(const MGridPos& pos) {
    if(has_point(pos)){
        return MGridPos(pos.x,0,pos.z);
    }
    if(pos.x < left && pos.z < top){
        return MGridPos(left,0,top);
    }
    if(pos.x>right && pos.z >bottom){
        return MGridPos(right, 0 , bottom);
    }
    if(pos.x>right && pos.z<top){
        return MGridPos(right, 0 , top);
    }
    if(pos.x<left && pos.z>bottom){
        return MGridPos(left, 0 , bottom);
    }
    if(pos.x<left){
        return MGridPos(left, 0, pos.z);
    }
    if(pos.z<top){
        return MGridPos(pos.x, 0, top);
    }
    if(pos.x>right){
        return MGridPos(right, 0 , pos.z);
    }
    return MGridPos(pos.x, 0 , bottom);
}

void MBound::grow_when_outside(const real_t diff_x, const real_t diff_z,const MGridPos& _grid_pos, const MBound& limit_bound,const int32_t base_grid_size){
    int32_t amount_x;
    int32_t amount_z;
    // If is outside and not coordinated in angoles
    if (!( (_grid_pos.x < left && _grid_pos.z < top) || (_grid_pos.x>right && _grid_pos.z >bottom) ||  (_grid_pos.x>right && _grid_pos.z<top) || (_grid_pos.x<left && _grid_pos.z>bottom) )){
        amount_z = abs(_grid_pos.x - center.x);
        amount_x = abs(_grid_pos.z - center.z);
        int32_t min_value = MIN(amount_x, amount_z);
        amount_x -= min_value;
        amount_z -= min_value;
    } else {
        real_t ax = abs(diff_z);
        real_t az = abs(diff_x);
        if(ax > az){
            ax -= az;
            ax = floor(ax);
            az = 0;
            amount_x = (int32_t)(ax/base_grid_size);
            amount_z = 0;
        } else {
            az -= ax;
            az = floor(az);
            ax = 0;
            amount_z = (int32_t)(az/base_grid_size);
            amount_x = 0;
        }
    }
    grow(limit_bound, amount_x, amount_z);
}

bool MBound::grow(const MBound& limit_bound,const int32_t amount_x,const int32_t amount_y) {
    if (*this == limit_bound){
        grow_left = false;
        grow_right = false;
        grow_bottom = false;
        grow_top = false;
        return false;
    }
    //left = CLAMP(left - 1, 0, limit_bound.left);
    //right = CLAMP(right + 1, 0, limit_bound.right);
    //top = CLAMP(top - 1, 0, limit_bound.top);
    //bottom = CLAMP(bottom + 1, 0, limit_bound.bottom);
    left -= amount_x;
    right += amount_x;
    top -= amount_y;
    bottom += amount_y;
    if (left < limit_bound.left)
    {
        left = limit_bound.left;
        grow_left = false;
    } else {
        grow_left = true;
    }
    if(right > limit_bound.right){
        right = limit_bound.right;
        grow_right = false;
    } else {
        grow_right = true;
    }
    if(top < limit_bound.top){
        top = limit_bound.top;
        grow_top = false;
    } else {
        grow_top = true;
    }
    if(bottom > limit_bound.bottom){
        bottom = limit_bound.bottom;
        grow_bottom = false;
    } else {
        grow_bottom = true;
    }
    return true;
}


MGridPos MBound::get_edge_point() {
    if(grow_left){
        return MGridPos(left, 0, center.z);
    }
    if(grow_right){
        return MGridPos(right,0 ,center.z);
    }
    if(grow_top){
        return MGridPos(center.x, 0, top);
    }
    return MGridPos(center.x, 0 , bottom);
}


bool MBound::grow_positive(const int32_t amount, const MBound& limit_bound) {
    right += amount;
    bottom += amount;
    if(right > limit_bound.right || bottom > limit_bound.bottom){
        return false;
    }
    return true;
}

bool MBound::get_next_region(const int32_t region_size, const MBound& limit_bound) {
    if(cursor.z == limit_bound.bottom){
        return false;
    }
    left = cursor.x;
    top = cursor.z;
    right = MIN(cursor.x + region_size, limit_bound.right);
    bottom = MIN(cursor.z + region_size,limit_bound.bottom);
    // in this case we reached at the end of row
    if(left == right){
        VariantUtilityFunctions::_print("next row");
        cursor.x = 0;
        cursor.z = bottom;
        return get_next_region(region_size, limit_bound);
    }
    cursor.x = right;
    cursor.z = top;
    return true;
}

bool MBound::get_next_shared_edge_region(const int32_t region_size, const MBound& limit_bound){
    left = cursor.x;
    top = cursor.z;
    right = MIN(cursor.x + region_size, limit_bound.right);
    bottom = MIN(cursor.z + region_size,limit_bound.bottom);
    cursor.x = right - 1;
    cursor.z = top - 1;
    return true;
}

