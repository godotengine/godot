#include "mpixel_region.h"

#include "core/variant/variant.h"
#include "core/variant/variant_utility.h"


MPixelRegion::MPixelRegion(){}

MPixelRegion::MPixelRegion(const uint32_t _width,const uint32_t _height) {
    //Minus one becuase pixels id start from zero
    right = _width - 1;
    bottom = _height - 1;
}

MPixelRegion::MPixelRegion(const uint32_t _left, const uint32_t _right, const uint32_t _top, const uint32_t _bottom){
    left = _left;
    right = _right;
    top = _top;
    bottom = _bottom;
}

void MPixelRegion::grow_all_side(const MPixelRegion& limit){
    if(left>0) left -=1;
    if(top>0) top -=1;
    right += 1;
    bottom +=1;
    if(left<limit.left) left = limit.left;
    if(top<limit.top) top = limit.top;
    if(right>limit.right) right = limit.right;
    if(bottom>limit.bottom) bottom = limit.bottom;
}

void MPixelRegion::grow_all_side(const MPixelRegion& limit,uint32_t amount){
    left = left > amount ? left - amount : 0;
    left = left < limit.left ? limit.left : left;
    top = top > amount ? top - amount : 0;
    top = top < limit.top ? limit.top : top;
    right += amount;
    bottom += amount;
    right = right > limit.right ? limit.right : right;
    bottom = bottom > limit.bottom ? limit.bottom : bottom;
}


bool MPixelRegion::grow_positve(const uint32_t xamount,const uint32_t yamount,const MPixelRegion& limit){
    if(left>limit.right || top>limit.bottom){
        return false;
    }
    right += xamount;
    bottom += yamount;
    if(right>limit.right) right = limit.right;
    if(bottom>limit.bottom) bottom = limit.bottom;
    return true;
}

Vector<MPixelRegion> MPixelRegion::devide(uint32_t amount) {
    Vector<MPixelRegion> output;
    uint32_t xamount = (right - left)/amount;
    uint32_t yamount = (bottom - top)/amount;
    uint32_t xpoint=left;
    uint32_t ypoint=top;
    uint32_t index = 0;
    while (true)
    {
        MPixelRegion r(xpoint,xpoint,ypoint,ypoint);
        if(r.grow_positve(xamount,yamount, *this)){
            output.append(r);
            xpoint = xpoint + xamount + 1;
        } else {
            xpoint=left;
            ypoint = ypoint + yamount + 1;
            MPixelRegion r2(xpoint,xpoint,ypoint,ypoint);
            if(r2.grow_positve(xamount,yamount, *this)){
                output.append(r2);
                xpoint = xpoint + xamount + 1;
            } else {
                break;
            }
        }
        index++;
        if(index>100){
            break;
        }
    }
    return output;
}

MPixelRegion MPixelRegion::get_local(MPixelRegion region){
    region.left -= left;
    region.right -= left;
    region.top -= top;
    region.bottom -= top;
    return region;
}

uint32_t MPixelRegion::get_height(){
    return bottom - top + 1;
}

uint32_t MPixelRegion::get_width(){
    return right - left + 1;
}

uint32_t MPixelRegion::get_pixel_amount(){
    return (right - left + 1)*(bottom - top + 1);
}

void MPixelRegion::clear(){
    left=0;
    right=0;
    top=0;
    bottom=0;
}

void MPixelRegion::print_region(String prefix){
    VariantUtilityFunctions::_print(prefix+" left ",left," right ",right," top ",top," bottom ",bottom);
}