#include "mresource.h"


//  https://github.com/phoboslab/qoi
#define QOI_IMPLEMENTATION
#define QOI_NO_STDIO
#include "thirdparty/qoi.h"

//#define PRINT_DEBUG
#include "core/io/image.h"
#include "core/io/file_access.h"
#include "core/variant/variant.h"
#include "core/io/compression.h"

#include "mimage.h"


MResource::QuadTreeRF::QuadTreeRF(MPixelRegion _px_region,float* _data,uint32_t _window_width, float _accuracy, MResource::QuadTreeRF* _root,uint8_t _depth,uint8_t _h_encoding)
:data(_data),window_width(_window_width),px_region(_px_region),accuracy(_accuracy),root(_root),depth(_depth),h_encoding(_h_encoding)
{
    update_min_max_height();
    if(!root){
        // Then we are root
        // And we should calculate how is the h_encoding
        double dh = max_height - min_height;
        if((dh/U4_MAX) <= accuracy){
            h_encoding = H_ENCODE_U4;
        } else if((dh/U8_MAX) <= accuracy) {
            h_encoding = H_ENCODE_U8;
        } else if((dh/U16_MAX) <= accuracy) {
            h_encoding = H_ENCODE_U16;
        } else {
            h_encoding = H_ENCODE_FLOAT;
        }
    }
}

MResource::QuadTreeRF::QuadTreeRF(MPixelRegion _px_region,float* _data,uint32_t _window_width,uint8_t _h_encoding,uint8_t _depth,MResource::QuadTreeRF* _root)
:px_region(_px_region),data(_data),window_width(_window_width),h_encoding(_h_encoding),depth(_depth),root(_root)
{

}

MResource::QuadTreeRF::~QuadTreeRF(){
    if(ne){
        memdelete(ne);
    }
    if(nw){
        memdelete(nw);
    }
    if(se){
        memdelete(se);
    }
    if(sw){
        memdelete(sw);
    }
}

void MResource::QuadTreeRF::update_min_max_height(){
    uint32_t px_index_top_left = px_region.left + (px_region.top*window_width);
    min_height = data[px_index_top_left];
    max_height = data[px_index_top_left];
    for(uint32_t y=px_region.top;y<=px_region.bottom;y++){
        for(uint32_t x=px_region.left;x<=px_region.right;x++){
            uint32_t px_index = x + (y*window_width);
            float val = data[px_index];
            if(IS_HOLE(val)){
                has_hole = true;
                continue;
            }
            if(IS_HOLE(max_height) || val>max_height){
                max_height = val;
            }
            if(IS_HOLE(min_height)  || val<min_height){
                min_height = val;
            }
        }
    }
}

void MResource::QuadTreeRF::divide_upto_leaf(){
    if(px_region.right - px_region.left + 1 <= MIN_BLOCK_SIZE_IN_QUADTREERF){
        return; // we reach the leaf
    }
    uint32_t bw = (px_region.right - px_region.left)/2; //Branch width minus one
    ERR_FAIL_COND(bw==0);
    MPixelRegion px_ne(px_region.left , px_region.left+bw , px_region.top , px_region.top+bw); // North West
    MPixelRegion px_nw(px_region.left+bw+1, px_region.right, px_region.top, px_region.top+bw); // North East
    MPixelRegion px_se(px_region.left, px_region.left+bw, px_region.top+bw+1 , px_region.bottom); // South West
    MPixelRegion px_sw(px_region.left+bw+1, px_region.right, px_region.top+bw+1 , px_region.bottom); // South East
    MResource::QuadTreeRF* who_is_root = root ? root : this;
    uint8_t new_depth = depth+1;
    ne = memnew(MResource::QuadTreeRF(px_ne,data,window_width,accuracy,who_is_root,new_depth,h_encoding));
    nw = memnew(MResource::QuadTreeRF(px_nw,data,window_width,accuracy,who_is_root,new_depth,h_encoding));
    se = memnew(MResource::QuadTreeRF(px_se,data,window_width,accuracy,who_is_root,new_depth,h_encoding));
    sw = memnew(MResource::QuadTreeRF(px_sw,data,window_width,accuracy,who_is_root,new_depth,h_encoding));

    ne->divide_upto_leaf();
    nw->divide_upto_leaf();
    se->divide_upto_leaf();
    sw->divide_upto_leaf();
}

uint32_t MResource::QuadTreeRF::get_only_hole_head_size(){
    return 1;
}

uint32_t MResource::QuadTreeRF::get_flat_head_size(){
    uint32_t size = 1; // one for block specifcation
    // Size for min and max height
    if(h_encoding==H_ENCODE_U4){
        size += 1;
    } else if(h_encoding==H_ENCODE_U8){
        size += 1;
    } else if(h_encoding==H_ENCODE_U16){
        size += 2;
    } else if(h_encoding==H_ENCODE_FLOAT){
        size +=4;
    } else {
        ERR_FAIL_V_MSG(size,"Unknown H Encoidng "+itos(h_encoding));
    }
    return size;
}

uint32_t MResource::QuadTreeRF::get_block_head_size(){
    uint32_t size = 1; // one for block specifcation
    // Size for min and max height
    if(h_encoding==H_ENCODE_U4){
        size += 1;
    } else if(h_encoding==H_ENCODE_U8){
        size += 2;
    } else if(h_encoding==H_ENCODE_U16){
        size += 4;
    } else if(h_encoding==H_ENCODE_FLOAT){
        size +=8;
    } else {
        ERR_FAIL_V_MSG(size,"Unknown H Encoidng "+itos(h_encoding));
    }
    return size;
}

uint32_t MResource::QuadTreeRF::get_optimal_non_divide_size(){
    if(IS_HOLE(min_height) || IS_HOLE(max_height)){
        data_encoding = DATA_ENCODE_ONLY_HOLE;
        return get_only_hole_head_size(); // Only one as we don't have only meta block
    }
    double dh = max_height - min_height;
    if(dh<accuracy && !has_hole){ // Flat Mode
        data_encoding = DATA_ENCODE_FLAT;
        // Size will remain the same as there is no data block here
        return get_flat_head_size();
    }
    uint32_t size = get_block_head_size();
    uint32_t px_amount = px_region.get_pixel_amount();
    double h_step;
    if(px_amount % 4 == 0){
        h_step = has_hole ? dh/HU2_MAX : dh/U2_MAX;
        if(h_step <= accuracy){
            data_encoding = DATA_ENCODE_U2;
            size += px_amount/4;
            return size;
        }
    }
    if(px_amount % 2 == 0){
        h_step = has_hole ? dh/HU4_MAX : dh/U4_MAX;
        if(h_step <= accuracy){
            data_encoding = DATA_ENCODE_U4;
            size += px_amount/2;
            return size;
        }
    }
    h_step = has_hole ? dh/HU8_MAX : dh/U8_MAX;
    if(h_step <= accuracy){
        data_encoding = DATA_ENCODE_U8;
        size += px_amount;
        return size;
    }
    h_step = has_hole ? dh/HU16_MAX : dh/U16_MAX;
    if(h_step <= accuracy){
        data_encoding = DATA_ENCODE_U16;
        size += px_amount*2;
        return size;
    }
    data_encoding = DATA_ENCODE_FLOAT;
    size = 1; // We don't have a min and max height only one byte for block specification
    size += px_amount*4;
    return size;
}

uint32_t MResource::QuadTreeRF::get_optimal_size(){
    uint32_t size = get_optimal_non_divide_size();
    uint32_t divid_size;
    if(nw){
        divid_size = ne->get_optimal_size();
        divid_size += nw->get_optimal_size();
        divid_size += se->get_optimal_size();
        divid_size += sw->get_optimal_size();
        if(divid_size < size){
            //Then we should divide as division get a better compression
            //Also in this case we Keep the children
            return divid_size;
        }
    }
    //So we should not divde and we should remove children
    if(nw){
        memdelete(ne);
        memdelete(nw);
        memdelete(se);
        memdelete(sw);
        ne = nullptr;
        nw = nullptr;
        se = nullptr;
        sw = nullptr;
    }
    return size;
}

void MResource::QuadTreeRF::encode_min_max_height(PackedByteArray& save_data,uint32_t& save_index){
    ERR_FAIL_COND(data_encoding>=DATA_ENCODE_MAX);
    if(data_encoding==DATA_ENCODE_FLOAT || data_encoding==DATA_ENCODE_ONLY_HOLE){
        return;
    }
    double main_min_height = root ? root->min_height : min_height;
    double dh_main = root ? root->max_height - root->min_height : max_height - min_height;
    if(h_encoding == H_ENCODE_U4){
        uint8_t minh_u4=0;
        uint8_t maxh_u4=0;
        if(dh_main>0.0000001){ // We should handle dh_main zero only here as that will not happen to othe H_ENCODING
            double h_step_main = dh_main/U4_MAX;
            float fmin = (min_height-main_min_height) / h_step_main;
            float fmax = (max_height-main_min_height) / h_step_main;
            minh_u4 = (uint8_t)std::min(fmin,(float)U4_MAX);
            maxh_u4 = (uint8_t)std::min(fmax,(float)U4_MAX);
        }
        encode_uint4(minh_u4,maxh_u4,save_data.ptrw()+save_index);
        // In this case only min height does not change anything as we store in one byte
        save_index++;
        return;
    }
    if(h_encoding == H_ENCODE_U8){
        double h_step_main = dh_main/U8_MAX;
        float fmin = (min_height-main_min_height) / h_step_main;
        float fmax = (max_height-main_min_height) / h_step_main;
        uint8_t minh_u8 = (uint8_t)std::min(fmin,(float)U8_MAX);
        uint8_t maxh_u8 = (uint8_t)std::min(fmax,(float)U8_MAX);
        save_data.write[save_index] = minh_u8;
        save_index++;
        if(data_encoding!=DATA_ENCODE_FLAT){
            save_data.write[save_index] = maxh_u8;
            save_index++;
        }
        return;
    }
    if(h_encoding == H_ENCODE_U16){
        double h_step_main = dh_main/U16_MAX;
        float fmin = (min_height-main_min_height) / h_step_main;
        float fmax = (max_height-main_min_height) / h_step_main;
        uint16_t minh_u16 = (uint16_t)std::min(fmin,(float)U16_MAX);
        uint16_t maxh_u16 = (uint16_t)std::min(fmax,(float)U16_MAX);
        encode_uint16(minh_u16,save_data.ptrw()+save_index);
        save_index+=2;
        if(data_encoding!=DATA_ENCODE_FLAT){
            encode_uint16(maxh_u16,save_data.ptrw()+save_index);
            save_index+=2;
        }
        return;
    }
    if(h_encoding == H_ENCODE_FLOAT){
        encode_float(min_height,save_data.ptrw()+save_index);
        save_index+=4;
        if(data_encoding!=DATA_ENCODE_FLAT){
            encode_float(max_height,save_data.ptrw()+save_index);
            save_index+=4;
        }
        return;
    }
    ERR_FAIL_MSG("H Encoding is not valid "+itos(h_encoding));
}

void MResource::QuadTreeRF::decode_min_max_height(const PackedByteArray& compress_data,uint32_t& decompress_index){
    ERR_FAIL_COND(data_encoding>=DATA_ENCODE_MAX);
    if(data_encoding==DATA_ENCODE_FLOAT || data_encoding==DATA_ENCODE_ONLY_HOLE){
        return;
    }
    double main_min_height = root ? root->min_height : min_height;
    double dh_main = root ? root->max_height - root->min_height : max_height - min_height;
    if(dh_main<0.00001){
        min_height = main_min_height;
        max_height = main_min_height;
        decompress_index++; //We have H_ENCODE_U4 in this case so one increase
        ERR_FAIL_COND(h_encoding!=H_ENCODE_U4);
        return;
    }
    if(h_encoding == H_ENCODE_U4){
        double h_step_main = dh_main/U4_MAX;
        uint8_t minh_u4=0;
        uint8_t maxh_u4=0;
        decode_uint4(minh_u4,maxh_u4,compress_data.ptr()+decompress_index);
        decompress_index++;
        min_height = main_min_height + (minh_u4*h_step_main);
        max_height = main_min_height + (maxh_u4*h_step_main);
        return;
    }
    if(h_encoding == H_ENCODE_U8){
        double h_step_main = dh_main/U8_MAX;
        uint8_t minh_u8 = compress_data[decompress_index];
        decompress_index++;
        min_height = main_min_height + (minh_u8*h_step_main);
        if(data_encoding==DATA_ENCODE_FLAT){
            max_height = min_height;
            return;
        }
        uint8_t maxh_u8 = compress_data[decompress_index];
        decompress_index++;
        max_height = main_min_height + (maxh_u8*h_step_main);
        return;
    }
    if(h_encoding == H_ENCODE_U16){
        double h_step_main = dh_main/U16_MAX;
        uint16_t minh_u16 = decode_uint16(compress_data.ptr()+decompress_index);
        decompress_index+=2;
        min_height = main_min_height + (minh_u16*h_step_main);
        if(data_encoding==DATA_ENCODE_FLAT){
            max_height = min_height;
            return;
        }
        uint16_t maxh_u16 = decode_uint16(compress_data.ptr()+decompress_index);
        decompress_index+=2;
        max_height = main_min_height + (maxh_u16*h_step_main);
        return;
    }
    if(h_encoding == H_ENCODE_FLOAT){
        min_height = decode_float(compress_data.ptr()+decompress_index);
        decompress_index+=4;
        if(data_encoding==DATA_ENCODE_FLAT){
            max_height = min_height;
            return;
        }
        max_height = decode_float(compress_data.ptr()+decompress_index);
        decompress_index+=4;
        return;
    }
    ERR_FAIL_MSG("Unknow H encoding in uncompress "+itos(h_encoding));
}

void MResource::QuadTreeRF::save_quad_tree_data(PackedByteArray& save_data,uint32_t& save_index){
    ERR_FAIL_COND(accuracy<0);
    if(nw){ // Then this is divided
        // Order of getting info matter
        ne->save_quad_tree_data(save_data,save_index);
        nw->save_quad_tree_data(save_data,save_index);
        se->save_quad_tree_data(save_data,save_index);
        sw->save_quad_tree_data(save_data,save_index);
        return;
    }
    // Creating Headers
    // meta-data
    uint8_t meta = 0;
    meta |= (depth << 4);
    meta |= (uint8_t)has_hole;
    meta |= (data_encoding<<1);
    save_data.write[save_index] = meta;
    save_index++;
    encode_min_max_height(save_data,save_index);
    if(data_encoding == DATA_ENCODE_FLOAT){
        encode_data_float(save_data,save_index);
        return;
    }
    if(data_encoding == DATA_ENCODE_FLAT || data_encoding == DATA_ENCODE_ONLY_HOLE){
        #ifdef PRINT_DEBUG
        if(data_encoding == DATA_ENCODE_FLAT){
            VariantUtilityFunctions::_print("EncodeFLAT L ",px_region.left," R ",px_region.right," T ",px_region.top, " B ",px_region.bottom, " save_index ",save_index, " ----- ");
        }
        else if(data_encoding == DATA_ENCODE_ONLY_HOLE){
            VariantUtilityFunctions::_print("EncodeOnlyHole L ",px_region.left," R ",px_region.right," T ",px_region.top, " B ",px_region.bottom, " save_index ",save_index, " ----- ");
        }
        #endif
        return;
    }
    // Otherwise on all condition the min and max height should be encoded
    double dh = max_height - min_height;
    if(data_encoding == DATA_ENCODE_U2){
        encode_data_u2(save_data,save_index);
        return;
    }
    if(data_encoding == DATA_ENCODE_U4){
        encode_data_u4(save_data,save_index);
        return;
    }
    if(data_encoding == DATA_ENCODE_U8){
        encode_data_u8(save_data,save_index);
        return;
    }
    if(data_encoding == DATA_ENCODE_U16){
        encode_data_u16(save_data,save_index);
        return;
    }
    ERR_FAIL_MSG("Unknow Data Encoding "+itos(data_encoding));
}

void MResource::QuadTreeRF::load_quad_tree_data(const PackedByteArray& compress_data,uint32_t& decompress_index){
    uint8_t meta = compress_data[decompress_index];
    uint8_t cdepth = meta >> 4;
    data_encoding = (meta & 0xE) >> 1;
    if(cdepth==depth){
        decompress_index++;
        data_encoding = (meta & 0xE) >> 1;
        has_hole = meta & 0x1;
        decode_min_max_height(compress_data,decompress_index);
        if(data_encoding==DATA_ENCODE_FLAT){
            decode_data_flat();
            return;
        }
        if(data_encoding==DATA_ENCODE_U2){
            decode_data_u2(compress_data,decompress_index);
            return;
        }
        if(data_encoding==DATA_ENCODE_U4){
            decode_data_u4(compress_data,decompress_index);
            return;
        }
        if(data_encoding==DATA_ENCODE_U8){
            decode_data_u8(compress_data,decompress_index);
            return;
        }
        if(data_encoding==DATA_ENCODE_U16){
            decode_data_u16(compress_data,decompress_index);
            return;
        }
        if(data_encoding==DATA_ENCODE_FLOAT){
            decode_data_float(compress_data,decompress_index);
            return;
        }
        if(data_encoding==DATA_ENCODE_ONLY_HOLE){
            decode_data_only_hole();
            return;
        }
        ERR_FAIL_MSG("Not a valid data encoding "+itos(data_encoding));
        return;
    }
    uint32_t bw = (px_region.right - px_region.left)/2; //Branch width minus one
    ERR_FAIL_COND(bw==0);
    MPixelRegion px_ne(px_region.left , px_region.left+bw , px_region.top , px_region.top+bw); // North West
    MPixelRegion px_nw(px_region.left+bw+1, px_region.right, px_region.top, px_region.top+bw); // North East
    MPixelRegion px_se(px_region.left, px_region.left+bw, px_region.top+bw+1 , px_region.bottom); // South West
    MPixelRegion px_sw(px_region.left+bw+1, px_region.right, px_region.top+bw+1 , px_region.bottom); // South East
    MResource::QuadTreeRF* who_is_root = root ? root : this;
    uint8_t new_depth = depth + 1;
    ne = memnew(MResource::QuadTreeRF(px_ne,data,window_width,h_encoding,new_depth,who_is_root));
    nw = memnew(MResource::QuadTreeRF(px_nw,data,window_width,h_encoding,new_depth,who_is_root));
    se = memnew(MResource::QuadTreeRF(px_se,data,window_width,h_encoding,new_depth,who_is_root));
    sw = memnew(MResource::QuadTreeRF(px_sw,data,window_width,h_encoding,new_depth,who_is_root));
    // Calling order matter
    ne->load_quad_tree_data(compress_data,decompress_index);
    nw->load_quad_tree_data(compress_data,decompress_index);
    se->load_quad_tree_data(compress_data,decompress_index);
    sw->load_quad_tree_data(compress_data,decompress_index);
}

void MResource::QuadTreeRF::encode_data_u2(PackedByteArray& save_data,uint32_t& save_index){
    #ifdef PRINT_DEBUG
    VariantUtilityFunctions::_print("EncodeU2 L ",px_region.left," R ",px_region.right," T ",px_region.top, " B ",px_region.bottom, " save_index ",save_index, " ----- ");
    #endif
    double dh = max_height - min_height;
    uint8_t maxu2 = has_hole ? HU2_MAX : U2_MAX;
    double h_step = dh/maxu2;
    uint8_t vals[4];
    uint8_t val_index =0;
    for(uint32_t y=px_region.top;y<=px_region.bottom;y++){
        for(uint32_t x=px_region.left;x<=px_region.right;x++){
            uint32_t px_index = x + (y*window_width);
            double val = data[px_index] - min_height;
            if(IS_HOLE(val)){
                vals[val_index] = U2_MAX;
            } else {
                vals[val_index] = (uint8_t)(val/h_step);
                vals[val_index] = std::min(vals[val_index],maxu2);
            }
            val_index++;
            if(val_index==4){
                val_index=0;
                encode_uint2(vals[0],vals[1],vals[2],vals[3],save_data.ptrw()+save_index);
                save_index++;
            }
        }
    }
}

void MResource::QuadTreeRF::encode_data_u4(PackedByteArray& save_data,uint32_t& save_index){
    #ifdef PRINT_DEBUG
    VariantUtilityFunctions::_print("EncodeU4 L ",px_region.left," R ",px_region.right," T ",px_region.top, " B ",px_region.bottom, " save_index ",save_index, " ----- ");
    #endif
    double dh = max_height - min_height;
    uint8_t maxu4 = has_hole ? HU4_MAX : U4_MAX;
    double h_step = dh/maxu4;
    //VariantUtilityFunctions::_print("Encode4 h step ",h_step, " dh ",dh);
    uint8_t vals[2];
    uint8_t val_index =0;
    for(uint32_t y=px_region.top;y<=px_region.bottom;y++){
        for(uint32_t x=px_region.left;x<=px_region.right;x++){
            uint32_t px_index = x + (y*window_width);
            double val = data[px_index] - min_height;
            if(IS_HOLE(val)){
                vals[val_index] = U4_MAX;
            } else {
                vals[val_index] = (uint8_t)(val / h_step);
                vals[val_index] = std::min(vals[val_index],maxu4);
            }
            //VariantUtilityFunctions::_print("Encode4 ",data[px_index]," -> ",vals[val_index]);
            val_index++;
            if(val_index==2){
                val_index=0;
                encode_uint4(vals[0],vals[1],save_data.ptrw()+save_index);
                save_index++;
                val_index = 0;
            }
        }
    }
}

void MResource::QuadTreeRF::encode_data_u8(PackedByteArray& save_data,uint32_t& save_index){
    #ifdef PRINT_DEBUG
    VariantUtilityFunctions::_print("EncodeU8 L ",px_region.left," R ",px_region.right," T ",px_region.top, " B ",px_region.bottom, " save_index ",save_index, " ----- ");
    #endif
    double dh = max_height - min_height;
    double maxu8 = has_hole ? HU8_MAX : U8_MAX;
    double h_step = dh/maxu8;
    for(uint32_t y=px_region.top;y<=px_region.bottom;y++){
        for(uint32_t x=px_region.left;x<=px_region.right;x++){
            uint32_t px_index = x + (y*window_width);
            double val = data[px_index] - min_height;
            val /= h_step;
            val = std::min(val,maxu8);
            uint8_t sval = IS_HOLE(val) ? UINT8_MAX : val;
            sval = std::min(sval,(uint8_t)U8_MAX);
            //VariantUtilityFunctions::_print("Encode8 ",val," -> ",sval);
            save_data.write[save_index] = sval;
            save_index++;
        }
    }
}

void MResource::QuadTreeRF::encode_data_u16(PackedByteArray& save_data,uint32_t& save_index){
    #ifdef PRINT_DEBUG
    VariantUtilityFunctions::_print("EncodeU16 L ",px_region.left," R ",px_region.right," T ",px_region.top, " B ",px_region.bottom, " save_index ",save_index, " ----- ");
    #endif
    double dh = max_height - min_height;
    double maxu16 = has_hole ? HU16_MAX : U16_MAX;
    double h_step = dh/maxu16;
    for(uint32_t y=px_region.top;y<=px_region.bottom;y++){
        for(uint32_t x=px_region.left;x<=px_region.right;x++){
            uint32_t px_index = x + (y*window_width);
            double val = data[px_index] - min_height;
            val /= h_step;
            val = std::min(val,maxu16);
            uint16_t sval = IS_HOLE(val) ? UINT16_MAX : val;
            sval = std::min(sval,(uint16_t)U16_MAX);
            //VariantUtilityFunctions::_print("Encode16 ",val," -> ",sval);
            encode_uint16(sval,save_data.ptrw()+save_index);
            save_index += 2;
        }
    }
}

void MResource::QuadTreeRF::encode_data_float(PackedByteArray& save_data,uint32_t& save_index){
    #ifdef PRINT_DEBUG
    VariantUtilityFunctions::_print("EncodeFloat L ",px_region.left," R ",px_region.right," T ",px_region.top, " B ",px_region.bottom, " save_index ",save_index, " ----- ");
    #endif
    for(uint32_t y=px_region.top;y<=px_region.bottom;y++){
        for(uint32_t x=px_region.left;x<=px_region.right;x++){
            uint32_t px_index = x + (y*window_width);
            float val = data[px_index];
            encode_float(val,save_data.ptrw()+save_index);
            save_index += 4;
        }
    }
}


void MResource::QuadTreeRF::decode_data_flat(){
    #ifdef PRINT_DEBUG
    VariantUtilityFunctions::_print("DecodeFlat L ",px_region.left," R ",px_region.right," T ",px_region.top, " B ",px_region.bottom, " ----- ");
    #endif
    for(uint32_t y=px_region.top;y<=px_region.bottom;y++){
        for(uint32_t x=px_region.left;x<=px_region.right;x++){
            uint32_t px_index = x + (y*window_width);
            data[px_index] = min_height;
        }
    }
}
void MResource::QuadTreeRF::decode_data_u2(const PackedByteArray& compress_data,uint32_t& decompress_index){
    #ifdef PRINT_DEBUG
    VariantUtilityFunctions::_print("DecodeU2 L ",px_region.left," R ",px_region.right," T ",px_region.top, " B ",px_region.bottom, " decompress_index ",decompress_index, " ----- ");
    #endif
    float dh = max_height - min_height;
    double h_step = has_hole ? dh/HU2_MAX : dh/U2_MAX;
    uint8_t vals[4];
    uint8_t vals_index=0;
    for(uint32_t y=px_region.top;y<=px_region.bottom;y++){
        for(uint32_t x=px_region.left;x<=px_region.right;x++){
            if(vals_index==0){
                decode_uint2(vals[0],vals[1],vals[2],vals[3],compress_data.ptr()+decompress_index);
                decompress_index++;
            }
            uint32_t px_index = x + (y*window_width);
            if(has_hole && vals[vals_index]==U2_MAX){
                data[px_index] = FLOAT_HOLE;
            } else {
                data[px_index] = min_height + vals[vals_index]*h_step;
            }
            vals_index++;
            if(vals_index==4){
                vals_index=0;
            }
        }
    }
}
void MResource::QuadTreeRF::decode_data_u4(const PackedByteArray& compress_data,uint32_t& decompress_index){
    #ifdef PRINT_DEBUG
    VariantUtilityFunctions::_print("DecodeU4 L ",px_region.left," R ",px_region.right," T ",px_region.top, " B ",px_region.bottom, " decompress_index ",decompress_index, " ----- ");
    #endif
    float dh = max_height - min_height;
    double h_step = has_hole ? dh/HU4_MAX : dh/U4_MAX;
    uint8_t vals[2];
    uint8_t vals_index=0;
    for(uint32_t y=px_region.top;y<=px_region.bottom;y++){
        for(uint32_t x=px_region.left;x<=px_region.right;x++){
            if(vals_index==0){
                decode_uint4(vals[0],vals[1],compress_data.ptr()+decompress_index);
                decompress_index++;
            }
            uint32_t px_index = x + (y*window_width);
            if(has_hole && vals[vals_index] == U4_MAX){
                data[px_index] = FLOAT_HOLE;
            } else {
                data[px_index] = min_height + vals[vals_index]*h_step;
            }
            vals_index++;
            if(vals_index==2){
                vals_index=0;
            }
        }
    }
    //VariantUtilityFunctions::_print("decompress index ",decompress_index);
}
void MResource::QuadTreeRF::decode_data_u8(const PackedByteArray& compress_data,uint32_t& decompress_index){
    #ifdef PRINT_DEBUG
    VariantUtilityFunctions::_print("DecodeU8 L ",px_region.left," R ",px_region.right," T ",px_region.top, " B ",px_region.bottom, " decompress_index ",decompress_index, " ----- ");
    #endif
    float dh = max_height - min_height;
    double h_step = has_hole ? dh/HU8_MAX : dh/U8_MAX;
    for(uint32_t y=px_region.top;y<=px_region.bottom;y++){
        for(uint32_t x=px_region.left;x<=px_region.right;x++){
            uint8_t val8 = compress_data[decompress_index];
            decompress_index++;
            uint32_t px_index = x + (y*window_width);
            if(has_hole && val8==U8_MAX){
                data[px_index] = FLOAT_HOLE;
                continue;
            }
            data[px_index] = min_height + (float)(val8*h_step);
        }
    }
}
void MResource::QuadTreeRF::decode_data_u16(const PackedByteArray& compress_data,uint32_t& decompress_index){
    #ifdef PRINT_DEBUG
    VariantUtilityFunctions::_print("DecodeU16 L ",px_region.left," R ",px_region.right," T ",px_region.top, " B ",px_region.bottom, " decompress_index ",decompress_index, " ----- ");
    #endif
    float dh = max_height - min_height;
    double h_step = has_hole ? dh/HU16_MAX : dh/U16_MAX;
    for(uint32_t y=px_region.top;y<=px_region.bottom;y++){
        for(uint32_t x=px_region.left;x<=px_region.right;x++){
            uint16_t val16 = decode_uint16(compress_data.ptr()+decompress_index);
            decompress_index+=2;
            uint32_t px_index = x + (y*window_width);
            if(has_hole && val16==U16_MAX){
                data[px_index] = FLOAT_HOLE;
                continue;
            }
            data[px_index] = min_height + (float)(val16*h_step);
            //VariantUtilityFunctions::_print("decoding16 ",val16, " , ",data[px_index]);
        }
    }
}
void MResource::QuadTreeRF::decode_data_float(const PackedByteArray& compress_data,uint32_t& decompress_index){
    #ifdef PRINT_DEBUG
    VariantUtilityFunctions::_print("DecodeFloat L ",px_region.left," R ",px_region.right," T ",px_region.top, " B ",px_region.bottom, " decompress_index ",decompress_index, " ----- ");
    #endif
    for(uint32_t y=px_region.top;y<=px_region.bottom;y++){
        for(uint32_t x=px_region.left;x<=px_region.right;x++){
            uint32_t px_index = x + (y*window_width);
            data[px_index] = decode_float(compress_data.ptr()+decompress_index);
            decompress_index+=4;
        }
    }
}

void MResource::QuadTreeRF::decode_data_only_hole(){
    #ifdef PRINT_DEBUG
    VariantUtilityFunctions::_print("DecodeOnlyHole L ",px_region.left," R ",px_region.right," T ",px_region.top, " B ",px_region.bottom, " ----- ");
    #endif
    for(uint32_t y=px_region.top;y<=px_region.bottom;y++){
        for(uint32_t x=px_region.left;x<=px_region.right;x++){
            uint32_t px_index = x + (y*window_width);
            data[px_index] = FLOAT_HOLE;
        }
    }
}

void MResource::_bind_methods(){
    ClassDB::bind_method(D_METHOD("set_compressed_data","input"), &MResource::set_compressed_data);
    ClassDB::bind_method(D_METHOD("get_compressed_data"), &MResource::get_compressed_data);
    ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY,"compressed_data"), "set_compressed_data","get_compressed_data");

    ClassDB::bind_method(D_METHOD("get_data_format","_name"), &MResource::get_data_format);
    ClassDB::bind_method(D_METHOD("get_data_width","name"), &MResource::get_data_width);
    ClassDB::bind_method(D_METHOD("get_heightmap_width"), &MResource::get_heightmap_width);
    ClassDB::bind_method(D_METHOD("get_min_height"), &MResource::get_min_height);
    ClassDB::bind_method(D_METHOD("get_max_height"), &MResource::get_max_height);

    ClassDB::bind_method(D_METHOD("insert_heightmap_rf","data","accuracy","compress_qtq","file_compress"), &MResource::insert_heightmap_rf);
    ClassDB::bind_method(D_METHOD("get_heightmap_rf","two_plus_one"), &MResource::get_heightmap_rf);

    ClassDB::bind_method(D_METHOD("insert_data","data","name","format","compress","file_compress"), &MResource::insert_data);
    ClassDB::bind_method(D_METHOD("get_data","name","two_plus_one"), &MResource::get_data);
    ClassDB::bind_method(D_METHOD("remove_data","name"), &MResource::remove_data);

    ClassDB::bind_method(D_METHOD("get_compress","name"), &MResource::get_compress);
    ClassDB::bind_method(D_METHOD("get_file_compress","name"), &MResource::get_file_compress);
    ClassDB::bind_method(D_METHOD("is_compress_qtq"), &MResource::is_compress_qtq);


    BIND_ENUM_CONSTANT(COMPRESS_NONE);
    BIND_ENUM_CONSTANT(COMPRESS_QOI);
    BIND_ENUM_CONSTANT(COMPRESS_PNG);

    BIND_ENUM_CONSTANT(FILE_COMPRESSION_NONE);
    BIND_ENUM_CONSTANT(FILE_COMPRESSION_FASTLZ);
    BIND_ENUM_CONSTANT(FILE_COMPRESSION_DEFLATE);
    BIND_ENUM_CONSTANT(FILE_COMPRESSION_ZSTD);
    BIND_ENUM_CONSTANT(FILE_COMPRESSION_GZIP);
}

MResource::MResource(){
}
MResource::~MResource(){
}

bool MResource::has_data(const StringName& _name){
    return compressed_data.has(_name);
}

void MResource::set_compressed_data(const Dictionary& data){
    compressed_data = data;
}
const Dictionary& MResource::get_compressed_data(){
    return compressed_data;
}

Image::Format MResource::get_data_format(const StringName& _name){
    ERR_FAIL_COND_V(!compressed_data.has(_name),Image::Format::FORMAT_MAX);
    if(format_cache.has(_name)){
        return (Image::Format)format_cache[_name];
    }
    PackedByteArray data = compressed_data[_name];
    format_cache.insert(_name,data[FORMAT_INDEX]);
    return (Image::Format)data[FORMAT_INDEX];
}

uint16_t MResource::get_data_width(const StringName& _name){
    ERR_FAIL_COND_V(!compressed_data.has(_name),0);
    if(width_cache.has(_name)){
        return width_cache[_name];
    }
    PackedByteArray data = compressed_data[_name];
    uint32_t width = decode_uint16(data.ptrw()+WIDTH_INDEX);
    width_cache.insert(_name,width);
    return width;
}

uint32_t MResource::get_heightmap_width(){
    return get_data_width(StringName("heightmap"));
}

float MResource::get_min_height(){
    if(!IS_HOLE(min_height_cache))
    {
        return min_height_cache;
    }
    StringName hname("heightmap");
    ERR_FAIL_COND_V(!compressed_data.has(hname),min_height_cache);
    PackedByteArray data = compressed_data[hname];
    min_height_cache = decode_float(data.ptr()+MIN_HEIGHT_INDEX);
    max_height_cache = decode_float(data.ptr()+MAX_HEIGHT_INDEX);
    return min_height_cache;
}

float MResource::get_max_height(){
    if(!IS_HOLE(max_height_cache))
    {
        return max_height_cache;
    }
    StringName hname("heightmap");
    ERR_FAIL_COND_V(!compressed_data.has(hname),max_height_cache);
    PackedByteArray data = compressed_data[hname];
    min_height_cache = decode_float(data.ptr()+MIN_HEIGHT_INDEX);
    max_height_cache = decode_float(data.ptr()+MAX_HEIGHT_INDEX);
    return max_height_cache;
}
	static PackedByteArray _PackedByteArray_compress(PackedByteArray *p_instance, int p_mode) {
		PackedByteArray compressed;

		if (p_instance->size() > 0) {
			Compression::Mode mode = (Compression::Mode)(p_mode);
			compressed.resize(Compression::get_max_compressed_buffer_size(p_instance->size(), mode));
			int result = Compression::compress(compressed.ptrw(), p_instance->ptr(), p_instance->size(), mode);

			result = result >= 0 ? result : 0;
			compressed.resize(result);
		}

		return compressed;
	}
	static PackedByteArray _PackedByteArray_decompress(PackedByteArray *p_instance, int64_t p_buffer_size, int p_mode) {
		PackedByteArray decompressed;
		Compression::Mode mode = (Compression::Mode)(p_mode);

		int64_t buffer_size = p_buffer_size;

		if (buffer_size <= 0) {
			ERR_FAIL_V_MSG(decompressed, "Decompression buffer size must be greater than zero.");
		}
		if (p_instance->size() == 0) {
			ERR_FAIL_V_MSG(decompressed, "Compressed buffer size must be greater than zero.");
		}

		decompressed.resize(buffer_size);
		int result = Compression::decompress(decompressed.ptrw(), buffer_size, p_instance->ptr(), p_instance->size(), mode);

		result = result >= 0 ? result : 0;
		decompressed.resize(result);

		return decompressed;
	}

void MResource::insert_data(const PackedByteArray& data, const StringName& _name,Image::Format format,MResource::Compress compress,MResource::FileCompress file_compress){
    if(format==Image::Format::FORMAT_R8 && compress==MResource::Compress::COMPRESS_PNG){
        format=Image::Format::FORMAT_L8;
        WARN_PRINT("PNG support format L8, so format R8 in "+String(_name)+" will convert to L8");
    }
    {
        const StringName& hname("heightmap");
        ERR_FAIL_COND_MSG(_name==hname,"Use insert_heightmap_rf function to insert heightmap");
    }
    uint32_t pixel_size = MImage::get_format_pixel_size(format);
    ERR_FAIL_COND_MSG(!pixel_size,"Unsported format");
    ERR_FAIL_COND(data.size() % pixel_size != 0);
    uint32_t pixel_amount = data.size() / pixel_size;
    uint32_t width = sqrt(pixel_amount);
    ERR_FAIL_COND(width<1);
    ERR_FAIL_COND(pixel_amount!=width*width);
    PackedByteArray final_data;
    if( ((width - 1) & (width - 2))==0 && width!=2){
        uint32_t new_width = width - 1;
        uint32_t new_size = new_width*new_width*pixel_size;
        final_data.resize(new_size);
        // Copy rows
        uint32_t new_row_size = new_width*pixel_size;
        for(uint32_t row=0;row<new_width;row++){
            uint32_t pos_old = row * width * pixel_size;
            uint32_t pos_new = row * new_width * pixel_size;
            memcpy(final_data.ptrw()+pos_new,data.ptr()+pos_old,new_row_size);
        }
        width--;
    } else if (((width & (width - 1)) == 0))
    {
        final_data = data;
    } else {
        ERR_FAIL_MSG("Image dimension should be in power of two");
        return;
    }
    // Creating compress data
    PackedByteArray new_compressed_data;
    // Creating header
    uint16_t flags=0;
    new_compressed_data.resize(MRESOURCE_HEADER_SIZE);
    {
        new_compressed_data.write[0] = MMAGIC_NUM;
        new_compressed_data.write[1] = CURRENT_MRESOURCE_VERSION;
        // FALGS WILL BE ADDED AT THE END
        new_compressed_data.write[4] = (uint8_t)format;
        encode_uint16(width,new_compressed_data.ptrw()+6);
    }
    uint32_t save_index = 0;
    PackedByteArray data_part;
    if(compress==Compress::COMPRESS_NONE){
        data_part = final_data;
    }
    else if(compress==Compress::COMPRESS_QOI){
        uint8_t channel_count = get_supported_qoi_format_channel_count(format);
        ERR_FAIL_COND_MSG(channel_count==0,"QOI only support these two format: RGBA8 and RGB8");
        flags |= FLAG_COMPRESSION_QOI;
        // QOI compression
        qoi_desc qoi_img;
        qoi_img.width = width;
        qoi_img.height = width;
        qoi_img.channels = channel_count;
        qoi_img.colorspace = 1;
        void *input_data = (void *)final_data.ptrw();
        int qoi_data_size;
        void* qoi_data = qoi_encode(input_data,&qoi_img,&qoi_data_size);
        data_part.resize(qoi_data_size);
        memcpy(data_part.ptrw(),qoi_data,qoi_data_size);
        ::free(qoi_data);
    }
    else if(compress==Compress::COMPRESS_PNG){
        ERR_FAIL_COND_MSG(!get_supported_png_format(format),"PNG support only RGB8, RGBA8 and L8");
        flags |= FLAG_COMPRESSION_PNG;
        Ref<Image> img;
        img = Image::create_from_data(width,width,false,format,final_data);
        data_part = img->save_png_to_buffer();
    }
    uint32_t data_size_before_file_compress = data_part.size();
    /// FILE COMPRESS
    if(file_compress==FileCompress::FILE_COMPRESSION_FASTLZ){
        flags |= FLAG_COMPRESSION_FASTLZ;
        data_part = _PackedByteArray_compress(&data_part,Compression::MODE_FASTLZ);
    }
    else if(file_compress==FileCompress::FILE_COMPRESSION_DEFLATE){
        flags |= FLAG_COMPRESSION_DEFLATE;
        data_part = _PackedByteArray_compress(&data_part,Compression::MODE_DEFLATE);
    }
    else if(file_compress==FileCompress::FILE_COMPRESSION_ZSTD){
        flags |= FLAG_COMPRESSION_ZSTD;
        data_part = _PackedByteArray_compress( &data_part,Compression::MODE_ZSTD);// data_part.compress(FileAccess::COMPRESSION_ZSTD);
    }
    else if(file_compress==FileCompress::FILE_COMPRESSION_GZIP){
        flags |= FLAG_COMPRESSION_GZIP;
        data_part = _PackedByteArray_compress (&data_part,Compression::MODE_GZIP);//data_part.compress(FileAccess::COMPRESSION_GZIP);
    }
    encode_uint16(flags,new_compressed_data.ptrw()+FLAGS_INDEX);
    encode_uint32(data_size_before_file_compress,new_compressed_data.ptrw()+DATA_SIZE_BEFORE_FILE_COMPRESS_INDEX);
    new_compressed_data.append_array(data_part);
    compressed_data[_name] = new_compressed_data;
}

PackedByteArray MResource::get_data(const StringName& _name,bool two_plus_one){
    PackedByteArray out;
    {
        StringName hname("heightmap");
        ERR_FAIL_COND_V_MSG(_name==hname,out,"Use get_heightmap_rf to get heightmap");
    }
    ERR_FAIL_COND_V(!compressed_data.has(_name),out);
    PackedByteArray comp_data = compressed_data[_name];
    //Getting Header
    ERR_FAIL_COND_V_MSG(comp_data[0]!=MMAGIC_NUM,out,"Magic number not found this file can be corrupted");
    ERR_FAIL_COND_V_MSG(comp_data[1]!=CURRENT_MRESOURCE_VERSION,out,"Resource version not match, Please export your data with version "+itos(comp_data[1])+" of MResource to a raw format and reimport that here");
    uint16_t flags = decode_uint16(comp_data.ptr()+FLAGS_INDEX);
    Image::Format format = (Image::Format)comp_data[4];
    format_cache.insert(_name,comp_data[4]);
    uint16_t width = decode_uint16(comp_data.ptr()+6);
    uint32_t data_size_before_file_compress = decode_uint32(comp_data.ptr()+DATA_SIZE_BEFORE_FILE_COMPRESS_INDEX);
    width_cache.insert(_name,width);
    uint8_t pixel_size = MImage::get_format_pixel_size(format);
    ERR_FAIL_COND_V(pixel_size==0,out);
    out.resize(width*width*pixel_size);
    //Finish Getting Header
    comp_data = comp_data.slice(MRESOURCE_HEADER_SIZE);
    uint32_t data_size = width*width*pixel_size;
    // FILE Compress
    if(flags & FLAG_COMPRESSION_FASTLZ){
        comp_data = _PackedByteArray_decompress(&comp_data,data_size_before_file_compress,FileAccess::COMPRESSION_FASTLZ);// comp_data.decompress(data_size_before_file_compress,FileAccess::COMPRESSION_FASTLZ);
    }
    else if(flags & FLAG_COMPRESSION_DEFLATE){
        comp_data = _PackedByteArray_decompress(&comp_data,data_size_before_file_compress,FileAccess::COMPRESSION_DEFLATE);// comp_data.decompress(data_size_before_file_compress,FileAccess::COMPRESSION_DEFLATE);
    }
    else if(flags & FLAG_COMPRESSION_ZSTD){
        comp_data = _PackedByteArray_decompress(&comp_data,data_size_before_file_compress,FileAccess::COMPRESSION_ZSTD);// comp_data.decompress(data_size_before_file_compress,FileAccess::COMPRESSION_ZSTD);
    }
    else if(flags & FLAG_COMPRESSION_GZIP){
        comp_data = _PackedByteArray_decompress(&comp_data,data_size_before_file_compress,FileAccess::COMPRESSION_GZIP);// comp_data.decompress(data_size_before_file_compress,FileAccess::COMPRESSION_GZIP);
    }
    //Compress
    if(flags & FLAG_COMPRESSION_QOI){
        uint8_t channel_count = get_supported_qoi_format_channel_count(format);
        qoi_desc qoi_img;
        qoi_img.width = width;
        qoi_img.height = width;
        qoi_img.channels = channel_count;
        qoi_img.colorspace = 1;
        void *comp_ptr = comp_data.ptrw();
        int size = comp_data.size();
        void *qoi_data = qoi_decode(comp_ptr,size,&qoi_img,channel_count);
        comp_data.resize(data_size);
        memcpy((void*)comp_data.ptrw(),qoi_data,data_size);
        ::free(qoi_data);
    }
    else if(flags & FLAG_COMPRESSION_PNG){
        Ref<Image> img = memnew(Image(width,width,false,format));
        img->load_png_from_buffer(comp_data);
        comp_data = img->get_data();
    }
    ERR_FAIL_COND_V(comp_data.size()!=data_size,comp_data);
    if(two_plus_one){
        comp_data = add_empty_pixels_to_right_bottom(comp_data,pixel_size,width);
    }
    return comp_data;
}

void MResource::remove_data(const StringName& _name){
    ERR_FAIL_COND(!compressed_data.has(_name));
    compressed_data.erase(_name);
}

void MResource::insert_heightmap_rf(const PackedByteArray& data,float accuracy,bool compress_qtq,MResource::FileCompress file_compress){
    const Image::Format format = Image::Format::FORMAT_RF;
    const StringName& _name("heightmap");
    uint32_t pixel_size = MImage::get_format_pixel_size(format);
    ERR_FAIL_COND_MSG(!pixel_size,"Unsported format");
    ERR_FAIL_COND(data.size() % pixel_size != 0);
    uint32_t pixel_amount = data.size() / pixel_size;
    uint32_t width = sqrt(pixel_amount);
    ERR_FAIL_COND(width<1);
    ERR_FAIL_COND(pixel_amount!=width*width);
    // Originally each image has a power of two plus one size in m terrain
    // But the stored image will always have power of two
    // The edge pixels will be corrected after loading the image
    // Here also we drop the edge pixel if our data have them
    // Also only images with power of two is acceptable for compressing
    PackedByteArray final_data;
    if( ((width - 1) & (width - 2))==0 && width!=2){
        uint32_t new_width = width - 1;
        uint32_t new_size = new_width*new_width*pixel_size;
        final_data.resize(new_size);
        // Copy rows
        uint32_t new_row_size = new_width*pixel_size;
        for(uint32_t row=0;row<new_width;row++){
            uint32_t pos_old = row * width * pixel_size;
            uint32_t pos_new = row * new_width * pixel_size;
            memcpy(final_data.ptrw()+pos_new,data.ptr()+pos_old,new_row_size);
        }
        width--;
    } else if (((width & (width - 1)) == 0))
    {
        final_data = data;
    } else {
        ERR_FAIL_MSG("Not a valid image size to compress");
        return;
    }
    // Creating compress data
    PackedByteArray new_compressed_data;
    // Creating header
    uint16_t flags=0;
    new_compressed_data.resize(MRESOURCE_HEIGHTMAP_HEADER_SIZE);
    {
        new_compressed_data.write[0] = MMAGIC_NUM;
        new_compressed_data.write[1] = CURRENT_MRESOURCE_VERSION;
        // FALGS WILL BE ADDED AT THE END
        new_compressed_data.write[4] = (uint8_t)format;
        encode_uint16(width,new_compressed_data.ptrw()+6);
    }
    uint32_t save_index = MRESOURCE_HEADER_SIZE;
    ERR_FAIL_COND_MSG(format!=Image::Format::FORMAT_RF,"You can insert heightmap only with this format: FORMAT_RF");
    flags |= FLAG_IS_HEIGHT_MAP;
    float* data_ptr = (float*)final_data.ptrw();
    // Min and Max height
    float min_height=data[0];
    float max_height=data[0];
    min_height_cache = min_height;
    max_height_cache = max_height;
    uint32_t px_amount = width*width;
    for(uint32_t i=1;i<px_amount;i++){
        if(min_height > data_ptr[i]){
            min_height = data_ptr[i];
        }
        if(max_height < data_ptr[i]){
            max_height = data_ptr[i];
        }
    }
    encode_float(min_height,new_compressed_data.ptrw()+MIN_HEIGHT_INDEX);
    encode_float(max_height,new_compressed_data.ptrw()+MAX_HEIGHT_INDEX);
    PackedByteArray data_part;
    save_index=0;
    if(compress_qtq){
        if(max_height-min_height > FLATTEN_MIN_HEIGHT_DIFF){
            flags |= FLAG_FLATTEN_OLS;
            /// FLATTEN OLS
            uint32_t devision = width/FLATTEN_SECTION_WIDTH;
            ERR_FAIL_COND_MSG(((devision & (devision - 1)) != 0),"Flatten OLS devision is not in power of two "+itos(devision));
            uint32_t flatten_header_data_size = (devision*devision*FLATTEN_SECTION_HEADER_SIZE) + FLATTEN_HEADER_SIZE;
            data_part.resize(data_part.size()+flatten_header_data_size);
            uint8_t devision_log2 = log2(devision);
            ERR_FAIL_COND_MSG(devision_log2>15,"The heightmap image dimension is too big");
            data_part.write[save_index] = devision_log2;
            save_index+=FLATTEN_HEADER_SIZE;
            Vector<uint32_t> flatten_section_header = flatten_ols((float*)final_data.ptrw(),width,devision);
            flatten_header_data_size-=FLATTEN_HEADER_SIZE; //resize to the header section data size only
            ERR_FAIL_COND(flatten_section_header.size()*4!=flatten_header_data_size);
            const uint8_t* flatten_section_header_ptr = (const uint8_t*)flatten_section_header.ptr();
            memcpy(data_part.ptrw()+save_index,flatten_section_header_ptr,flatten_header_data_size);
            flatten_section_header.resize(0);
            save_index+=flatten_header_data_size;
            // Finish setting FLATTEN OLS
        }
        flags |= FLAG_COMPRESSION_QTQ;
        compress_qtq_rf(final_data,data_part,width,save_index,accuracy);
    }
    else
    {
        data_part = final_data;
    }
    uint32_t data_size_before_file_compress=data_part.size();
    /// FILE COMPRESS
    if(file_compress==FileCompress::FILE_COMPRESSION_FASTLZ){
        flags |= FLAG_COMPRESSION_FASTLZ;
        data_part = _PackedByteArray_compress(&data_part,FileAccess::COMPRESSION_FASTLZ);// data_part.compress(FileAccess::COMPRESSION_FASTLZ);
    }
    else if(file_compress==FileCompress::FILE_COMPRESSION_DEFLATE){
        flags |= FLAG_COMPRESSION_DEFLATE;
        data_part = _PackedByteArray_compress(&data_part,FileAccess::COMPRESSION_DEFLATE);// data_part.compress(FileAccess::COMPRESSION_DEFLATE);
    }
    else if(file_compress==FileCompress::FILE_COMPRESSION_ZSTD){
        flags |= FLAG_COMPRESSION_ZSTD;
        data_part = _PackedByteArray_compress(&data_part,FileAccess::COMPRESSION_ZSTD);// data_part.compress(FileAccess::COMPRESSION_ZSTD);
    }
    else if(file_compress==FileCompress::FILE_COMPRESSION_GZIP){
        flags |= FLAG_COMPRESSION_GZIP;
        data_part = _PackedByteArray_compress(&data_part,FileAccess::COMPRESSION_GZIP);// data_part.compress(FileAccess::COMPRESSION_GZIP);
    }
    encode_uint16(flags,new_compressed_data.ptrw()+FLAGS_INDEX);
    encode_uint32(data_size_before_file_compress,new_compressed_data.ptrw()+DATA_SIZE_BEFORE_FILE_COMPRESS_INDEX);
    new_compressed_data.append_array(data_part);
    compressed_data[_name] = new_compressed_data;
}


PackedByteArray MResource::get_heightmap_rf(bool two_plus_one){
    PackedByteArray out;
    StringName _name("heightmap");
    ERR_FAIL_COND_V(!compressed_data.has(_name),out);
    PackedByteArray comp_data = compressed_data[_name];
    //Getting Header
    ERR_FAIL_COND_V_MSG(comp_data[0]!=MMAGIC_NUM,out,"Magic number not found this file can be corrupted");
    ERR_FAIL_COND_V_MSG(comp_data[1]!=CURRENT_MRESOURCE_VERSION,out,"Resource version not match, Please export your data with version "+itos(comp_data[1])+" of MResource to a raw format and reimport that here");
    uint16_t flags = decode_uint16(comp_data.ptr()+FLAGS_INDEX);
    Image::Format format = (Image::Format)comp_data[4];
    ERR_FAIL_COND_V(format!=Image::Format::FORMAT_RF,out);
    uint16_t width = decode_uint16(comp_data.ptr()+6);
    uint32_t data_size_before_file_compress = decode_uint32(comp_data.ptr()+DATA_SIZE_BEFORE_FILE_COMPRESS_INDEX);
    width_cache.insert(_name,width);
    uint8_t pixel_size = MImage::get_format_pixel_size(format);
    ERR_FAIL_COND_V(pixel_size==0,out);
    //Finish Getting Header
    out.resize(width*width*pixel_size);
    //Getting min and max height
    float min_height = decode_float(comp_data.ptrw()+MIN_HEIGHT_INDEX);
    float max_height = decode_float(comp_data.ptrw()+MAX_HEIGHT_INDEX);
    min_height_cache = min_height;
    max_height_cache = max_height;
    comp_data = comp_data.slice(MRESOURCE_HEIGHTMAP_HEADER_SIZE);
    uint32_t decompress_index=0;
    uint32_t data_size = width*width*pixel_size;
    if(flags & FLAG_COMPRESSION_FASTLZ){
        comp_data = _PackedByteArray_decompress(&comp_data,data_size_before_file_compress,FileAccess::COMPRESSION_FASTLZ);// comp_data.decompress(data_size_before_file_compress,FileAccess::COMPRESSION_FASTLZ);
    }
    else if(flags & FLAG_COMPRESSION_DEFLATE){
        comp_data = _PackedByteArray_decompress (&comp_data,data_size_before_file_compress,FileAccess::COMPRESSION_DEFLATE);// comp_data.decompress(data_size_before_file_compress,FileAccess::COMPRESSION_DEFLATE);
    }
    else if(flags & FLAG_COMPRESSION_ZSTD){
        comp_data = _PackedByteArray_decompress (&comp_data,data_size_before_file_compress,FileAccess::COMPRESSION_ZSTD);// comp_data.decompress(data_size_before_file_compress,FileAccess::COMPRESSION_ZSTD);
    }
    else if(flags & FLAG_COMPRESSION_GZIP){
        comp_data = _PackedByteArray_decompress (&comp_data,data_size_before_file_compress,FileAccess::COMPRESSION_GZIP);// comp_data.decompress(data_size_before_file_compress,FileAccess::COMPRESSION_GZIP);
    }
    ////////////// Compress QTQ
    if(!(flags & FLAG_COMPRESSION_QTQ)){
        memcpy(out.ptrw(),comp_data.ptrw()+decompress_index,out.size());
        PackedByteArray out_plus_one = add_empty_pixels_to_right_bottom(out,pixel_size,width);
        return out_plus_one;
    }
    // Getting Flatten header if Flatten_OLS FLAG is active, This will also increate the decompress index
    Vector<uint32_t> flatten_section_header;
    uint16_t devision;
    if(flags & FLAG_FLATTEN_OLS){ // This should be at bottom later
        uint8_t devision_log2 = comp_data[decompress_index];
        decompress_index += FLATTEN_HEADER_SIZE;
        ERR_FAIL_COND_V(devision_log2==0 || devision_log2>15 ,out);
        devision = 1 << devision_log2;
        // Size bellow does not contain FLATTEN_HEADER_SIZE
        uint32_t flatten_header_size = (devision*devision*FLATTEN_SECTION_HEADER_SIZE);
        flatten_section_header.resize(flatten_header_size/FLATTEN_SECTION_HEADER_SIZE);
        memcpy((uint8_t *)flatten_section_header.ptrw(),comp_data.ptr()+decompress_index,flatten_header_size);
        decompress_index+=flatten_header_size;
    }
    decompress_qtq_rf(comp_data,out,width,decompress_index);
    if(flags & FLAG_FLATTEN_OLS){ 
        unflatten_ols((float*)out.ptrw(),width,devision,flatten_section_header);
    }
    if(two_plus_one){
        PackedByteArray out_plus_one = add_empty_pixels_to_right_bottom(out,pixel_size,width);
        return out_plus_one;
    }
    return out;
}

PackedByteArray MResource::add_empty_pixels_to_right_bottom(const PackedByteArray& data, uint8_t pixel_size, uint32_t width){
    PackedByteArray out;
    // Check if size of data match
    ERR_FAIL_COND_V(data.size()!=width*width*pixel_size,out);
    uint32_t new_width = width + 1;
    out.resize(new_width*new_width*pixel_size);
    for(uint32_t y=0;y<width;y++){
        uint32_t old_index = y*width*pixel_size;
        uint32_t new_index = y*new_width*pixel_size;
        memcpy(out.ptrw()+new_index,data.ptr()+old_index,width*pixel_size);
        //Correcting edges in each of row
        uint32_t old_end_row_index = old_index + (width - 1)*pixel_size;
        uint32_t new_end_row_index = new_index + (new_width - 1)*pixel_size;
        memcpy(out.ptrw()+new_end_row_index,data.ptr()+old_end_row_index,pixel_size);
    }
    {
        // Correcting edge in last row
        uint32_t old_last_row_index = (width - 1)*width*pixel_size;
        uint32_t new_last_row_index = (new_width - 1)*new_width*pixel_size;
        memcpy(out.ptrw()+new_last_row_index,data.ptr()+old_last_row_index,width*pixel_size);
    }
    {
        // Correcting bottom right corner
        uint32_t old_last_index = ((width - 1) + (width - 1)*width)*pixel_size;
        uint32_t new_last_index = ((new_width - 1) + (new_width - 1)*new_width)*pixel_size;
        memcpy(out.ptrw()+new_last_index,data.ptr()+old_last_index,pixel_size);
    }
    return out;
}

void MResource::compress_qtq_rf(PackedByteArray& uncompress_data,PackedByteArray& compress_data,uint32_t window_width,uint32_t& save_index,float accuracy){
    //Building the QuadTree
    MPixelRegion window_px_region(window_width,window_width);
    float* ptr = (float*)uncompress_data.ptrw();
    MResource::QuadTreeRF* quad_tree = memnew(MResource::QuadTreeRF(window_px_region,ptr,window_width,accuracy));
    // Creating compress_QTQ data Header
    {
        compress_data.resize(save_index + COMPRESSION_QTQ_HEADER_SIZE);
        uint8_t* ptrw = compress_data.ptrw();
        ptrw += save_index;
        encode_float(quad_tree->min_height,ptrw);
        ptrw += 4;
        encode_float(quad_tree->max_height,ptrw);
        ptrw += 4;
        ptrw[0] = quad_tree->h_encoding;
        save_index += COMPRESSION_QTQ_HEADER_SIZE;
    }
    quad_tree->divide_upto_leaf();
    uint32_t size = quad_tree->get_optimal_size();
    compress_data.resize(save_index+size);
    uint8_t before_save_index = save_index;
    #ifdef PRINT_DEBUG
    dump_header(compress_data);
    dump_qtq_header(compress_data);
    VariantUtilityFunctions::_print("Start to save ----------------------------------------------",save_index);
    #endif
    quad_tree->save_quad_tree_data(compress_data,save_index);
    #ifdef PRINT_DEBUG
    VariantUtilityFunctions::_print("End saving ----------------------------------------------",save_index);
    #endif
    memdelete(quad_tree);
}

void MResource::decompress_qtq_rf(const PackedByteArray& compress_data,PackedByteArray& uncompress_data,uint32_t window_width,uint32_t decompress_index){
    float main_min_height = decode_float(compress_data.ptr()+decompress_index);
    decompress_index+=4;
    float main_max_height = decode_float(compress_data.ptr()+decompress_index);
    decompress_index+=4;
    uint8_t h_encoding = compress_data[decompress_index];
    decompress_index++;
    MPixelRegion px_region(window_width,window_width);
    //(MPixelRegion _px_region,uint32_t _window_width,uint8_t _h_encoding,uint8_t _depth=0,MResource::QuadTreeRF* _root=nullptr)
    float* ptrw = (float*)uncompress_data.ptrw();
    MResource::QuadTreeRF* quad_tree = memnew(MResource::QuadTreeRF(px_region,ptrw,window_width,h_encoding));
    quad_tree->min_height = main_min_height;
    quad_tree->max_height = main_max_height;
    #ifdef PRINT_DEBUG
    dump_header(compress_data);
    dump_qtq_header(compress_data);
    VariantUtilityFunctions::_print("Start to Load ----------------------------------------------",decompress_index);
    #endif
    quad_tree->load_quad_tree_data(compress_data,decompress_index);
    #ifdef PRINT_DEBUG
    VariantUtilityFunctions::_print("End Loading ----------------------------------------------",decompress_index);
    #endif
    memdelete(quad_tree);
}

// https://research.usq.edu.au/download/987c0abcd3777bef266467fee9e205061f6eb905c44526215eb21a7d6404aebf/272457/TS04D_scarmana_8433.pdf
//--- z is height
//--- x and y are position in x y plane
/* MATRIX A
sum_i x[i]*x[i],    sum_i x[i]*y[i],    sum_i x[i]
sum_i x[i]*y[i],    sum_i y[i]*y[i],    sum_i y[i]
sum_i x[i],         sum_i y[i],         n
*/
/* MATRIX C
{sum_i x[i]*z[i],   sum_i y[i]*z[i],    sum_i z[i]}
*/
/* MATRIX B -> These are plane coefficient
{b1, b2, b2} 
*/
// A*B = C -> Then -> B = A_INVERSE*C
/* PLANE FORMULA
    z = b1*x + b2*y + b3
*/
// As we have a uniform grid we have below rules
// sumx2=sumy2
// sumx=sumy
// sumxy=sumyx
Vector<uint32_t> MResource::flatten_ols(float* data,uint32_t witdth,uint16_t devision){
    uint32_t section_width = witdth/devision;
    Basis matrix_a_invers;
    //As matrix A is same among all sections we calculate that here for better performance
    {
        uint64_t sumx = get_sumx(section_width);
        uint64_t sumx2 = get_sumx2(section_width);
        uint64_t sumxy = get_sumxy(section_width);
        Vector3 row0(sumx2,sumxy,sumx);
        Vector3 row1(sumxy,sumx2,sumx);
        Vector3 row2(sumx,sumx,section_width*section_width);
        matrix_a_invers = Basis(row0,row1,row2).inverse();
    }
    Vector<uint32_t> planes;
    for(uint32_t dy=0;dy<devision;dy++){
        for(uint32_t dx=0;dx<devision;dx++){
            MPixelRegion px_reg;
            px_reg.left = dx*section_width;
            px_reg.right = ((dx+1)*section_width) - 1;
            px_reg.top = dy*section_width;
            px_reg.bottom = ((dy+1)*section_width) - 1;
            uint32_t p = flatten_section_ols(data,px_reg,witdth,matrix_a_invers);
            planes.push_back(p);
        }
    }
    return planes;
}

void MResource::unflatten_ols(float* data,uint32_t witdth,uint16_t devision,const Vector<uint32_t>& headers){
    ERR_FAIL_COND(headers.size()!=devision*devision);
    uint32_t section_width = witdth/devision;
    uint32_t header_index = 0;
    for(uint32_t dy=0;dy<devision;dy++){
        for(uint32_t dx=0;dx<devision;dx++){
            MPixelRegion px_reg;
            px_reg.left = dx*section_width;
            px_reg.right = ((dx+1)*section_width) - 1;
            px_reg.top = dy*section_width;
            px_reg.bottom = ((dy+1)*section_width) - 1;
            unflatten_section_ols(data,px_reg,witdth,headers[header_index]);
            header_index++;
        }
    }
}
uint32_t MResource::flatten_section_ols(float* data,MPixelRegion px_region,uint32_t window_width,Basis matrix_a_invers){
    uint32_t header;
    //Ignore if there is a hole in section
    for(uint32_t gy=px_region.top;gy<=px_region.bottom;gy++){
        for(uint32_t gx=px_region.left;gx<=px_region.right;gx++){
            uint32_t index = gx + gy*window_width;
            if(IS_HOLE(data[index])){
                float af = FLOAT_HOLE;
                uint16_t ah = float_to_half(af);
                uint8_t* header_ptr = (uint8_t*)&header;
                encode_uint16(ah,header_ptr);
                encode_uint16(ah,header_ptr+2);
                return header;
            }
        }
    }
    // Calculating sum(xz)
    double sumxz=0;
    for(uint32_t gy=px_region.top;gy<=px_region.bottom;gy++){
        for(uint32_t gx=px_region.left;gx<=px_region.right;gx++){
            // Local x to this section
            uint32_t x = gx - px_region.left;
            uint32_t index = gx + gy*window_width;
            sumxz += data[index]*x;
        }
    }
    // Calculating sum(yz)
    double sumyz=0;
    for(uint32_t gy=px_region.top;gy<=px_region.bottom;gy++){
        for(uint32_t gx=px_region.left;gx<=px_region.right;gx++){
            // Local x to this section
            uint32_t y = gy - px_region.top;
            uint32_t index = gx + gy*window_width;
            sumyz += data[index]*y;
        }
    }
    // Calculating sum(z)
    double sumz=0;
    for(uint32_t gy=px_region.top;gy<=px_region.bottom;gy++){
        for(uint32_t gx=px_region.left;gx<=px_region.right;gx++){
            uint32_t index = gx + gy*window_width;
            sumz += data[index];
        }
    }
    // Creating C Matrix
    Vector3 matrix_c(sumxz,sumyz,sumz);
    // Claculating Matrix B
    Vector3 matrix_b = matrix_a_invers.xform(matrix_c);

    uint16_t b1_half = float_to_half(matrix_b.x);
    uint16_t b2_half = float_to_half(matrix_b.y);
    // We should convert back so to float so we get the same result when decompressing
    // This is due to error while convert from half to float
    float b1 = half_to_float(b1_half);
    float b2 = half_to_float(b2_half);

    for(uint32_t gy=px_region.top;gy<=px_region.bottom;gy++){
        for(uint32_t gx=px_region.left;gx<=px_region.right;gx++){
            uint32_t index = gx + gy*window_width;
            // We should use local x and y
            float x = (float)(gx - px_region.left);
            float y = (float)(gy - px_region.top);
            float plane_z = b1*x + b2*y;
            //plane_z += matrix_b.z;
            data[index] -= plane_z;
        }
    }
    uint8_t* header_ptr = (uint8_t*)&header;
    encode_uint16(b1_half,header_ptr);
    encode_uint16(b2_half,header_ptr+2);
    return header;
}

void MResource::unflatten_section_ols(float* data,MPixelRegion px_region,uint32_t window_width,uint32_t header){
    float b1;
    float b2;
    {
        const uint8_t* header_ptr = (const uint8_t*)&header;
        uint16_t half_b1 = decode_uint16(header_ptr);
        uint16_t half_b2 = decode_uint16(header_ptr+2);
        b1 = half_to_float(half_b1);
        b2 = half_to_float(half_b2);
    }
    if(IS_HOLE(b1) || IS_HOLE(b2)){
        return; // Ignore in case there is hole or in case b1,b2 is NAN
    }
    for(uint32_t gy=px_region.top;gy<=px_region.bottom;gy++){
        for(uint32_t gx=px_region.left;gx<=px_region.right;gx++){
            uint32_t index = gx + gy*window_width;
            // We should use local x and y
            float x = (float)(gx - px_region.left);
            float y = (float)(gy - px_region.top);
            float plane_z = b1*x + b2*y;
            data[index] += plane_z;
        }
    }
}
// x = 0,1,2,...,n -> sum(x) = n(n+1)/2 --> Gauss formula
uint64_t MResource::get_sumx(uint64_t n){
    n--; // as this n contain also x in zero position
    uint64_t sum = (n*(n+1)/2); // only one row
    return sum*(n+1);
}

uint64_t MResource::get_sumxy(uint64_t n){
    uint64_t sum=0;
    for(uint32_t i=0;i<n;i++){
        for(uint32_t j=0;j<n;j++){
            sum+=i*j;
        }
    }
    return sum;
}

uint64_t MResource::get_sumx2(uint64_t n){
    uint64_t sum=0;
    for(uint32_t i=0;i<n;i++){
        sum += i*i;
    } // only one row
    return sum*n;
}


void MResource::encode_uint2(uint8_t a,uint8_t b,uint8_t c,uint8_t d, uint8_t *p_arr){
    *p_arr=0;
    *p_arr = a | (b << 2) | (c << 4) | (d << 6);
}

void MResource::decode_uint2(uint8_t& a,uint8_t& b,uint8_t& c,uint8_t& d,const uint8_t *p_arr){
    a = *p_arr & 0x3;
    b = ((*p_arr & 0xC) >> 2);
    c = ((*p_arr & 0x30) >> 4);
    d = ((*p_arr & 0xC0)>>6);
}

void MResource::encode_uint4(uint8_t a,uint8_t b,uint8_t *p_arr){
    *p_arr = a | (b << 4);
}

void MResource::decode_uint4(uint8_t& a,uint8_t& b,const uint8_t *p_arr){
    a = *p_arr & 0xF;
    b = ((*p_arr & 0xF0)>>4);
}

void MResource::encode_uint16(uint16_t p_uint, uint8_t *p_arr){
	for (int i = 0; i < 2; i++) {
		*p_arr = p_uint & 0xFF;
		p_arr++;
		p_uint >>= 8;
	}
}

uint16_t MResource::decode_uint16(const uint8_t *p_arr){
	uint16_t u = 0;
	for (int i = 0; i < 2; i++) {
		uint16_t b = *p_arr;
		b <<= (i * 8);
		u |= b;
		p_arr++;
	}
    return u;
}

void MResource::encode_uint32(uint32_t p_uint, uint8_t *p_arr){
	for (int i = 0; i < 4; i++) {
		*p_arr = p_uint & 0xFF;
		p_arr++;
		p_uint >>= 8;
	}
}

uint32_t MResource::decode_uint32(const uint8_t *p_arr){
	uint32_t u = 0;
	for (int i = 0; i < 4; i++) {
		uint32_t b = *p_arr;
		b <<= (i * 8);
		u |= b;
		p_arr++;
	}
    return u;
}

void MResource::encode_uint64(uint64_t p_uint, uint8_t *p_arr){
	for (int i = 0; i < 8; i++) {
		*p_arr = p_uint & 0xFF;
		p_arr++;
		p_uint >>= 8;
	}
}

uint64_t MResource::decode_uint64(const uint8_t *p_arr){
	uint64_t u = 0;
	for (int i = 0; i < 8; i++) {
		uint64_t b = (*p_arr) & 0xFF;
		b <<= (i * 8);
		u |= b;
		p_arr++;
	}
    return u;
}

void MResource::encode_float(float p_float, uint8_t *p_arr){
    encode_uint32(reinterpret_cast<uint32_t&>(p_float) , p_arr);
}

float MResource::decode_float(const uint8_t *p_arr){
    uint32_t u = decode_uint32(p_arr);
    return reinterpret_cast<float&>(u);
}

uint16_t MResource::float_to_half(float f) {
	union {
			float fv;
			uint32_t ui;
		} ci;
		ci.fv = f;

		uint32_t x = ci.ui;
		uint32_t sign = (unsigned short)(x >> 31);
		uint32_t mantissa;
		uint32_t exponent;
		uint16_t hf;

		// get mantissa
		mantissa = x & ((1 << 23) - 1);
		// get exponent bits
		exponent = x & (0xFF << 23);
		if (exponent >= 0x47800000) {
			// check if the original single precision float number is a NaN
			if (mantissa && (exponent == (0xFF << 23))) {
				// we have a single precision NaN
				mantissa = (1 << 23) - 1;
			} else {
				// 16-bit half-float representation stores number as Inf
				mantissa = 0;
			}
			hf = (((uint16_t)sign) << 15) | (uint16_t)((0x1F << 10)) |
					(uint16_t)(mantissa >> 13);
		}
		// check if exponent is <= -15
		else if (exponent <= 0x38000000) {
			/*
			// store a denorm half-float value or zero
			exponent = (0x38000000 - exponent) >> 23;
			mantissa >>= (14 + exponent);

			hf = (((uint16_t)sign) << 15) | (uint16_t)(mantissa);
			*/
			hf = 0; //denormals do not work for 3D, convert to zero
		} else {
			hf = (((uint16_t)sign) << 15) |
					(uint16_t)((exponent - 0x38000000) >> 13) |
					(uint16_t)(mantissa >> 13);
		}

		return hf;
}

float MResource::half_to_float(uint16_t h){
    union {
        uint32_t u32;
        float f32;
    } u;

    u.u32 = half_to_uint32(h);
    return u.f32;
}

uint32_t MResource::half_to_uint32(uint16_t h){
    uint16_t h_exp, h_sig;
    uint32_t f_sgn, f_exp, f_sig;

    h_exp = (h & 0x7c00u);
    f_sgn = ((uint32_t)h & 0x8000u) << 16;
    switch (h_exp) {
        case 0x0000u: /* 0 or subnormal */
            h_sig = (h & 0x03ffu);
            /* Signed zero */
            if (h_sig == 0) {
                return f_sgn;
            }
            /* Subnormal */
            h_sig <<= 1;
            while ((h_sig & 0x0400u) == 0) {
                h_sig <<= 1;
                h_exp++;
            }
            f_exp = ((uint32_t)(127 - 15 - h_exp)) << 23;
            f_sig = ((uint32_t)(h_sig & 0x03ffu)) << 13;
            return f_sgn + f_exp + f_sig;
        case 0x7c00u: /* inf or NaN */
            /* All-ones exponent and a copy of the significand */
            return f_sgn + 0x7f800000u + (((uint32_t)(h & 0x03ffu)) << 13);
        default: /* normalized */
            /* Just need to adjust the exponent and shift */
            return f_sgn + (((uint32_t)(h & 0x7fffu) + 0x1c000u) << 13);
    }
}



int MResource::get_supported_qoi_format_channel_count(Image::Format p_format) {
	switch (p_format) {
		case Image::FORMAT_RGB8:
			return 3;
		case Image::FORMAT_RGBA8:
			return 4;
	}
	return 0;
}

bool MResource::get_supported_png_format(Image::Format format){
	switch (format) {
		case Image::FORMAT_RGB8:
			return true;
		case Image::FORMAT_RGBA8:
			return true;
        case Image::FORMAT_L8:
            return true;
	}
	return false;
}


MResource::Compress MResource::get_compress(const StringName& _name){
    ERR_FAIL_COND_V(!compressed_data.has(_name),MResource::Compress::COMPRESS_NONE);
    uint16_t flag;
    {
        const PackedByteArray data = compressed_data[_name];
        flag = decode_uint16(data.ptr()+FLAGS_INDEX);
    }
    if(flag & FLAG_COMPRESSION_QOI){
        return MResource::Compress::COMPRESS_QOI;
    }
    if(flag & FLAG_COMPRESSION_PNG){
        return MResource::Compress::COMPRESS_PNG;
    }
    return MResource::Compress::COMPRESS_NONE;
}

MResource::FileCompress MResource::get_file_compress(const StringName& _name){
    ERR_FAIL_COND_V(!compressed_data.has(_name),MResource::FileCompress::FILE_COMPRESSION_NONE);
    uint16_t flag;
    {
        const PackedByteArray data = compressed_data[_name];
        flag = decode_uint16(data.ptr()+FLAGS_INDEX);
    }
    if(flag & FLAG_COMPRESSION_FASTLZ){
       return MResource::FileCompress::FILE_COMPRESSION_FASTLZ;
    }
    if(flag & FLAG_COMPRESSION_DEFLATE){
        return MResource::FileCompress::FILE_COMPRESSION_DEFLATE;
    }
    if(flag & FLAG_COMPRESSION_ZSTD){
        return MResource::FileCompress::FILE_COMPRESSION_ZSTD;
    }
    if(flag & FLAG_COMPRESSION_GZIP){
        return MResource::FileCompress::FILE_COMPRESSION_GZIP;
    }
    return MResource::FileCompress::FILE_COMPRESSION_NONE;
}

bool MResource::is_compress_qtq(){
    ERR_FAIL_COND_V(!compressed_data.has(HEIGHTMAP_NAME),false);
    uint16_t flag;
    {
        const PackedByteArray data = compressed_data[HEIGHTMAP_NAME];
        flag = decode_uint16(data.ptr()+FLAGS_INDEX);
    }
    return flag & FLAG_COMPRESSION_QTQ;
}
