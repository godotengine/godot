#ifndef MRESOURCE
#define MRESOURCE

#define CURRENT_MRESOURCE_VERSION 0 // Unless change which cause compatiblity break this will no change

#define MRESOURCE_HEADER_SIZE 12
#define MMAGIC_NUM 77
#define FLAGS_INDEX 2
#define FORMAT_INDEX 4
#define WIDTH_INDEX 6
#define DATA_SIZE_BEFORE_FILE_COMPRESS_INDEX 8
// INDEX    DATA HEADER --- common in all formats
// 0     uint8_t -> Magic num -> 77
// 1     uint8_t -> Version NUMBER
// 2     uint16_t FLAGS -> some flags for determining which compression is applied and whether data is heightmap or other image type
// 4     uint8_t image_format -> same as Image::Format in image class in Godot
// 6     uint16_t width (LE) -> width and height are equale so height is this
// 8     uint32_t data_size (LE) -> data size before applying file compression

// IF FLAG_IS_HEIGHT_MAP is active -> in total 4 byte
#define MRESOURCE_HEIGHTMAP_HEADER_SIZE 20
#define MIN_HEIGHT_INDEX 12
#define MAX_HEIGHT_INDEX 16
// float min_height
// float max_height

// In case Flatten flag is active
#define FLATTEN_MIN_HEIGHT_DIFF 10 //If difference between min and max height is less than this the FLATTEN will no be applied as heightmap is allready queit flat
#define FLATTEN_SECTION_WIDTH 8 // must be in power of two
#define FLATTEN_HEADER_SIZE 1 // two half floating
#define FLATTEN_SECTION_HEADER_SIZE 4 // two half floating
// next pos is the devision_log2 amount for FLATTEN in uint8_t format
// DATA amount for FLATTEN section (devision*devision)*FLATTEN_SECTION_HEADER_SIZE + FLATTEN_HEADER_SIZE

#define COMPRESSION_QTQ_HEADER_SIZE 9
// COMPRESSION_QTQ header start from 4th-byte position (As always aplly first)
// float min_height -> 4 byte
// float max_height -> 4 byte
// uint8_t h_encoding -> show how min and max height in each block is encoded
////////// 0 -> encoded as uint4
////////// 1 -> encoded as uint8
////////// 2 -> encoded as uint16
////////// 3 -> encoded as float

// COMPRESSION_QTQ Block start after above
// 0xF0 the most left byte is the depth inside the QuadTreeRF
// 0x0F the remaining 4 right bytes are as Follows
// 1 left byte show if we have a hole inside that block in case there is a hole inside that block the last number uintX will reserve for hole and in case it is float the hole represent NAN value
// 3 right byte of that show the encoding of block which is as follow: (DEFINED BY DATA_ENCODE_)
////////// 0 -> flat and the entire block has the same height and its determine by min height, in this case we don't have max-height, and there is no data in block
////////// 2 -> height are encoded as uint2
////////// 3 -> height are encoded as uint4
////////// 4 -> height are encoded as uint8
////////// 5 -> height are encoded as uint16
////////// 6 -> height are encoded as float -> in this case the is no min and max height in header


// Min and max height in each block divid the main min and max block from zero to maximum number which they can handle

#define DATA_ENCODE_FLAT 0
#define DATA_ENCODE_U2 1
#define DATA_ENCODE_U4 2
#define DATA_ENCODE_U8 3
#define DATA_ENCODE_U16 4
#define DATA_ENCODE_FLOAT 5
#define DATA_ENCODE_ONLY_HOLE 6
#define DATA_ENCODE_MAX 7

// one last number in each encoding is reserved for terrain holes
#define U2_MAX 3
#define U4_MAX 15
#define U8_MAX 255
#define U16_MAX 65535

#define MIN_U4(value) (((value)>U4_MAX) ? U4_MAX : value);

// In case there is hole in terrain the last number of each data encoding is hole
// This is only true for data_encoding not h_encoding
#define HU2_MAX 2
#define HU4_MAX 14
#define HU8_MAX 254
#define HU16_MAX 65534

#define H_ENCODE_U4 0
#define H_ENCODE_U8 1
#define H_ENCODE_U16 2
#define H_ENCODE_FLOAT 3
#define H_ENCODE_MAX 4


#define MIN_BLOCK_SIZE_IN_QUADTREERF 4

// If more compression will be added in future
// All compression must be be lossless except COMPRESSION_QTQ or any comression which is in first order
#define FLAG_IS_HEIGHT_MAP 1 // QuadTreeRF Quantazation compression
#define FLAG_FLATTEN_OLS 2 // QuadTreeRF Quantazation compression
#define FLAG_COMPRESSION_QTQ 4 // QuadTreeRF Quantazation compression
#define FLAG_COMPRESSION_QOI 8
#define FLAG_COMPRESSION_PNG 16
#define FLAG_COMPRESSION_FASTLZ 32
#define FLAG_COMPRESSION_DEFLATE 64
#define FLAG_COMPRESSION_ZSTD 128
#define FLAG_COMPRESSION_GZIP 256

#include "core/variant/variant.h"
#include "core/variant/dictionary.h"
#include "core/io/image.h"
#include "core/templates/hash_map.h"
#include "mpixel_region.h"
#include "mconfig.h"



class MResource : public Resource {
    GDCLASS(MResource,Resource);

    protected:
    static void _bind_methods();

    private:
    struct QuadTreeRF
    {
        MResource::QuadTreeRF* ne = nullptr; //North east
        MResource::QuadTreeRF* nw = nullptr; //North west
        MResource::QuadTreeRF* se = nullptr; //South east
        MResource::QuadTreeRF* sw = nullptr; //South west
        bool has_hole = false;
        MPixelRegion px_region;
        float* data=nullptr; // Point always to uncompress data
        uint32_t window_width;
        uint8_t depth = 0;
        uint8_t h_encoding;
        uint8_t data_encoding=255;
        float accuracy=-1;
        float min_height;
        float max_height;
        QuadTreeRF* root=nullptr;
        QuadTreeRF(MPixelRegion _px_region,float* _data,uint32_t _window_width, float _accuracy, MResource::QuadTreeRF* _root=nullptr,uint8_t _depth=0,uint8_t _h_encoding=255);
        //Bellow constructor is used for decompression
        QuadTreeRF(MPixelRegion _px_region,float* _data,uint32_t _window_width,uint8_t _h_encoding,uint8_t _depth=0,MResource::QuadTreeRF* _root=nullptr);
        ~QuadTreeRF();
        void update_min_max_height(); // If min max height remian NAN after this the entire section is hole
        void divide_upto_leaf();
        _FORCE_INLINE_ uint32_t get_only_hole_head_size();
        _FORCE_INLINE_ uint32_t get_flat_head_size();
        _FORCE_INLINE_ uint32_t get_block_head_size();
        uint32_t get_optimal_non_divide_size(); // This will also determine data encoding
        //Bellow will call above method so data encoding will be determined
        //Also bellow will determine if we should divide or not
        uint32_t get_optimal_size();
        private:
        void encode_min_max_height(PackedByteArray& save_data,uint32_t& save_index);
        void decode_min_max_height(const PackedByteArray& compress_data,uint32_t& decompress_index);
        public:
        void save_quad_tree_data(PackedByteArray& save_data,uint32_t& save_index);
        void load_quad_tree_data(const PackedByteArray& compress_data,uint32_t& decompress_index);
        
        private:
        void encode_data_u2(PackedByteArray& save_data,uint32_t& save_index);
        void encode_data_u4(PackedByteArray& save_data,uint32_t& save_index);
        void encode_data_u8(PackedByteArray& save_data,uint32_t& save_index);
        void encode_data_u16(PackedByteArray& save_data,uint32_t& save_index);
        void encode_data_float(PackedByteArray& save_data,uint32_t& save_index);

        void decode_data_flat();
        void decode_data_u2(const PackedByteArray& compress_data,uint32_t& decompress_index);
        void decode_data_u4(const PackedByteArray& compress_data,uint32_t& decompress_index);
        void decode_data_u8(const PackedByteArray& compress_data,uint32_t& decompress_index);
        void decode_data_u16(const PackedByteArray& compress_data,uint32_t& decompress_index);
        void decode_data_float(const PackedByteArray& compress_data,uint32_t& decompress_index);
        void decode_data_only_hole();
    };
    Dictionary compressed_data;
    //As if we want to grab these bellow from compress data we should copy them entire compress data
    //Into packedByteArray we cache them for better performance
    //In case in future there will possible to get data compressed_data[key] without coppy we can remove them
    HashMap<StringName,uint8_t> format_cache;
    HashMap<StringName,uint16_t> width_cache;
    float min_height_cache = FLOAT_HOLE;
    float max_height_cache = FLOAT_HOLE;

    public:
    MResource();
    ~MResource();
    enum Compress {
        COMPRESS_NONE = 0,
        COMPRESS_QOI = 1,
        COMPRESS_PNG = 2
    };
    enum FileCompress {
        FILE_COMPRESSION_NONE = 0,
        FILE_COMPRESSION_FASTLZ = 1,
        FILE_COMPRESSION_DEFLATE = 2,
        FILE_COMPRESSION_ZSTD = 3,
        FILE_COMPRESSION_GZIP = 4
    };
    Array get_data_names();
    bool has_data(const StringName& name);
    void set_compressed_data(const Dictionary& data);
    const Dictionary& get_compressed_data();

    Image::Format get_data_format(const StringName& name);
    uint16_t get_data_width(const StringName& name);
    uint32_t get_heightmap_width();
    float get_min_height();
    float get_max_height();

    void insert_data(const PackedByteArray& data, const StringName& name,Image::Format format,MResource::Compress compress,MResource::FileCompress file_compress);
    PackedByteArray get_data(const StringName& name,bool two_plus_one=true);
    void remove_data(const StringName& name);

    void insert_heightmap_rf(const PackedByteArray& data,float accuracy,bool compress_qtq = true,MResource::FileCompress file_compress=MResource::FileCompress::FILE_COMPRESSION_NONE);
    PackedByteArray get_heightmap_rf(bool two_plus_one=true);



    private:
    //Will add on empty pixels rows and coulums to right and bottom
    PackedByteArray add_empty_pixels_to_right_bottom(const PackedByteArray& data, uint8_t pixel_size, uint32_t width);
    // Compresion base on 
    void compress_qtq_rf(PackedByteArray& uncompress_data,PackedByteArray& compress_data,uint32_t window_width,uint32_t& save_index,float accuracy);
    void decompress_qtq_rf(const PackedByteArray& compress_data,PackedByteArray& uncompress_data,uint32_t window_width,uint32_t decompress_index);
    // Will add and remove Linear Regression with least square method
    Vector<uint32_t> flatten_ols(float* data,uint32_t witdth,uint16_t devision); // Will ignore holes
    void unflatten_ols(float* data,uint32_t witdth,uint16_t devision,const Vector<uint32_t>& headers); 
    uint32_t flatten_section_ols(float* data,MPixelRegion px_region,uint32_t window_width,Basis matrix_a_invers);
    void unflatten_section_ols(float* data,MPixelRegion px_region,uint32_t window_width,uint32_t header);
    _FORCE_INLINE_ uint64_t get_sumx(uint64_t n);
    _FORCE_INLINE_ uint64_t get_sumxy(uint64_t n);
    _FORCE_INLINE_ uint64_t get_sumx2(uint64_t n);
    public:
    // Too much overhead if use these in PackedByteArray
    static _FORCE_INLINE_ void encode_uint2(uint8_t a,uint8_t b,uint8_t c,uint8_t d, uint8_t *p_arr);
    static _FORCE_INLINE_ void decode_uint2(uint8_t& a,uint8_t& b,uint8_t& c,uint8_t& d,const uint8_t *p_arr);
    static _FORCE_INLINE_ void encode_uint4(uint8_t a,uint8_t b, uint8_t *p_arr);
    static _FORCE_INLINE_ void decode_uint4(uint8_t& a,uint8_t& b,const uint8_t *p_arr);
    static _FORCE_INLINE_ void encode_uint16(uint16_t p_uint, uint8_t *p_arr);
    static _FORCE_INLINE_ uint16_t decode_uint16(const uint8_t *p_arr);
    static _FORCE_INLINE_ void encode_uint32(uint32_t p_uint, uint8_t *p_arr);
    static _FORCE_INLINE_ uint32_t decode_uint32(const uint8_t *p_arr);
    static _FORCE_INLINE_ void encode_uint64(uint64_t p_uint, uint8_t *p_arr);
    static _FORCE_INLINE_ uint64_t decode_uint64(const uint8_t *p_arr);
    static _FORCE_INLINE_ void encode_float(float p_float, uint8_t *p_arr);
    static _FORCE_INLINE_ float decode_float(const uint8_t *p_arr);
    static _FORCE_INLINE_ uint16_t float_to_half(float f);
    static _FORCE_INLINE_ float half_to_float(uint16_t h);
    static _FORCE_INLINE_ uint32_t half_to_uint32(uint16_t f);

    int get_supported_qoi_format_channel_count(Image::Format format);
    bool get_supported_png_format(Image::Format format);

    MResource::Compress get_compress(const StringName& name);
    MResource::FileCompress get_file_compress(const StringName& name);
    bool is_compress_qtq();
};

VARIANT_ENUM_CAST(MResource::Compress);
VARIANT_ENUM_CAST(MResource::FileCompress);

#endif