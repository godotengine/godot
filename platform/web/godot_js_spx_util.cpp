#include "modules/spx/gdextension_spx_ext.h"
#include "core/extension/gdextension.h"

#include "godot_js_spx_util.h"
#include <string.h>
#include <cstdlib>
#include <cstdint>
#include <emscripten/emscripten.h>

static ObjectPool<GdVec2> vec2Pool(100);
static ObjectPool<GdString> stringPool(100);
static ObjectPool<GdObj> objPool(100);
static ObjectPool<GdInt> intPool(100);
static ObjectPool<GdFloat> floatPool(100);
static ObjectPool<GdBool> boolPool(100);
static ObjectPool<GdVec3> vec3Pool(100);
static ObjectPool<GdVec4> vec4Pool(100);
static ObjectPool<GdColor> colorPool(100);
static ObjectPool<GdRect2> rect2Pool(100);
static ObjectPool<GdArray> arrayPool(100);

// Check if the machine is little-endian
inline bool isLittleEndian() {
    static const uint32_t test = 0x12345678;
    return *reinterpret_cast<const uint8_t*>(&test) == 0x78;
}

// LittleEnd read functions
uint64_t readUint64LE(const uint8_t* bytes) {
    if (isLittleEndian()) {
        return *reinterpret_cast<const uint64_t*>(bytes);
    }
    return (uint64_t)bytes[0] |
           ((uint64_t)bytes[1] << 8) |
           ((uint64_t)bytes[2] << 16) |
           ((uint64_t)bytes[3] << 24) |
           ((uint64_t)bytes[4] << 32) |
           ((uint64_t)bytes[5] << 40) |
           ((uint64_t)bytes[6] << 48) |
           ((uint64_t)bytes[7] << 56);
}

uint32_t readUint32LE(const uint8_t* bytes) {
    if (isLittleEndian()) {
        return *reinterpret_cast<const uint32_t*>(bytes);
    }
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) |
           ((uint32_t)bytes[3] << 24);
}

void writeUint64LE(uint8_t* bytes, uint64_t value) {
    if (isLittleEndian()) {
        *reinterpret_cast<uint64_t*>(bytes) = value;
        return;
    }
    bytes[0] = value & 0xFF;
    bytes[1] = (value >> 8) & 0xFF;
    bytes[2] = (value >> 16) & 0xFF;
    bytes[3] = (value >> 24) & 0xFF;
    bytes[4] = (value >> 32) & 0xFF;
    bytes[5] = (value >> 40) & 0xFF;
    bytes[6] = (value >> 48) & 0xFF;
    bytes[7] = (value >> 56) & 0xFF;
}

void writeUint32LE(uint8_t* bytes, uint32_t value) {
    if (isLittleEndian()) {
        *reinterpret_cast<uint32_t*>(bytes) = value;
        return;
    }
    bytes[0] = value & 0xFF;
    bytes[1] = (value >> 8) & 0xFF;
    bytes[2] = (value >> 16) & 0xFF;
    bytes[3] = (value >> 24) & 0xFF;
}

extern "C" {

// other functions
EMSCRIPTEN_KEEPALIVE
float gdspx_get_value(float* array, int idx) {
    return array[idx];
}


// bool functions
EMSCRIPTEN_KEEPALIVE
GdBool* gdspx_alloc_bool() {
    return boolPool.acquire();
}

EMSCRIPTEN_KEEPALIVE
GdBool* gdspx_new_bool(bool val) {
    GdBool* ptr = gdspx_alloc_bool();
    *ptr = (GdBool)val;
    return ptr;
}

EMSCRIPTEN_KEEPALIVE
void gdspx_free_bool(GdBool* b) {
    boolPool.release(b);
}


// float functions
EMSCRIPTEN_KEEPALIVE
GdFloat* gdspx_alloc_float() {
    return floatPool.acquire();
}

EMSCRIPTEN_KEEPALIVE
GdFloat* gdspx_new_float(float val) {
    GdFloat* ptr = gdspx_alloc_float();
    *ptr = (GdFloat)val;
    return ptr;
}

EMSCRIPTEN_KEEPALIVE
void gdspx_free_float(GdFloat* f) {
    floatPool.release(f);
}

// int functions
EMSCRIPTEN_KEEPALIVE
GdInt* gdspx_alloc_int() {
    return intPool.acquire();
}

EMSCRIPTEN_KEEPALIVE
GdInt* gdspx_new_int(uint32_t high,uint32_t low) {
    GdInt* ptr = gdspx_alloc_int();
    int64_t val = int64_t(high)<<32 | int64_t(low);
    *ptr = (GdInt)val;
    return ptr;
}

EMSCRIPTEN_KEEPALIVE
void gdspx_free_int(GdInt* i) {
    *i = 0;
    intPool.release(i);
}

// object functions
EMSCRIPTEN_KEEPALIVE
GdObj* gdspx_alloc_obj() {
    return objPool.acquire();
}

EMSCRIPTEN_KEEPALIVE
GdObj* gdspx_new_obj(uint32_t high,uint32_t low) {
    GdObj* ptr = gdspx_alloc_obj();
    int64_t val = int64_t(high)<<32 | int64_t(low);
    *ptr = (GdObj)val;
    return ptr;
}

EMSCRIPTEN_KEEPALIVE
void gdspx_free_obj(GdObj* obj) {
    *obj = 0;
    objPool.release(obj);
}

// vec2 functions
EMSCRIPTEN_KEEPALIVE
GdVec2* gdspx_alloc_vec2() {
    return vec2Pool.acquire();
}

EMSCRIPTEN_KEEPALIVE
GdVec2* gdspx_new_vec2(float x, float y) {
    GdVec2* ptr = gdspx_alloc_vec2();
    ptr->x = x;
    ptr->y = y;
    return ptr;
}

EMSCRIPTEN_KEEPALIVE
void gdspx_free_vec2(GdVec2* vec) {
    vec2Pool.release(vec);
}

// vec3 functions
EMSCRIPTEN_KEEPALIVE
GdVec3* gdspx_alloc_vec3() {
    return vec3Pool.acquire();
}

EMSCRIPTEN_KEEPALIVE
GdVec3* gdspx_new_vec3(float x, float y, float z) {
    GdVec3* ptr= gdspx_alloc_vec3();
    ptr->x = x;
    ptr->y = y;
    ptr->z = z;
    return ptr;
}

EMSCRIPTEN_KEEPALIVE
void gdspx_free_vec3(GdVec3* vec) {
    vec3Pool.release(vec);
}

// vec4 functions
EMSCRIPTEN_KEEPALIVE
GdVec4* gdspx_alloc_vec4() {
    return vec4Pool.acquire();
}

EMSCRIPTEN_KEEPALIVE
GdVec4* gdspx_new_vec4(float x, float y, float z, float w) {
    GdVec4* ptr = gdspx_alloc_vec4();
    ptr->x = x;
    ptr->y = y;
    ptr->z = z;
    ptr->w = w;
    return ptr;
}

EMSCRIPTEN_KEEPALIVE
void gdspx_free_vec4(GdVec4* vec) {
    vec4Pool.release(vec);
}

// color functions
EMSCRIPTEN_KEEPALIVE
GdColor* gdspx_alloc_color() {
    return colorPool.acquire();
}

EMSCRIPTEN_KEEPALIVE
GdColor* gdspx_new_color(float r, float g, float b, float a) {
    GdColor* ptr = gdspx_alloc_color();
    ptr->r = r;
    ptr->g = g;
    ptr->b = b;
    ptr->a = a;
    return ptr;
}

EMSCRIPTEN_KEEPALIVE
void gdspx_free_color(GdColor* color) {
    colorPool.release(color);
}

// rect2 functions
EMSCRIPTEN_KEEPALIVE
GdRect2* gdspx_alloc_rect2() {
    return rect2Pool.acquire();
}

EMSCRIPTEN_KEEPALIVE
GdRect2* gdspx_new_rect2(float x, float y, float width, float height) {
    GdRect2* ptr = gdspx_alloc_rect2();
    ptr->position.x = x;
    ptr->position.y = y;
    ptr->size.width = width;
    ptr->size.height = height;
    return ptr;
}

EMSCRIPTEN_KEEPALIVE
void gdspx_free_rect2(GdRect2* rect) {
    rect2Pool.release(rect);
}

// string functions
EMSCRIPTEN_KEEPALIVE
GdString* gdspx_alloc_string() {
    return stringPool.acquire();
}

EMSCRIPTEN_KEEPALIVE
GdString* gdspx_new_string(const char* str, uint32_t len) {
    GdString* ptr = gdspx_alloc_string();
    char* result = (char*)malloc(len + 1);
	strcpy(result, str);
    *ptr = result;
    return ptr;
}

EMSCRIPTEN_KEEPALIVE
const char* gdspx_get_string(GdString* ptr) {
    return (const char *)(*ptr);
}

EMSCRIPTEN_KEEPALIVE
void gdspx_free_cstr(const char* str) {
    free((void*)str);
}

EMSCRIPTEN_KEEPALIVE
int32_t gdspx_get_string_len(GdString* ptr) {
    auto length = strlen((*(const char **)ptr)) + 1;
    return length;
}

EMSCRIPTEN_KEEPALIVE
void gdspx_free_string(GdString* p_gdstr) {
    if(p_gdstr == nullptr || *p_gdstr == nullptr) {
        print_line("gdspx_free_stringptr: null pointer");
        return;
    }
    free((void*)*p_gdstr);
    *p_gdstr = nullptr;
    stringPool.release(p_gdstr);
}



// string functions
EMSCRIPTEN_KEEPALIVE
GdArray* gdspx_alloc_array() {
    return arrayPool.acquire();
}


EMSCRIPTEN_KEEPALIVE
void gdspx_free_array(GdArray* p_gdstr) {
    if (p_gdstr == nullptr || *p_gdstr == nullptr) {
        return;
    }
    
    GdArrayInfo* info = *p_gdstr;
    
    if (info->data != nullptr) {
        if (info->type == GD_ARRAY_TYPE_STRING) {
            char** strings = (char**)info->data;
            for (int64_t i = 0; i < info->size; i++) {
                if (strings[i] != nullptr) {
                    free(strings[i]);
                }
            }
        }
        free(info->data);
    }
    
    free(info);
    *p_gdstr = nullptr;
    
    arrayPool.release(p_gdstr);
}

GdArrayInfo* deserializeGdArray(uint8_t* bytes, int byteSize) {
    if (byteSize < 8) {
        return nullptr;
    }
    
    GdArrayInfo* info = (GdArrayInfo*)malloc(sizeof(GdArrayInfo));
    if (info == nullptr) {
        return nullptr;
    }
    
    // 8字节header: [size:4][type:4]
    info->size = (int32_t)readUint32LE(bytes);
    info->type = (int32_t)readUint32LE(bytes + 4);
    
    uint8_t* dataBytes = bytes + 8;
    int dataSize = byteSize - 8;
    
    switch (info->type) {
        case GD_ARRAY_TYPE_INT64:
        case GD_ARRAY_TYPE_GDOBJ: {
            int64_t* data = (int64_t*)malloc(info->size * sizeof(int64_t));
            if (isLittleEndian()) {
                memcpy(data, dataBytes, info->size * sizeof(int64_t));
            } else {
                for (int64_t i = 0; i < info->size; i++) {
                    data[i] = (int64_t)readUint64LE(dataBytes + i * 8);
                }
            }
            info->data = data;
            break;
        }
        case GD_ARRAY_TYPE_FLOAT: {
            float* data = (float*)malloc(info->size * sizeof(float));
            if (isLittleEndian()) {
                memcpy(data, dataBytes, info->size * sizeof(float));
            } else {
                for (int64_t i = 0; i < info->size; i++) {
                    uint32_t bits = readUint32LE(dataBytes + i * 4);
                    data[i] = *(float*)&bits;
                }
            }
            info->data = data;
            break;
        }
        case GD_ARRAY_TYPE_BOOL: {
            bool* data = (bool*)malloc(info->size * sizeof(bool));
            for (int64_t i = 0; i < info->size; i++) {
                data[i] = dataBytes[i] != 0;
            }
            info->data = data;
            break;
        }
        case GD_ARRAY_TYPE_BYTE: {
            uint8_t* data = (uint8_t*)malloc(info->size);
            memcpy(data, dataBytes, info->size);
            info->data = data;
            break;
        }
        case GD_ARRAY_TYPE_STRING: {
            char** strings = (char**)malloc(info->size * sizeof(char*));
            int offset = 0;
            
            for (int64_t i = 0; i < info->size; i++) {
                if (offset + 4 > dataSize) {
                    for (int64_t j = 0; j < i; j++) {
                        free(strings[j]);
                    }
                    free(strings);
                    free(info);
                    return nullptr;
                }
                
                uint32_t strLen = readUint32LE(dataBytes + offset);
                offset += 4;
                
                if (offset + strLen > dataSize) {
                    for (int64_t j = 0; j < i; j++) {
                        free(strings[j]);
                    }
                    free(strings);
                    free(info);
                    return nullptr;
                }
                
                // alloc and copy strings
                strings[i] = (char*)malloc(strLen + 1);
                memcpy(strings[i], dataBytes + offset, strLen);
                strings[i][strLen] = '\0';
                offset += strLen;
            }
            info->data = strings;
            break;
        }
        default:
            free(info);
            return nullptr;
    }
    
    return info;
}

uint8_t* serializeGdArray(GdArrayInfo* info, int* outSize) {
    if (info == nullptr) {
        print_line("serializeGdArray null");
        *outSize = 0;
        return nullptr;
    }
    
    int dataSize = 0;
    
    switch (info->type) {
        case GD_ARRAY_TYPE_INT64:
        case GD_ARRAY_TYPE_GDOBJ:
            dataSize = info->size * 8;
            break;
        case GD_ARRAY_TYPE_FLOAT:
            dataSize = info->size * 4;
            break;
        case GD_ARRAY_TYPE_BOOL:
        case GD_ARRAY_TYPE_BYTE:
            dataSize = info->size;
            break;
        case GD_ARRAY_TYPE_STRING: {
            char** strings = (char**)info->data;
            dataSize = 0;
            for (int64_t i = 0; i < info->size; i++) {
                dataSize += 4 + strlen(strings[i]);
            }
            break;
        }
        default:
            *outSize = 0;
            print_line("error: Unknown array type", info->type);
            return nullptr;
    }
    
    int totalSize = 8 + dataSize;
    uint8_t* result = (uint8_t*)malloc(totalSize);
    if (result == nullptr) {
        *outSize = 0;
        return nullptr;
    }
    
    // 8字节header: [size:4][type:4]
    writeUint32LE(result, (uint32_t)info->size);
    writeUint32LE(result + 4, (uint32_t)info->type);
    
    uint8_t* dataPtr = result + 8;
    switch (info->type) {
        case GD_ARRAY_TYPE_INT64:
        case GD_ARRAY_TYPE_GDOBJ: {
            int64_t* data = (int64_t*)info->data;
            if (isLittleEndian()) {
                memcpy(dataPtr, data, info->size * sizeof(int64_t));
            } else {
                for (int64_t i = 0; i < info->size; i++) {
                    writeUint64LE(dataPtr + i * 8, (uint64_t)data[i]);
                }
            }
            break;
        }
        case GD_ARRAY_TYPE_FLOAT: {
            float* data = (float*)info->data;
            if (isLittleEndian()) {
                memcpy(dataPtr, data, info->size * sizeof(float));
            } else {
                for (int64_t i = 0; i < info->size; i++) {
                    uint32_t bits = *(uint32_t*)&data[i];
                    writeUint32LE(dataPtr + i * 4, bits);
                }
            }
            break;
        }
        case GD_ARRAY_TYPE_BOOL: {
            bool* data = (bool*)info->data;
            for (int64_t i = 0; i < info->size; i++) {
                dataPtr[i] = data[i] ? 1 : 0;
            }
            break;
        }
        case GD_ARRAY_TYPE_BYTE: {
            memcpy(dataPtr, info->data, info->size);
            break;
        }
        case GD_ARRAY_TYPE_STRING: {
            char** strings = (char**)info->data;
            int offset = 0;
            for (int64_t i = 0; i < info->size; i++) {
                uint32_t strLen = strlen(strings[i]);
                writeUint32LE(dataPtr + offset, strLen);
                offset += 4;
                memcpy(dataPtr + offset, strings[i], strLen);
                offset += strLen;
            }
            break;
        }
    }
    
    *outSize = totalSize;
    return result;
}

EMSCRIPTEN_KEEPALIVE
uint8_t* gdspx_to_js_array(GdArray* p_gdstr, int* outSize) {
    if (p_gdstr == nullptr || *p_gdstr == nullptr) {
        return nullptr;
    }
    GdArrayInfo* info = *p_gdstr;
    return serializeGdArray(info,outSize);
}
EMSCRIPTEN_KEEPALIVE
GdArray* gdspx_to_gd_array(uint8_t* bytes, int byteSize) {
    GdArray* p_gdstr = gdspx_alloc_array();
    GdArrayInfo* info = deserializeGdArray(bytes, byteSize);
    *p_gdstr = info;
    return p_gdstr;
}

}// extern "C"
