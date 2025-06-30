#include "plutovg-private.h"
#include "plutovg-utils.h"

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "plutovg-stb-image-write.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "plutovg-stb-image.h"

static plutovg_surface_t* plutovg_surface_create_uninitialized(int width, int height)
{
    if(width > STBI_MAX_DIMENSIONS || height > STBI_MAX_DIMENSIONS)
        return NULL;
    const size_t size = width * height * 4;
    plutovg_surface_t* surface = malloc(size + sizeof(plutovg_surface_t));
    if(surface == NULL)
        return NULL;
    surface->ref_count = 1;
    surface->width = width;
    surface->height = height;
    surface->stride = width * 4;
    surface->data = (uint8_t*)(surface + 1);
    return surface;
}

plutovg_surface_t* plutovg_surface_create(int width, int height)
{
    plutovg_surface_t* surface = plutovg_surface_create_uninitialized(width, height);
    if(surface)
        memset(surface->data, 0, surface->height * surface->stride);
    return surface;
}

plutovg_surface_t* plutovg_surface_create_for_data(unsigned char* data, int width, int height, int stride)
{
    plutovg_surface_t* surface = malloc(sizeof(plutovg_surface_t));
    surface->ref_count = 1;
    surface->width = width;
    surface->height = height;
    surface->stride = stride;
    surface->data = data;
    return surface;
}

static plutovg_surface_t* plutovg_surface_load_from_image(stbi_uc* image, int width, int height)
{
    plutovg_surface_t* surface = plutovg_surface_create_uninitialized(width, height);
    if(surface)
        plutovg_convert_rgba_to_argb(surface->data, image, surface->width, surface->height, surface->stride);
    stbi_image_free(image);
    return surface;
}

plutovg_surface_t* plutovg_surface_load_from_image_file(const char* filename)
{
    int width, height, channels;
    stbi_uc* image = stbi_load(filename, &width, &height, &channels, STBI_rgb_alpha);
    if(image == NULL)
        return NULL;
    return plutovg_surface_load_from_image(image, width, height);
}

plutovg_surface_t* plutovg_surface_load_from_image_data(const void* data, int length)
{
    int width, height, channels;
    stbi_uc* image = stbi_load_from_memory(data, length, &width, &height, &channels, STBI_rgb_alpha);
    if(image == NULL)
        return NULL;
    return plutovg_surface_load_from_image(image, width, height);
}

static const uint8_t base64_table[128] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x3E, 0x00, 0x00, 0x00, 0x3F,
    0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B,
    0x3C, 0x3D, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
    0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
    0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
    0x17, 0x18, 0x19, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,
    0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30,
    0x31, 0x32, 0x33, 0x00, 0x00, 0x00, 0x00, 0x00
};

plutovg_surface_t* plutovg_surface_load_from_image_base64(const char* data, int length)
{
    plutovg_surface_t* surface = NULL;
    uint8_t* output_data = NULL;
    size_t output_length = 0;

    size_t equals_sign_count = 0;
    size_t sidx = 0;
    size_t didx = 0;

    if(length == -1)
        length = strlen(data);
    output_data = malloc(length);
    if(output_data == NULL)
        return NULL;
    for(int i = 0; i < length; ++i) {
        uint8_t cc = data[i];
        if(cc == '=') {
            ++equals_sign_count;
        } else if(cc == '+' || cc == '/' || PLUTOVG_IS_ALNUM(cc)) {
            if(equals_sign_count > 0)
                goto cleanup;
            output_data[output_length++] = base64_table[cc];
        } else if(!PLUTOVG_IS_WS(cc)) {
            goto cleanup;
        }
    }

    if(output_length == 0 || equals_sign_count > 2 || (output_length % 4) == 1)
        goto cleanup;
    output_length -= (output_length + 3) / 4;
    if(output_length == 0) {
        goto cleanup;
    }

    if(output_length > 1) {
        while(didx < output_length - 2) {
            output_data[didx + 0] = (((output_data[sidx + 0] << 2) & 255) | ((output_data[sidx + 1] >> 4) & 003));
            output_data[didx + 1] = (((output_data[sidx + 1] << 4) & 255) | ((output_data[sidx + 2] >> 2) & 017));
            output_data[didx + 2] = (((output_data[sidx + 2] << 6) & 255) | ((output_data[sidx + 3] >> 0) & 077));
            sidx += 4;
            didx += 3;
        }
    }

    if(didx < output_length)
        output_data[didx] = (((output_data[sidx + 0] << 2) & 255) | ((output_data[sidx + 1] >> 4) & 003));
    if(++didx < output_length) {
        output_data[didx] = (((output_data[sidx + 1] << 4) & 255) | ((output_data[sidx + 2] >> 2) & 017));
    }

    surface = plutovg_surface_load_from_image_data(output_data, output_length);
cleanup:
    free(output_data);
    return surface;
}

plutovg_surface_t* plutovg_surface_reference(plutovg_surface_t* surface)
{
    if(surface == NULL)
        return NULL;
    ++surface->ref_count;
    return surface;
}

void plutovg_surface_destroy(plutovg_surface_t* surface)
{
    if(surface == NULL)
        return;
    if(--surface->ref_count == 0) {
        free(surface);
    }
}

int plutovg_surface_get_reference_count(const plutovg_surface_t* surface)
{
    if(surface)
        return surface->ref_count;
    return 0;
}

unsigned char* plutovg_surface_get_data(const plutovg_surface_t* surface)
{
    return surface->data;
}

int plutovg_surface_get_width(const plutovg_surface_t* surface)
{
    return surface->width;
}

int plutovg_surface_get_height(const plutovg_surface_t* surface)
{
    return surface->height;
}

int plutovg_surface_get_stride(const plutovg_surface_t* surface)
{
    return surface->stride;
}

void plutovg_surface_clear(plutovg_surface_t* surface, const plutovg_color_t* color)
{
    uint32_t pixel = plutovg_premultiply_argb(plutovg_color_to_argb32(color));
    for(int y = 0; y < surface->height; y++) {
        uint32_t* pixels = (uint32_t*)(surface->data + surface->stride * y);
        plutovg_memfill32(pixels, surface->width, pixel);
    }
}

static void plutovg_surface_write_begin(const plutovg_surface_t* surface)
{
    plutovg_convert_argb_to_rgba(surface->data, surface->data, surface->width, surface->height, surface->stride);
}

static void plutovg_surface_write_end(const plutovg_surface_t* surface)
{
    plutovg_convert_rgba_to_argb(surface->data, surface->data, surface->width, surface->height, surface->stride);
}

bool plutovg_surface_write_to_png(const plutovg_surface_t* surface, const char* filename)
{
    plutovg_surface_write_begin(surface);
    int success = stbi_write_png(filename, surface->width, surface->height, 4, surface->data, surface->stride);
    plutovg_surface_write_end(surface);
    return success;
}

bool plutovg_surface_write_to_jpg(const plutovg_surface_t* surface, const char* filename, int quality)
{
    plutovg_surface_write_begin(surface);
    int success = stbi_write_jpg(filename, surface->width, surface->height, 4, surface->data, quality);
    plutovg_surface_write_end(surface);
    return success;
}

bool plutovg_surface_write_to_png_stream(const plutovg_surface_t* surface, plutovg_write_func_t write_func, void* closure)
{
    plutovg_surface_write_begin(surface);
    int success = stbi_write_png_to_func(write_func, closure, surface->width, surface->height, 4, surface->data, surface->stride);
    plutovg_surface_write_end(surface);
    return success;
}

bool plutovg_surface_write_to_jpg_stream(const plutovg_surface_t* surface, plutovg_write_func_t write_func, void* closure, int quality)
{
    plutovg_surface_write_begin(surface);
    int success = stbi_write_jpg_to_func(write_func, closure, surface->width, surface->height, 4, surface->data, quality);
    plutovg_surface_write_end(surface);
    return success;
}

void plutovg_convert_argb_to_rgba(unsigned char* dst, const unsigned char* src, int width, int height, int stride)
{
    for(int y = 0; y < height; y++) {
        const uint32_t* src_row = (const uint32_t*)(src + stride * y);
        uint32_t* dst_row = (uint32_t*)(dst + stride * y);
        for(int x = 0; x < width; x++) {
            uint32_t pixel = src_row[x];
            uint32_t a = (pixel >> 24) & 0xFF;
            if(a == 0) {
                dst_row[x] = 0x00000000;
            } else {
                uint32_t r = (pixel >> 16) & 0xFF;
                uint32_t g = (pixel >> 8) & 0xFF;
                uint32_t b = (pixel >> 0) & 0xFF;
                if(a != 255) {
                    r = (r * 255) / a;
                    g = (g * 255) / a;
                    b = (b * 255) / a;
                }

                dst_row[x] = (a << 24) | (b << 16) | (g << 8) | r;
            }
        }
    }
}

void plutovg_convert_rgba_to_argb(unsigned char* dst, const unsigned char* src, int width, int height, int stride)
{
    for(int y = 0; y < height; y++) {
        const uint32_t* src_row = (const uint32_t*)(src + stride * y);
        uint32_t* dst_row = (uint32_t*)(dst + stride * y);
        for(int x = 0; x < width; x++) {
            uint32_t pixel = src_row[x];
            uint32_t a = (pixel >> 24) & 0xFF;
            if(a == 0) {
                dst_row[x] = 0x00000000;
            } else {
                uint32_t b = (pixel >> 16) & 0xFF;
                uint32_t g = (pixel >> 8) & 0xFF;
                uint32_t r = (pixel >> 0) & 0xFF;
                if(a != 255) {
                    r = (r * a) / 255;
                    g = (g * a) / 255;
                    b = (b * a) / 255;
                }

                dst_row[x] = (a << 24) | (r << 16) | (g << 8) | b;
            }
        }
    }
}
