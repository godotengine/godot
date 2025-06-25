#include "plutovg.h"
#include "plutovg-utils.h"

#include <stdio.h>
#include <assert.h>

#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#include "plutovg-stb-truetype.h"

static int plutovg_text_iterator_length(const void* data, int length, plutovg_text_encoding_t encoding)
{
    if(length != -1)
        return length;
    length = 0;
    switch(encoding) {
    case PLUTOVG_TEXT_ENCODING_UTF8:
    case PLUTOVG_TEXT_ENCODING_LATIN1: {
        const uint8_t* text = data;
        while(*text++)
            length++;
        break;
    } case PLUTOVG_TEXT_ENCODING_UTF16: {
        const uint16_t* text = data;
        while(*text++)
            length++;
        break;
    } case PLUTOVG_TEXT_ENCODING_UTF32: {
        const uint32_t* text = data;
        while(*text++)
            length++;
        break;
    } default:
        assert(false);
    }

    return length;
}

void plutovg_text_iterator_init(plutovg_text_iterator_t* it, const void* text, int length, plutovg_text_encoding_t encoding)
{
    it->text = text;
    it->length = plutovg_text_iterator_length(text, length, encoding);
    it->encoding = encoding;
    it->index = 0;
}

bool plutovg_text_iterator_has_next(const plutovg_text_iterator_t* it)
{
    return it->index < it->length;
}

plutovg_codepoint_t plutovg_text_iterator_next(plutovg_text_iterator_t* it)
{
    plutovg_codepoint_t codepoint = 0;
    switch(it->encoding) {
    case PLUTOVG_TEXT_ENCODING_UTF8: {
        static const uint8_t trailing[256] = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5
        };

        static const uint32_t offsets[6] = {
            0x00000000, 0x00003080, 0x000E2080, 0x03C82080, 0xFA082080, 0x82082080
        };

        const uint8_t* text = it->text;
        int trailing_bytes = trailing[text[it->index]];
        if(it->index + trailing_bytes >= it->length)
            trailing_bytes = 0;
        switch(trailing_bytes) {
        case 5: codepoint += text[it->index++]; codepoint <<= 6;
        case 4: codepoint += text[it->index++]; codepoint <<= 6;
        case 3: codepoint += text[it->index++]; codepoint <<= 6;
        case 2: codepoint += text[it->index++]; codepoint <<= 6;
        case 1: codepoint += text[it->index++]; codepoint <<= 6;
        case 0: codepoint += text[it->index++];
        }

        codepoint -= offsets[trailing_bytes];
        break;
    } case PLUTOVG_TEXT_ENCODING_UTF16: {
        const uint16_t* text = it->text;
        codepoint = text[it->index++];
        if(((codepoint) & 0xfffffc00) == 0xd800) {
            if(it->index < it->length && (((codepoint) & 0xfffffc00) == 0xdc00)) {
                uint16_t trail = text[it->index++];
                codepoint = (codepoint << 10) + trail - ((0xD800u << 10) - 0x10000u + 0xDC00u);
            }
        }

        break;
    } case PLUTOVG_TEXT_ENCODING_UTF32: {
        const uint32_t* text = it->text;
        codepoint = text[it->index++];
        break;
    } case PLUTOVG_TEXT_ENCODING_LATIN1: {
        const uint8_t* text = it->text;
        codepoint = text[it->index++];
        break;
    } default:
        assert(false);
    }

    return codepoint;
}

typedef struct {
    stbtt_vertex* vertices;
    int nvertices;
    int index;
    int advance_width;
    int left_side_bearing;
    int x1;
    int y1;
    int x2;
    int y2;
} glyph_t;

#define GLYPH_CACHE_SIZE 256
struct plutovg_font_face {
    int ref_count;
    int ascent;
    int descent;
    int line_gap;
    int x1;
    int y1;
    int x2;
    int y2;
    stbtt_fontinfo info;
    glyph_t** glyphs[GLYPH_CACHE_SIZE];
    plutovg_destroy_func_t destroy_func;
    void* closure;
};

plutovg_font_face_t* plutovg_font_face_load_from_file(const char* filename, int ttcindex)
{
    FILE* fp = fopen(filename, "rb");
    if(fp == NULL) {
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    long length = ftell(fp);
    if(length == -1L) {
        fclose(fp);
        return NULL;
    }

    void* data = malloc(length);
    if(data == NULL) {
        fclose(fp);
        return NULL;
    }

    fseek(fp, 0, SEEK_SET);
    size_t nread = fread(data, 1, length, fp);
    fclose(fp);

    if(nread != length) {
        free(data);
        return NULL;
    }

    return plutovg_font_face_load_from_data(data, length, ttcindex, free, data);
}

plutovg_font_face_t* plutovg_font_face_load_from_data(const void* data, unsigned int length, int ttcindex, plutovg_destroy_func_t destroy_func, void* closure)
{
    stbtt_fontinfo info;
    int offset = stbtt_GetFontOffsetForIndex(data, ttcindex);
    if(offset == -1 || !stbtt_InitFont(&info, data, offset)) {
        if(destroy_func)
            destroy_func(closure);
        return NULL;
    }

    plutovg_font_face_t* face = malloc(sizeof(plutovg_font_face_t));
    face->ref_count = 1;
    face->info = info;
    stbtt_GetFontVMetrics(&face->info, &face->ascent, &face->descent, &face->line_gap);
    stbtt_GetFontBoundingBox(&face->info, &face->x1, &face->y1, &face->x2, &face->y2);
    memset(face->glyphs, 0, sizeof(face->glyphs));
    face->destroy_func = destroy_func;
    face->closure = closure;
    return face;
}

plutovg_font_face_t* plutovg_font_face_reference(plutovg_font_face_t* face)
{
    if(face == NULL)
        return NULL;
    ++face->ref_count;
    return face;
}

void plutovg_font_face_destroy(plutovg_font_face_t* face)
{
    if(face == NULL)
        return;
    if(--face->ref_count == 0) {
        for(int i = 0; i < GLYPH_CACHE_SIZE; i++) {
            if(face->glyphs[i] == NULL)
                continue;
            for(int j = 0; j < GLYPH_CACHE_SIZE; j++) {
                glyph_t* glyph = face->glyphs[i][j];
                if(glyph == NULL)
                    continue;
                stbtt_FreeShape(&face->info, glyph->vertices);
                free(glyph);
            }

            free(face->glyphs[i]);
        }

        if(face->destroy_func)
            face->destroy_func(face->closure);
        free(face);
    }
}

int plutovg_font_face_get_reference_count(const plutovg_font_face_t* face)
{
    if(face)
        return face->ref_count;
    return 0;
}

static float plutovg_font_face_get_scale(const plutovg_font_face_t* face, float size)
{
    return stbtt_ScaleForMappingEmToPixels(&face->info, size);
}

void plutovg_font_face_get_metrics(const plutovg_font_face_t* face, float size, float* ascent, float* descent, float* line_gap, plutovg_rect_t* extents)
{
    float scale = plutovg_font_face_get_scale(face, size);
    if(ascent) *ascent = face->ascent * scale;
    if(descent) *descent = face->descent * scale;
    if(line_gap) *line_gap = face->line_gap * scale;
    if(extents) {
        extents->x = face->x1 * scale;
        extents->y = face->y2 * -scale;
        extents->w = (face->x2 - face->x1) * scale;
        extents->h = (face->y1 - face->y2) * -scale;
    }
}

static glyph_t* plutovg_font_face_get_glyph(plutovg_font_face_t* face, plutovg_codepoint_t codepoint)
{
    unsigned int msb = (codepoint >> 8) & 0xFF;
    if(face->glyphs[msb] == NULL) {
        face->glyphs[msb] = calloc(GLYPH_CACHE_SIZE, sizeof(glyph_t*));
    }

    unsigned int lsb = codepoint & 0xFF;
    if(face->glyphs[msb][lsb] == NULL) {
        glyph_t* glyph = malloc(sizeof(glyph_t));
        glyph->index = stbtt_FindGlyphIndex(&face->info, codepoint);
        glyph->nvertices = stbtt_GetGlyphShape(&face->info, glyph->index, &glyph->vertices);
        stbtt_GetGlyphHMetrics(&face->info, glyph->index, &glyph->advance_width, &glyph->left_side_bearing);
        if(!stbtt_GetGlyphBox(&face->info, glyph->index, &glyph->x1, &glyph->y1, &glyph->x2, &glyph->y2))
            glyph->x1 = glyph->y1 = glyph->x2 = glyph->y2 = 0;
        face->glyphs[msb][lsb] = glyph;
    }

    return face->glyphs[msb][lsb];
}

void plutovg_font_face_get_glyph_metrics(plutovg_font_face_t* face, float size, plutovg_codepoint_t codepoint, float* advance_width, float* left_side_bearing, plutovg_rect_t* extents)
{
    float scale = plutovg_font_face_get_scale(face, size);
    glyph_t* glyph = plutovg_font_face_get_glyph(face, codepoint);
    if(advance_width) *advance_width = glyph->advance_width * scale;
    if(left_side_bearing) *left_side_bearing = glyph->left_side_bearing * scale;
    if(extents) {
        extents->x = glyph->x1 * scale;
        extents->y = glyph->y2 * -scale;
        extents->w = (glyph->x2 - glyph->x1) * scale;
        extents->h = (glyph->y1 - glyph->y2) * -scale;
    }
}

static void glyph_traverse_func(void* closure, plutovg_path_command_t command, const plutovg_point_t* points, int npoints)
{
    plutovg_path_t* path = (plutovg_path_t*)(closure);
    switch(command) {
    case PLUTOVG_PATH_COMMAND_MOVE_TO:
        plutovg_path_move_to(path, points[0].x, points[0].y);
        break;
    case PLUTOVG_PATH_COMMAND_LINE_TO:
        plutovg_path_line_to(path, points[0].x, points[0].y);
        break;
    case PLUTOVG_PATH_COMMAND_CUBIC_TO:
        plutovg_path_cubic_to(path, points[0].x, points[0].y, points[1].x, points[1].y, points[2].x, points[2].y);
        break;
    case PLUTOVG_PATH_COMMAND_CLOSE:
        assert(false);
    }
}

float plutovg_font_face_get_glyph_path(plutovg_font_face_t* face, float size, float x, float y, plutovg_codepoint_t codepoint, plutovg_path_t* path)
{
    return plutovg_font_face_traverse_glyph_path(face, size, x, y, codepoint, glyph_traverse_func, path);
}

float plutovg_font_face_traverse_glyph_path(plutovg_font_face_t* face, float size, float x, float y, plutovg_codepoint_t codepoint, plutovg_path_traverse_func_t traverse_func, void* closure)
{
    float scale = plutovg_font_face_get_scale(face, size);
    plutovg_matrix_t matrix;
    plutovg_matrix_init_translate(&matrix, x, y);
    plutovg_matrix_scale(&matrix, scale, -scale);

    plutovg_point_t points[3];
    plutovg_point_t current_point = {0, 0};
    glyph_t* glyph = plutovg_font_face_get_glyph(face, codepoint);
    for(int i = 0; i < glyph->nvertices; i++) {
        switch(glyph->vertices[i].type) {
        case STBTT_vmove:
            points[0].x = glyph->vertices[i].x;
            points[0].y = glyph->vertices[i].y;
            current_point = points[0];
            plutovg_matrix_map_points(&matrix, points, points, 1);
            traverse_func(closure, PLUTOVG_PATH_COMMAND_MOVE_TO, points, 1);
            break;
        case STBTT_vline:
            points[0].x = glyph->vertices[i].x;
            points[0].y = glyph->vertices[i].y;
            current_point = points[0];
            plutovg_matrix_map_points(&matrix, points, points, 1);
            traverse_func(closure, PLUTOVG_PATH_COMMAND_LINE_TO, points, 1);
            break;
        case STBTT_vcurve:
            points[0].x = 2.f / 3.f * glyph->vertices[i].cx + 1.f / 3.f * current_point.x;
            points[0].y = 2.f / 3.f * glyph->vertices[i].cy + 1.f / 3.f * current_point.y;
            points[1].x = 2.f / 3.f * glyph->vertices[i].cx + 1.f / 3.f * glyph->vertices[i].x;
            points[1].y = 2.f / 3.f * glyph->vertices[i].cy + 1.f / 3.f * glyph->vertices[i].y;
            points[2].x = glyph->vertices[i].x;
            points[2].y = glyph->vertices[i].y;
            current_point = points[2];
            plutovg_matrix_map_points(&matrix, points, points, 3);
            traverse_func(closure, PLUTOVG_PATH_COMMAND_CUBIC_TO, points, 3);
            break;
        case STBTT_vcubic:
            points[0].x = glyph->vertices[i].cx;
            points[0].y = glyph->vertices[i].cy;
            points[1].x = glyph->vertices[i].cx1;
            points[1].y = glyph->vertices[i].cy1;
            points[2].x = glyph->vertices[i].x;
            points[2].y = glyph->vertices[i].y;
            current_point = points[2];
            plutovg_matrix_map_points(&matrix, points, points, 3);
            traverse_func(closure, PLUTOVG_PATH_COMMAND_CUBIC_TO, points, 3);
            break;
        default:
            assert(false);
        }
    }

    return glyph->advance_width * scale;
}

float plutovg_font_face_text_extents(plutovg_font_face_t* face, float size, const void* text, int length, plutovg_text_encoding_t encoding, plutovg_rect_t* extents)
{
    plutovg_text_iterator_t it;
    plutovg_text_iterator_init(&it, text, length, encoding);
    plutovg_rect_t* text_extents = NULL;
    float total_advance_width = 0.f;
    while(plutovg_text_iterator_has_next(&it)) {
        plutovg_codepoint_t codepoint = plutovg_text_iterator_next(&it);

        float advance_width;
        if(extents == NULL) {
            plutovg_font_face_get_glyph_metrics(face, size, codepoint, &advance_width, NULL, NULL);
            total_advance_width += advance_width;
            continue;
        }

        plutovg_rect_t glyph_extents;
        plutovg_font_face_get_glyph_metrics(face, size, codepoint, &advance_width, NULL, &glyph_extents);

        glyph_extents.x += total_advance_width;
        total_advance_width += advance_width;
        if(text_extents == NULL) {
            text_extents = extents;
            *text_extents = glyph_extents;
            continue;
        }

        float x1 = plutovg_min(text_extents->x, glyph_extents.x);
        float y1 = plutovg_min(text_extents->y, glyph_extents.y);
        float x2 = plutovg_max(text_extents->x + text_extents->w, glyph_extents.x + glyph_extents.w);
        float y2 = plutovg_max(text_extents->y + text_extents->h, glyph_extents.y + glyph_extents.h);

        text_extents->x = x1;
        text_extents->y = y1;
        text_extents->w = x2 - x1;
        text_extents->h = y2 - y1;
    }

    if(extents && !text_extents) {
        extents->x = 0;
        extents->y = 0;
        extents->w = 0;
        extents->h = 0;
    }

    return total_advance_width;
}
