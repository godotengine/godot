#include "plutovg-private.h"
#include "plutovg-utils.h"

#include <ctype.h>

void plutovg_color_init_rgb(plutovg_color_t* color, float r, float g, float b)
{
    plutovg_color_init_rgba(color, r, g, b, 1.f);
}

void plutovg_color_init_rgba(plutovg_color_t* color, float r, float g, float b, float a)
{
    color->r = plutovg_clamp(r, 0.f, 1.f);
    color->g = plutovg_clamp(g, 0.f, 1.f);
    color->b = plutovg_clamp(b, 0.f, 1.f);
    color->a = plutovg_clamp(a, 0.f, 1.f);
}

void plutovg_color_init_rgb8(plutovg_color_t* color, int r, int g, int b)
{
    plutovg_color_init_rgba8(color, r, g, b, 255);
}

void plutovg_color_init_rgba8(plutovg_color_t* color, int r, int g, int b, int a)
{
    plutovg_color_init_rgba(color, r / 255.f, g / 255.f, b / 255.f, a / 255.f);
}

void plutovg_color_init_rgba32(plutovg_color_t* color, unsigned int value)
{
    uint8_t r = (value >> 24) & 0xFF;
    uint8_t g = (value >> 16) & 0xFF;
    uint8_t b = (value >>  8) & 0xFF;
    uint8_t a = (value >>  0) & 0xFF;
    plutovg_color_init_rgba8(color, r, g, b, a);
}

void plutovg_color_init_argb32(plutovg_color_t* color, unsigned int value)
{
    uint8_t a = (value >> 24) & 0xFF;
    uint8_t r = (value >> 16) & 0xFF;
    uint8_t g = (value >>  8) & 0xFF;
    uint8_t b = (value >>  0) & 0xFF;
    plutovg_color_init_rgba8(color, r, g, b, a);
}

unsigned int plutovg_color_to_rgba32(const plutovg_color_t* color)
{
    uint32_t r = lroundf(color->r * 255);
    uint32_t g = lroundf(color->g * 255);
    uint32_t b = lroundf(color->b * 255);
    uint32_t a = lroundf(color->a * 255);
    return (r << 24) | (g << 16) | (b << 8) | (a);
}

unsigned int plutovg_color_to_argb32(const plutovg_color_t* color)
{
    uint32_t a = lroundf(color->a * 255);
    uint32_t r = lroundf(color->r * 255);
    uint32_t g = lroundf(color->g * 255);
    uint32_t b = lroundf(color->b * 255);
    return (a << 24) | (r << 16) | (g << 8) | (b);
}

static inline uint8_t hex_digit(uint8_t c)
{
    if(c >= '0' && c <= '9')
        return c - '0';
    if(c >= 'a' && c <= 'f')
        return 10 + c - 'a';
    if(c >= 'A' && c <= 'F')
        return 10 + c - 'A';
    return 0;
}

static inline uint8_t hex_expand(uint8_t c)
{
    uint8_t h = hex_digit(c);
    return (h << 4) | h;
}

static inline uint8_t hex_combine(uint8_t c1, uint8_t c2)
{
    uint8_t h1 = hex_digit(c1);
    uint8_t h2 = hex_digit(c2);
    return (h1 << 4) | h2;
}

#define MAX_NAME 24
typedef struct {
    char name[MAX_NAME];
    uint32_t value;
} color_entry_t;

static int color_entry_compare(const void* a, const void* b)
{
    const char* name = a;
    const color_entry_t* entry = b;
    return strcmp(name, entry->name);
}

static bool parse_rgb_component(const char** begin, const char* end, int* component)
{
    float value = 0;
    if(!plutovg_parse_number(begin, end, &value))
        return false;
    if(plutovg_skip_delim(begin, end, '%'))
        value *= 2.55f;
    value = plutovg_clamp(value, 0.f, 255.f);
    *component = lroundf(value);
    return true;
}

static bool parse_alpha_component(const char** begin, const char* end, int* component)
{
    float value = 0;
    if(!plutovg_parse_number(begin, end, &value))
        return false;
    if(plutovg_skip_delim(begin, end, '%'))
        value /= 100.f;
    value = plutovg_clamp(value, 0.f, 1.f);
    *component = lroundf(value * 255.f);
    return true;
}

int plutovg_color_parse(plutovg_color_t* color, const char* data, int length)
{
    if(length == -1)
        length = strlen(data);
    const char* it = data;
    const char* end = it + length;
    plutovg_skip_ws(&it, end);
    if(plutovg_skip_delim(&it, end, '#')) {
        int r, g, b, a = 255;
        const char* begin = it;
        while(it < end && isxdigit(*it))
            ++it;
        int count = it - begin;
        if(count == 3 || count == 4) {
            r = hex_expand(begin[0]);
            g = hex_expand(begin[1]);
            b = hex_expand(begin[2]);
            if(count == 4) {
                a = hex_expand(begin[3]);
            }
        } else if(count == 6 || count == 8) {
            r = hex_combine(begin[0], begin[1]);
            g = hex_combine(begin[2], begin[3]);
            b = hex_combine(begin[4], begin[5]);
            if(count == 8) {
                a = hex_combine(begin[6], begin[7]);
            }
        } else {
            return 0;
        }

        plutovg_color_init_rgba8(color, r, g, b, a);
    } else {
        int name_length = 0;
        char name[MAX_NAME + 1];
        while(it < end && name_length < MAX_NAME && isalpha(*it))
            name[name_length++] = tolower(*it++);
        name[name_length] = '\0';

        if(strcmp(name, "transparent") == 0) {
            plutovg_color_init_rgba(color, 0, 0, 0, 0);
        } else if(strcmp(name, "rgb") == 0 || strcmp(name, "rgba") == 0) {
            if(!plutovg_skip_ws_and_delim(&it, end, '('))
                return 0;
            int r, g, b, a = 255;
            if(!parse_rgb_component(&it, end, &r)
                || !plutovg_skip_ws_and_comma(&it, end)
                || !parse_rgb_component(&it, end, &g)
                || !plutovg_skip_ws_and_comma(&it, end)
                || !parse_rgb_component(&it, end, &b)) {
                return 0;
            }

            if(plutovg_skip_ws_and_comma(&it, end)
                && !parse_alpha_component(&it, end, &a)) {
                return 0;
            }

            plutovg_skip_ws(&it, end);
            if(!plutovg_skip_delim(&it, end, ')'))
                return 0;
            plutovg_color_init_rgba8(color, r, g, b, a);
        } else {
            static const color_entry_t colormap[] = {
                {"aliceblue", 0xF0F8FF},
                {"antiquewhite", 0xFAEBD7},
                {"aqua", 0x00FFFF},
                {"aquamarine", 0x7FFFD4},
                {"azure", 0xF0FFFF},
                {"beige", 0xF5F5DC},
                {"bisque", 0xFFE4C4},
                {"black", 0x000000},
                {"blanchedalmond", 0xFFEBCD},
                {"blue", 0x0000FF},
                {"blueviolet", 0x8A2BE2},
                {"brown", 0xA52A2A},
                {"burlywood", 0xDEB887},
                {"cadetblue", 0x5F9EA0},
                {"chartreuse", 0x7FFF00},
                {"chocolate", 0xD2691E},
                {"coral", 0xFF7F50},
                {"cornflowerblue", 0x6495ED},
                {"cornsilk", 0xFFF8DC},
                {"crimson", 0xDC143C},
                {"cyan", 0x00FFFF},
                {"darkblue", 0x00008B},
                {"darkcyan", 0x008B8B},
                {"darkgoldenrod", 0xB8860B},
                {"darkgray", 0xA9A9A9},
                {"darkgreen", 0x006400},
                {"darkgrey", 0xA9A9A9},
                {"darkkhaki", 0xBDB76B},
                {"darkmagenta", 0x8B008B},
                {"darkolivegreen", 0x556B2F},
                {"darkorange", 0xFF8C00},
                {"darkorchid", 0x9932CC},
                {"darkred", 0x8B0000},
                {"darksalmon", 0xE9967A},
                {"darkseagreen", 0x8FBC8F},
                {"darkslateblue", 0x483D8B},
                {"darkslategray", 0x2F4F4F},
                {"darkslategrey", 0x2F4F4F},
                {"darkturquoise", 0x00CED1},
                {"darkviolet", 0x9400D3},
                {"deeppink", 0xFF1493},
                {"deepskyblue", 0x00BFFF},
                {"dimgray", 0x696969},
                {"dimgrey", 0x696969},
                {"dodgerblue", 0x1E90FF},
                {"firebrick", 0xB22222},
                {"floralwhite", 0xFFFAF0},
                {"forestgreen", 0x228B22},
                {"fuchsia", 0xFF00FF},
                {"gainsboro", 0xDCDCDC},
                {"ghostwhite", 0xF8F8FF},
                {"gold", 0xFFD700},
                {"goldenrod", 0xDAA520},
                {"gray", 0x808080},
                {"green", 0x008000},
                {"greenyellow", 0xADFF2F},
                {"grey", 0x808080},
                {"honeydew", 0xF0FFF0},
                {"hotpink", 0xFF69B4},
                {"indianred", 0xCD5C5C},
                {"indigo", 0x4B0082},
                {"ivory", 0xFFFFF0},
                {"khaki", 0xF0E68C},
                {"lavender", 0xE6E6FA},
                {"lavenderblush", 0xFFF0F5},
                {"lawngreen", 0x7CFC00},
                {"lemonchiffon", 0xFFFACD},
                {"lightblue", 0xADD8E6},
                {"lightcoral", 0xF08080},
                {"lightcyan", 0xE0FFFF},
                {"lightgoldenrodyellow", 0xFAFAD2},
                {"lightgray", 0xD3D3D3},
                {"lightgreen", 0x90EE90},
                {"lightgrey", 0xD3D3D3},
                {"lightpink", 0xFFB6C1},
                {"lightsalmon", 0xFFA07A},
                {"lightseagreen", 0x20B2AA},
                {"lightskyblue", 0x87CEFA},
                {"lightslategray", 0x778899},
                {"lightslategrey", 0x778899},
                {"lightsteelblue", 0xB0C4DE},
                {"lightyellow", 0xFFFFE0},
                {"lime", 0x00FF00},
                {"limegreen", 0x32CD32},
                {"linen", 0xFAF0E6},
                {"magenta", 0xFF00FF},
                {"maroon", 0x800000},
                {"mediumaquamarine", 0x66CDAA},
                {"mediumblue", 0x0000CD},
                {"mediumorchid", 0xBA55D3},
                {"mediumpurple", 0x9370DB},
                {"mediumseagreen", 0x3CB371},
                {"mediumslateblue", 0x7B68EE},
                {"mediumspringgreen", 0x00FA9A},
                {"mediumturquoise", 0x48D1CC},
                {"mediumvioletred", 0xC71585},
                {"midnightblue", 0x191970},
                {"mintcream", 0xF5FFFA},
                {"mistyrose", 0xFFE4E1},
                {"moccasin", 0xFFE4B5},
                {"navajowhite", 0xFFDEAD},
                {"navy", 0x000080},
                {"oldlace", 0xFDF5E6},
                {"olive", 0x808000},
                {"olivedrab", 0x6B8E23},
                {"orange", 0xFFA500},
                {"orangered", 0xFF4500},
                {"orchid", 0xDA70D6},
                {"palegoldenrod", 0xEEE8AA},
                {"palegreen", 0x98FB98},
                {"paleturquoise", 0xAFEEEE},
                {"palevioletred", 0xDB7093},
                {"papayawhip", 0xFFEFD5},
                {"peachpuff", 0xFFDAB9},
                {"peru", 0xCD853F},
                {"pink", 0xFFC0CB},
                {"plum", 0xDDA0DD},
                {"powderblue", 0xB0E0E6},
                {"purple", 0x800080},
                {"rebeccapurple", 0x663399},
                {"red", 0xFF0000},
                {"rosybrown", 0xBC8F8F},
                {"royalblue", 0x4169E1},
                {"saddlebrown", 0x8B4513},
                {"salmon", 0xFA8072},
                {"sandybrown", 0xF4A460},
                {"seagreen", 0x2E8B57},
                {"seashell", 0xFFF5EE},
                {"sienna", 0xA0522D},
                {"silver", 0xC0C0C0},
                {"skyblue", 0x87CEEB},
                {"slateblue", 0x6A5ACD},
                {"slategray", 0x708090},
                {"slategrey", 0x708090},
                {"snow", 0xFFFAFA},
                {"springgreen", 0x00FF7F},
                {"steelblue", 0x4682B4},
                {"tan", 0xD2B48C},
                {"teal", 0x008080},
                {"thistle", 0xD8BFD8},
                {"tomato", 0xFF6347},
                {"turquoise", 0x40E0D0},
                {"violet", 0xEE82EE},
                {"wheat", 0xF5DEB3},
                {"white", 0xFFFFFF},
                {"whitesmoke", 0xF5F5F5},
                {"yellow", 0xFFFF00},
                {"yellowgreen", 0x9ACD32}
            };

            const color_entry_t* entry = bsearch(name, colormap, sizeof(colormap) / sizeof(color_entry_t), sizeof(color_entry_t), color_entry_compare);
            if(entry == NULL)
                return 0;
            plutovg_color_init_argb32(color, 0xFF000000 | entry->value);
        }
    }

    plutovg_skip_ws(&it, end);
    return it - data;
}

static void* plutovg_paint_create(plutovg_paint_type_t type, size_t size)
{
    plutovg_paint_t* paint = malloc(size);
    paint->ref_count = 1;
    paint->type = type;
    return paint;
}

plutovg_paint_t* plutovg_paint_create_rgb(float r, float g, float b)
{
    return plutovg_paint_create_rgba(r, g, b, 1.f);
}

plutovg_paint_t* plutovg_paint_create_rgba(float r, float g, float b, float a)
{
    plutovg_solid_paint_t* solid = plutovg_paint_create(PLUTOVG_PAINT_TYPE_COLOR, sizeof(plutovg_solid_paint_t));
    solid->color.r = plutovg_clamp(r, 0.f, 1.f);
    solid->color.g = plutovg_clamp(g, 0.f, 1.f);
    solid->color.b = plutovg_clamp(b, 0.f, 1.f);
    solid->color.a = plutovg_clamp(a, 0.f, 1.f);
    return &solid->base;
}

plutovg_paint_t* plutovg_paint_create_color(const plutovg_color_t* color)
{
    return plutovg_paint_create_rgba(color->r, color->g, color->b, color->a);
}

static plutovg_gradient_paint_t* plutovg_gradient_create(plutovg_gradient_type_t type, plutovg_spread_method_t spread, const plutovg_gradient_stop_t* stops, int nstops, const plutovg_matrix_t* matrix)
{
    plutovg_gradient_paint_t* gradient = plutovg_paint_create(PLUTOVG_PAINT_TYPE_GRADIENT, sizeof(plutovg_gradient_paint_t) + nstops * sizeof(plutovg_gradient_stop_t));
    gradient->type = type;
    gradient->spread = spread;
    gradient->matrix = matrix ? *matrix : PLUTOVG_IDENTITY_MATRIX;
    gradient->stops = (plutovg_gradient_stop_t*)(gradient + 1);
    gradient->nstops = nstops;

    float prev_offset = 0.f;
    for(int i = 0; i < nstops; ++i) {
        const plutovg_gradient_stop_t* stop = stops + i;
        gradient->stops[i].offset = plutovg_max(prev_offset, plutovg_clamp(stop->offset, 0.f, 1.f));
        gradient->stops[i].color.r = plutovg_clamp(stop->color.r, 0.f, 1.f);
        gradient->stops[i].color.g = plutovg_clamp(stop->color.g, 0.f, 1.f);
        gradient->stops[i].color.b = plutovg_clamp(stop->color.b, 0.f, 1.f);
        gradient->stops[i].color.a = plutovg_clamp(stop->color.a, 0.f, 1.f);
        prev_offset = gradient->stops[i].offset;
    }

    return gradient;
}

plutovg_paint_t* plutovg_paint_create_linear_gradient(float x1, float y1, float x2, float y2, plutovg_spread_method_t spread, const plutovg_gradient_stop_t* stops, int nstops, const plutovg_matrix_t* matrix)
{
    plutovg_gradient_paint_t* gradient = plutovg_gradient_create(PLUTOVG_GRADIENT_TYPE_LINEAR, spread, stops, nstops, matrix);
    gradient->values[0] = x1;
    gradient->values[1] = y1;
    gradient->values[2] = x2;
    gradient->values[3] = y2;
    return &gradient->base;
}

plutovg_paint_t* plutovg_paint_create_radial_gradient(float cx, float cy, float cr, float fx, float fy, float fr, plutovg_spread_method_t spread, const plutovg_gradient_stop_t* stops, int nstops, const plutovg_matrix_t* matrix)
{
    plutovg_gradient_paint_t* gradient = plutovg_gradient_create(PLUTOVG_GRADIENT_TYPE_RADIAL, spread, stops, nstops, matrix);
    gradient->values[0] = cx;
    gradient->values[1] = cy;
    gradient->values[2] = cr;
    gradient->values[3] = fx;
    gradient->values[4] = fy;
    gradient->values[5] = fr;
    return &gradient->base;
}

plutovg_paint_t* plutovg_paint_create_texture(plutovg_surface_t* surface, plutovg_texture_type_t type, float opacity, const plutovg_matrix_t* matrix)
{
    plutovg_texture_paint_t* texture = plutovg_paint_create(PLUTOVG_PAINT_TYPE_TEXTURE, sizeof(plutovg_texture_paint_t));
    texture->type = type;
    texture->opacity = plutovg_clamp(opacity, 0.f, 1.f);
    texture->matrix = matrix ? *matrix : PLUTOVG_IDENTITY_MATRIX;
    texture->surface = plutovg_surface_reference(surface);
    return &texture->base;
}

plutovg_paint_t* plutovg_paint_reference(plutovg_paint_t* paint)
{
    if(paint == NULL)
        return NULL;
    ++paint->ref_count;
    return paint;
}

void plutovg_paint_destroy(plutovg_paint_t* paint)
{
    if(paint == NULL)
        return;
    if(--paint->ref_count == 0) {
        if(paint->type == PLUTOVG_PAINT_TYPE_TEXTURE) {
            plutovg_texture_paint_t* texture = (plutovg_texture_paint_t*)(paint);
            plutovg_surface_destroy(texture->surface);
        }

        free(paint);
    }
}

int plutovg_paint_get_reference_count(const plutovg_paint_t* paint)
{
    if(paint)
        return paint->ref_count;
    return 0;
}
