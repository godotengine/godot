#include "plutovg-private.h"

#include <limits.h>
#include <math.h>

#define COLOR_TABLE_SIZE 1024
typedef struct {
    plutovg_spread_method_t spread;
    plutovg_matrix_t matrix;
    uint32_t colortable[COLOR_TABLE_SIZE];
    union {
        struct {
            double x1, y1;
            double x2, y2;
        } linear;
        struct {
            double cx, cy, cr;
            double fx, fy, fr;
        } radial;
    };
} gradient_data_t;

typedef struct {
    plutovg_matrix_t matrix;
    uint8_t* data;
    int width;
    int height;
    int stride;
    int const_alpha;
} texture_data_t;

typedef struct {
    double dx;
    double dy;
    double l;
    double off;
} linear_gradient_values_t;

typedef struct {
    double dx;
    double dy;
    double dr;
    double sqrfr;
    double a;
    double inv2a;
    int extended;
} radial_gradient_values_t;

static inline uint32_t premultiply_color(const plutovg_color_t* color, double opacity)
{
    uint32_t alpha = (uint8_t)(color->a * opacity * 255);
    uint32_t pr = (uint8_t)(color->r * alpha);
    uint32_t pg = (uint8_t)(color->g * alpha);
    uint32_t pb = (uint8_t)(color->b * alpha);

    return (alpha << 24) | (pr << 16) | (pg << 8) | (pb);
}

static inline uint32_t interpolate_pixel(uint32_t x, uint32_t a, uint32_t y, uint32_t b)
{
    uint32_t t = (x & 0xff00ff) * a + (y & 0xff00ff) * b;
    t >>= 8;
    t &= 0xff00ff;
    x = ((x >> 8) & 0xff00ff) * a + ((y >> 8) & 0xff00ff) * b;
    x &= 0xff00ff00;
    x |= t;
    return x;
}

static inline uint32_t BYTE_MUL(uint32_t x, uint32_t a)
{
    uint32_t t = (x & 0xff00ff) * a;
    t = (t + ((t >> 8) & 0xff00ff) + 0x800080) >> 8;
    t &= 0xff00ff;
    x = ((x >> 8) & 0xff00ff) * a;
    x = (x + ((x >> 8) & 0xff00ff) + 0x800080);
    x &= 0xff00ff00;
    x |= t;
    return x;
}

static inline void memfill32(uint32_t* dest, uint32_t value, int length)
{
    for(int i = 0 ;i < length;i++)
        *dest++ = value;
}

static inline int gradient_clamp(const gradient_data_t* gradient, int ipos)
{
    if(gradient->spread == plutovg_spread_method_repeat)
    {
        ipos = ipos % COLOR_TABLE_SIZE;
        ipos = ipos < 0 ? COLOR_TABLE_SIZE + ipos : ipos;
    }
    else if(gradient->spread == plutovg_spread_method_reflect)
    {
        const int limit = COLOR_TABLE_SIZE * 2;
        ipos = ipos % limit;
        ipos = ipos < 0 ? limit + ipos : ipos;
        ipos = ipos >= COLOR_TABLE_SIZE ? limit - 1 - ipos : ipos;
    }
    else
    {
        if(ipos < 0)
            ipos = 0;
        else if(ipos >= COLOR_TABLE_SIZE)
            ipos = COLOR_TABLE_SIZE - 1;
    }

    return ipos;
}

#define FIXPT_BITS 8
#define FIXPT_SIZE (1 << FIXPT_BITS)
static inline uint32_t gradient_pixel_fixed(const gradient_data_t* gradient, int fixed_pos)
{
    int ipos = (fixed_pos + (FIXPT_SIZE / 2)) >> FIXPT_BITS;
    return gradient->colortable[gradient_clamp(gradient, ipos)];
}

static inline uint32_t gradient_pixel(const gradient_data_t* gradient, double pos)
{
    int ipos = (int)(pos * (COLOR_TABLE_SIZE - 1) + 0.5);
    return gradient->colortable[gradient_clamp(gradient, ipos)];
}

static void fetch_linear_gradient(uint32_t* buffer, const linear_gradient_values_t* v, const gradient_data_t* gradient, int y, int x, int length)
{
    double t, inc;
    double rx = 0, ry = 0;

    if(v->l == 0.0)
    {
        t = inc = 0;
    }
    else
    {
        rx = gradient->matrix.m01 * (y + 0.5) + gradient->matrix.m00 * (x + 0.5) + gradient->matrix.m02;
        ry = gradient->matrix.m11 * (y + 0.5) + gradient->matrix.m10 * (x + 0.5) + gradient->matrix.m12;
        t = v->dx * rx + v->dy * ry + v->off;
        inc = v->dx * gradient->matrix.m00 + v->dy * gradient->matrix.m10;
        t *= (COLOR_TABLE_SIZE - 1);
        inc *= (COLOR_TABLE_SIZE - 1);
    }

    const uint32_t* end = buffer + length;
    if(inc > -1e-5 && inc < 1e-5)
    {
        memfill32(buffer, gradient_pixel_fixed(gradient, (int)(t * FIXPT_SIZE)), length);
    }
    else
    {
        if(t + inc * length < (double)(INT_MAX >> (FIXPT_BITS + 1)) && t + inc * length > (double)(INT_MIN >> (FIXPT_BITS + 1)))
        {
            int t_fixed = (int)(t * FIXPT_SIZE);
            int inc_fixed = (int)(inc * FIXPT_SIZE);
            while(buffer < end)
            {
                *buffer = gradient_pixel_fixed(gradient, t_fixed);
                t_fixed += inc_fixed;
                ++buffer;
            }
        }
        else
        {
            while(buffer < end)
            {
                *buffer = gradient_pixel(gradient, t / COLOR_TABLE_SIZE);
                t += inc;
                ++buffer;
            }
        }
    }
}

static void fetch_radial_gradient(uint32_t* buffer, const radial_gradient_values_t* v, const gradient_data_t* gradient, int y, int x, int length)
{
    if(v->a == 0.0)
    {
        memfill32(buffer, 0, length);
        return;
    }

    double rx = gradient->matrix.m01 * (y + 0.5) + gradient->matrix.m02 + gradient->matrix.m00 * (x + 0.5);
    double ry = gradient->matrix.m11 * (y + 0.5) + gradient->matrix.m12 + gradient->matrix.m10 * (x + 0.5);
    uint32_t* end = buffer + length;
    rx -= gradient->radial.fx;
    ry -= gradient->radial.fy;

    double inv_a = 1 / (2 * v->a);
    const double delta_rx = gradient->matrix.m00;
    const double delta_ry = gradient->matrix.m10;

    double b = 2 * (v->dr * gradient->radial.fr + rx * v->dx + ry * v->dy);
    double delta_b = 2 * (delta_rx * v->dx + delta_ry * v->dy);
    const double b_delta_b = 2 * b * delta_b;
    const double delta_b_delta_b = 2 * delta_b * delta_b;

    const double bb = b * b;
    const double delta_bb = delta_b * delta_b;

    b *= inv_a;
    delta_b *= inv_a;

    const double rxrxryry = rx * rx + ry * ry;
    const double delta_rxrxryry = delta_rx * delta_rx + delta_ry * delta_ry;
    const double rx_plus_ry = 2 * (rx * delta_rx + ry * delta_ry);
    const double delta_rx_plus_ry = 2 * delta_rxrxryry;

    inv_a *= inv_a;

    double det = (bb - 4 * v->a * (v->sqrfr - rxrxryry)) * inv_a;
    double delta_det = (b_delta_b + delta_bb + 4 * v->a * (rx_plus_ry + delta_rxrxryry)) * inv_a;
    const double delta_delta_det = (delta_b_delta_b + 4 * v->a * delta_rx_plus_ry) * inv_a;

    if(v->extended)
    {
        while(buffer < end)
        {
            uint32_t result = 0;
            if(det >= 0)
            {
                double w = sqrt(det) - b;
                if(gradient->radial.fr + v->dr * w >= 0)
                    result = gradient_pixel(gradient, w);
            }

            *buffer = result;
            det += delta_det;
            delta_det += delta_delta_det;
            b += delta_b;
            ++buffer;
        }
    }
    else
    {
        while(buffer < end)
        {
            *buffer++ = gradient_pixel(gradient, sqrt(det) - b);
            det += delta_det;
            delta_det += delta_delta_det;
            b += delta_b;
        }
    }
}

#define ALPHA(c) ((c) >> 24)
static void composition_solid_source(uint32_t* dest, int length, uint32_t color, uint32_t alpha)
{
    if(alpha == 255)
    {
        memfill32(dest, color, length);
    }
    else
    {
        uint32_t ialpha = 255 - alpha;
        color = BYTE_MUL(color, alpha);
        for(int i = 0;i < length;i++)
            dest[i] = color + BYTE_MUL(dest[i], ialpha);
    }
}

static void composition_solid_source_over(uint32_t* dest, int length, uint32_t color, uint32_t const_alpha)
{
    if(const_alpha != 255) color = BYTE_MUL(color, const_alpha);
    uint32_t ialpha = 255 - ALPHA(color);
    for(int i = 0;i < length;i++)
        dest[i] = color + BYTE_MUL(dest[i], ialpha);
}

static void composition_solid_destination_in(uint32_t* dest, int length, uint32_t color, uint32_t const_alpha)
{
    uint32_t a = ALPHA(color);
    if(const_alpha != 255) a = BYTE_MUL(a, const_alpha) + 255 - const_alpha;
    for(int i = 0;i < length;i++)
        dest[i] = BYTE_MUL(dest[i], a);
}

static void composition_solid_destination_out(uint32_t* dest, int length, uint32_t color, uint32_t const_alpha)
{
    uint32_t a = ALPHA(~color);
    if(const_alpha != 255) a = BYTE_MUL(a, const_alpha) + 255 - const_alpha;
    for(int i = 0; i < length;i++)
        dest[i] = BYTE_MUL(dest[i], a);
}

static void composition_source(uint32_t* dest, int length, const uint32_t* src, uint32_t const_alpha)
{
    if(const_alpha == 255)
    {
        memcpy(dest, src, (size_t)(length) * sizeof(uint32_t));
    }
    else
    {
        uint32_t ialpha = 255 - const_alpha;
        for(int i = 0;i < length;i++)
            dest[i] = interpolate_pixel(src[i], const_alpha, dest[i], ialpha);
    }
}

static void composition_source_over(uint32_t* dest, int length, const uint32_t* src, uint32_t const_alpha)
{
    uint32_t s, sia;
    if(const_alpha == 255)
    {
        for(int i = 0;i < length;i++)
        {
            s = src[i];
            if(s >= 0xff000000)
                dest[i] = s;
            else if(s != 0)
            {
                sia = ALPHA(~s);
                dest[i] = s + BYTE_MUL(dest[i], sia);
            }
        }
    }
    else
    {
        for(int i = 0;i < length;i++)
        {
            s = BYTE_MUL(src[i], const_alpha);
            sia = ALPHA(~s);
            dest[i] = s + BYTE_MUL(dest[i], sia);
        }
    }
}

static void composition_destination_in(uint32_t* dest, int length, const uint32_t* src, uint32_t const_alpha)
{
    if(const_alpha == 255)
    {
        for(int i = 0; i < length;i++)
            dest[i] = BYTE_MUL(dest[i], ALPHA(src[i]));
    }
    else
    {
        uint32_t cia = 255 - const_alpha;
        uint32_t a;
        for(int i = 0;i < length;i++)
        {
            a = BYTE_MUL(ALPHA(src[i]), const_alpha) + cia;
            dest[i] = BYTE_MUL(dest[i], a);
        }
    }
}

static void composition_destination_out(uint32_t* dest, int length, const uint32_t* src, uint32_t const_alpha)
{
    if(const_alpha == 255)
    {
        for(int i = 0;i < length;i++)
            dest[i] = BYTE_MUL(dest[i], ALPHA(~src[i]));
    }
    else
    {
        uint32_t cia = 255 - const_alpha;
        uint32_t sia;
        for(int i = 0;i < length;i++)
        {
            sia = BYTE_MUL(ALPHA(~src[i]), const_alpha) + cia;
            dest[i] = BYTE_MUL(dest[i], sia);
        }
    }
}

typedef void(*composition_solid_function_t)(uint32_t* dest, int length, uint32_t color, uint32_t const_alpha);
typedef void(*composition_function_t)(uint32_t* dest, int length, const uint32_t* src, uint32_t const_alpha);

static const composition_solid_function_t composition_solid_map[] = {
    composition_solid_source,
    composition_solid_source_over,
    composition_solid_destination_in,
    composition_solid_destination_out
};

static const composition_function_t composition_map[] = {
    composition_source,
    composition_source_over,
    composition_destination_in,
    composition_destination_out
};

static void blend_solid(plutovg_surface_t* surface, plutovg_operator_t op, const plutovg_rle_t* rle, uint32_t solid)
{
    composition_solid_function_t func = composition_solid_map[op];
    int count = rle->spans.size;
    const plutovg_span_t* spans = rle->spans.data;
    while(count--)
    {
        uint32_t* target = (uint32_t*)(surface->data + spans->y * surface->stride) + spans->x;
        func(target, spans->len, solid, spans->coverage);
        ++spans;
    }
}

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define BUFFER_SIZE 1024
static void blend_linear_gradient(plutovg_surface_t* surface, plutovg_operator_t op, const plutovg_rle_t* rle, const gradient_data_t* gradient)
{
    composition_function_t func = composition_map[op];
    unsigned int buffer[BUFFER_SIZE];

    linear_gradient_values_t v;
    v.dx = gradient->linear.x2 - gradient->linear.x1;
    v.dy = gradient->linear.y2 - gradient->linear.y1;
    v.l = v.dx * v.dx + v.dy * v.dy;
    v.off = 0.0;
    if(v.l != 0.0)
    {
        v.dx /= v.l;
        v.dy /= v.l;
        v.off = -v.dx * gradient->linear.x1 - v.dy * gradient->linear.y1;
    }

    int count = rle->spans.size;
    const plutovg_span_t* spans = rle->spans.data;
    while(count--)
    {
        int length = spans->len;
        int x = spans->x;
        while(length)
        {
            int l = MIN(length, BUFFER_SIZE);
            fetch_linear_gradient(buffer, &v, gradient, spans->y, x, l);
            uint32_t* target = (uint32_t*)(surface->data + spans->y * surface->stride) + x;
            func(target, l, buffer, spans->coverage);
            x += l;
            length -= l;
        }

        ++spans;
    }
}

static void blend_radial_gradient(plutovg_surface_t* surface, plutovg_operator_t op, const plutovg_rle_t* rle, const gradient_data_t* gradient)
{
    composition_function_t func = composition_map[op];
    unsigned int buffer[BUFFER_SIZE];

    radial_gradient_values_t v;
    v.dx = gradient->radial.cx - gradient->radial.fx;
    v.dy = gradient->radial.cy - gradient->radial.fy;

    v.dr = gradient->radial.cr - gradient->radial.fr;
    v.sqrfr = gradient->radial.fr * gradient->radial.fr;

    v.a = v.dr * v.dr - v.dx * v.dx - v.dy * v.dy;
    v.inv2a = 1 / (2 * v.a);

    v.extended = gradient->radial.fr != 0.0 || v.a <= 0.0;

    int count = rle->spans.size;
    const plutovg_span_t* spans = rle->spans.data;
    while(count--)
    {
        int length = spans->len;
        int x = spans->x;
        while(length)
        {
            int l = MIN(length, BUFFER_SIZE);
            fetch_radial_gradient(buffer, &v, gradient, spans->y, x, l);
            uint32_t* target = (uint32_t*)(surface->data + spans->y * surface->stride) + x;
            func(target, l, buffer, spans->coverage);
            x += l;
            length -= l;
        }

        ++spans;
    }
}

#define CLAMP(v, lo, hi) ((v) < (lo) ? (lo) : (hi) < (v) ? (hi) : (v))
#define FIXED_SCALE (1 << 16)
static void blend_transformed_argb(plutovg_surface_t* surface, plutovg_operator_t op, const plutovg_rle_t* rle, const texture_data_t* texture)
{
    composition_function_t func = composition_map[op];
    uint32_t buffer[BUFFER_SIZE];

    int image_width = texture->width;
    int image_height = texture->height;

    int fdx = (int)(texture->matrix.m00 * FIXED_SCALE);
    int fdy = (int)(texture->matrix.m10 * FIXED_SCALE);

    int count = rle->spans.size;
    const plutovg_span_t* spans = rle->spans.data;
    while(count--)
    {
        uint32_t* target = (uint32_t*)(surface->data + spans->y * surface->stride) + spans->x;

        const double cx = spans->x + 0.5;
        const double cy = spans->y + 0.5;

        int x = (int)((texture->matrix.m01 * cy + texture->matrix.m00 * cx + texture->matrix.m02) * FIXED_SCALE);
        int y = (int)((texture->matrix.m11 * cy + texture->matrix.m10 * cx + texture->matrix.m12) * FIXED_SCALE);

        int length = spans->len;
        const int coverage = (spans->coverage * texture->const_alpha) >> 8;
        while(length)
        {
            int l = MIN(length, BUFFER_SIZE);
            const uint32_t* end = buffer + l;
            uint32_t* b = buffer;
            while(b < end)
            {
                int px = CLAMP(x >> 16, 0, image_width - 1);
                int py = CLAMP(y >> 16, 0, image_height - 1);
                *b = ((const uint32_t*)(texture->data + py * texture->stride))[px];

                x += fdx;
                y += fdy;
                ++b;
            }

            func(target, l, buffer, coverage);
            target += l;
            length -= l;
        }

        ++spans;
    }
}

static void blend_untransformed_argb(plutovg_surface_t* surface, plutovg_operator_t op, const plutovg_rle_t* rle, const texture_data_t* texture)
{
    composition_function_t func = composition_map[op];

    const int image_width = texture->width;
    const int image_height = texture->height;

    int xoff = (int)(texture->matrix.m02);
    int yoff = (int)(texture->matrix.m12);

    int count = rle->spans.size;
    const plutovg_span_t* spans = rle->spans.data;
    while(count--)
    {
        int x = spans->x;
        int length = spans->len;
        int sx = xoff + x;
        int sy = yoff + spans->y;
        if(sy >= 0 && sy < image_height && sx < image_width)
        {
            if(sx < 0)
            {
                x -= sx;
                length += sx;
                sx = 0;
            }
            if(sx + length > image_width) length = image_width - sx;
            if(length > 0)
            {
                const int coverage = (spans->coverage * texture->const_alpha) >> 8;
                const uint32_t* src = (const uint32_t*)(texture->data + sy * texture->stride) + sx;
                uint32_t* dest = (uint32_t*)(surface->data + spans->y * surface->stride) + x;
                func(dest, length, src, coverage);
            }
        }

        ++spans;
    }
}

static void blend_untransformed_tiled_argb(plutovg_surface_t* surface, plutovg_operator_t op, const plutovg_rle_t* rle, const texture_data_t* texture)
{
    composition_function_t func = composition_map[op];

    int image_width = texture->width;
    int image_height = texture->height;

    int xoff = (int)(texture->matrix.m02) % image_width;
    int yoff = (int)(texture->matrix.m12) % image_height;

    if(xoff < 0)
        xoff += image_width;
    if(yoff < 0)
        yoff += image_height;

    int count = rle->spans.size;
    const plutovg_span_t* spans = rle->spans.data;
    while(count--)
    {
        int x = spans->x;
        int length = spans->len;
        int sx = (xoff + spans->x) % image_width;
        int sy = (spans->y + yoff) % image_height;
        if(sx < 0)
            sx += image_width;
        if(sy < 0)
            sy += image_height;

        const int coverage = (spans->coverage * texture->const_alpha) >> 8;
        while(length)
        {
            int l = MIN(image_width - sx, length);
            if(BUFFER_SIZE < l)
                l = BUFFER_SIZE;
            const uint32_t* src = (const uint32_t*)(texture->data + sy * texture->stride) + sx;
            uint32_t* dest = (uint32_t*)(surface->data + spans->y * surface->stride) + x;
            func(dest, l, src, coverage);
            x += l;
            length -= l;
            sx = 0;
        }

        ++spans;
    }
}

static void blend_transformed_tiled_argb(plutovg_surface_t* surface, plutovg_operator_t op, const plutovg_rle_t* rle, const texture_data_t* texture)
{
    composition_function_t func = composition_map[op];
    uint32_t buffer[BUFFER_SIZE];

    int image_width = texture->width;
    int image_height = texture->height;
    const int scanline_offset = texture->stride / 4;

    int fdx = (int)(texture->matrix.m00 * FIXED_SCALE);
    int fdy = (int)(texture->matrix.m10 * FIXED_SCALE);

    int count = rle->spans.size;
    const plutovg_span_t* spans = rle->spans.data;
    while(count--)
    {
        uint32_t* target = (uint32_t*)(surface->data + spans->y * surface->stride) + spans->x;
        const uint32_t* image_bits = (const uint32_t*)texture->data;

        const double cx = spans->x + 0.5;
        const double cy = spans->y + 0.5;

        int x = (int)((texture->matrix.m01 * cy + texture->matrix.m00 * cx + texture->matrix.m02) * FIXED_SCALE);
        int y = (int)((texture->matrix.m11 * cy + texture->matrix.m10 * cx + texture->matrix.m12) * FIXED_SCALE);

        const int coverage = (spans->coverage * texture->const_alpha) >> 8;
        int length = spans->len;
        while(length)
        {
            int l = MIN(length, BUFFER_SIZE);
            const uint32_t* end = buffer + l;
            uint32_t* b = buffer;
            int px16 = x % (image_width << 16);
            int py16 = y % (image_height << 16);
            int px_delta = fdx % (image_width << 16);
            int py_delta = fdy % (image_height << 16);
            while(b < end)
            {
                if(px16 < 0) px16 += image_width << 16;
                if(py16 < 0) py16 += image_height << 16;
                int px = px16 >> 16;
                int py = py16 >> 16;
                int y_offset = py * scanline_offset;

                *b = image_bits[y_offset + px];
                x += fdx;
                y += fdy;
                px16 += px_delta;
                if(px16 >= image_width << 16)
                    px16 -= image_width << 16;
                py16 += py_delta;
                if(py16 >= image_height << 16)
                    py16 -= image_height << 16;
                ++b;
            }

            func(target, l, buffer, coverage);
            target += l;
            length -= l;
        }

        ++spans;
    }
}

void plutovg_blend(plutovg_t* pluto, const plutovg_rle_t* rle)
{
    plutovg_paint_t* source = pluto->state->source;
    if(source->type==plutovg_paint_type_color)
    {
        const plutovg_color_t* color = plutovg_paint_get_color(source);
        plutovg_blend_color(pluto, rle, color);
    }
    else if(source->type==plutovg_paint_type_gradient)
    {
        const plutovg_gradient_t* gradient = plutovg_paint_get_gradient(source);
        plutovg_blend_gradient(pluto, rle, gradient);
    }
    else
    {
        const plutovg_texture_t* texture = plutovg_paint_get_texture(source);
        plutovg_blend_texture(pluto, rle, texture);
    }
}

void plutovg_blend_color(plutovg_t* pluto, const plutovg_rle_t* rle, const plutovg_color_t* color)
{
    plutovg_state_t* state = pluto->state;
    uint32_t solid = premultiply_color(color, state->opacity);
    blend_solid(pluto->surface, state->op, rle, solid);
}

void plutovg_blend_gradient(plutovg_t* pluto, const plutovg_rle_t* rle, const plutovg_gradient_t* gradient)
{
    plutovg_state_t* state = pluto->state;
    gradient_data_t data;
    if(gradient->type==plutovg_gradient_type_linear)
    {
        data.linear.x1 = gradient->values[0];
        data.linear.y1 = gradient->values[1];
        data.linear.x2 = gradient->values[2];
        data.linear.y2 = gradient->values[3];
    }
    else
    {
        data.radial.cx = gradient->values[0];
        data.radial.cy = gradient->values[1];
        data.radial.cr = gradient->values[2];
        data.radial.fx = gradient->values[3];
        data.radial.fy = gradient->values[4];
        data.radial.fr = gradient->values[5];
    }

    int dist, idist, pos = 0;
    int i;
    int nstop = gradient->stops.size;
    const plutovg_gradient_stop_t *curr, *next, *start;
    uint32_t curr_color, next_color;
    double delta, t, incr, fpos;
    double opacity = state->opacity * gradient->opacity;

    start = gradient->stops.data;
    curr = start;
    curr_color = premultiply_color(&curr->color, opacity);
    incr = 1.0 / COLOR_TABLE_SIZE;
    fpos = 1.5 * incr;

    data.colortable[pos++] = curr_color;

    while(fpos <= curr->offset)
    {
        data.colortable[pos] = data.colortable[pos - 1];
        pos++;
        fpos += incr;
    }

    for(i = 0;i < nstop - 1;i++)
    {
        curr = (start + i);
        next = (start + i + 1);
        delta = 1.0 / (next->offset - curr->offset);
        next_color = premultiply_color(&next->color, opacity);
        while(fpos < next->offset && pos < COLOR_TABLE_SIZE)
        {
            t = (fpos - curr->offset) * delta;
            dist = (int)(255 * t);
            idist = 255 - dist;
            data.colortable[pos] = interpolate_pixel(curr_color, (uint32_t)idist, next_color, (uint32_t)dist);
            ++pos;
            fpos += incr;
        }

        curr_color = next_color;
    }

    for(;pos < COLOR_TABLE_SIZE;++pos)
        data.colortable[pos] = curr_color;

    data.spread = gradient->spread;
    data.matrix = gradient->matrix;
    plutovg_matrix_multiply(&data.matrix, &data.matrix, &state->matrix);
    plutovg_matrix_invert(&data.matrix);

    if(gradient->type==plutovg_gradient_type_linear)
        blend_linear_gradient(pluto->surface, state->op, rle, &data);
    else
        blend_radial_gradient(pluto->surface, state->op, rle, &data);
}

void plutovg_blend_texture(plutovg_t* pluto, const plutovg_rle_t* rle, const plutovg_texture_t* texture)
{
    plutovg_state_t* state = pluto->state;
    texture_data_t data;
    data.data = texture->surface->data;
    data.width = texture->surface->width;
    data.height = texture->surface->height;
    data.stride = texture->surface->stride;
    data.const_alpha = (int)(state->opacity * texture->opacity * 256.0);

    data.matrix = texture->matrix;
    plutovg_matrix_multiply(&data.matrix, &data.matrix, &state->matrix);
    plutovg_matrix_invert(&data.matrix);

    const plutovg_matrix_t* matrix = &data.matrix;
    int translating = (matrix->m00==1.0 && matrix->m10==0.0 && matrix->m01==0.0 && matrix->m11==1.0);
    if(translating)
    {
        if(texture->type==plutovg_texture_type_plain)
            blend_untransformed_argb(pluto->surface, state->op, rle, &data);
        else
            blend_untransformed_tiled_argb(pluto->surface, state->op, rle, &data);
    }
    else
    {
        if(texture->type==plutovg_texture_type_plain)
            blend_transformed_argb(pluto->surface, state->op, rle, &data);
        else
            blend_transformed_tiled_argb(pluto->surface, state->op, rle, &data);
    }
}
