#ifndef PLUTOVG_PRIVATE_H
#define PLUTOVG_PRIVATE_H

#include "plutovg.h"

struct plutovg_surface {
    int ref_count;
    int width;
    int height;
    int stride;
    unsigned char* data;
};

struct plutovg_path {
    int ref_count;
    int num_curves;
    int num_contours;
    int num_points;
    plutovg_point_t start_point;
    struct {
        plutovg_path_element_t* data;
        int size;
        int capacity;
    } elements;
};

typedef enum {
    PLUTOVG_PAINT_TYPE_COLOR,
    PLUTOVG_PAINT_TYPE_GRADIENT,
    PLUTOVG_PAINT_TYPE_TEXTURE
} plutovg_paint_type_t;

struct plutovg_paint {
    int ref_count;
    plutovg_paint_type_t type;
};

typedef struct {
    plutovg_paint_t base;
    plutovg_color_t color;
} plutovg_solid_paint_t;

typedef enum {
    PLUTOVG_GRADIENT_TYPE_LINEAR,
    PLUTOVG_GRADIENT_TYPE_RADIAL
} plutovg_gradient_type_t;

typedef struct {
    plutovg_paint_t base;
    plutovg_gradient_type_t type;
    plutovg_spread_method_t spread;
    plutovg_matrix_t matrix;
    plutovg_gradient_stop_t* stops;
    int nstops;
    float values[6];
} plutovg_gradient_paint_t;

typedef struct {
    plutovg_paint_t base;
    plutovg_texture_type_t type;
    float opacity;
    plutovg_matrix_t matrix;
    plutovg_surface_t* surface;
} plutovg_texture_paint_t;

typedef struct {
    int x;
    int len;
    int y;
    unsigned char coverage;
} plutovg_span_t;

typedef struct {
    struct {
        plutovg_span_t* data;
        int size;
        int capacity;
    } spans;

    int x;
    int y;
    int w;
    int h;
} plutovg_span_buffer_t;

typedef struct {
    float offset;
    struct {
        float* data;
        int size;
        int capacity;
    } array;
} plutovg_stroke_dash_t;

typedef struct {
    float width;
    plutovg_line_cap_t cap;
    plutovg_line_join_t join;
    float miter_limit;
} plutovg_stroke_style_t;

typedef struct {
    plutovg_stroke_style_t style;
    plutovg_stroke_dash_t dash;
} plutovg_stroke_data_t;

typedef struct plutovg_state {
    plutovg_paint_t* paint;
    plutovg_color_t color;
    plutovg_matrix_t matrix;
    plutovg_stroke_data_t stroke;
    plutovg_operator_t op;
    plutovg_fill_rule_t winding;
    plutovg_span_buffer_t clip_spans;
    plutovg_font_face_t* font_face;
    float font_size;
    float opacity;
    bool clipping;
    struct plutovg_state* next;
} plutovg_state_t;

struct plutovg_canvas {
    int ref_count;
    plutovg_surface_t* surface;
    plutovg_path_t* path;
    plutovg_state_t* state;
    plutovg_state_t* freed_state;
    plutovg_rect_t clip_rect;
    plutovg_span_buffer_t clip_spans;
    plutovg_span_buffer_t fill_spans;
};

void plutovg_span_buffer_init(plutovg_span_buffer_t* span_buffer);
void plutovg_span_buffer_init_rect(plutovg_span_buffer_t* span_buffer, int x, int y, int width, int height);
void plutovg_span_buffer_reset(plutovg_span_buffer_t* span_buffer);
void plutovg_span_buffer_destroy(plutovg_span_buffer_t* span_buffer);
void plutovg_span_buffer_copy(plutovg_span_buffer_t* span_buffer, const plutovg_span_buffer_t* source);
void plutovg_span_buffer_extents(plutovg_span_buffer_t* span_buffer, plutovg_rect_t* extents);
void plutovg_span_buffer_intersect(plutovg_span_buffer_t* span_buffer, const plutovg_span_buffer_t* a, const plutovg_span_buffer_t* b);

void plutovg_rasterize(plutovg_span_buffer_t* span_buffer, const plutovg_path_t* path, const plutovg_matrix_t* matrix, const plutovg_rect_t* clip_rect, const plutovg_stroke_data_t* stroke_data, plutovg_fill_rule_t winding);
void plutovg_blend(plutovg_canvas_t* canvas, const plutovg_span_buffer_t* span_buffer);
void plutovg_memfill32(unsigned int* dest, int length, unsigned int value);

#endif // PLUTOVG_PRIVATE_H
