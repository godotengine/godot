#include "plutovg-private.h"
#include "plutovg-utils.h"

int plutovg_version(void)
{
    return PLUTOVG_VERSION;
}

const char* plutovg_version_string(void)
{
    return PLUTOVG_VERSION_STRING;
}

static void plutovg_stroke_data_reset(plutovg_stroke_data_t* stroke)
{
    plutovg_array_clear(stroke->dash.array);
    stroke->dash.offset = 0.f;
    stroke->style.width = 1.f;
    stroke->style.cap = PLUTOVG_LINE_CAP_BUTT;
    stroke->style.join = PLUTOVG_LINE_JOIN_MITER;
    stroke->style.miter_limit = 10.f;
}

static void plutovg_stroke_data_copy(plutovg_stroke_data_t* stroke, const plutovg_stroke_data_t* source)
{
    plutovg_array_clear(stroke->dash.array);
    plutovg_array_append(stroke->dash.array, source->dash.array);
    stroke->dash.offset = source->dash.offset;
    stroke->style.width = source->style.width;
    stroke->style.cap = source->style.cap;
    stroke->style.join = source->style.join;
    stroke->style.miter_limit = source->style.miter_limit;
}

static void plutovg_state_reset(plutovg_state_t* state)
{
    plutovg_paint_destroy(state->paint);
    plutovg_matrix_init_identity(&state->matrix);
    plutovg_stroke_data_reset(&state->stroke);
    plutovg_span_buffer_reset(&state->clip_spans);
    plutovg_font_face_destroy(state->font_face);
    state->paint = NULL;
    state->color = PLUTOVG_BLACK_COLOR;
    state->font_face = NULL;
    state->font_size = 12.f;
    state->op = PLUTOVG_OPERATOR_SRC_OVER;
    state->winding = PLUTOVG_FILL_RULE_NON_ZERO;
    state->clipping = false;
    state->opacity = 1.f;
}

static void plutovg_state_copy(plutovg_state_t* state, const plutovg_state_t* source)
{
    plutovg_stroke_data_copy(&state->stroke, &source->stroke);
    plutovg_span_buffer_copy(&state->clip_spans, &source->clip_spans);
    state->paint = plutovg_paint_reference(source->paint);
    state->font_face = plutovg_font_face_reference(source->font_face);
    state->color = source->color;
    state->matrix = source->matrix;
    state->font_size = source->font_size;
    state->op = source->op;
    state->winding = source->winding;
    state->clipping = source->clipping;
    state->opacity = source->opacity;
}

static plutovg_state_t* plutovg_state_create(void)
{
    plutovg_state_t* state = malloc(sizeof(plutovg_state_t));
    memset(state, 0, sizeof(plutovg_state_t));
    plutovg_state_reset(state);
    return state;
}

static void plutovg_state_destroy(plutovg_state_t* state)
{
    plutovg_paint_destroy(state->paint);
    plutovg_array_destroy(state->stroke.dash.array);
    plutovg_span_buffer_destroy(&state->clip_spans);
    free(state);
}

plutovg_canvas_t* plutovg_canvas_create(plutovg_surface_t* surface)
{
    plutovg_canvas_t* canvas = malloc(sizeof(plutovg_canvas_t));
    canvas->ref_count = 1;
    canvas->surface = plutovg_surface_reference(surface);
    canvas->path = plutovg_path_create();
    canvas->state = plutovg_state_create();
    canvas->freed_state = NULL;
    canvas->clip_rect = PLUTOVG_MAKE_RECT(0, 0, surface->width, surface->height);
    plutovg_span_buffer_init(&canvas->clip_spans);
    plutovg_span_buffer_init(&canvas->fill_spans);
    return canvas;
}

plutovg_canvas_t* plutovg_canvas_reference(plutovg_canvas_t* canvas)
{
    if(canvas == NULL)
        return NULL;
    ++canvas->ref_count;
    return canvas;
}

void plutovg_canvas_destroy(plutovg_canvas_t* canvas)
{
    if(canvas == NULL)
        return;
    if(--canvas->ref_count == 0) {
        while(canvas->state) {
            plutovg_state_t* state = canvas->state;
            canvas->state = state->next;
            plutovg_state_destroy(state);
        }

        while(canvas->freed_state) {
            plutovg_state_t* state = canvas->freed_state;
            canvas->freed_state = state->next;
            plutovg_state_destroy(state);
        }

        plutovg_span_buffer_destroy(&canvas->fill_spans);
        plutovg_span_buffer_destroy(&canvas->clip_spans);
        plutovg_surface_destroy(canvas->surface);
        plutovg_path_destroy(canvas->path);
        free(canvas);
    }
}

int plutovg_canvas_get_reference_count(const plutovg_canvas_t* canvas)
{
    if(canvas == NULL)
        return 0;
    return canvas->ref_count;
}

plutovg_surface_t* plutovg_canvas_get_surface(const plutovg_canvas_t* canvas)
{
    return canvas->surface;
}

void plutovg_canvas_save(plutovg_canvas_t* canvas)
{
    plutovg_state_t* new_state = canvas->freed_state;
    if(new_state == NULL)
        new_state = plutovg_state_create();
    else
        canvas->freed_state = new_state->next;
    plutovg_state_copy(new_state, canvas->state);
    new_state->next = canvas->state;
    canvas->state = new_state;
}

void plutovg_canvas_restore(plutovg_canvas_t* canvas)
{
    if(canvas->state->next == NULL)
        return;
    plutovg_state_t* old_state = canvas->state;
    canvas->state = old_state->next;
    plutovg_state_reset(old_state);
    old_state->next = canvas->freed_state;
    canvas->freed_state = old_state;
}

void plutovg_canvas_set_rgb(plutovg_canvas_t* canvas, float r, float g, float b)
{
    plutovg_canvas_set_rgba(canvas, r, g, b, 1.f);
}

void plutovg_canvas_set_rgba(plutovg_canvas_t* canvas, float r, float g, float b, float a)
{
    plutovg_color_init_rgba(&canvas->state->color, r, g, b, a);
    plutovg_canvas_set_paint(canvas, NULL);
}

void plutovg_canvas_set_color(plutovg_canvas_t* canvas, const plutovg_color_t* color)
{
    plutovg_canvas_set_rgba(canvas, color->r, color->g, color->b, color->a);
}

void plutovg_canvas_set_linear_gradient(plutovg_canvas_t* canvas, float x1, float y1, float x2, float y2, plutovg_spread_method_t spread, const plutovg_gradient_stop_t* stops, int nstops, const plutovg_matrix_t* matrix)
{
    plutovg_paint_t* paint = plutovg_paint_create_linear_gradient(x1, y1, x2, y2, spread, stops, nstops, matrix);
    plutovg_canvas_set_paint(canvas, paint);
    plutovg_paint_destroy(paint);
}

void plutovg_canvas_set_radial_gradient(plutovg_canvas_t* canvas, float cx, float cy, float cr, float fx, float fy, float fr, plutovg_spread_method_t spread, const plutovg_gradient_stop_t* stops, int nstops, const plutovg_matrix_t* matrix)
{
    plutovg_paint_t* paint = plutovg_paint_create_radial_gradient(cx, cy, cr, fx, fy, fr, spread, stops, nstops, matrix);
    plutovg_canvas_set_paint(canvas, paint);
    plutovg_paint_destroy(paint);
}

void plutovg_canvas_set_texture(plutovg_canvas_t* canvas, plutovg_surface_t* surface, plutovg_texture_type_t type, float opacity, const plutovg_matrix_t* matrix)
{
    plutovg_paint_t* paint = plutovg_paint_create_texture(surface, type, opacity, matrix);
    plutovg_canvas_set_paint(canvas, paint);
    plutovg_paint_destroy(paint);
}

void plutovg_canvas_set_paint(plutovg_canvas_t* canvas, plutovg_paint_t* paint)
{
    paint = plutovg_paint_reference(paint);
    plutovg_paint_destroy(canvas->state->paint);
    canvas->state->paint = paint;
}

plutovg_paint_t* plutovg_canvas_get_paint(const plutovg_canvas_t* canvas, plutovg_color_t* color)
{
    if(color)
        *color = canvas->state->color;
    return canvas->state->paint;
}

void plutovg_canvas_set_font(plutovg_canvas_t* canvas, plutovg_font_face_t* face, float size)
{
    plutovg_canvas_set_font_face(canvas, face);
    plutovg_canvas_set_font_size(canvas, size);
}

void plutovg_canvas_set_font_face(plutovg_canvas_t* canvas, plutovg_font_face_t* face)
{
    face = plutovg_font_face_reference(face);
    plutovg_font_face_destroy(canvas->state->font_face);
    canvas->state->font_face = face;
}

plutovg_font_face_t* plutovg_canvas_get_font_face(const plutovg_canvas_t* canvas)
{
    return canvas->state->font_face;
}

void plutovg_canvas_set_font_size(plutovg_canvas_t* canvas, float size)
{
    canvas->state->font_size = size;
}

float plutovg_canvas_get_font_size(const plutovg_canvas_t* canvas)
{
    return canvas->state->font_size;
}

void plutovg_canvas_set_fill_rule(plutovg_canvas_t* canvas, plutovg_fill_rule_t winding)
{
    canvas->state->winding = winding;
}

plutovg_fill_rule_t plutovg_canvas_get_fill_rule(const plutovg_canvas_t* canvas)
{
    return canvas->state->winding;
}

void plutovg_canvas_set_operator(plutovg_canvas_t* canvas, plutovg_operator_t op)
{
    canvas->state->op = op;
}

plutovg_operator_t plutovg_canvas_get_operator(const plutovg_canvas_t* canvas)
{
    return canvas->state->op;
}

void plutovg_canvas_set_opacity(plutovg_canvas_t* canvas, float opacity)
{
    canvas->state->opacity = plutovg_clamp(opacity, 0.f, 1.f);
}

float plutovg_canvas_get_opacity(const plutovg_canvas_t* canvas)
{
    return canvas->state->opacity;
}

void plutovg_canvas_set_line_width(plutovg_canvas_t* canvas, float line_width)
{
    canvas->state->stroke.style.width = line_width;
}

float plutovg_canvas_get_line_width(const plutovg_canvas_t* canvas)
{
    return canvas->state->stroke.style.width;
}

void plutovg_canvas_set_line_cap(plutovg_canvas_t* canvas, plutovg_line_cap_t line_cap)
{
    canvas->state->stroke.style.cap = line_cap;
}

plutovg_line_cap_t plutovg_canvas_get_line_cap(const plutovg_canvas_t* canvas)
{
    return canvas->state->stroke.style.cap;
}

void plutovg_canvas_set_line_join(plutovg_canvas_t* canvas, plutovg_line_join_t line_join)
{
    canvas->state->stroke.style.join = line_join;
}

plutovg_line_join_t plutovg_canvas_get_line_join(const plutovg_canvas_t* canvas)
{
    return canvas->state->stroke.style.join;
}

void plutovg_canvas_set_miter_limit(plutovg_canvas_t* canvas, float miter_limit)
{
    canvas->state->stroke.style.miter_limit = miter_limit;
}

float plutovg_canvas_get_miter_limit(const plutovg_canvas_t* canvas)
{
    return canvas->state->stroke.style.miter_limit;
}

void plutovg_canvas_set_dash(plutovg_canvas_t* canvas, float offset, const float* dashes, int ndashes)
{
    plutovg_canvas_set_dash_offset(canvas, offset);
    plutovg_canvas_set_dash_array(canvas, dashes, ndashes);
}

void plutovg_canvas_set_dash_offset(plutovg_canvas_t* canvas, float offset)
{
    canvas->state->stroke.dash.offset = offset;
}

float plutovg_canvas_get_dash_offset(const plutovg_canvas_t* canvas)
{
    return canvas->state->stroke.dash.offset;
}

void plutovg_canvas_set_dash_array(plutovg_canvas_t* canvas, const float* dashes, int ndashes)
{
    plutovg_array_clear(canvas->state->stroke.dash.array);
    plutovg_array_append_data(canvas->state->stroke.dash.array, dashes, ndashes);
}

int plutovg_canvas_get_dash_array(const plutovg_canvas_t* canvas, const float** dashes)
{
    if(dashes)
        *dashes = canvas->state->stroke.dash.array.data;
    return canvas->state->stroke.dash.array.size;
}

void plutovg_canvas_translate(plutovg_canvas_t* canvas, float tx, float ty)
{
    plutovg_matrix_translate(&canvas->state->matrix, tx, ty);
}

void plutovg_canvas_scale(plutovg_canvas_t* canvas, float sx, float sy)
{
    plutovg_matrix_scale(&canvas->state->matrix, sx, sy);
}

void plutovg_canvas_shear(plutovg_canvas_t* canvas, float shx, float shy)
{
    plutovg_matrix_shear(&canvas->state->matrix, shx, shy);
}

void plutovg_canvas_rotate(plutovg_canvas_t* canvas, float angle)
{
    plutovg_matrix_rotate(&canvas->state->matrix, angle);
}

void plutovg_canvas_transform(plutovg_canvas_t* canvas, const plutovg_matrix_t* matrix)
{
    plutovg_matrix_multiply(&canvas->state->matrix, matrix, &canvas->state->matrix);
}

void plutovg_canvas_reset_matrix(plutovg_canvas_t* canvas)
{
    plutovg_matrix_init_identity(&canvas->state->matrix);
}

void plutovg_canvas_set_matrix(plutovg_canvas_t* canvas, const plutovg_matrix_t* matrix)
{
    canvas->state->matrix = matrix ? *matrix : PLUTOVG_IDENTITY_MATRIX;
}

void plutovg_canvas_get_matrix(const plutovg_canvas_t* canvas, plutovg_matrix_t* matrix)
{
    *matrix = canvas->state->matrix;
}

void plutovg_canvas_map(const plutovg_canvas_t* canvas, float x, float y, float* xx, float* yy)
{
    plutovg_matrix_map(&canvas->state->matrix, x, y, xx, yy);
}

void plutovg_canvas_map_point(const plutovg_canvas_t* canvas, const plutovg_point_t* src, plutovg_point_t* dst)
{
    plutovg_matrix_map_point(&canvas->state->matrix, src, dst);
}

void plutovg_canvas_map_rect(const plutovg_canvas_t* canvas, const plutovg_rect_t* src, plutovg_rect_t* dst)
{
    plutovg_matrix_map_rect(&canvas->state->matrix, src, dst);
}

void plutovg_canvas_move_to(plutovg_canvas_t* canvas, float x, float y)
{
    plutovg_path_move_to(canvas->path, x, y);
}

void plutovg_canvas_line_to(plutovg_canvas_t* canvas, float x, float y)
{
    plutovg_path_line_to(canvas->path, x, y);
}

void plutovg_canvas_quad_to(plutovg_canvas_t* canvas, float x1, float y1, float x2, float y2)
{
    plutovg_path_quad_to(canvas->path, x1, y1, x2, y2);
}

void plutovg_canvas_cubic_to(plutovg_canvas_t* canvas, float x1, float y1, float x2, float y2, float x3, float y3)
{
    plutovg_path_cubic_to(canvas->path, x1, y1, x2, y2, x3, y3);
}

void plutovg_canvas_arc_to(plutovg_canvas_t* canvas, float rx, float ry, float angle, bool large_arc_flag, bool sweep_flag, float x, float y)
{
    plutovg_path_arc_to(canvas->path, rx, ry, angle, large_arc_flag, sweep_flag, x, y);
}

void plutovg_canvas_rect(plutovg_canvas_t* canvas, float x, float y, float w, float h)
{
    plutovg_path_add_rect(canvas->path, x, y, w, h);
}

void plutovg_canvas_round_rect(plutovg_canvas_t* canvas, float x, float y, float w, float h, float rx, float ry)
{
    plutovg_path_add_round_rect(canvas->path, x, y, w, h, rx, ry);
}

void plutovg_canvas_ellipse(plutovg_canvas_t* canvas, float cx, float cy, float rx, float ry)
{
    plutovg_path_add_ellipse(canvas->path, cx, cy, rx, ry);
}

void plutovg_canvas_circle(plutovg_canvas_t* canvas, float cx, float cy, float r)
{
    plutovg_path_add_circle(canvas->path, cx, cy, r);
}

void plutovg_canvas_arc(plutovg_canvas_t* canvas, float cx, float cy, float r, float a0, float a1, bool ccw)
{
    plutovg_path_add_arc(canvas->path, cx, cy, r, a0, a1, ccw);
}

void plutovg_canvas_add_path(plutovg_canvas_t* canvas, const plutovg_path_t* path)
{
    plutovg_path_add_path(canvas->path, path, NULL);
}

void plutovg_canvas_new_path(plutovg_canvas_t* canvas)
{
    plutovg_path_reset(canvas->path);
}

void plutovg_canvas_close_path(plutovg_canvas_t* canvas)
{
    plutovg_path_close(canvas->path);
}

void plutovg_canvas_get_current_point(const plutovg_canvas_t* canvas, float* x, float* y)
{
    plutovg_path_get_current_point(canvas->path, x, y);
}

plutovg_path_t* plutovg_canvas_get_path(const plutovg_canvas_t* canvas)
{
    return canvas->path;
}

void plutovg_canvas_fill_extents(const plutovg_canvas_t* canvas, plutovg_rect_t* extents)
{
    plutovg_path_extents(canvas->path, extents, true);
    plutovg_canvas_map_rect(canvas, extents, extents);
}

void plutovg_canvas_stroke_extents(const plutovg_canvas_t* canvas, plutovg_rect_t* extents)
{
    plutovg_stroke_data_t* stroke = &canvas->state->stroke;
    float cap_limit = stroke->style.width / 2.f;
    if(stroke->style.cap == PLUTOVG_LINE_CAP_SQUARE)
        cap_limit *= PLUTOVG_SQRT2;
    float join_limit = stroke->style.width / 2.f;
    if(stroke->style.join == PLUTOVG_LINE_JOIN_MITER) {
        join_limit *= stroke->style.miter_limit;
    }

    float delta = plutovg_max(cap_limit, join_limit);
    plutovg_path_extents(canvas->path, extents, true);
    extents->x -= delta;
    extents->y -= delta;
    extents->w += delta * 2.f;
    extents->h += delta * 2.f;
    plutovg_canvas_map_rect(canvas, extents, extents);
}

void plutovg_canvas_clip_extents(const plutovg_canvas_t* canvas, plutovg_rect_t* extents)
{
    if(canvas->state->clipping) {
        plutovg_span_buffer_extents(&canvas->state->clip_spans, extents);
    } else {
        extents->x = canvas->clip_rect.x;
        extents->y = canvas->clip_rect.y;
        extents->w = canvas->clip_rect.w;
        extents->h = canvas->clip_rect.h;
    }
}

void plutovg_canvas_fill(plutovg_canvas_t* canvas)
{
    plutovg_canvas_fill_preserve(canvas);
    plutovg_canvas_new_path(canvas);
}

void plutovg_canvas_stroke(plutovg_canvas_t* canvas)
{
    plutovg_canvas_stroke_preserve(canvas);
    plutovg_canvas_new_path(canvas);
}

void plutovg_canvas_clip(plutovg_canvas_t* canvas)
{
    plutovg_canvas_clip_preserve(canvas);
    plutovg_canvas_new_path(canvas);
}

void plutovg_canvas_paint(plutovg_canvas_t* canvas)
{
    if(canvas->state->clipping) {
        plutovg_blend(canvas, &canvas->state->clip_spans);
    } else {
        plutovg_span_buffer_init_rect(&canvas->clip_spans, 0, 0, canvas->surface->width, canvas->surface->height);
        plutovg_blend(canvas, &canvas->clip_spans);
    }
}

void plutovg_canvas_fill_preserve(plutovg_canvas_t* canvas)
{
    plutovg_rasterize(&canvas->fill_spans, canvas->path, &canvas->state->matrix, &canvas->clip_rect, NULL, canvas->state->winding);
    if(canvas->state->clipping) {
        plutovg_span_buffer_intersect(&canvas->clip_spans, &canvas->fill_spans, &canvas->state->clip_spans);
        plutovg_blend(canvas, &canvas->clip_spans);
    } else {
        plutovg_blend(canvas, &canvas->fill_spans);
    }
}

void plutovg_canvas_stroke_preserve(plutovg_canvas_t* canvas)
{
    plutovg_rasterize(&canvas->fill_spans, canvas->path, &canvas->state->matrix, &canvas->clip_rect, &canvas->state->stroke, PLUTOVG_FILL_RULE_NON_ZERO);
    if(canvas->state->clipping) {
        plutovg_span_buffer_intersect(&canvas->clip_spans, &canvas->fill_spans, &canvas->state->clip_spans);
        plutovg_blend(canvas, &canvas->clip_spans);
    } else {
        plutovg_blend(canvas, &canvas->fill_spans);
    }
}

void plutovg_canvas_clip_preserve(plutovg_canvas_t* canvas)
{
    if(canvas->state->clipping) {
        plutovg_rasterize(&canvas->fill_spans, canvas->path, &canvas->state->matrix, &canvas->clip_rect, NULL, canvas->state->winding);
        plutovg_span_buffer_intersect(&canvas->clip_spans, &canvas->fill_spans, &canvas->state->clip_spans);
        plutovg_span_buffer_copy(&canvas->state->clip_spans, &canvas->clip_spans);
    } else {
        plutovg_rasterize(&canvas->state->clip_spans, canvas->path, &canvas->state->matrix, &canvas->clip_rect, NULL, canvas->state->winding);
        canvas->state->clipping = true;
    }
}

void plutovg_canvas_fill_rect(plutovg_canvas_t* canvas, float x, float y, float w, float h)
{
    plutovg_canvas_new_path(canvas);
    plutovg_canvas_rect(canvas, x, y, w, h);
    plutovg_canvas_fill(canvas);
}

void plutovg_canvas_fill_path(plutovg_canvas_t* canvas, const plutovg_path_t* path)
{
    plutovg_canvas_new_path(canvas);
    plutovg_canvas_add_path(canvas, path);
    plutovg_canvas_fill(canvas);
}

void plutovg_canvas_stroke_rect(plutovg_canvas_t* canvas, float x, float y, float w, float h)
{
    plutovg_canvas_new_path(canvas);
    plutovg_canvas_rect(canvas, x, y, w, h);
    plutovg_canvas_stroke(canvas);
}

void plutovg_canvas_stroke_path(plutovg_canvas_t* canvas, const plutovg_path_t* path)
{
    plutovg_canvas_new_path(canvas);
    plutovg_canvas_add_path(canvas, path);
    plutovg_canvas_stroke(canvas);
}

void plutovg_canvas_clip_rect(plutovg_canvas_t* canvas, float x, float y, float w, float h)
{
    plutovg_canvas_new_path(canvas);
    plutovg_canvas_rect(canvas, x, y, w, h);
    plutovg_canvas_clip(canvas);
}

void plutovg_canvas_clip_path(plutovg_canvas_t* canvas, const plutovg_path_t* path)
{
    plutovg_canvas_new_path(canvas);
    plutovg_canvas_add_path(canvas, path);
    plutovg_canvas_clip(canvas);
}

float plutovg_canvas_add_glyph(plutovg_canvas_t* canvas, plutovg_codepoint_t codepoint, float x, float y)
{
    plutovg_state_t* state = canvas->state;
    if(state->font_face && state->font_size > 0.f)
        return plutovg_font_face_get_glyph_path(state->font_face, state->font_size, x, y, codepoint, canvas->path);
    return 0.f;
}

float plutovg_canvas_add_text(plutovg_canvas_t* canvas, const void* text, int length, plutovg_text_encoding_t encoding, float x, float y)
{
    plutovg_state_t* state = canvas->state;
    if(state->font_face == NULL || state->font_size <= 0.f)
        return 0.f;
    plutovg_text_iterator_t it;
    plutovg_text_iterator_init(&it, text, length, encoding);
    float advance_width = 0.f;
    while(plutovg_text_iterator_has_next(&it)) {
        plutovg_codepoint_t codepoint = plutovg_text_iterator_next(&it);
        advance_width += plutovg_font_face_get_glyph_path(state->font_face, state->font_size, x + advance_width, y, codepoint, canvas->path);
    }

    return advance_width;
}

float plutovg_canvas_fill_text(plutovg_canvas_t* canvas, const void* text, int length, plutovg_text_encoding_t encoding, float x, float y)
{
    plutovg_canvas_new_path(canvas);
    float advance_width = plutovg_canvas_add_text(canvas, text, length, encoding, x, y);
    plutovg_canvas_fill(canvas);
    return advance_width;
}

float plutovg_canvas_stroke_text(plutovg_canvas_t* canvas, const void* text, int length, plutovg_text_encoding_t encoding, float x, float y)
{
    plutovg_canvas_new_path(canvas);
    float advance_width = plutovg_canvas_add_text(canvas, text, length, encoding, x, y);
    plutovg_canvas_stroke(canvas);
    return advance_width;
}

float plutovg_canvas_clip_text(plutovg_canvas_t* canvas, const void* text, int length, plutovg_text_encoding_t encoding, float x, float y)
{
    plutovg_canvas_new_path(canvas);
    float advance_width = plutovg_canvas_add_text(canvas, text, length, encoding, x, y);
    plutovg_canvas_clip(canvas);
    return advance_width;
}

void plutovg_canvas_font_metrics(const plutovg_canvas_t* canvas, float* ascent, float* descent, float* line_gap, plutovg_rect_t* extents)
{
    plutovg_state_t* state = canvas->state;
    if(state->font_face && state->font_size > 0.f) {
        plutovg_font_face_get_metrics(state->font_face, state->font_size, ascent, descent, line_gap, extents);
        return;
    }

    if(ascent) *ascent = 0.f;
    if(descent) *descent = 0.f;
    if(line_gap) *line_gap = 0.f;
    if(extents) {
        extents->x = 0.f;
        extents->y = 0.f;
        extents->w = 0.f;
        extents->h = 0.f;
    }
}

void plutovg_canvas_glyph_metrics(plutovg_canvas_t* canvas, plutovg_codepoint_t codepoint, float* advance_width, float* left_side_bearing, plutovg_rect_t* extents)
{
    plutovg_state_t* state = canvas->state;
    if(state->font_face && state->font_size > 0.f) {
        plutovg_font_face_get_glyph_metrics(state->font_face, state->font_size, codepoint, advance_width, left_side_bearing, extents);
        return;
    }

    if(advance_width) *advance_width = 0.f;
    if(left_side_bearing) *left_side_bearing = 0.f;
    if(extents) {
        extents->x = 0.f;
        extents->y = 0.f;
        extents->w = 0.f;
        extents->h = 0.f;
    }
}

float plutovg_canvas_text_extents(plutovg_canvas_t* canvas, const void* text, int length, plutovg_text_encoding_t encoding, plutovg_rect_t* extents)
{
    plutovg_state_t* state = canvas->state;
    if(state->font_face && state->font_size > 0.f) {
        return plutovg_font_face_text_extents(state->font_face, state->font_size, text, length, encoding, extents);
    }

    if(extents) {
        extents->x = 0.f;
        extents->y = 0.f;
        extents->w = 0.f;
        extents->h = 0.f;
    }

    return 0.f;
}
