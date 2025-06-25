#include "plutovg-private.h"
#include "plutovg-utils.h"

#include "plutovg-ft-raster.h"
#include "plutovg-ft-stroker.h"

#include <limits.h>

void plutovg_span_buffer_init(plutovg_span_buffer_t* span_buffer)
{
    plutovg_array_init(span_buffer->spans);
    plutovg_span_buffer_reset(span_buffer);
}

void plutovg_span_buffer_init_rect(plutovg_span_buffer_t* span_buffer, int x, int y, int width, int height)
{
    plutovg_array_clear(span_buffer->spans);
    plutovg_array_ensure(span_buffer->spans, height);
    plutovg_span_t* spans = span_buffer->spans.data;
    for(int i = 0; i < height; i++) {
        spans[i].x = x;
        spans[i].y = y + i;
        spans[i].len = width;
        spans[i].coverage = 255;
    }

    span_buffer->x = x;
    span_buffer->y = y;
    span_buffer->w = width;
    span_buffer->h = height;
    span_buffer->spans.size = height;
}

void plutovg_span_buffer_reset(plutovg_span_buffer_t* span_buffer)
{
    plutovg_array_clear(span_buffer->spans);
    span_buffer->x = 0;
    span_buffer->y = 0;
    span_buffer->w = -1;
    span_buffer->h = -1;
}

void plutovg_span_buffer_destroy(plutovg_span_buffer_t* span_buffer)
{
    plutovg_array_destroy(span_buffer->spans);
}

void plutovg_span_buffer_copy(plutovg_span_buffer_t* span_buffer, const plutovg_span_buffer_t* source)
{
    plutovg_array_clear(span_buffer->spans);
    plutovg_array_append(span_buffer->spans, source->spans);
    span_buffer->x = source->x;
    span_buffer->y = source->y;
    span_buffer->w = source->w;
    span_buffer->h = source->h;
}

static void plutovg_span_buffer_update_extents(plutovg_span_buffer_t* span_buffer)
{
    if(span_buffer->w != -1 && span_buffer->h != -1)
        return;
    if(span_buffer->spans.size == 0) {
        span_buffer->x = 0;
        span_buffer->y = 0;
        span_buffer->w = 0;
        span_buffer->h = 0;
        return;
    }

    plutovg_span_t* spans = span_buffer->spans.data;
    int x1 = INT_MAX;
    int y1 = spans[0].y;
    int x2 = 0;
    int y2 = spans[span_buffer->spans.size - 1].y;
    for(int i = 0; i < span_buffer->spans.size; i++) {
        if(spans[i].x < x1) x1 = spans[i].x;
        if(spans[i].x + spans[i].len > x2) x2 = spans[i].x + spans[i].len;
    }

    span_buffer->x = x1;
    span_buffer->y = y1;
    span_buffer->w = x2 - x1;
    span_buffer->h = y2 - y1 + 1;
}

void plutovg_span_buffer_extents(plutovg_span_buffer_t* span_buffer, plutovg_rect_t* extents)
{
    plutovg_span_buffer_update_extents(span_buffer);
    extents->x = span_buffer->x;
    extents->y = span_buffer->y;
    extents->w = span_buffer->w;
    extents->h = span_buffer->h;
}

void plutovg_span_buffer_intersect(plutovg_span_buffer_t* span_buffer, const plutovg_span_buffer_t* a, const plutovg_span_buffer_t* b)
{
    plutovg_span_buffer_reset(span_buffer);
    plutovg_array_ensure(span_buffer->spans, plutovg_max(a->spans.size, b->spans.size));

    plutovg_span_t* a_spans = a->spans.data;
    plutovg_span_t* a_end = a_spans + a->spans.size;

    plutovg_span_t* b_spans = b->spans.data;
    plutovg_span_t* b_end = b_spans + b->spans.size;
    while(a_spans < a_end && b_spans < b_end) {
        if(b_spans->y > a_spans->y) {
            ++a_spans;
            continue;
        }

        if(a_spans->y != b_spans->y) {
            ++b_spans;
            continue;
        }

        int ax1 = a_spans->x;
        int ax2 = ax1 + a_spans->len;
        int bx1 = b_spans->x;
        int bx2 = bx1 + b_spans->len;
        if(bx1 < ax1 && bx2 < ax1) {
            ++b_spans;
            continue;
        }

        if(ax1 < bx1 && ax2 < bx1) {
            ++a_spans;
            continue;
        }

        int x = plutovg_max(ax1, bx1);
        int len = plutovg_min(ax2, bx2) - x;
        if(len) {
            plutovg_array_ensure(span_buffer->spans, 1);
            plutovg_span_t* span = span_buffer->spans.data + span_buffer->spans.size;
            span->x = x;
            span->len = len;
            span->y = a_spans->y;
            span->coverage = plutovg_div255(a_spans->coverage * b_spans->coverage);
            span_buffer->spans.size += 1;
        }

        if(ax2 < bx2) {
            ++a_spans;
        } else {
            ++b_spans;
        }
    }
}

#define ALIGN_SIZE(size) (((size) + 7ul) & ~7ul)
static PVG_FT_Outline* ft_outline_create(int points, int contours)
{
    size_t points_size = ALIGN_SIZE((points + contours) * sizeof(PVG_FT_Vector));
    size_t tags_size = ALIGN_SIZE((points + contours) * sizeof(char));
    size_t contours_size = ALIGN_SIZE(contours * sizeof(int));
    size_t contours_flag_size = ALIGN_SIZE(contours * sizeof(char));
    PVG_FT_Outline* outline = malloc(points_size + tags_size + contours_size + contours_flag_size + sizeof(PVG_FT_Outline));

    PVG_FT_Byte* outline_data = (PVG_FT_Byte*)(outline + 1);
    outline->points = (PVG_FT_Vector*)(outline_data);
    outline->tags = (char*)(outline_data + points_size);
    outline->contours = (int*)(outline_data + points_size + tags_size);
    outline->contours_flag = (char*)(outline_data + points_size + tags_size + contours_size);
    outline->n_points = 0;
    outline->n_contours = 0;
    outline->flags = 0x0;
    return outline;
}

static void ft_outline_destroy(PVG_FT_Outline* outline)
{
    free(outline);
}

#define FT_COORD(x) (PVG_FT_Pos)((x) * 64)
static void ft_outline_move_to(PVG_FT_Outline* ft, float x, float y)
{
    ft->points[ft->n_points].x = FT_COORD(x);
    ft->points[ft->n_points].y = FT_COORD(y);
    ft->tags[ft->n_points] = PVG_FT_CURVE_TAG_ON;
    if(ft->n_points) {
        ft->contours[ft->n_contours] = ft->n_points - 1;
        ft->n_contours++;
    }

    ft->contours_flag[ft->n_contours] = 1;
    ft->n_points++;
}

static void ft_outline_line_to(PVG_FT_Outline* ft, float x, float y)
{
    ft->points[ft->n_points].x = FT_COORD(x);
    ft->points[ft->n_points].y = FT_COORD(y);
    ft->tags[ft->n_points] = PVG_FT_CURVE_TAG_ON;
    ft->n_points++;
}

static void ft_outline_cubic_to(PVG_FT_Outline* ft, float x1, float y1, float x2, float y2, float x3, float y3)
{
    ft->points[ft->n_points].x = FT_COORD(x1);
    ft->points[ft->n_points].y = FT_COORD(y1);
    ft->tags[ft->n_points] = PVG_FT_CURVE_TAG_CUBIC;
    ft->n_points++;

    ft->points[ft->n_points].x = FT_COORD(x2);
    ft->points[ft->n_points].y = FT_COORD(y2);
    ft->tags[ft->n_points] = PVG_FT_CURVE_TAG_CUBIC;
    ft->n_points++;

    ft->points[ft->n_points].x = FT_COORD(x3);
    ft->points[ft->n_points].y = FT_COORD(y3);
    ft->tags[ft->n_points] = PVG_FT_CURVE_TAG_ON;
    ft->n_points++;
}

static void ft_outline_close(PVG_FT_Outline* ft)
{
    ft->contours_flag[ft->n_contours] = 0;
    int index = ft->n_contours ? ft->contours[ft->n_contours - 1] + 1 : 0;
    if(index == ft->n_points)
        return;
    ft->points[ft->n_points].x = ft->points[index].x;
    ft->points[ft->n_points].y = ft->points[index].y;
    ft->tags[ft->n_points] = PVG_FT_CURVE_TAG_ON;
    ft->n_points++;
}

static void ft_outline_end(PVG_FT_Outline* ft)
{
    if(ft->n_points) {
        ft->contours[ft->n_contours] = ft->n_points - 1;
        ft->n_contours++;
    }
}

static PVG_FT_Outline* ft_outline_convert_stroke(const plutovg_path_t* path, const plutovg_matrix_t* matrix, const plutovg_stroke_data_t* stroke_data);

static PVG_FT_Outline* ft_outline_convert(const plutovg_path_t* path, const plutovg_matrix_t* matrix, const plutovg_stroke_data_t* stroke_data)
{
    if(stroke_data) {
        return ft_outline_convert_stroke(path, matrix, stroke_data);
    }

    plutovg_path_iterator_t it;
    plutovg_path_iterator_init(&it, path);

    plutovg_point_t points[3];
    PVG_FT_Outline* outline = ft_outline_create(path->num_points, path->num_contours);
    while(plutovg_path_iterator_has_next(&it)) {
        switch(plutovg_path_iterator_next(&it, points)) {
        case PLUTOVG_PATH_COMMAND_MOVE_TO:
            plutovg_matrix_map_points(matrix, points, points, 1);
            ft_outline_move_to(outline, points[0].x, points[0].y);
            break;
        case PLUTOVG_PATH_COMMAND_LINE_TO:
            plutovg_matrix_map_points(matrix, points, points, 1);
            ft_outline_line_to(outline, points[0].x, points[0].y);
            break;
        case PLUTOVG_PATH_COMMAND_CUBIC_TO:
            plutovg_matrix_map_points(matrix, points, points, 3);
            ft_outline_cubic_to(outline, points[0].x, points[0].y, points[1].x, points[1].y, points[2].x, points[2].y);
            break;
        case PLUTOVG_PATH_COMMAND_CLOSE:
            ft_outline_close(outline);
            break;
        }
    }

    ft_outline_end(outline);
    return outline;
}

static PVG_FT_Outline* ft_outline_convert_dash(const plutovg_path_t* path, const plutovg_matrix_t* matrix, const plutovg_stroke_dash_t* stroke_dash)
{
    if(stroke_dash->array.size == 0)
        return ft_outline_convert(path, matrix, NULL);
    plutovg_path_t* dashed = plutovg_path_clone_dashed(path, stroke_dash->offset, stroke_dash->array.data, stroke_dash->array.size);
    PVG_FT_Outline* outline = ft_outline_convert(dashed, matrix, NULL);
    plutovg_path_destroy(dashed);
    return outline;
}

static PVG_FT_Outline* ft_outline_convert_stroke(const plutovg_path_t* path, const plutovg_matrix_t* matrix, const plutovg_stroke_data_t* stroke_data)
{
    double scale_x = sqrt(matrix->a * matrix->a + matrix->b * matrix->b);
    double scale_y = sqrt(matrix->c * matrix->c + matrix->d * matrix->d);

    double scale = hypot(scale_x, scale_y) / PLUTOVG_SQRT2;
    double width = stroke_data->style.width * scale;

    PVG_FT_Fixed ftWidth = (PVG_FT_Fixed)(width * 0.5 * (1 << 6));
    PVG_FT_Fixed ftMiterLimit = (PVG_FT_Fixed)(stroke_data->style.miter_limit * (1 << 16));

    PVG_FT_Stroker_LineCap ftCap;
    switch(stroke_data->style.cap) {
    case PLUTOVG_LINE_CAP_SQUARE:
        ftCap = PVG_FT_STROKER_LINECAP_SQUARE;
        break;
    case PLUTOVG_LINE_CAP_ROUND:
        ftCap = PVG_FT_STROKER_LINECAP_ROUND;
        break;
    default:
        ftCap = PVG_FT_STROKER_LINECAP_BUTT;
        break;
    }

    PVG_FT_Stroker_LineJoin ftJoin;
    switch(stroke_data->style.join) {
    case PLUTOVG_LINE_JOIN_BEVEL:
        ftJoin = PVG_FT_STROKER_LINEJOIN_BEVEL;
        break;
    case PLUTOVG_LINE_JOIN_ROUND:
        ftJoin = PVG_FT_STROKER_LINEJOIN_ROUND;
        break;
    default:
        ftJoin = PVG_FT_STROKER_LINEJOIN_MITER_FIXED;
        break;
    }

    PVG_FT_Stroker stroker;
    PVG_FT_Stroker_New(&stroker);
    PVG_FT_Stroker_Set(stroker, ftWidth, ftCap, ftJoin, ftMiterLimit);

    PVG_FT_Outline* outline = ft_outline_convert_dash(path, matrix, &stroke_data->dash);
    PVG_FT_Stroker_ParseOutline(stroker, outline);

    PVG_FT_UInt points;
    PVG_FT_UInt contours;
    PVG_FT_Stroker_GetCounts(stroker, &points, &contours);

    PVG_FT_Outline* stroke_outline = ft_outline_create(points, contours);
    PVG_FT_Stroker_Export(stroker, stroke_outline);

    PVG_FT_Stroker_Done(stroker);
    ft_outline_destroy(outline);
    return stroke_outline;
}

static void spans_generation_callback(int count, const PVG_FT_Span* spans, void* user)
{
    plutovg_span_buffer_t* span_buffer = (plutovg_span_buffer_t*)(user);
    plutovg_array_append_data(span_buffer->spans, spans, count);
}

void plutovg_rasterize(plutovg_span_buffer_t* span_buffer, const plutovg_path_t* path, const plutovg_matrix_t* matrix, const plutovg_rect_t* clip_rect, const plutovg_stroke_data_t* stroke_data, plutovg_fill_rule_t winding)
{
    PVG_FT_Raster_Params params;
    params.flags = PVG_FT_RASTER_FLAG_DIRECT | PVG_FT_RASTER_FLAG_AA;
    params.gray_spans = spans_generation_callback;
    params.user = span_buffer;
    if(clip_rect) {
        params.flags |= PVG_FT_RASTER_FLAG_CLIP;
        params.clip_box.xMin = (PVG_FT_Pos)clip_rect->x;
        params.clip_box.yMin = (PVG_FT_Pos)clip_rect->y;
        params.clip_box.xMax = (PVG_FT_Pos)(clip_rect->x + clip_rect->w);
        params.clip_box.yMax = (PVG_FT_Pos)(clip_rect->y + clip_rect->h);
    }

    PVG_FT_Outline* outline = ft_outline_convert(path, matrix, stroke_data);
    if(stroke_data) {
        outline->flags = PVG_FT_OUTLINE_NONE;
    } else {
        switch(winding) {
        case PLUTOVG_FILL_RULE_EVEN_ODD:
            outline->flags = PVG_FT_OUTLINE_EVEN_ODD_FILL;
            break;
        default:
            outline->flags = PVG_FT_OUTLINE_NONE;
            break;
        }
    }

    params.source = outline;
    plutovg_span_buffer_reset(span_buffer);
    PVG_FT_Raster_Render(&params);
    ft_outline_destroy(outline);
}
