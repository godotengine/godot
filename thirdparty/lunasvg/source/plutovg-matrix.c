#include "plutovg.h"
#include "plutovg-utils.h"

void plutovg_matrix_init(plutovg_matrix_t* matrix, float a, float b, float c, float d, float e, float f)
{
    matrix->a = a; matrix->b = b;
    matrix->c = c; matrix->d = d;
    matrix->e = e; matrix->f = f;
}

void plutovg_matrix_init_identity(plutovg_matrix_t* matrix)
{
    matrix->a = 1; matrix->b = 0;
    matrix->c = 0; matrix->d = 1;
    matrix->e = 0; matrix->f = 0;
}

void plutovg_matrix_init_translate(plutovg_matrix_t* matrix, float tx, float ty)
{
    plutovg_matrix_init(matrix, 1, 0, 0, 1, tx, ty);
}

void plutovg_matrix_init_scale(plutovg_matrix_t* matrix, float sx, float sy)
{
    plutovg_matrix_init(matrix, sx, 0, 0, sy, 0, 0);
}

void plutovg_matrix_init_rotate(plutovg_matrix_t* matrix, float angle)
{
    float c = cosf(angle);
    float s = sinf(angle);
    plutovg_matrix_init(matrix, c, s, -s, c, 0, 0);
}

void plutovg_matrix_init_shear(plutovg_matrix_t* matrix, float shx, float shy)
{
    plutovg_matrix_init(matrix, 1, tanf(shy), tanf(shx), 1, 0, 0);
}

void plutovg_matrix_translate(plutovg_matrix_t* matrix, float tx, float ty)
{
    plutovg_matrix_t m;
    plutovg_matrix_init_translate(&m, tx, ty);
    plutovg_matrix_multiply(matrix, &m, matrix);
}

void plutovg_matrix_scale(plutovg_matrix_t* matrix, float sx, float sy)
{
    plutovg_matrix_t m;
    plutovg_matrix_init_scale(&m, sx, sy);
    plutovg_matrix_multiply(matrix, &m, matrix);
}

void plutovg_matrix_rotate(plutovg_matrix_t* matrix, float angle)
{
    plutovg_matrix_t m;
    plutovg_matrix_init_rotate(&m, angle);
    plutovg_matrix_multiply(matrix, &m, matrix);
}

void plutovg_matrix_shear(plutovg_matrix_t* matrix, float shx, float shy)
{
    plutovg_matrix_t m;
    plutovg_matrix_init_shear(&m, shx, shy);
    plutovg_matrix_multiply(matrix, &m, matrix);
}

void plutovg_matrix_multiply(plutovg_matrix_t* matrix, const plutovg_matrix_t* left, const plutovg_matrix_t* right)
{
    float a = left->a * right->a + left->b * right->c;
    float b = left->a * right->b + left->b * right->d;
    float c = left->c * right->a + left->d * right->c;
    float d = left->c * right->b + left->d * right->d;
    float e = left->e * right->a + left->f * right->c + right->e;
    float f = left->e * right->b + left->f * right->d + right->f;
    plutovg_matrix_init(matrix, a, b, c, d, e, f);
}

bool plutovg_matrix_invert(const plutovg_matrix_t* matrix, plutovg_matrix_t* inverse)
{
    float det = (matrix->a * matrix->d - matrix->b * matrix->c);
    if(det == 0.f)
        return false;
    if(inverse) {
        float inv_det = 1.f / det;
        float a = matrix->a * inv_det;
        float b = matrix->b * inv_det;
        float c = matrix->c * inv_det;
        float d = matrix->d * inv_det;
        float e = (matrix->c * matrix->f - matrix->d * matrix->e) * inv_det;
        float f = (matrix->b * matrix->e - matrix->a * matrix->f) * inv_det;
        plutovg_matrix_init(inverse, d, -b, -c, a, e, f);
    }

    return true;
}

void plutovg_matrix_map(const plutovg_matrix_t* matrix, float x, float y, float* xx, float* yy)
{
    *xx = x * matrix->a + y * matrix->c + matrix->e;
    *yy = x * matrix->b + y * matrix->d + matrix->f;
}

void plutovg_matrix_map_point(const plutovg_matrix_t* matrix, const plutovg_point_t* src, plutovg_point_t* dst)
{
    plutovg_matrix_map(matrix, src->x, src->y, &dst->x, &dst->y);
}

void plutovg_matrix_map_points(const plutovg_matrix_t* matrix, const plutovg_point_t* src, plutovg_point_t* dst, int count)
{
    for(int i = 0; i < count; ++i) {
        plutovg_matrix_map_point(matrix, &src[i], &dst[i]);
    }
}

void plutovg_matrix_map_rect(const plutovg_matrix_t* matrix, const plutovg_rect_t* src, plutovg_rect_t* dst)
{
    plutovg_point_t p[4];
    p[0].x = src->x;
    p[0].y = src->y;
    p[1].x = src->x + src->w;
    p[1].y = src->y;
    p[2].x = src->x + src->w;
    p[2].y = src->y + src->h;
    p[3].x = src->x;
    p[3].y = src->y + src->h;
    plutovg_matrix_map_points(matrix, p, p, 4);

    float l = p[0].x;
    float t = p[0].y;
    float r = p[0].x;
    float b = p[0].y;

    for(int i = 1; i < 4; i++) {
        if(p[i].x < l) l = p[i].x;
        if(p[i].x > r) r = p[i].x;
        if(p[i].y < t) t = p[i].y;
        if(p[i].y > b) b = p[i].y;
    }

    dst->x = l;
    dst->y = t;
    dst->w = r - l;
    dst->h = b - t;
}

static int parse_matrix_parameters(const char** begin, const char* end, float values[6], int required, int optional)
{
    if(!plutovg_skip_ws_and_delim(begin, end, '('))
        return 0;
    int count = 0;
    int max_count = required + optional;
    for(; count < max_count; ++count) {
        if(!plutovg_parse_number(begin, end, values + count))
            break;
        plutovg_skip_ws_or_comma(begin, end);
    }

    if((count == required || count == max_count) && plutovg_skip_delim(begin, end, ')'))
        return count;
    return 0;
}

bool plutovg_matrix_parse(plutovg_matrix_t* matrix, const char* data, int length)
{
    float values[6];
    plutovg_matrix_init_identity(matrix);
    if(length == -1)
        length = strlen(data);
    const char* it = data;
    const char* end = it + length;
    plutovg_skip_ws(&it, end);
    while(it < end) {
        if(plutovg_skip_string(&it, end, "matrix")) {
            int count = parse_matrix_parameters(&it, end, values, 6, 0);
            if(count == 0)
                return false;
            plutovg_matrix_t m = { values[0], values[1], values[2], values[3], values[4], values[5] };
            plutovg_matrix_multiply(matrix, &m, matrix);
        } else if(plutovg_skip_string(&it, end, "translate")) {
            int count = parse_matrix_parameters(&it, end, values, 1, 1);
            if(count == 0)
                return false;
            if(count == 1) {
                plutovg_matrix_translate(matrix, values[0], 0);
            } else {
                plutovg_matrix_translate(matrix, values[0], values[1]);
            }
        } else if(plutovg_skip_string(&it, end, "scale")) {
            int count = parse_matrix_parameters(&it, end, values, 1, 1);
            if(count == 0)
                return false;
            if(count == 1) {
                plutovg_matrix_scale(matrix, values[0], values[0]);
            } else {
                plutovg_matrix_scale(matrix, values[0], values[1]);
            }
        } else if(plutovg_skip_string(&it, end, "rotate")) {
            int count = parse_matrix_parameters(&it, end, values, 1, 2);
            if(count == 0)
                return false;
            if(count == 3)
                plutovg_matrix_translate(matrix, values[1], values[2]);
            plutovg_matrix_rotate(matrix, PLUTOVG_DEG2RAD(values[0]));
            if(count == 3) {
                plutovg_matrix_translate(matrix, -values[1], -values[2]);
            }
        } else if(plutovg_skip_string(&it, end, "skewX")) {
            int count = parse_matrix_parameters(&it, end, values, 1, 0);
            if(count == 0)
                return false;
            plutovg_matrix_shear(matrix, PLUTOVG_DEG2RAD(values[0]), 0);
        } else if(plutovg_skip_string(&it, end, "skewY")) {
            int count = parse_matrix_parameters(&it, end, values, 1, 0);
            if(count == 0)
                return false;
            plutovg_matrix_shear(matrix, 0, PLUTOVG_DEG2RAD(values[0]));
        } else {
            return false;
        }

        plutovg_skip_ws_or_comma(&it, end);
    }

    return true;
}
