#include "plutovg-private.h"
#include "plutovg-utils.h"

#include <assert.h>

void plutovg_path_iterator_init(plutovg_path_iterator_t* it, const plutovg_path_t* path)
{
    it->elements = path->elements.data;
    it->size = path->elements.size;
    it->index = 0;
}

bool plutovg_path_iterator_has_next(const plutovg_path_iterator_t* it)
{
    return it->index < it->size;
}

plutovg_path_command_t plutovg_path_iterator_next(plutovg_path_iterator_t* it, plutovg_point_t points[3])
{
    const plutovg_path_element_t* elements = it->elements + it->index;
    switch(elements[0].header.command) {
    case PLUTOVG_PATH_COMMAND_MOVE_TO:
    case PLUTOVG_PATH_COMMAND_LINE_TO:
    case PLUTOVG_PATH_COMMAND_CLOSE:
        points[0] = elements[1].point;
        break;
    case PLUTOVG_PATH_COMMAND_CUBIC_TO:
        points[0] = elements[1].point;
        points[1] = elements[2].point;
        points[2] = elements[3].point;
        break;
    }

    it->index += elements[0].header.length;
    return elements[0].header.command;
}

plutovg_path_t* plutovg_path_create(void)
{
    plutovg_path_t* path = malloc(sizeof(plutovg_path_t));
    path->ref_count = 1;
    path->num_curves = 0;
    path->num_contours = 0;
    path->num_points = 0;
    path->start_point.x = 0;
    path->start_point.y = 0;
    plutovg_array_init(path->elements);
    return path;
}

plutovg_path_t* plutovg_path_reference(plutovg_path_t* path)
{
    if(path == NULL)
        return NULL;
    ++path->ref_count;
    return path;
}

void plutovg_path_destroy(plutovg_path_t* path)
{
    if(path == NULL)
        return;
    if(--path->ref_count == 0) {
        plutovg_array_destroy(path->elements);
        free(path);
    }
}

int plutovg_path_get_reference_count(const plutovg_path_t* path)
{
    if(path)
        return path->ref_count;
    return 0;
}

int plutovg_path_get_elements(const plutovg_path_t* path, const plutovg_path_element_t** elements)
{
    if(elements)
        *elements = path->elements.data;
    return path->elements.size;
}

static plutovg_path_element_t* plutovg_path_add_command(plutovg_path_t* path, plutovg_path_command_t command, int npoints)
{
    const int length = npoints + 1;
    plutovg_array_ensure(path->elements, length);
    plutovg_path_element_t* elements = path->elements.data + path->elements.size;
    elements->header.command = command;
    elements->header.length = length;
    path->elements.size += length;
    path->num_points += npoints;
    return elements + 1;
}

void plutovg_path_move_to(plutovg_path_t* path, float x, float y)
{
    plutovg_path_element_t* elements = plutovg_path_add_command(path, PLUTOVG_PATH_COMMAND_MOVE_TO, 1);
    elements[0].point.x = x;
    elements[0].point.y = y;

    path->start_point.x = x;
    path->start_point.y = y;
    path->num_contours += 1;
}

void plutovg_path_line_to(plutovg_path_t* path, float x, float y)
{
    if(path->elements.size == 0)
        plutovg_path_move_to(path, 0, 0);
    plutovg_path_element_t* elements = plutovg_path_add_command(path, PLUTOVG_PATH_COMMAND_LINE_TO, 1);
    elements[0].point.x = x;
    elements[0].point.y = y;
}

void plutovg_path_quad_to(plutovg_path_t* path, float x1, float y1, float x2, float y2)
{
    float current_x, current_y;
    plutovg_path_get_current_point(path, &current_x, &current_y);
    float cp1x = 2.f / 3.f * x1 + 1.f / 3.f * current_x;
    float cp1y = 2.f / 3.f * y1 + 1.f / 3.f * current_y;
    float cp2x = 2.f / 3.f * x1 + 1.f / 3.f * x2;
    float cp2y = 2.f / 3.f * y1 + 1.f / 3.f * y2;
    plutovg_path_cubic_to(path, cp1x, cp1y, cp2x, cp2y, x2, y2);
}

void plutovg_path_cubic_to(plutovg_path_t* path, float x1, float y1, float x2, float y2, float x3, float y3)
{
    if(path->elements.size == 0)
        plutovg_path_move_to(path, 0, 0);
    plutovg_path_element_t* elements = plutovg_path_add_command(path, PLUTOVG_PATH_COMMAND_CUBIC_TO, 3);
    elements[0].point.x = x1;
    elements[0].point.y = y1;
    elements[1].point.x = x2;
    elements[1].point.y = y2;
    elements[2].point.x = x3;
    elements[2].point.y = y3;
    path->num_curves += 1;
}

void plutovg_path_arc_to(plutovg_path_t* path, float rx, float ry, float angle, bool large_arc_flag, bool sweep_flag, float x, float y)
{
    float current_x, current_y;
    plutovg_path_get_current_point(path, &current_x, &current_y);
    if(rx == 0.f || ry == 0.f || (current_x == x && current_y == y)) {
        plutovg_path_line_to(path, x, y);
        return;
    }

    if(rx < 0) rx = -rx;
    if(ry < 0) ry = -ry;

    float dx = current_x - x;
    float dy = current_y - y;

    dx *= 0.5f;
    dy *= 0.5f;

    plutovg_matrix_t matrix;
    plutovg_matrix_init_rotate(&matrix, -angle);
    plutovg_matrix_map(&matrix, dx, dy, &dx, &dy);

    float rxrx = rx * rx;
    float ryry = ry * ry;
    float dxdx = dx * dx;
    float dydy = dy * dy;
    float radius = dxdx / rxrx + dydy / ryry;
    if(radius > 1.f) {
        rx *= sqrtf(radius);
        ry *= sqrtf(radius);
    }

    plutovg_matrix_init_scale(&matrix, 1.f / rx, 1.f / ry);
    plutovg_matrix_rotate(&matrix, -angle);

    float x1, y1;
    float x2, y2;
    plutovg_matrix_map(&matrix, current_x, current_y, &x1, &y1);
    plutovg_matrix_map(&matrix, x, y, &x2, &y2);

    float dx1 = x2 - x1;
    float dy1 = y2 - y1;
    float d = dx1 * dx1 + dy1 * dy1;
    float scale_sq = 1 / d - 0.25f;
    if(scale_sq < 0) scale_sq = 0;
    float scale = sqrtf(scale_sq);
    if(sweep_flag == large_arc_flag)
        scale = -scale;
    dx1 *= scale;
    dy1 *= scale;

    float cx1 = 0.5f * (x1 + x2) - dy1;
    float cy1 = 0.5f * (y1 + y2) + dx1;

    float th1 = atan2f(y1 - cy1, x1 - cx1);
    float th2 = atan2f(y2 - cy1, x2 - cx1);
    float th_arc = th2 - th1;
    if(th_arc < 0 && sweep_flag)
        th_arc += PLUTOVG_TWO_PI;
    else if(th_arc > 0 && !sweep_flag)
        th_arc -= PLUTOVG_TWO_PI;
    plutovg_matrix_init_rotate(&matrix, angle);
    plutovg_matrix_scale(&matrix, rx, ry);
    int segments = (int)(ceilf(fabsf(th_arc / (PLUTOVG_HALF_PI + 0.001f))));
    for(int i = 0; i < segments; i++) {
        float th_start = th1 + i * th_arc / segments;
        float th_end = th1 + (i + 1) * th_arc / segments;
        float t = (8.f / 6.f) * tanf(0.25f * (th_end - th_start));

        float x3 = cosf(th_end) + cx1;
        float y3 = sinf(th_end) + cy1;

        float cp2x = x3 + t * sinf(th_end);
        float cp2y = y3 - t * cosf(th_end);

        float cp1x = cosf(th_start) - t * sinf(th_start);
        float cp1y = sinf(th_start) + t * cosf(th_start);

        cp1x += cx1;
        cp1y += cy1;

        plutovg_matrix_map(&matrix, cp1x, cp1y, &cp1x, &cp1y);
        plutovg_matrix_map(&matrix, cp2x, cp2y, &cp2x, &cp2y);
        plutovg_matrix_map(&matrix, x3, y3, &x3, &y3);

        plutovg_path_cubic_to(path, cp1x, cp1y, cp2x, cp2y, x3, y3);
    }
}

void plutovg_path_close(plutovg_path_t* path)
{
    if(path->elements.size == 0)
        return;
    plutovg_path_element_t* elements = plutovg_path_add_command(path, PLUTOVG_PATH_COMMAND_CLOSE, 1);
    elements[0].point.x = path->start_point.x;
    elements[0].point.y = path->start_point.y;
}

void plutovg_path_get_current_point(const plutovg_path_t* path, float* x, float* y)
{
    float xx = 0.f;
    float yy = 0.f;
    if(path->num_points > 0) {
        xx = path->elements.data[path->elements.size - 1].point.x;
        yy = path->elements.data[path->elements.size - 1].point.y;
    }

    if(x) *x = xx;
    if(y) *y = yy;
}

void plutovg_path_reserve(plutovg_path_t* path, int count)
{
    plutovg_array_ensure(path->elements, count);
}

void plutovg_path_reset(plutovg_path_t* path)
{
    plutovg_array_clear(path->elements);
    path->start_point.x = 0;
    path->start_point.y = 0;
    path->num_contours = 0;
    path->num_points = 0;
    path->num_curves = 0;
}

void plutovg_path_add_rect(plutovg_path_t* path, float x, float y, float w, float h)
{
    plutovg_path_reserve(path, 6 * 2);
    plutovg_path_move_to(path, x, y);
    plutovg_path_line_to(path, x + w, y);
    plutovg_path_line_to(path, x + w, y + h);
    plutovg_path_line_to(path, x, y + h);
    plutovg_path_line_to(path, x, y);
    plutovg_path_close(path);
}

void plutovg_path_add_round_rect(plutovg_path_t* path, float x, float y, float w, float h, float rx, float ry)
{
    rx = plutovg_min(rx, w * 0.5f);
    ry = plutovg_min(ry, h * 0.5f);
    if(rx == 0.f && ry == 0.f) {
        plutovg_path_add_rect(path, x, y, w, h);
        return;
    }

    float right = x + w;
    float bottom = y + h;

    float cpx = rx * PLUTOVG_KAPPA;
    float cpy = ry * PLUTOVG_KAPPA;

    plutovg_path_reserve(path, 6 * 2 + 4 * 4);
    plutovg_path_move_to(path, x, y+ry);
    plutovg_path_cubic_to(path, x, y+ry-cpy, x+rx-cpx, y, x+rx, y);
    plutovg_path_line_to(path, right-rx, y);
    plutovg_path_cubic_to(path, right-rx+cpx, y, right, y+ry-cpy, right, y+ry);
    plutovg_path_line_to(path, right, bottom-ry);
    plutovg_path_cubic_to(path, right, bottom-ry+cpy, right-rx+cpx, bottom, right-rx, bottom);
    plutovg_path_line_to(path, x+rx, bottom);
    plutovg_path_cubic_to(path, x+rx-cpx, bottom, x, bottom-ry+cpy, x, bottom-ry);
    plutovg_path_line_to(path, x, y+ry);
    plutovg_path_close(path);
}

void plutovg_path_add_ellipse(plutovg_path_t* path, float cx, float cy, float rx, float ry)
{
    float left = cx - rx;
    float top = cy - ry;
    float right = cx + rx;
    float bottom = cy + ry;

    float cpx = rx * PLUTOVG_KAPPA;
    float cpy = ry * PLUTOVG_KAPPA;

    plutovg_path_reserve(path, 2 * 2 + 4 * 4);
    plutovg_path_move_to(path, cx, top);
    plutovg_path_cubic_to(path, cx+cpx, top, right, cy-cpy, right, cy);
    plutovg_path_cubic_to(path, right, cy+cpy, cx+cpx, bottom, cx, bottom);
    plutovg_path_cubic_to(path, cx-cpx, bottom, left, cy+cpy, left, cy);
    plutovg_path_cubic_to(path, left, cy-cpy, cx-cpx, top, cx, top);
    plutovg_path_close(path);
}

void plutovg_path_add_circle(plutovg_path_t* path, float cx, float cy, float r)
{
    plutovg_path_add_ellipse(path, cx, cy, r, r);
}

void plutovg_path_add_arc(plutovg_path_t* path, float cx, float cy, float r, float a0, float a1, bool ccw)
{
    float da = a1 - a0;
    if(fabsf(da) > PLUTOVG_TWO_PI) {
        da = PLUTOVG_TWO_PI;
    } else if(da != 0.f && ccw != (da < 0.f)) {
        da += PLUTOVG_TWO_PI * (ccw ? -1 : 1);
    }

    int seg_n = (int)(ceilf(fabsf(da) / PLUTOVG_HALF_PI));
    if(seg_n == 0)
        return;
    float a = a0;
    float ax = cx + cosf(a) * r;
    float ay = cy + sinf(a) * r;

    float seg_a = da / seg_n;
    float d = (seg_a / PLUTOVG_HALF_PI) * PLUTOVG_KAPPA * r;
    float dx = -sinf(a) * d;
    float dy = cosf(a) * d;

    plutovg_path_reserve(path, 2 + 4 * seg_n);
    if(path->elements.size == 0) {
        plutovg_path_move_to(path, ax, ay);
    } else {
        plutovg_path_line_to(path, ax, ay);
    }

    for(int i = 0; i < seg_n; i++) {
        float cp1x = ax + dx;
        float cp1y = ay + dy;

        a += seg_a;
        ax = cx + cosf(a) * r;
        ay = cy + sinf(a) * r;

        dx = -sinf(a) * d;
        dy = cosf(a) * d;

        float cp2x = ax - dx;
        float cp2y = ay - dy;

        plutovg_path_cubic_to(path, cp1x, cp1y, cp2x, cp2y, ax, ay);
    }
}

void plutovg_path_transform(plutovg_path_t* path, const plutovg_matrix_t* matrix)
{
    plutovg_path_element_t* elements = path->elements.data;
    for(int i = 0; i < path->elements.size; i += elements[i].header.length) {
        switch(elements[i].header.command) {
        case PLUTOVG_PATH_COMMAND_MOVE_TO:
        case PLUTOVG_PATH_COMMAND_LINE_TO:
        case PLUTOVG_PATH_COMMAND_CLOSE:
            plutovg_matrix_map_point(matrix, &elements[i + 1].point, &elements[i + 1].point);
            break;
        case PLUTOVG_PATH_COMMAND_CUBIC_TO:
            plutovg_matrix_map_point(matrix, &elements[i + 1].point, &elements[i + 1].point);
            plutovg_matrix_map_point(matrix, &elements[i + 2].point, &elements[i + 2].point);
            plutovg_matrix_map_point(matrix, &elements[i + 3].point, &elements[i + 3].point);
            break;
        }
    }
}

void plutovg_path_add_path(plutovg_path_t* path, const plutovg_path_t* source, const plutovg_matrix_t* matrix)
{
    if(matrix == NULL) {
        plutovg_array_append(path->elements, source->elements);
        path->num_curves += source->num_curves;
        path->num_contours += source->num_contours;
        path->num_points += source->num_points;
        path->start_point = source->start_point;
        return;
    }

    plutovg_path_iterator_t it;
    plutovg_path_iterator_init(&it, source);

    plutovg_point_t points[3];
    plutovg_array_ensure(path->elements, source->elements.size);
    while(plutovg_path_iterator_has_next(&it)) {
        switch(plutovg_path_iterator_next(&it, points)) {
        case PLUTOVG_PATH_COMMAND_MOVE_TO:
            plutovg_matrix_map_points(matrix, points, points, 1);
            plutovg_path_move_to(path, points[0].x, points[0].y);
            break;
        case PLUTOVG_PATH_COMMAND_LINE_TO:
            plutovg_matrix_map_points(matrix, points, points, 1);
            plutovg_path_line_to(path, points[0].x, points[0].y);
            break;
        case PLUTOVG_PATH_COMMAND_CUBIC_TO:
            plutovg_matrix_map_points(matrix, points, points, 3);
            plutovg_path_cubic_to(path, points[0].x, points[0].y, points[1].x, points[1].y, points[2].x, points[2].y);
            break;
        case PLUTOVG_PATH_COMMAND_CLOSE:
            plutovg_path_close(path);
            break;
        }
    }
}

void plutovg_path_traverse(const plutovg_path_t* path, plutovg_path_traverse_func_t traverse_func, void* closure)
{
    plutovg_path_iterator_t it;
    plutovg_path_iterator_init(&it, path);

    plutovg_point_t points[3];
    while(plutovg_path_iterator_has_next(&it)) {
        switch(plutovg_path_iterator_next(&it, points)) {
        case PLUTOVG_PATH_COMMAND_MOVE_TO:
            traverse_func(closure, PLUTOVG_PATH_COMMAND_MOVE_TO, points, 1);
            break;
        case PLUTOVG_PATH_COMMAND_LINE_TO:
            traverse_func(closure, PLUTOVG_PATH_COMMAND_LINE_TO, points, 1);
            break;
        case PLUTOVG_PATH_COMMAND_CUBIC_TO:
            traverse_func(closure, PLUTOVG_PATH_COMMAND_CUBIC_TO, points, 3);
            break;
        case PLUTOVG_PATH_COMMAND_CLOSE:
            traverse_func(closure, PLUTOVG_PATH_COMMAND_CLOSE, points, 1);
            break;
        }
    }
}

typedef struct {
    float x1; float y1;
    float x2; float y2;
    float x3; float y3;
    float x4; float y4;
} bezier_t;

static inline void split_bezier(const bezier_t* b, bezier_t* first, bezier_t* second)
{
    float c = (b->x2 + b->x3) * 0.5f;
    first->x2 = (b->x1 + b->x2) * 0.5f;
    second->x3 = (b->x3 + b->x4) * 0.5f;
    first->x1 = b->x1;
    second->x4 = b->x4;
    first->x3 = (first->x2 + c) * 0.5f;
    second->x2 = (second->x3 + c) * 0.5f;
    first->x4 = second->x1 = (first->x3 + second->x2) * 0.5f;

    c = (b->y2 + b->y3) * 0.5f;
    first->y2 = (b->y1 + b->y2) * 0.5f;
    second->y3 = (b->y3 + b->y4) * 0.5f;
    first->y1 = b->y1;
    second->y4 = b->y4;
    first->y3 = (first->y2 + c) * 0.5f;
    second->y2 = (second->y3 + c) * 0.5f;
    first->y4 = second->y1 = (first->y3 + second->y2) * 0.5f;
}

void plutovg_path_traverse_flatten(const plutovg_path_t* path, plutovg_path_traverse_func_t traverse_func, void* closure)
{
    if(path->num_curves == 0) {
        plutovg_path_traverse(path, traverse_func, closure);
        return;
    }

    const float threshold = 0.25f;

    plutovg_path_iterator_t it;
    plutovg_path_iterator_init(&it, path);

    bezier_t beziers[32];
    plutovg_point_t points[3];
    plutovg_point_t current_point = {0, 0};
    while(plutovg_path_iterator_has_next(&it)) {
        plutovg_path_command_t command = plutovg_path_iterator_next(&it, points);
        switch(command) {
        case PLUTOVG_PATH_COMMAND_MOVE_TO:
        case PLUTOVG_PATH_COMMAND_LINE_TO:
        case PLUTOVG_PATH_COMMAND_CLOSE:
            traverse_func(closure, command, points, 1);
            current_point = points[0];
            break;
        case PLUTOVG_PATH_COMMAND_CUBIC_TO:
            beziers[0].x1 = current_point.x;
            beziers[0].y1 = current_point.y;
            beziers[0].x2 = points[0].x;
            beziers[0].y2 = points[0].y;
            beziers[0].x3 = points[1].x;
            beziers[0].y3 = points[1].y;
            beziers[0].x4 = points[2].x;
            beziers[0].y4 = points[2].y;
            bezier_t* b = beziers;
            while(b >= beziers) {
                float y4y1 = b->y4 - b->y1;
                float x4x1 = b->x4 - b->x1;
                float l = fabsf(x4x1) + fabsf(y4y1);
                float d;
                if(l > 1.f) {
                    d = fabsf((x4x1)*(b->y1 - b->y2) - (y4y1)*(b->x1 - b->x2)) + fabsf((x4x1)*(b->y1 - b->y3) - (y4y1)*(b->x1 - b->x3));
                } else {
                    d = fabsf(b->x1 - b->x2) + fabsf(b->y1 - b->y2) + fabsf(b->x1 - b->x3) + fabsf(b->y1 - b->y3);
                    l = 1.f;
                }

                if(d < threshold*l || b == beziers + 31) {
                    plutovg_point_t p = { b->x4, b->y4 };
                    traverse_func(closure, PLUTOVG_PATH_COMMAND_LINE_TO, &p, 1);
                    --b;
                } else {
                    split_bezier(b, b + 1, b);
                    ++b;
                }
            }

            current_point = points[2];
            break;
        }
    }
}

typedef struct {
    const float* dashes; int ndashes;
    float start_phase; float phase;
    int start_index; int index;
    bool start_toggle; bool toggle;
    plutovg_point_t current_point;
    plutovg_path_traverse_func_t traverse_func;
    void* closure;
} dasher_t;

static void dash_traverse_func(void* closure, plutovg_path_command_t command, const plutovg_point_t* points, int npoints)
{
    dasher_t* dasher = (dasher_t*)(closure);
    if(command == PLUTOVG_PATH_COMMAND_MOVE_TO) {
        if(dasher->start_toggle)
            dasher->traverse_func(dasher->closure, PLUTOVG_PATH_COMMAND_MOVE_TO, points, npoints);
        dasher->current_point = points[0];
        dasher->phase = dasher->start_phase;
        dasher->index = dasher->start_index;
        dasher->toggle = dasher->start_toggle;
        return;
    }

    assert(command == PLUTOVG_PATH_COMMAND_LINE_TO || command == PLUTOVG_PATH_COMMAND_CLOSE);
    plutovg_point_t p0 = dasher->current_point;
    plutovg_point_t p1 = points[0];
    float dx = p1.x - p0.x;
    float dy = p1.y - p0.y;
    float dist0 = sqrtf(dx*dx + dy*dy);
    float dist1 = 0.f;
    while(dist0 - dist1 > dasher->dashes[dasher->index % dasher->ndashes] - dasher->phase) {
        dist1 += dasher->dashes[dasher->index % dasher->ndashes] - dasher->phase;
        float a = dist1 / dist0;
        plutovg_point_t p = { p0.x + a * dx, p0.y + a * dy };
        if(dasher->toggle) {
            dasher->traverse_func(dasher->closure, PLUTOVG_PATH_COMMAND_LINE_TO, &p, 1);
        } else {
            dasher->traverse_func(dasher->closure, PLUTOVG_PATH_COMMAND_MOVE_TO, &p, 1);
        }

        dasher->phase = 0.f;
        dasher->toggle = !dasher->toggle;
        dasher->index++;
    }

    if(dasher->toggle) {
        dasher->traverse_func(dasher->closure, PLUTOVG_PATH_COMMAND_LINE_TO, &p1, 1);
    }

    dasher->phase += dist0 - dist1;
    dasher->current_point = p1;
}

void plutovg_path_traverse_dashed(const plutovg_path_t* path, float offset, const float* dashes, int ndashes, plutovg_path_traverse_func_t traverse_func, void* closure)
{
    float dash_sum = 0.f;
    for(int i = 0; i < ndashes; ++i)
        dash_sum += dashes[i];
    if(ndashes % 2 == 1)
        dash_sum *= 2.f;
    if(dash_sum <= 0.f) {
        plutovg_path_traverse(path, traverse_func, closure);
        return;
    }

    dasher_t dasher;
    dasher.dashes = dashes;
    dasher.ndashes = ndashes;
    dasher.start_phase = fmodf(offset, dash_sum);
    if(dasher.start_phase < 0.f)
        dasher.start_phase += dash_sum;
    dasher.start_index = 0;
    dasher.start_toggle = true;
    while(dasher.start_phase > 0.f && dasher.start_phase >= dasher.dashes[dasher.start_index % dasher.ndashes]) {
        dasher.start_phase -= dashes[dasher.start_index % dasher.ndashes];
        dasher.start_toggle = !dasher.start_toggle;
        dasher.start_index++;
    }

    dasher.phase = dasher.start_phase;
    dasher.index = dasher.start_index;
    dasher.toggle = dasher.start_toggle;
    dasher.current_point.x = 0;
    dasher.current_point.y = 0;
    dasher.traverse_func = traverse_func;
    dasher.closure = closure;
    plutovg_path_traverse_flatten(path, dash_traverse_func, &dasher);
}

plutovg_path_t* plutovg_path_clone(const plutovg_path_t* path)
{
    plutovg_path_t* clone = plutovg_path_create();
    plutovg_array_append(clone->elements, path->elements);
    clone->start_point = path->start_point;
    clone->num_contours = path->num_contours;
    clone->num_points = path->num_points;
    clone->num_curves = path->num_curves;
    return clone;
}

static void clone_traverse_func(void* closure, plutovg_path_command_t command, const plutovg_point_t* points, int npoints)
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
        plutovg_path_close(path);
        break;
    }
}

plutovg_path_t* plutovg_path_clone_flatten(const plutovg_path_t* path)
{
    plutovg_path_t* clone = plutovg_path_create();
    plutovg_path_reserve(clone, path->elements.size + path->num_curves * 32);
    plutovg_path_traverse_flatten(path, clone_traverse_func, clone);
    return clone;
}

plutovg_path_t* plutovg_path_clone_dashed(const plutovg_path_t* path, float offset, const float* dashes, int ndashes)
{
    plutovg_path_t* clone = plutovg_path_create();
    plutovg_path_reserve(clone, path->elements.size + path->num_curves * 32);
    plutovg_path_traverse_dashed(path, offset, dashes, ndashes, clone_traverse_func, clone);
    return clone;
}

typedef struct {
    plutovg_point_t current_point;
    bool is_first_point;
    float length;
    float x1;
    float y1;
    float x2;
    float y2;
} extents_calculator_t;

static void extents_traverse_func(void* closure, plutovg_path_command_t command, const plutovg_point_t* points, int npoints)
{
    extents_calculator_t* calculator = (extents_calculator_t*)(closure);
    if(calculator->is_first_point) {
        assert(command == PLUTOVG_PATH_COMMAND_MOVE_TO);
        calculator->is_first_point = false;
        calculator->current_point = points[0];
        calculator->x1 = points[0].x;
        calculator->y1 = points[0].y;
        calculator->x2 = points[0].x;
        calculator->y2 = points[0].y;
        calculator->length = 0;
        return;
    }

    for(int i = 0; i < npoints; ++i) {
        calculator->x1 = plutovg_min(calculator->x1, points[i].x);
        calculator->y1 = plutovg_min(calculator->y1, points[i].y);
        calculator->x2 = plutovg_max(calculator->x2, points[i].x);
        calculator->y2 = plutovg_max(calculator->y2, points[i].y);
        if(command != PLUTOVG_PATH_COMMAND_MOVE_TO)
            calculator->length += hypotf(points[i].x - calculator->current_point.x, points[i].y - calculator->current_point.y);
        calculator->current_point = points[i];
    }
}

float plutovg_path_extents(const plutovg_path_t* path, plutovg_rect_t* extents, bool tight)
{
    extents_calculator_t calculator = {{0, 0}, true, 0, 0, 0, 0, 0};
    if(tight) {
        plutovg_path_traverse_flatten(path, extents_traverse_func, &calculator);
    } else {
        plutovg_path_traverse(path, extents_traverse_func, &calculator);
    }

    if(extents) {
        extents->x = calculator.x1;
        extents->y = calculator.y1;
        extents->w = calculator.x2 - calculator.x1;
        extents->h = calculator.y2 - calculator.y1;
    }

    return calculator.length;
}

float plutovg_path_length(const plutovg_path_t* path)
{
    return plutovg_path_extents(path, NULL, true);
}

static inline bool parse_arc_flag(const char** begin, const char* end, bool* flag)
{
    if(plutovg_skip_delim(begin, end, '0'))
        *flag = 0;
    else if(plutovg_skip_delim(begin, end, '1'))
        *flag = 1;
    else
        return false;
    plutovg_skip_ws_or_comma(begin, end);
    return true;
}

static inline bool parse_path_coordinates(const char** begin, const char* end, float values[6], int offset, int count)
{
    for(int i = 0; i < count; i++) {
        if(!plutovg_parse_number(begin, end, values + offset + i))
            return false;
        plutovg_skip_ws_or_comma(begin, end);
    }

    return true;
}

bool plutovg_path_parse(plutovg_path_t* path, const char* data, int length)
{
    if(length == -1)
        length = strlen(data);
    const char* it = data;
    const char* end = it + length;

    float values[6];
    bool flags[2];

    float start_x = 0;
    float start_y = 0;
    float last_control_x = 0;
    float last_control_y = 0;
    float current_x = 0;
    float current_y = 0;

    char command = 0;
    char last_command = 0;
    plutovg_skip_ws(&it, end);
    while(it < end) {
        if(PLUTOVG_IS_ALPHA(*it)) {
            command = *it++;
            plutovg_skip_ws(&it, end);
        }

        if(!last_command && !(command == 'M' || command == 'm'))
            return false;
        if(command == 'M' || command == 'm') {
            if(!parse_path_coordinates(&it, end, values, 0, 2))
                return false;
            if(command == 'm') {
                values[0] += current_x;
                values[1] += current_y;
            }

            plutovg_path_move_to(path, values[0], values[1]);
            current_x = start_x = values[0];
            current_y = start_y = values[1];
            command = command == 'm' ? 'l' : 'L';
        } else if(command == 'L' || command == 'l') {
            if(!parse_path_coordinates(&it, end, values, 0, 2))
                return false;
            if(command == 'l') {
                values[0] += current_x;
                values[1] += current_y;
            }

            plutovg_path_line_to(path, values[0], values[1]);
            current_x = values[0];
            current_y = values[1];
        } else if(command == 'H' || command == 'h') {
            if(!parse_path_coordinates(&it, end, values, 0, 1))
                return false;
            if(command == 'h') {
                values[0] += current_x;
            }

            plutovg_path_line_to(path, values[0], current_y);
            current_x = values[0];
        } else if(command == 'V' || command == 'v') {
            if(!parse_path_coordinates(&it, end, values, 1, 1))
                return false;
            if(command == 'v') {
                values[1] += current_y;
            }

            plutovg_path_line_to(path, current_x, values[1]);
            current_y = values[1];
        } else if(command == 'Q' || command == 'q') {
            if(!parse_path_coordinates(&it, end, values, 0, 4))
                return false;
            if(command == 'q') {
                values[0] += current_x;
                values[1] += current_y;
                values[2] += current_x;
                values[3] += current_y;
            }

            plutovg_path_quad_to(path, values[0], values[1], values[2], values[3]);
            last_control_x = values[0];
            last_control_y = values[1];
            current_x = values[2];
            current_y = values[3];
        } else if(command == 'C' || command == 'c') {
            if(!parse_path_coordinates(&it, end, values, 0, 6))
                return false;
            if(command == 'c') {
                values[0] += current_x;
                values[1] += current_y;
                values[2] += current_x;
                values[3] += current_y;
                values[4] += current_x;
                values[5] += current_y;
            }

            plutovg_path_cubic_to(path, values[0], values[1], values[2], values[3], values[4], values[5]);
            last_control_x = values[2];
            last_control_y = values[3];
            current_x = values[4];
            current_y = values[5];
        } else if(command == 'T' || command == 't') {
            if(last_command != 'Q' && last_command != 'q' && last_command != 'T' && last_command != 't') {
                values[0] = current_x;
                values[1] = current_y;
            } else {
                values[0] = 2 * current_x - last_control_x;
                values[1] = 2 * current_y - last_control_y;
            }

            if(!parse_path_coordinates(&it, end, values, 2, 2))
                return false;
            if(command == 't') {
                values[2] += current_x;
                values[3] += current_y;
            }

            plutovg_path_quad_to(path, values[0], values[1], values[2], values[3]);
            last_control_x = values[0];
            last_control_y = values[1];
            current_x = values[2];
            current_y = values[3];
        } else if(command == 'S' || command == 's') {
            if(last_command != 'C' && last_command != 'c' && last_command != 'S' && last_command != 's') {
                values[0] = current_x;
                values[1] = current_y;
            } else {
                values[0] = 2 * current_x - last_control_x;
                values[1] = 2 * current_y - last_control_y;
            }

            if(!parse_path_coordinates(&it, end, values, 2, 4))
                return false;
            if(command == 's') {
                values[2] += current_x;
                values[3] += current_y;
                values[4] += current_x;
                values[5] += current_y;
            }

            plutovg_path_cubic_to(path, values[0], values[1], values[2], values[3], values[4], values[5]);
            last_control_x = values[2];
            last_control_y = values[3];
            current_x = values[4];
            current_y = values[5];
        } else if(command == 'A' || command == 'a') {
            if(!parse_path_coordinates(&it, end, values, 0, 3)
                || !parse_arc_flag(&it, end, &flags[0])
                || !parse_arc_flag(&it, end, &flags[1])
                || !parse_path_coordinates(&it, end, values, 3, 2)) {
                return false;
            }

            if(command == 'a') {
                values[3] += current_x;
                values[4] += current_y;
            }

            plutovg_path_arc_to(path, values[0], values[1], PLUTOVG_DEG2RAD(values[2]), flags[0], flags[1], values[3], values[4]);
            current_x = values[3];
            current_y = values[4];
        } else if(command == 'Z' || command == 'z') {
            if(last_command == 'Z' || last_command == 'z')
                return false;
            plutovg_path_close(path);
            current_x = start_x;
            current_y = start_y;
        } else {
            return false;
        }

        plutovg_skip_ws_or_comma(&it, end);
        last_command = command;
    }

    return true;
}
