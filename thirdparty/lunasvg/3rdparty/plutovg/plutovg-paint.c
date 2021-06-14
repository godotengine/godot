#include "plutovg-private.h"

void plutovg_color_init_rgb(plutovg_color_t* color, double r, double g, double b)
{
    plutovg_color_init_rgba(color, r, g,  b, 1.0);
}

void plutovg_color_init_rgba(plutovg_color_t* color, double r, double g, double b, double a)
{
    color->r = r;
    color->g = g;
    color->b = b;
    color->a = a;
}

plutovg_gradient_t* plutovg_gradient_create_linear(double x1, double y1, double x2, double y2)
{
    plutovg_gradient_t* gradient = malloc(sizeof(plutovg_gradient_t));
    gradient->ref = 1;
    gradient->type = plutovg_gradient_type_linear;
    gradient->spread = plutovg_spread_method_pad;
    gradient->opacity = 1.0;
    plutovg_array_init(gradient->stops);
    plutovg_matrix_init_identity(&gradient->matrix);

    gradient->values[0] = x1;
    gradient->values[1] = y1;
    gradient->values[2] = x2;
    gradient->values[3] = y2;

    return gradient;
}

plutovg_gradient_t* plutovg_gradient_create_radial(double cx, double cy, double cr, double fx, double fy, double fr)
{
    plutovg_gradient_t* gradient = malloc(sizeof(plutovg_gradient_t));
    gradient->ref = 1;
    gradient->type = plutovg_gradient_type_radial;
    gradient->spread = plutovg_spread_method_pad;
    gradient->opacity = 1.0;
    plutovg_array_init(gradient->stops);
    plutovg_matrix_init_identity(&gradient->matrix);

    gradient->values[0] = cx;
    gradient->values[1] = cy;
    gradient->values[2] = cr;
    gradient->values[3] = fx;
    gradient->values[4] = fy;
    gradient->values[5] = fr;

    return gradient;
}

plutovg_gradient_t* plutovg_gradient_reference(plutovg_gradient_t* gradient)
{
    ++gradient->ref;
    return gradient;
}

void plutovg_gradient_destroy(plutovg_gradient_t* gradient)
{
    if(gradient==NULL)
        return;

    if(--gradient->ref==0)
    {
        free(gradient->stops.data);
        free(gradient);
    }
}

int plutovg_gradient_get_reference_count(const plutovg_gradient_t* gradient)
{
    return gradient->ref;
}

void plutovg_gradient_set_spread(plutovg_gradient_t* gradient, plutovg_spread_method_t spread)
{
    gradient->spread = spread;
}

plutovg_spread_method_t plutovg_gradient_get_spread(const plutovg_gradient_t* gradient)
{
    return gradient->spread;
}

void plutovg_gradient_set_matrix(plutovg_gradient_t* gradient, const plutovg_matrix_t* matrix)
{
    memcpy(&gradient->matrix, matrix, sizeof(plutovg_matrix_t));
}

void plutovg_gradient_get_matrix(const plutovg_gradient_t* gradient, plutovg_matrix_t *matrix)
{
    memcpy(matrix, &gradient->matrix, sizeof(plutovg_matrix_t));
}

void plutovg_gradient_add_stop_rgb(plutovg_gradient_t* gradient, double offset, double r, double g, double b)
{
    plutovg_gradient_add_stop_rgba(gradient, offset, r, g, b, 1.0);
}

void plutovg_gradient_add_stop_rgba(plutovg_gradient_t* gradient, double offset, double r, double g, double b, double a)
{
    plutovg_array_ensure(gradient->stops, 1);
    plutovg_gradient_stop_t* stops = gradient->stops.data;
    int nstops = gradient->stops.size;
    int i;
    for(i = 0;i < nstops;i++)
    {
        if(offset < stops[i].offset)
        {
            memmove(&stops[i+1], &stops[i], (size_t)(nstops - i) * sizeof(plutovg_gradient_stop_t));
            break;
        }
    }

    stops[i].offset = offset;
    stops[i].color.r = r;
    stops[i].color.g = g;
    stops[i].color.b = b;
    stops[i].color.a = a;

    gradient->stops.size++;
}

void plutovg_gradient_add_stop(plutovg_gradient_t* gradient, const plutovg_gradient_stop_t* stop)
{
    plutovg_gradient_add_stop_rgba(gradient, stop->offset, stop->color.r, stop->color.g, stop->color.b, stop->color.a);
}

void plutovg_gradient_clear_stops(plutovg_gradient_t* gradient)
{
    gradient->stops.size = 0;
}

int plutovg_gradient_get_stop_count(const plutovg_gradient_t* gradient)
{
    return gradient->stops.size;
}

plutovg_gradient_stop_t* plutovg_gradient_get_stops(const plutovg_gradient_t* gradient)
{
    return gradient->stops.data;
}

plutovg_gradient_type_t plutovg_gradient_get_type(const plutovg_gradient_t* gradient)
{
    return gradient->type;
}

void plutovg_gradient_get_values_linear(const plutovg_gradient_t* gradient, double* x1, double* y1, double* x2, double* y2)
{
    *x1 = gradient->values[0];
    *y1 = gradient->values[1];
    *x2 = gradient->values[2];
    *y2 = gradient->values[3];
}

void plutovg_gradient_get_values_radial(const plutovg_gradient_t* gradient, double* cx, double* cy, double* cr, double* fx, double* fy, double* fr)
{
    *cx = gradient->values[0];
    *cy = gradient->values[1];
    *cr = gradient->values[2];
    *fx = gradient->values[3];
    *fy = gradient->values[4];
    *fr = gradient->values[5];
}

void plutovg_gradient_set_values_linear(plutovg_gradient_t* gradient, double x1, double y1, double x2, double y2)
{
    gradient->values[0] = x1;
    gradient->values[1] = y1;
    gradient->values[2] = x2;
    gradient->values[3] = y2;
}

void plutovg_gradient_set_values_radial(plutovg_gradient_t* gradient, double cx, double cy, double cr, double fx, double fy, double fr)
{
    gradient->values[0] = cx;
    gradient->values[1] = cy;
    gradient->values[2] = cr;
    gradient->values[3] = fx;
    gradient->values[4] = fy;
    gradient->values[5] = fr;
}

void plutovg_gradient_set_opacity(plutovg_gradient_t* gradient, double opacity)
{
    gradient->opacity = opacity;
}

double plutovg_gradient_get_opacity(const plutovg_gradient_t* gradient)
{
    return gradient->opacity;
}

plutovg_texture_t* plutovg_texture_create(plutovg_surface_t* surface)
{
    plutovg_texture_t* texture = malloc(sizeof(plutovg_texture_t));
    texture->ref = 1;
    texture->type = plutovg_texture_type_plain;
    texture->surface = plutovg_surface_reference(surface);
    texture->opacity = 1.0;
    plutovg_matrix_init_identity(&texture->matrix);
    return texture;
}

plutovg_texture_t* plutovg_texture_reference(plutovg_texture_t* texture)
{
    ++texture->ref;
    return texture;
}

void plutovg_texture_destroy(plutovg_texture_t* texture)
{
    if(texture==NULL)
        return;

    if(--texture->ref==0)
    {
        plutovg_surface_destroy(texture->surface);
        free(texture);
    }
}

int plutovg_texture_get_reference_count(const plutovg_texture_t* texture)
{
    return texture->ref;
}

void plutovg_texture_set_type(plutovg_texture_t* texture, plutovg_texture_type_t type)
{
    texture->type = type;
}

plutovg_texture_type_t plutovg_texture_get_type(const plutovg_texture_t* texture)
{
    return texture->type;
}

void plutovg_texture_set_matrix(plutovg_texture_t* texture, const plutovg_matrix_t* matrix)
{
    memcpy(&texture->matrix, matrix, sizeof(plutovg_matrix_t));
}

void plutovg_texture_get_matrix(const plutovg_texture_t* texture, plutovg_matrix_t* matrix)
{
    memcpy(matrix, &texture->matrix, sizeof(plutovg_matrix_t));
}

void plutovg_texture_set_surface(plutovg_texture_t* texture, plutovg_surface_t* surface)
{
    surface = plutovg_surface_reference(surface);
    plutovg_surface_destroy(texture->surface);
    texture->surface = surface;
}

plutovg_surface_t* plutovg_texture_get_surface(const plutovg_texture_t* texture)
{
    return texture->surface;
}

void plutovg_texture_set_opacity(plutovg_texture_t* texture, double opacity)
{
    texture->opacity = opacity;
}

double plutovg_texture_get_opacity(const plutovg_texture_t* texture)
{
    return texture->opacity;
}

plutovg_paint_t* plutovg_paint_create_rgb(double r, double g, double b)
{
    return plutovg_paint_create_rgba(r, g, b, 1.0);
}

plutovg_paint_t* plutovg_paint_create_rgba(double r, double g, double b, double a)
{
    plutovg_paint_t* paint = malloc(sizeof(plutovg_paint_t));
    paint->ref = 1;
    paint->type = plutovg_paint_type_color;
    paint->color = malloc(sizeof(plutovg_color_t));
    plutovg_color_init_rgba(paint->color, r, g, b, a);
    return paint;
}

plutovg_paint_t* plutovg_paint_create_linear(double x1, double y1, double x2, double y2)
{
    plutovg_gradient_t* gradient = plutovg_gradient_create_linear(x1, y1, x2, y2);
    plutovg_paint_t* paint = plutovg_paint_create_gradient(gradient);
    plutovg_gradient_destroy(gradient);
    return paint;
}

plutovg_paint_t* plutovg_paint_create_radial(double cx, double cy, double cr, double fx, double fy, double fr)
{
    plutovg_gradient_t* gradient = plutovg_gradient_create_radial(cx, cy, cr, fx, fy, fr);
    plutovg_paint_t* paint = plutovg_paint_create_gradient(gradient);
    plutovg_gradient_destroy(gradient);
    return paint;
}

plutovg_paint_t* plutovg_paint_create_for_surface(plutovg_surface_t* surface)
{
    plutovg_texture_t* texture = plutovg_texture_create(surface);
    plutovg_paint_t* paint = plutovg_paint_create_texture(texture);
    plutovg_texture_destroy(texture);
    return paint;
}

plutovg_paint_t* plutovg_paint_create_color(const plutovg_color_t* color)
{
    return plutovg_paint_create_rgba(color->r, color->g, color->b, color->a);
}

plutovg_paint_t* plutovg_paint_create_gradient(plutovg_gradient_t* gradient)
{
    plutovg_paint_t* paint = malloc(sizeof(plutovg_paint_t));
    paint->ref = 1;
    paint->type = plutovg_paint_type_gradient;
    paint->gradient = plutovg_gradient_reference(gradient);
    return paint;
}

plutovg_paint_t* plutovg_paint_create_texture(plutovg_texture_t* texture)
{
    plutovg_paint_t* paint = malloc(sizeof(plutovg_paint_t));
    paint->ref = 1;
    paint->type = plutovg_paint_type_texture;
    paint->texture = plutovg_texture_reference(texture);
    return paint;
}

plutovg_paint_t* plutovg_paint_reference(plutovg_paint_t* paint)
{
    ++paint->ref;
    return paint;
}

void plutovg_paint_destroy(plutovg_paint_t* paint)
{
    if(paint==NULL)
        return;

    if(--paint->ref==0)
    {
        if(paint->type==plutovg_paint_type_color)
            free(paint->color);
        if(paint->type==plutovg_paint_type_gradient)
            plutovg_gradient_destroy(paint->gradient);
        if(paint->type==plutovg_paint_type_texture)
            plutovg_texture_destroy(paint->texture);
        free(paint);
    }
}

int plutovg_paint_get_reference_count(const plutovg_paint_t* paint)
{
    return paint->ref;
}

plutovg_paint_type_t plutovg_paint_get_type(const plutovg_paint_t* paint)
{
    return paint->type;
}

plutovg_color_t* plutovg_paint_get_color(const plutovg_paint_t* paint)
{
    return paint->type==plutovg_paint_type_color?paint->color:NULL;
}

plutovg_gradient_t* plutovg_paint_get_gradient(const plutovg_paint_t* paint)
{
    return paint->type==plutovg_paint_type_gradient?paint->gradient:NULL;
}

plutovg_texture_t* plutovg_paint_get_texture(const plutovg_paint_t* paint)
{
    return paint->type==plutovg_paint_type_texture?paint->texture:NULL;
}
