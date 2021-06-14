#ifndef PLUTOVG_H
#define PLUTOVG_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct plutovg_surface plutovg_surface_t;

plutovg_surface_t* plutovg_surface_create(int width, int height);
plutovg_surface_t* plutovg_surface_create_for_data(unsigned char* data, int width, int height, int stride);
plutovg_surface_t* plutovg_surface_reference(plutovg_surface_t* surface);
void plutovg_surface_destroy(plutovg_surface_t* surface);
int plutovg_surface_get_reference_count(const plutovg_surface_t* surface);
unsigned char* plutovg_surface_get_data(const plutovg_surface_t* surface);
int plutovg_surface_get_width(const plutovg_surface_t* surface);
int plutovg_surface_get_height(const plutovg_surface_t* surface);
int plutovg_surface_get_stride(const plutovg_surface_t* surface);

typedef struct {
    double x;
    double y;
} plutovg_point_t;

typedef struct {
    double x;
    double y;
    double w;
    double h;
} plutovg_rect_t;

typedef struct {
    double m00; double m10;
    double m01; double m11;
    double m02; double m12;
} plutovg_matrix_t;

void plutovg_matrix_init(plutovg_matrix_t* matrix, double m00, double m10, double m01, double m11, double m02, double m12);
void plutovg_matrix_init_identity(plutovg_matrix_t* matrix);
void plutovg_matrix_init_translate(plutovg_matrix_t* matrix, double x, double y);
void plutovg_matrix_init_scale(plutovg_matrix_t* matrix, double x, double y);
void plutovg_matrix_init_shear(plutovg_matrix_t* matrix, double x, double y);
void plutovg_matrix_init_rotate(plutovg_matrix_t* matrix, double radians, double x, double y);
void plutovg_matrix_translate(plutovg_matrix_t* matrix, double x, double y);
void plutovg_matrix_scale(plutovg_matrix_t* matrix, double x, double y);
void plutovg_matrix_shear(plutovg_matrix_t* matrix, double x, double y);
void plutovg_matrix_rotate(plutovg_matrix_t* matrix, double radians, double x, double y);
void plutovg_matrix_multiply(plutovg_matrix_t* matrix, const plutovg_matrix_t* a, const plutovg_matrix_t* b);
int plutovg_matrix_invert(plutovg_matrix_t* matrix);
void plutovg_matrix_map(const plutovg_matrix_t* matrix, double x, double y, double* _x, double* _y);
void plutovg_matrix_map_point(const plutovg_matrix_t* matrix, const plutovg_point_t* src, plutovg_point_t* dst);
void plutovg_matrix_map_rect(const plutovg_matrix_t* matrix, const plutovg_rect_t* src, plutovg_rect_t* dst);

typedef struct plutovg_path plutovg_path_t;

typedef enum {
    plutovg_path_element_move_to,
    plutovg_path_element_line_to,
    plutovg_path_element_cubic_to,
    plutovg_path_element_close
} plutovg_path_element_t;

plutovg_path_t* plutovg_path_create(void);
plutovg_path_t* plutovg_path_reference(plutovg_path_t* path);
void plutovg_path_destroy(plutovg_path_t* path);
int plutovg_path_get_reference_count(const plutovg_path_t* path);
void plutovg_path_move_to(plutovg_path_t* path, double x, double y);
void plutovg_path_line_to(plutovg_path_t* path, double x, double y);
void plutovg_path_quad_to(plutovg_path_t* path, double x1, double y1, double x2, double y2);
void plutovg_path_cubic_to(plutovg_path_t* path, double x1, double y1, double x2, double y2, double x3, double y3);
void plutovg_path_close(plutovg_path_t* path);
void plutovg_path_rel_move_to(plutovg_path_t* path, double x, double y);
void plutovg_path_rel_line_to(plutovg_path_t* path, double x, double y);
void plutovg_path_rel_quad_to(plutovg_path_t* path, double x1, double y1, double x2, double y2);
void plutovg_path_rel_cubic_to(plutovg_path_t* path, double x1, double y1, double x2, double y2, double x3, double y3);
void plutovg_path_add_rect(plutovg_path_t* path, double x, double y, double w, double h);
void plutovg_path_add_round_rect(plutovg_path_t* path, double x, double y, double w, double h, double rx, double ry);
void plutovg_path_add_ellipse(plutovg_path_t* path, double cx, double cy, double rx, double ry);
void plutovg_path_add_circle(plutovg_path_t* path, double cx, double cy, double r);
void plutovg_path_add_path(plutovg_path_t* path, const plutovg_path_t* source, const plutovg_matrix_t* matrix);
void plutovg_path_transform(plutovg_path_t* path, const plutovg_matrix_t* matrix);
void plutovg_path_get_current_point(const plutovg_path_t* path, double* x, double* y);
int plutovg_path_get_element_count(const plutovg_path_t* path);
plutovg_path_element_t* plutovg_path_get_elements(const plutovg_path_t* path);
int plutovg_path_get_point_count(const plutovg_path_t* path);
plutovg_point_t* plutovg_path_get_points(const plutovg_path_t* path);
void plutovg_path_clear(plutovg_path_t* path);
int plutovg_path_empty(const plutovg_path_t* path);
plutovg_path_t* plutovg_path_clone(const plutovg_path_t* path);
plutovg_path_t* plutovg_path_clone_flat(const plutovg_path_t* path);

typedef struct {
    double r;
    double g;
    double b;
    double a;
} plutovg_color_t;

void plutovg_color_init_rgb(plutovg_color_t* color, double r, double g, double b);
void plutovg_color_init_rgba(plutovg_color_t* color, double r, double g, double b, double a);

typedef enum {
    plutovg_spread_method_pad,
    plutovg_spread_method_reflect,
    plutovg_spread_method_repeat
} plutovg_spread_method_t;

typedef struct plutovg_gradient plutovg_gradient_t;

typedef enum {
    plutovg_gradient_type_linear,
    plutovg_gradient_type_radial
} plutovg_gradient_type_t;

typedef struct {
    double offset;
    plutovg_color_t color;
} plutovg_gradient_stop_t;

plutovg_gradient_t* plutovg_gradient_create_linear(double x1, double y1, double x2, double y2);
plutovg_gradient_t* plutovg_gradient_create_radial(double cx, double cy, double cr, double fx, double fy, double fr);
plutovg_gradient_t* plutovg_gradient_reference(plutovg_gradient_t* gradient);
void plutovg_gradient_destroy(plutovg_gradient_t* gradient);
int plutovg_gradient_get_reference_count(const plutovg_gradient_t* gradient);
void plutovg_gradient_set_spread(plutovg_gradient_t* gradient, plutovg_spread_method_t spread);
plutovg_spread_method_t plutovg_gradient_get_spread(const plutovg_gradient_t* gradient);
void plutovg_gradient_set_matrix(plutovg_gradient_t* gradient, const plutovg_matrix_t* matrix);
void plutovg_gradient_get_matrix(const plutovg_gradient_t* gradient, plutovg_matrix_t* matrix);
void plutovg_gradient_add_stop_rgb(plutovg_gradient_t* gradient, double offset, double r, double g, double b);
void plutovg_gradient_add_stop_rgba(plutovg_gradient_t* gradient, double offset, double r, double g, double b, double a);
void plutovg_gradient_add_stop(plutovg_gradient_t* gradient, const plutovg_gradient_stop_t* stop);
void plutovg_gradient_clear_stops(plutovg_gradient_t* gradient);
int plutovg_gradient_get_stop_count(const plutovg_gradient_t* gradient);
plutovg_gradient_stop_t* plutovg_gradient_get_stops(const plutovg_gradient_t* gradient);
plutovg_gradient_type_t plutovg_gradient_get_type(const plutovg_gradient_t* gradient);
void plutovg_gradient_get_values_linear(const plutovg_gradient_t* gradient, double* x1, double* y1, double* x2, double* y2);
void plutovg_gradient_get_values_radial(const plutovg_gradient_t* gradient, double* cx, double* cy, double* cr, double* fx, double* fy, double* fr);
void plutovg_gradient_set_values_linear(plutovg_gradient_t* gradient, double x1, double y1, double x2, double y2);
void plutovg_gradient_set_values_radial(plutovg_gradient_t* gradient, double cx, double cy, double cr, double fx, double fy, double fr);
void plutovg_gradient_set_opacity(plutovg_gradient_t* paint, double opacity);
double plutovg_gradient_get_opacity(const plutovg_gradient_t* paint);

typedef struct plutovg_texture plutovg_texture_t;

typedef enum {
    plutovg_texture_type_plain,
    plutovg_texture_type_tiled
} plutovg_texture_type_t;

plutovg_texture_t* plutovg_texture_create(plutovg_surface_t* surface);
plutovg_texture_t* plutovg_texture_reference(plutovg_texture_t* texture);
void plutovg_texture_destroy(plutovg_texture_t* texture);
int plutovg_texture_get_reference_count(const plutovg_texture_t* texture);
void plutovg_texture_set_type(plutovg_texture_t* texture, plutovg_texture_type_t type);
plutovg_texture_type_t plutovg_texture_get_type(const plutovg_texture_t* texture);
void plutovg_texture_set_matrix(plutovg_texture_t* texture, const plutovg_matrix_t* matrix);
void plutovg_texture_get_matrix(const plutovg_texture_t* texture, plutovg_matrix_t* matrix);
void plutovg_texture_set_surface(plutovg_texture_t* texture, plutovg_surface_t* surface);
plutovg_surface_t* plutovg_texture_get_surface(const plutovg_texture_t* texture);
void plutovg_texture_set_opacity(plutovg_texture_t* texture, double opacity);
double plutovg_texture_get_opacity(const plutovg_texture_t* texture);

typedef struct plutovg_paint plutovg_paint_t;

typedef enum {
    plutovg_paint_type_color,
    plutovg_paint_type_gradient,
    plutovg_paint_type_texture
} plutovg_paint_type_t;

plutovg_paint_t* plutovg_paint_create_rgb(double r, double g, double b);
plutovg_paint_t* plutovg_paint_create_rgba(double r, double g, double b, double a);
plutovg_paint_t* plutovg_paint_create_linear(double x1, double y1, double x2, double y2);
plutovg_paint_t* plutovg_paint_create_radial(double cx, double cy, double cr, double fx, double fy, double fr);
plutovg_paint_t* plutovg_paint_create_for_surface(plutovg_surface_t* surface);
plutovg_paint_t* plutovg_paint_create_color(const plutovg_color_t* color);
plutovg_paint_t* plutovg_paint_create_gradient(plutovg_gradient_t* gradient);
plutovg_paint_t* plutovg_paint_create_texture(plutovg_texture_t* texture);
plutovg_paint_t* plutovg_paint_reference(plutovg_paint_t* paint);
void plutovg_paint_destroy(plutovg_paint_t* paint);
int plutovg_paint_get_reference_count(const plutovg_paint_t* paint);
plutovg_paint_type_t plutovg_paint_get_type(const plutovg_paint_t* paint);
plutovg_color_t* plutovg_paint_get_color(const plutovg_paint_t* paint);
plutovg_gradient_t* plutovg_paint_get_gradient(const plutovg_paint_t* paint);
plutovg_texture_t* plutovg_paint_get_texture(const plutovg_paint_t* paint);

typedef enum {
    plutovg_line_cap_butt,
    plutovg_line_cap_round,
    plutovg_line_cap_square
} plutovg_line_cap_t;

typedef enum {
    plutovg_line_join_miter,
    plutovg_line_join_round,
    plutovg_line_join_bevel
} plutovg_line_join_t;

typedef enum {
    plutovg_fill_rule_non_zero,
    plutovg_fill_rule_even_odd
} plutovg_fill_rule_t;

typedef enum {
    plutovg_operator_src,
    plutovg_operator_src_over,
    plutovg_operator_dst_in,
    plutovg_operator_dst_out
} plutovg_operator_t;

typedef struct plutovg plutovg_t;

plutovg_t* plutovg_create(plutovg_surface_t* surface);
plutovg_t* plutovg_reference(plutovg_t* pluto);
void plutovg_destroy(plutovg_t* pluto);
int plutovg_get_reference_count(const plutovg_t* pluto);
void plutovg_save(plutovg_t* pluto);
void plutovg_restore(plutovg_t* pluto);
void plutovg_set_source_rgb(plutovg_t* pluto, double r, double g, double b);
void plutovg_set_source_rgba(plutovg_t* pluto, double r, double g, double b, double a);
void plutovg_set_source_surface(plutovg_t* pluto, plutovg_surface_t* surface, double x, double y);
void plutovg_set_source_color(plutovg_t* pluto, const plutovg_color_t* color);
void plutovg_set_source_gradient(plutovg_t* pluto, plutovg_gradient_t* gradient);
void plutovg_set_source_texture(plutovg_t* pluto, plutovg_texture_t* texture);
void plutovg_set_source(plutovg_t* pluto, plutovg_paint_t* source);
plutovg_paint_t* plutovg_get_source(const plutovg_t* pluto);

void plutovg_set_operator(plutovg_t* pluto, plutovg_operator_t op);
void plutovg_set_opacity(plutovg_t* pluto, double opacity);
void plutovg_set_fill_rule(plutovg_t* pluto, plutovg_fill_rule_t fill_rule);
plutovg_operator_t plutovg_get_operator(const plutovg_t* pluto);
double plutovg_get_opacity(const plutovg_t* pluto);
plutovg_fill_rule_t plutovg_get_fill_rule(const plutovg_t* pluto);

void plutovg_set_line_width(plutovg_t* pluto, double width);
void plutovg_set_line_cap(plutovg_t* pluto, plutovg_line_cap_t cap);
void plutovg_set_line_join(plutovg_t* pluto, plutovg_line_join_t join);
void plutovg_set_miter_limit(plutovg_t* pluto, double limit);
void plutovg_set_dash(plutovg_t* pluto, double offset, const double* data, int size);
double plutovg_get_line_width(const plutovg_t* pluto);
plutovg_line_cap_t plutovg_get_line_cap(const plutovg_t* pluto);
plutovg_line_join_t plutovg_get_line_join(const plutovg_t* pluto);
double plutovg_get_miter_limit(const plutovg_t* pluto);

void plutovg_translate(plutovg_t* pluto, double x, double y);
void plutovg_scale(plutovg_t* pluto, double x, double y);
void plutovg_rotate(plutovg_t* pluto, double radians, double x, double y);
void plutovg_transform(plutovg_t* pluto, const plutovg_matrix_t* matrix);
void plutovg_set_matrix(plutovg_t* pluto, const plutovg_matrix_t* matrix);
void plutovg_identity_matrix(plutovg_t* pluto);
void plutovg_get_matrix(const plutovg_t* pluto, plutovg_matrix_t* matrix);

void plutovg_move_to(plutovg_t* pluto, double x, double y);
void plutovg_line_to(plutovg_t* pluto, double x, double y);
void plutovg_quad_to(plutovg_t* pluto, double x1, double y1, double x2, double y2);
void plutovg_cubic_to(plutovg_t* pluto, double x1, double y1, double x2, double y2, double x3, double y3);
void plutovg_rel_move_to(plutovg_t* pluto, double x, double y);
void plutovg_rel_line_to(plutovg_t* pluto, double x, double y);
void plutovg_rel_quad_to(plutovg_t* pluto, double x1, double y1, double x2, double y2);
void plutovg_rel_cubic_to(plutovg_t* pluto, double x1, double y1, double x2, double y2, double x3, double y3);
void plutovg_rect(plutovg_t* pluto, double x, double y, double w, double h);
void plutovg_round_rect(plutovg_t* pluto, double x, double y, double w, double h, double rx, double ry);
void plutovg_ellipse(plutovg_t* pluto, double cx, double cy, double rx, double ry);
void plutovg_circle(plutovg_t* pluto, double cx, double cy, double r);
void plutovg_add_path(plutovg_t* pluto, const plutovg_path_t* path);
void plutovg_new_path(plutovg_t* pluto);
void plutovg_close_path(plutovg_t* pluto);
plutovg_path_t* plutovg_get_path(const plutovg_t* pluto);

void plutovg_fill(plutovg_t* pluto);
void plutovg_stroke(plutovg_t* pluto);
void plutovg_clip(plutovg_t* pluto);
void plutovg_paint(plutovg_t* pluto);

void plutovg_fill_preserve(plutovg_t* pluto);
void plutovg_stroke_preserve(plutovg_t* pluto);
void plutovg_clip_preserve(plutovg_t* pluto);

#ifdef __cplusplus
}
#endif

#endif // PLUTOVG_H
