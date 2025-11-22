#ifndef UFBXWT_TESTING_UTILS_INCLUDED
#define UFBXWT_TESTING_UTILS_INCLUDED

#include <math.h>

typedef struct {
	size_t num;
	double sum;
	double max;
} ufbxwt_diff_error;

static void ufbxwt_assert_close_real(ufbxwt_diff_error *p_err, ufbx_real a, ufbx_real b)
{
	ufbx_real err = (ufbx_real)fabs(a - b);
	ufbx_real scale = (ufbx_real)fmax(fmax(fabs(a), fabs(b)), 1.0f);
	ufbxwt_assert(err < scale * 0.00001);
	p_err->num++;
	p_err->sum += err;
	if (err > p_err->max) p_err->max = err;
}

static void ufbxwt_assert_close_vec2(ufbxwt_diff_error *p_err, ufbxw_vec2 a, ufbxw_vec2 b)
{
	ufbxwt_assert_close_real(p_err, a.x, b.x);
	ufbxwt_assert_close_real(p_err, a.y, b.y);
}

static void ufbxwt_assert_close_vec3(ufbxwt_diff_error *p_err, ufbxw_vec3 a, ufbxw_vec3 b)
{
	ufbxwt_assert_close_real(p_err, a.x, b.x);
	ufbxwt_assert_close_real(p_err, a.y, b.y);
	ufbxwt_assert_close_real(p_err, a.z, b.z);
}

static void ufbxwt_assert_close_vec4(ufbxwt_diff_error *p_err, ufbxw_vec4 a, ufbxw_vec4 b)
{
	ufbxwt_assert_close_real(p_err, a.x, b.x);
	ufbxwt_assert_close_real(p_err, a.y, b.y);
	ufbxwt_assert_close_real(p_err, a.z, b.z);
	ufbxwt_assert_close_real(p_err, a.w, b.w);
}

static void ufbxwt_assert_close_uvec2(ufbxwt_diff_error *p_err, ufbx_vec2 a, ufbx_vec2 b)
{
	ufbxwt_assert_close_real(p_err, a.x, b.x);
	ufbxwt_assert_close_real(p_err, a.y, b.y);
}

static void ufbxwt_assert_close_uvec3(ufbxwt_diff_error *p_err, ufbx_vec3 a, ufbx_vec3 b)
{
	ufbxwt_assert_close_real(p_err, a.x, b.x);
	ufbxwt_assert_close_real(p_err, a.y, b.y);
	ufbxwt_assert_close_real(p_err, a.z, b.z);
}

static void ufbxwt_assert_close_uvec4(ufbxwt_diff_error *p_err, ufbx_vec4 a, ufbx_vec4 b)
{
	ufbxwt_assert_close_real(p_err, a.x, b.x);
	ufbxwt_assert_close_real(p_err, a.y, b.y);
	ufbxwt_assert_close_real(p_err, a.z, b.z);
	ufbxwt_assert_close_real(p_err, a.w, b.w);
}

static void ufbxwt_log_diff(ufbxwt_diff_error err)
{
	if (err.num > 0) {
		double avg = err.sum / (double)err.num;
		ufbxwt_logf(".. Absolute diff: avg %.3g, max %.3g (%zu tests)", avg, err.max, err.num);
	}
}

static bool ufbxwt_equal_vec2(ufbxw_vec2 a, ufbxw_vec2 b)
{
	return a.x == b.x && a.y == b.y;
}

static bool ufbxwt_equal_vec3(ufbxw_vec3 a, ufbxw_vec3 b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

static bool ufbxwt_equal_vec4(ufbxw_vec4 a, ufbxw_vec4 b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

static ufbxw_node ufbxwt_create_node(ufbxw_scene *scene, const char *name)
{
	ufbxw_node node = ufbxw_create_node(scene);
	ufbxw_set_name(scene, node.id, name);
	return node;
}

static void ufbxwt_assert_error(ufbxw_error *error, ufbxw_error_type type, const char *func, const char *desc)
{
	ufbxwt_assert(error->type == type);
	ufbxwt_assert(!strcmp(error->function.data, func));
	ufbxwt_assert(!strcmp(error->description, desc));
}

static void ufbxwt_assert_string(ufbxw_string str, const char *ref)
{
	ufbxwt_assert(str.length == strlen(str.data));
	ufbxwt_assert(!strcmp(str.data, ref));
}

#endif
