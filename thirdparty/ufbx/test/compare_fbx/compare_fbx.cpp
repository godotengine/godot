#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>

#include "compare_fbx.h"
#include "../ufbx/ufbx.h"
#include "../util/im_arg.h"

static compare_fbx_opts g_opts;

static void ufbxwt_assert_fail(const char *file, uint32_t line, const char *expr) {
	fprintf(stderr, "assert fail: %s (%s:%u)\n", expr, file, line);
	exit(1);
}

#define ufbxwt_assert(cond) do { \
		if (!(cond)) ufbxwt_assert_fail(__FILE__, __LINE__, #cond); \
	} while (0)

template <typename T> static T min(T a, T b) { return a < b ? a : b; }
template <typename T> static T max(T a, T b) { return a < b ? b : a; }

void format(char *&dst, char *end, const char *value)
{
	dst += snprintf(dst, end - dst, "%s", value);
}

void format(char *&dst, char *end, bool value)
{
	dst += snprintf(dst, end - dst, "%s", value ? "true" : "false");
}

void format(char *&dst, char *end, int32_t value)
{
	dst += snprintf(dst, end - dst, "%" PRIi32, value);
}

void format(char *&dst, char *end, uint32_t value)
{
	dst += snprintf(dst, end - dst, "%" PRIu32, value);
}

void format(char *&dst, char *end, int64_t value)
{
	dst += snprintf(dst, end - dst, "%" PRIi64, value);
}

void format(char *&dst, char *end, uint64_t value)
{
	dst += snprintf(dst, end - dst, "%" PRIu64, value);
}

void format(char *&dst, char *end, double value)
{
	dst += snprintf(dst, end - dst, "%g", value);
}

void format(char *&dst, char *end, ufbx_vec2 value)
{
	dst += snprintf(dst, end - dst, "(%g, %g)", value.x, value.y);
}

void format(char *&dst, char *end, ufbx_vec3 value)
{
	dst += snprintf(dst, end - dst, "(%g, %g, %g)", value.x, value.y, value.z);
}

void format(char *&dst, char *end, ufbx_vec4 value)
{
	dst += snprintf(dst, end - dst, "(%g, %g, %g, %g)", value.x, value.y, value.z, value.w);
}

void format(char *&dst, char *end, ufbx_quat value)
{
	dst += snprintf(dst, end - dst, "(%g, %g, %g, %g)", value.x, value.y, value.z, value.w);
}

void format(char *&dst, char *end, ufbx_edge value)
{
	dst += snprintf(dst, end - dst, "(%u, %u)", value.a, value.b);
}

void format(char *&dst, char *end, ufbx_face value)
{
	dst += snprintf(dst, end - dst, "(%u, %u)", value.index_begin, value.num_indices);
}

void format(char *&dst, char *end, ufbx_string value)
{
	dst += snprintf(dst, end - dst, "\"%s\"", value.data);
}

struct check_scope
{
	char name[128];
};

#define MAX_SCOPES 64

check_scope g_scopes[MAX_SCOPES];
size_t g_num_scopes = 0;

static void compare_push_scope_v(const char *fmt, va_list args)
{
	ufbxwt_assert(g_num_scopes < MAX_SCOPES);
	check_scope *scope = &g_scopes[g_num_scopes++];
	vsnprintf(scope->name, sizeof(scope->name), fmt, args);
}

static void compare_pop_scope()
{
	g_num_scopes--;
}

static void compare_push_scope(const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	compare_push_scope_v(fmt, args);
	va_end(args);
}

struct compare_scope
{
	compare_scope(const char *fmt, ...)
	{
		va_list args;
		va_start(args, fmt);
		compare_push_scope_v(fmt, args);
		va_end(args);
	}

	~compare_scope()
	{
		compare_pop_scope();
	}
};

static void format_args_imp(char *&dst, char *end) { }

template <typename Arg, typename... Args>
static void format_args_imp(char *&dst, char *end, Arg arg, Args... args)
{
	format(dst, end, arg);
	format_args_imp(dst, end, args...);
}

template <typename... Args>
static void format_args(char *dst, size_t dst_size, Args... args)
{
	format_args_imp(dst, dst + dst_size, args...);
}

const char *find_filename(const char *path)
{
	const char *end = path + strlen(path);
	while (end != path) {
		if (end[-1] == '/' || end[-1] == '\\') {
			break;
		}
		end--;
	}
	return end;
}

size_t g_check_total_elements = 0;
size_t g_check_total_count = 0;
size_t g_check_fail_count = 0;

template <typename... Args>
static void check_fail_imp(const char *file, int line, Args... args)
{
	g_check_fail_count++;
	if (g_check_fail_count >= 100) {
		return;
	}

	char buf[512];
	format_args(buf, sizeof(buf), args...);

	char scope[256];

	if (g_num_scopes == 0) {
		snprintf(scope, sizeof(scope), "scene");
	} else {
		char *scope_ptr = scope;
		char *scope_end = scope + sizeof(scope);
		for (size_t i = 0; i < g_num_scopes; i++) {
			if (i > 0) {
				scope_ptr += snprintf(scope_ptr, scope_end - scope_ptr, " ");
			}
			scope_ptr += snprintf(scope_ptr, scope_end - scope_ptr, "%s", g_scopes[i].name);
		}
	}

	const char *filename = find_filename(file);
	printf("[FAIL] %s %s (%s:%d)\n", scope, buf, filename, line);
}

#define check_fail(...) check_fail_imp(__FILE__, __LINE__, __VA_ARGS__)

template <typename T>
static bool equals(T a, T b) { return a == b; }
static bool equals(double a, double b) {
	if (isnan(a)) return isnan(b);
	return a == b;
}

static bool equals(ufbx_string a, ufbx_string b) { return !strcmp(a.data, b.data); }
static bool equals(ufbx_vec2 a, ufbx_vec2 b) { return a.x == b.x && a.y == b.y; }
static bool equals(ufbx_vec3 a, ufbx_vec3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
static bool equals(ufbx_vec4 a, ufbx_vec4 b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }
static bool equals(ufbx_quat a, ufbx_quat b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }
static bool equals(ufbx_edge a, ufbx_edge b) { return a.a == b.a && a.b == b.b; }
static bool equals(ufbx_face a, ufbx_face b) { return a.index_begin == b.index_begin && a.num_indices == b.num_indices; }

static bool approx(double a, double b) {
	if (isnan(a)) return isnan(b);

	double err = fabs(a - b);
	return err <= fmax(fmax(fabs(a), fabs(b)), 1.0) * g_opts.approx_epsilon;
}

static bool approx(ufbx_vec2 a, ufbx_vec2 b) { return approx(a.x, b.x) && approx(a.y, b.y); }
static bool approx(ufbx_vec3 a, ufbx_vec3 b) { return approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z); }
static bool approx(ufbx_vec4 a, ufbx_vec4 b) { return approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z) && approx(a.w, b.w); }
static bool approx(ufbx_quat a, ufbx_quat b) {
	a = ufbx_quat_fix_antipodal(a, b);
	return approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z) && approx(a.w, b.w);
}

#define check(m_cond) do { \
		g_check_total_count++; \
		if (!(m_cond)) { \
			check_fail(#m_cond); \
		} \
	} while (0)

#define check_equal(m_src, m_ref, m_field) do { \
		g_check_total_count++; \
		if (!equals((m_src)->m_field, (m_ref)->m_field)) { \
			check_fail(#m_field, ": ", (m_src)->m_field, " != ", (m_ref)->m_field); \
		} \
	} while (0)

#define check_approx(m_src, m_ref, m_field) do { \
		g_check_total_count++; \
		if (!approx((m_src)->m_field, (m_ref)->m_field)) { \
			check_fail(#m_field, ": ", (m_src)->m_field, " != ", (m_ref)->m_field); \
		} \
	} while (0)

template <typename T>
static void check_list_equal_imp(const char *file, int line, const char *field, T a, T b)
{
	g_check_total_count++;
	if (a.count != b.count) {
		check_fail_imp(file, line, field, " count: ", a.count, " != ", b.count);
	}

	g_check_total_count++;
	size_t count = min(a.count, b.count);
	for (size_t i = 0; i < count; i++) {
		if (!equals(a[i], b[i])) {
			check_fail_imp(file, line, field, "[", i, "]: ", a[i], " != ", b[i]);
			break;
		}
	}
}

template <typename T>
static void check_list_approx_imp(const char *file, int line, const char *field, T a, T b)
{
	g_check_total_count++;
	if (a.count != b.count) {
		check_fail_imp(file, line, field, " count: ", a.count, " != ", b.count);
	}

	g_check_total_count++;
	size_t count = min(a.count, b.count);
	g_check_total_elements += count;
	for (size_t i = 0; i < count; i++) {
		if (!approx(a[i], b[i])) {
			check_fail_imp(file, line, field, "[", i, "]: ", a[i], " != ", b[i]);
			break;
		}
	}
}

template <typename T>
static void check_vertex_attrib_imp(const char *file, int line, const char *field, T a, T b)
{
	g_check_total_count++;
	if (a.exists != b.exists) {
		check_fail_imp(file, line, field, " exists: ", a.exists, " != ", b.exists);
	}

	g_check_total_count++;
	if (a.indices.count != b.indices.count) {
		check_fail_imp(file, line, field, " indices.count: ", a.indices.count, " != ", b.indices.count);
	}

	g_check_total_count++;
	size_t count = min(a.indices.count, b.indices.count);
	g_check_total_elements += count;
	for (size_t i = 0; i < count; i++) {
		if (!approx(a[i], b[i])) {
			check_fail_imp(file, line, field, "[", i, "]: ", a[i], " != ", b[i]);
			break;
		}
	}
}

#define check_list_equal(m_src, m_ref, m_field) \
	check_list_equal_imp(__FILE__, __LINE__, #m_field, (m_src)->m_field, (m_ref)->m_field)

#define check_list_approx(m_src, m_ref, m_field) \
	check_list_approx_imp(__FILE__, __LINE__, #m_field, (m_src)->m_field, (m_ref)->m_field)

#define check_vertex_attrib(m_src, m_ref, m_field) \
	check_vertex_attrib_imp(__FILE__, __LINE__, #m_field, (m_src)->m_field, (m_ref)->m_field)

static void compare_mesh(ufbx_mesh *src_mesh, ufbx_mesh *ref_mesh)
{
	check_equal(src_mesh, ref_mesh, num_vertices);
	check_equal(src_mesh, ref_mesh, num_indices);
	check_equal(src_mesh, ref_mesh, num_edges);

	check_list_equal(src_mesh, ref_mesh, vertex_indices);
	check_list_approx(src_mesh, ref_mesh, vertices);

	check_vertex_attrib(src_mesh, ref_mesh, vertex_normal);
	check_vertex_attrib(src_mesh, ref_mesh, vertex_crease);

	check_equal(src_mesh, ref_mesh, uv_sets.count);
	for (size_t set_ix = 0; set_ix < min(src_mesh->uv_sets.count, ref_mesh->uv_sets.count); set_ix++) {
		compare_scope scope { "uv set %zu", set_ix };

		ufbx_uv_set *src_set = &src_mesh->uv_sets[set_ix];
		ufbx_uv_set *ref_set = &ref_mesh->uv_sets[set_ix];
		check_equal(src_set, ref_set, name);

		check_vertex_attrib(src_set, ref_set, vertex_uv);
		check_vertex_attrib(src_set, ref_set, vertex_tangent);
		check_vertex_attrib(src_set, ref_set, vertex_bitangent);
	}

	check_equal(src_mesh, ref_mesh, color_sets.count);
	for (size_t set_ix = 0; set_ix < min(src_mesh->color_sets.count, ref_mesh->color_sets.count); set_ix++) {
		compare_scope scope { "color set %zu", set_ix };

		ufbx_color_set *src_set = &src_mesh->color_sets[set_ix];
		ufbx_color_set *ref_set = &ref_mesh->color_sets[set_ix];
		check_equal(src_set, ref_set, name);

		check_vertex_attrib(src_set, ref_set, vertex_color);
	}
}

static void compare_light(ufbx_light *src_light, ufbx_light *ref_light)
{
	check_approx(src_light, ref_light, color);
	check_approx(src_light, ref_light, intensity);
	check_approx(src_light, ref_light, type);
	check_approx(src_light, ref_light, decay);
	check_approx(src_light, ref_light, area_shape);
	check_approx(src_light, ref_light, inner_angle);
	check_approx(src_light, ref_light, outer_angle);
}

static void compare_node(ufbx_node *src_node, ufbx_node *ref_node, bool full)
{
	check_equal(src_node, ref_node, name);

	if (ref_node->parent) {
		check(src_node->parent);
		if (src_node->parent) {
			check_equal(src_node, ref_node, parent->typed_id);
		}
	} else {
		check(!src_node->parent);
	}

	check_approx(src_node, ref_node, local_transform.translation);
	check_approx(src_node, ref_node, local_transform.rotation);
	check_approx(src_node, ref_node, local_transform.scale);

	check_approx(src_node, ref_node, geometry_transform.translation);
	check_approx(src_node, ref_node, geometry_transform.rotation);
	check_approx(src_node, ref_node, geometry_transform.scale);

	if (ref_node->light) {
		compare_scope scope { "light" };
		check(src_node->light);
		if (src_node->light) {
			compare_light(src_node->light, ref_node->light);
		}
	}

	if (full) {
		if (ref_node->mesh) {
			compare_scope scope { "mesh" };
			check(src_node->mesh);
			if (src_node->mesh) {
				compare_mesh(src_node->mesh, ref_node->mesh);
			}
		}
	}
}

static void compare_scene(ufbx_scene *src_scene, ufbx_scene *ref_scene, bool full)
{
	check_equal(src_scene, ref_scene, nodes.count);
	for (size_t node_ix = 0; node_ix < min(src_scene->nodes.count, ref_scene->nodes.count); node_ix++) {
		ufbx_node *src_node = src_scene->nodes[node_ix];
		ufbx_node *ref_node = ref_scene->nodes[node_ix];

		compare_scope scope { "node '%s'", ref_node->name.data };
		compare_node(src_node, ref_node, full);
	}
}

static void compare_anim(ufbx_scene *src_scene, ufbx_anim *src_anim, ufbx_scene *ref_scene, ufbx_anim *ref_anim)
{
	double frame_begin = ref_anim->time_begin * ref_scene->settings.frames_per_second;
	double frame_end = ref_anim->time_end * ref_scene->settings.frames_per_second;

	size_t num_samples = 8;
	for (size_t sample_ix = 0; sample_ix < num_samples; sample_ix++) {
		double frame = frame_begin + (frame_end - frame_begin) * sample_ix / (num_samples - 1);
		double time = frame / ref_scene->settings.frames_per_second;

		ufbx_scene *src_state = ufbx_evaluate_scene(src_scene, src_anim, time, NULL, NULL);
		ufbx_scene *ref_state = ufbx_evaluate_scene(ref_scene, ref_anim, time, NULL, NULL);
		ufbxwt_assert(src_state);
		ufbxwt_assert(ref_state);

		compare_scope scope { "frame %.2f", frame };
		compare_scene(src_state, ref_state, false);

		ufbx_free_scene(src_state);
		ufbx_free_scene(ref_state);
	}
}

extern "C" bool compare_fbx(const char *src_path, const char *ref_path, const compare_fbx_opts *opts)
{
	g_opts = *opts;

	ufbx_load_opts load_opts = { 0 };
	load_opts.retain_vertex_attrib_w = true;

	ufbx_error load_error;

	ufbx_scene *src_scene = ufbx_load_file(src_path, &load_opts, &load_error);
	if (!src_scene) {
		char err_buf[512];
		ufbx_format_error(err_buf, sizeof(err_buf), &load_error);
		fprintf(stderr, "Failed to load source scene: %s\n", err_buf);
		return false;
	}

	ufbx_scene *ref_scene = ufbx_load_file(ref_path, &load_opts, &load_error);
	if (!ref_scene) {
		char err_buf[512];
		ufbx_format_error(err_buf, sizeof(err_buf), &load_error);
		fprintf(stderr, "Failed to load source scene: %s\n", err_buf);
		return false;
	}

	compare_scene(src_scene, ref_scene, true);

#if 0
	check_equal(src_scene, ref_scene, anim_stacks.count);
	for (size_t stack_ix = 0; stack_ix < min(src_scene->anim_stacks.count, ref_scene->anim_stacks.count); stack_ix++) {
		ufbx_anim_stack *src_stack = src_scene->anim_stacks[stack_ix];
		ufbx_anim_stack *ref_stack = ref_scene->anim_stacks[stack_ix];

		compare_scope scope { "anim '%s'", ref_stack->name.data };

		check_equal(ref_stack, src_stack, name);
		check_approx(ref_stack, src_stack, time_begin);
		check_approx(ref_stack, src_stack, time_end);

		compare_anim(src_scene, src_stack->anim, ref_scene, ref_stack->anim);
	}
#endif

	ufbx_free_scene(src_scene);
	ufbx_free_scene(ref_scene);

	if (g_check_fail_count > 0) {
		printf("\n");
	}

	size_t ok_count = g_check_total_count - g_check_fail_count;
	printf("%zu/%zu checks ok (%zu elements)\n", ok_count, g_check_total_count, g_check_total_elements);

	return g_check_fail_count == 0;
}
