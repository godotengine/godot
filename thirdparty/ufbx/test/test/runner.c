#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>

#include <stdint.h>
#include <stdbool.h>
void ufbxwt_assert_fail_imp(const char *file, uint32_t line, const char *expr);
static void ufbxwt_assert_fail(const char *file, uint32_t line, const char *expr) {
	ufbxwt_assert_fail_imp(file, line, expr);
}

#include "../../ufbx_write.h"
#include "../ufbx/ufbx.h"

#include "../util/ufbxwt_util.h"

#define ufbxwt_arraycount(arr) (sizeof(arr) / sizeof(*(arr)))

// -- Thread local

#define UFBXWT_HAS_THREADLOCAL 1

#if defined(_MSC_VER)
	#define ufbxwt_threadlocal __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
	#define ufbxwt_threadlocal __thread
#else
	#define ufbxwt_threadlocal
	#undef UFBXWT_HAS_THREADLOCAL
	#define UFBXWT_HAS_THREADLOCAL 0
#endif

#ifndef USE_SETJMP
#if !defined(__wasm__) && UFBXWT_HAS_THREADLOCAL
	#define USE_SETJMP 1
#else
	#define USE_SETJMP 0
#endif
#endif

#if USE_SETJMP

#include <setjmp.h>

#define ufbxwt_jmp_buf jmp_buf
#define ufbxwt_setjmp(env) setjmp(env)
#define ufbxwt_longjmp(env, status, file, line, expr) longjmp(env, status)

#else

#define ufbxwt_jmp_buf int
#define ufbxwt_setjmp(env) (0)

static void ufbxwt_longjmp(int env, int value, const char *file, uint32_t line, const char *expr)
{
	fprintf(stderr, "\nAssertion failed: %s:%u: %s\n", file, line, expr);
	exit(1);
}

#endif

#define ufbxwt_assert(cond) do { \
		if (!(cond)) ufbxwt_assert_fail_imp(__FILE__, __LINE__, #cond); \
	} while (0)

// Avoid ufbxw_assert()
#undef ufbxw_assert

typedef struct {
	bool failed;
	const char *file;
	uint32_t line;
	const char *expr;
} ufbxwt_fail;

typedef struct {
	const char *group;
	const char *name;
	void (*func)(void);

	ufbxwt_fail fail;
} ufbxwt_test;

ufbxwt_test *g_current_test;

char data_root[256];
char output_root[256];

int g_verbose;
ufbxw_error g_error;

char g_log_buf[16*1024];
uint32_t g_log_pos;

char g_hint[8*1024];

uint32_t g_file_version;
const char *g_file_format;

void ufbxwt_logf(const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	if (g_log_pos < sizeof(g_log_buf)) {
		g_log_pos += vsnprintf(g_log_buf + g_log_pos,
			sizeof(g_log_buf) - g_log_pos, fmt, args);
		if (g_log_pos < sizeof(g_log_buf)) {
			g_log_buf[g_log_pos] = '\n';
			g_log_pos++;
		}
	}
	va_end(args);
}

void ufbxwt_hintf(const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	vsnprintf(g_hint, sizeof(g_hint), fmt, args);
	va_end(args);
}

void ufbxwt_log_flush(bool print_always)
{
	if ((g_verbose || print_always) && g_log_pos > 0) {
		int prev_newline = 1;
		for (uint32_t i = 0; i < g_log_pos; i++) {
			if (i >= sizeof(g_log_buf)) break;
			char ch = g_log_buf[i];
			if (ch == '\n') {
				putchar('\n');
				prev_newline = 1;
			} else {
				if (prev_newline) {
					putchar(' ');
					putchar(' ');
				}
				prev_newline = 0;
				putchar(ch);
			}
		}
	}
	g_log_pos = 0;
}

void ufbxwt_log_uerror(ufbx_error *err)
{
	if (!err) return;
	if (err->info_length > 0) {
		ufbxwt_logf("Error: %s (%s)", err->description.data, err->info);
	} else {
		ufbxwt_logf("Error: %s", err->description.data);
	}
	for (size_t i = 0; i < err->stack_size; i++) {
		ufbx_error_frame *f = &err->stack[i];
		ufbxwt_logf("Line %u %s: %s", f->source_line, f->function.data, f->description.data);
	}
}

void ufbxwt_log_error(const ufbxw_error *err)
{
	ufbxwt_logf("Error: %s(): %s", err->function.data, err->description);
}

#include "testing_utils.h"

bool ufbxwt_check_scene_error_imp(ufbxw_scene *scene, const char *file, int line);
void ufbxwt_do_scene_test(const char *name, void (*test_fn)(ufbxw_scene *scene, ufbxwt_diff_error *err), void (*check_fn)(ufbx_scene *scene, ufbxwt_diff_error *err), const ufbxw_scene_opts *user_opts, uint32_t flags);

#define ufbxwt_check_error(scene) do { if (ufbxwt_check_scene_error_imp((scene), __FILE__, __LINE__)) return; } while (0)

#define UFBXWT_IMPL 1

#define UFBXWT_TEST(name) void ufbxwt_test_fn_##name(void)

#define UFBXWT_SCENE_TEST_FLAGS(name, flags) \
	void ufbxwt_test_fn_imp_scene_##name(ufbxw_scene *scene, ufbxwt_diff_error *err); \
	void ufbxwt_check_scene_##name(ufbx_scene *scene, ufbxwt_diff_error *err); \
	void ufbxwt_test_fn_scene_##name(void) { \
		ufbxwt_do_scene_test(#name, &ufbxwt_test_fn_imp_scene_##name, &ufbxwt_check_scene_##name, NULL, flags); \
	} \
	void ufbxwt_test_fn_imp_scene_##name(ufbxw_scene *scene, ufbxwt_diff_error *err)

#define UFBXWT_SCENE_CHECK_FLAGS(name, flags) void ufbxwt_check_scene_##name(ufbx_scene *scene, ufbxwt_diff_error *err)

#define UFBXWT_TEST_GROUP ""

#define UFBXWT_SCENE_TEST(name) UFBXWT_SCENE_TEST_FLAGS(name, 0)
#define UFBXWT_SCENE_CHECK(name) UFBXWT_SCENE_CHECK_FLAGS(name, 0)

#include "all_tests.h"

#undef UFBXWT_IMPL
#undef UFBXWT_TEST
#undef UFBXWT_SCENE_TEST_FLAGS
#undef UFBXWT_SCENE_CHECK_FLAGS
#undef UFBXWT_TEST_GROUP

#define UFBXWT_IMPL 0
#define UFBXWT_TEST(name) { UFBXWT_TEST_GROUP, #name, &ufbxwt_test_fn_##name },
#define UFBXWT_SCENE_TEST_FLAGS(name, flags) { UFBXWT_TEST_GROUP, #name, &ufbxwt_test_fn_scene_##name },
#define UFBXWT_SCENE_CHECK_FLAGS(name, flags)
#define UFBXWT_TEST_GROUP ""

ufbxwt_test g_tests[] = {
	#include "all_tests.h"
};

ufbxwt_jmp_buf g_test_jmp;

#undef UFBXWT_IMPL
#undef UFBXWT_TEST
#undef UFBXWT_SCENE_TEST_FLAGS
#undef UFBXWT_SCENE_CHECK_FLAGS
#undef UFBXWT_TEST_GROUP

typedef struct {
	const char *name;
	uint32_t num_total;
	uint32_t num_ran;
	uint32_t num_ok;
} ufbxwt_test_stats;

ufbxwt_test_stats g_test_groups[ufbxwt_arraycount(g_tests)];
size_t g_num_groups = 0;

ufbxwt_test_stats *ufbxwt_get_test_group(const char *name)
{
	for (size_t i = g_num_groups; i > 0; --i) {
		ufbxwt_test_stats *group = &g_test_groups[i - 1];
		if (!strcmp(group->name, name)) return group;
	}

	ufbxwt_test_stats *group = &g_test_groups[g_num_groups++];
	group->name = name;
	return group;
}

ufbxwt_threadlocal ufbxwt_jmp_buf *t_jmp_buf;

void ufbxwt_assert_fail_imp(const char *file, uint32_t line, const char *expr)
{
	if (t_jmp_buf) {
		ufbxwt_longjmp(*t_jmp_buf, 1, file, line, expr);
	}

	g_current_test->fail.failed = 1;
	g_current_test->fail.file = file;
	g_current_test->fail.line = line;
	g_current_test->fail.expr = expr;

	ufbxwt_longjmp(g_test_jmp, 1, file, line, expr);
}

bool g_fuzz = false;
bool g_allow_scene_error = false;

bool ufbxwt_check_scene_error_imp(ufbxw_scene *scene, const char *file, int line)
{
	ufbxw_error error;
	if (!ufbxw_get_error(scene, &error)) {
		// No error: Keep going
		return false;
	}

	if (g_allow_scene_error) {
		// We allow errors and hit one, return to not break the rest of the test
		return true;
	}

	ufbxwt_log_error(&error);

	// This will longjmp/abort out of the function
	ufbxwt_assert_fail(file, line, "ufbxwt_check_error()");
	return false;
}

static void ufbxwt_error_callback(void *user, const ufbxw_error *error)
{
	ufbxwt_log_error(error);
	ufbxwt_assert(0 && "error");
}

void ufbxwt_do_scene_test(const char *name, void (*test_fn)(ufbxw_scene *scene, ufbxwt_diff_error *err), void (*check_fn)(ufbx_scene *scene, ufbxwt_diff_error *err), const ufbxw_scene_opts *user_opts, uint32_t flags)
{
	ufbxw_scene_opts scene_opts = { 0 };
	if (user_opts) {
		scene_opts = *user_opts;
	}

	ufbxw_scene *scene = ufbxw_create_scene(&scene_opts);
	ufbxw_set_error_callback(scene, &ufbxwt_error_callback, NULL);

	ufbxwt_diff_error err = { 0 };

	g_allow_scene_error = false;

	// Create the scene
	test_fn(scene, &err);

	ufbxw_prepare_scene(scene, NULL);

	ufbxw_memory_stats memory_stats = ufbxw_get_memory_stats(scene);
	ufbxwt_logf(".. Scene %.1fkB (%zu allocs, %zu blocks)",
		(double)memory_stats.allocated_bytes * 1e-3,
		memory_stats.allocation_count,
		memory_stats.block_allocation_count
	);

	static const uint32_t versions[] = { 7400, 7500 };
	static const ufbxw_save_format formats[] = { UFBXW_SAVE_FORMAT_ASCII, UFBXW_SAVE_FORMAT_BINARY };

	for (int version_ix = 0; version_ix < ufbxwt_arraycount(versions); version_ix++) {
		for (int format_ix = 0; format_ix < ufbxwt_arraycount(formats); format_ix++) {
			ufbxw_save_opts save_opts = { 0 };
			save_opts.version = versions[version_ix];
			save_opts.format = formats[format_ix];

			uint32_t version = save_opts.version;
			const char *format = save_opts.format == UFBXW_SAVE_FORMAT_ASCII ? "ascii" : "binary";

			if (g_file_version && version != g_file_version) continue;
			if (g_file_format && strcmp(format, g_file_format) != 0) continue;

			ufbxwt_logf(".. saving %u %s", version, format);

			char output_path[256];
			snprintf(output_path, sizeof(output_path), "%s/%s_%u_%s.fbx", output_root, name, version, format);

			ufbxw_error save_error;
			ufbxw_save_file(scene, output_path, &save_opts, &save_error);
			if (save_error.type != UFBXW_ERROR_NONE) {
				ufbxwt_log_error(&save_error);
			}

			if (check_fn) {
				ufbx_load_opts load_opts = { 0 };
				ufbx_error load_error;

				ufbx_scene *loaded_scene = ufbx_load_file(output_path, &load_opts, &load_error);
				if (!loaded_scene) {
					ufbxwt_log_uerror(&load_error);
				}
				ufbxwt_assert(loaded_scene);

				check_fn(loaded_scene, &err);

				ufbx_free_scene(loaded_scene);
			}
		}
	}

	ufbxw_free_scene(scene);

	if (g_fuzz) {
		for (size_t max_allocs = 1; max_allocs < memory_stats.allocation_count; max_allocs++) {
			ufbxw_scene_opts fuzz_opts = scene_opts;
			ufbxwt_hintf("max_allocs=%zu", max_allocs);

			fuzz_opts.max_allocations = max_allocs;

			ufbxw_scene *fuzz_scene = ufbxw_create_scene(&fuzz_opts);
			ufbxwt_diff_error fuzz_err = { 0 };

			g_allow_scene_error = true;
			test_fn(fuzz_scene, &fuzz_err);

			if (!ufbxw_get_error(fuzz_scene, NULL)) {
				ufbxw_prepare_scene(fuzz_scene, NULL);
			}

			ufbxw_error fuzz_error;
			ufbxwt_assert(ufbxw_get_error(fuzz_scene, &fuzz_error));
			ufbxwt_assert(fuzz_error.type == UFBXW_ERROR_ALLOCATION_LIMIT);

			ufbxw_free_scene(fuzz_scene);
		}
	}
}

int ufbxwt_run_test(ufbxwt_test *test)
{
	printf("%s: ", test->name);
	fflush(stdout);

	memset(&g_error, 0, sizeof(g_error));
	g_hint[0] = '\0';

	g_current_test = test;
	if (!ufbxwt_setjmp(g_test_jmp)) {
		test->func();
		printf("OK\n");
		fflush(stdout);
		return 1;
	} else {
		printf("FAIL\n");
		fflush(stdout);

		if (g_hint[0]) {
			printf("Hint: %s\n", g_hint);
		}
		if (g_error.type != UFBXW_ERROR_NONE) {
			ufbxwt_log_error(&g_error);
		}

		return 0;
	}
}

int main(int argc, char **argv)
{
	uint32_t num_tests = ufbxwt_arraycount(g_tests);
	uint32_t num_ok = 0;
	const char *test_filter = NULL;
	const char *test_group = NULL;

	snprintf(data_root, sizeof(data_root), "data");
	snprintf(output_root, sizeof(output_root), "output");

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verbose")) {
			g_verbose = 1;
		}
		if (!strcmp(argv[i], "-t") || !strcmp(argv[i], "--test")) {
			if (++i < argc) {
				test_filter = argv[i];
			}
		}
		if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--format")) {
			if (++i < argc) g_file_version = (uint32_t)atoi(argv[i]);
			if (++i < argc) g_file_format = argv[i];
		}
		if (!strcmp(argv[i], "-d") || !strcmp(argv[i], "--data")) {
			if (++i < argc) {
				size_t len = strlen(argv[i]);
				if (len == 0) {
					fprintf(stderr, "-d: Expected data root");
					return 1;
				}
				if (len + 2 > sizeof(data_root)) {
					fprintf(stderr, "-d: Data root too long");
					return 1;
				}
				memcpy(data_root, argv[i], len);
				char end = argv[i][len - 1];
				if (end == '/' || end == '\\') {
					data_root[len - 1] = '\0';
				}
			}
		}
		if (!strcmp(argv[i], "-o") || !strcmp(argv[i], "--output")) {
			if (++i < argc) {
				size_t len = strlen(argv[i]);
				if (len == 0) {
					fprintf(stderr, "-d: Expected output root");
					return 1;
				}
				if (len + 2 > sizeof(output_root)) {
					fprintf(stderr, "-d: Output root too long");
					return 1;
				}
				memcpy(output_root, argv[i], len);
				char end = argv[i][len - 1];
				if (end == '/' || end == '\\') {
					output_root[len - 1] = '\0';
				}
			}
		}
		if (!strcmp(argv[i], "--fuzz")) {
			g_fuzz = true;
		}
		if (!strcmp(argv[i], "-g") || !strcmp(argv[i], "--group")) {
			if (++i < argc) {
				test_group = argv[i];
			}
		}
	}

	uint32_t num_ran = 0;
	for (uint32_t i = 0; i < num_tests; i++) {
		ufbxwt_test *test = &g_tests[i];
		ufbxwt_test_stats *group_stats = ufbxwt_get_test_group(test->group);
		group_stats->num_total++;

		if (test_filter && strcmp(test->name, test_filter)) {
			continue;
		}
		if (test_group && strcmp(test->group, test_group)) {
			continue;
		}

		group_stats->num_ran++;
		num_ran++;
		bool print_always = false;
		if (ufbxwt_run_test(test)) {
			num_ok++;
			group_stats->num_ok++;
		} else {
			print_always = true;
		}

		ufbxwt_log_flush(print_always);
	}

	if (num_ok < num_tests) {
		printf("\n");
		for (uint32_t i = 0; i < num_tests; i++) {
			ufbxwt_test *test = &g_tests[i];
			if (test->fail.failed) {
				ufbxwt_fail *fail = &test->fail;
				const char *file = fail->file, *find;
				find = strrchr(file, '/');
				file = find ? find + 1 : file;
				find = strrchr(file, '\\');
				file = find ? find + 1 : file;
				printf("(%s) %s:%u: %s\n", test->name,
					file, fail->line, fail->expr);
			}
		}
	}

	printf("\nTests passed: %u/%u\n", num_ok, num_ran);

	if (g_verbose) {
		size_t num_skipped = 0;
		for (size_t i = 0; i < g_num_groups; i++) {
			ufbxwt_test_stats *group = &g_test_groups[i];
			if (group->num_ran == 0) {
				num_skipped++;
				continue;
			}
			printf("  %s: %u/%u\n", group->name, group->num_ok, group->num_ran);
		}
		if (num_skipped > 0) {
			printf("  .. skipped %zu groups\n", num_skipped);
		}
	}

	return num_ok == num_ran ? 0 : 1;
}


