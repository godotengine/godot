#ifndef IM_ARG_INCLUDED_H
#define IM_ARG_INCLUDED_H

#include <stdbool.h>
#include <stddef.h>
#include <limits.h>
#include <stdarg.h>

#ifndef IM_ARG_MAX_ALIAS 
#define IM_ARG_MAX_ALIAS 4
#endif

typedef enum im_arg_default {
	IM_ARG_DEFAULT_HELP = 0x1,
	IM_ARG_DEFAULT_USAGE = 0x2,
} im_arg_default;

#if defined(__cplusplus)
extern "C" {
#endif

void im_arg_begin(const char *exe, size_t argc, const char *const *argv);
void im_arg_begin_c(int argc, char **argv);

bool im_arg_next();

void im_arg_show_help();
bool im_arg_help(const char *help, const char *description);

void im_arg_helpf(const char *fmt, ...);
void im_arg_vhelpf(const char *fmt, va_list args);

size_t im_arg_count();
int im_arg_int(size_t index);
const char *im_arg_str(size_t index);

void im_arg_fail(const char *description);
void im_arg_failf(const char *fmt, ...);

void im_arg_check(bool condition, const char *description);
void im_arg_checkf(bool condition, const char *fmt, ...);

void im_arg_category(const char *description);

bool im_arg(const char *fmt, const char *description);
bool im_arg_unknown();

typedef struct im_arg_context im_arg_context;

void im_arg_begin_ctx(im_arg_context *ctx, const char *exe, size_t argc, const char *const *argv);
void im_arg_begin_c_ctx(im_arg_context *ctx, int argc, char **argv);

size_t im_arg_count_ctx(im_arg_context *ctx);
int im_arg_int_ctx(im_arg_context *ctx, size_t index);
const char *im_arg_str_ctx(im_arg_context *ctx, size_t index);

void im_arg_show_help_ctx(im_arg_context *ctx);
bool im_arg_help_ctx(im_arg_context *ctx, const char *help, const char *description);

void im_arg_helpf_ctx(im_arg_context *ctx, const char *fmt, ...);
void im_arg_vhelpf_ctx(im_arg_context *ctx, const char *fmt, va_list args);

bool im_arg_next_ctx(im_arg_context *ctx);
void im_arg_category_ctx(im_arg_context *ctx, const char *description);
bool im_arg_ctx(im_arg_context *ctx, const char *fmt, const char *description);
bool im_arg_unknown_ctx(im_arg_context *ctx);

#if defined(__cplusplus)
}
#endif

#endif

#ifdef IM_ARG_IMPLEMENTATION
#ifndef IM_ARG_IMPLEMENTED_H
#define IM_ARG_IMPLEMENTED_H

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

typedef enum {
	IM_ARG__STATE_INITIAL,
	IM_ARG__STATE_PARSE,
	IM_ARG__STATE_MAIN_OPT,
	IM_ARG__STATE_MAIN_ARG,
	IM_ARG__STATE_CHECK,
	IM_ARG__STATE_PRE_HELP,
	IM_ARG__STATE_HELP_MEASURE,
	IM_ARG__STATE_HELP,
} im_arg__state;

typedef enum {
	IM_ARG__COUNT_UNSPECIFIED,
	IM_ARG__COUNT_ANY,
	IM_ARG__COUNT_ZERO_OR_ONE,
	IM_ARG__COUNT_ONE,
	IM_ARG__COUNT_ONE_OR_MORE,
} im_arg__count;

static void im_arg__expand_count(size_t *min_count, size_t *max_count, im_arg__count count)
{
	switch (count) {
	case IM_ARG__COUNT_UNSPECIFIED: break;
	case IM_ARG__COUNT_ANY: *min_count = 0; *max_count = SIZE_MAX; break;
	case IM_ARG__COUNT_ZERO_OR_ONE: *min_count = 0; *max_count = 1; break;
	case IM_ARG__COUNT_ONE: *min_count = 1; *max_count = 1; break;
	case IM_ARG__COUNT_ONE_OR_MORE: *min_count = 1; *max_count = SIZE_MAX; break;
	}
}

typedef struct {
	uint16_t offset;
	uint16_t length;
	im_arg__count count;
} im_arg__arg;

typedef struct {
	uint16_t alias_count;
	uint16_t alias_offset[IM_ARG_MAX_ALIAS];
	uint16_t alias_length[IM_ARG_MAX_ALIAS];
	uint16_t alias_hash[IM_ARG_MAX_ALIAS];

	uint16_t positional_length;

	uint16_t arg_count;
	uint32_t arg_offset;

	im_arg__count count;

	size_t seen_count;
} im_arg__opt;

typedef struct {
	size_t option_width;
} im_arg__cat;

typedef struct {
	const char *ptr;
	int len;
} im_arg__str_span;

struct im_arg_context {
	size_t argc;
	const char *const *argv;

	const char *opt;
	size_t opt_length;
	uint16_t opt_hash;

	im_arg__state state;
	size_t start_index;
	size_t index;
	bool handled;

	bool *argv_handled;

	size_t arg_base;
	size_t arg_count;

	size_t opt_index;
	size_t cat_index;

	im_arg__cat *cat;
	im_arg__cat default_cat;

	im_arg__opt *opts;
	size_t num_opts;
	size_t cap_opts;

	im_arg__arg *args;
	size_t num_args;
	size_t cap_args;

	im_arg__cat *cats;
	size_t num_cats;
	size_t cap_cats;

	size_t max_width;
};

#define im_arg__assert(cond) ((void)0)

#define im_arg__grow(p_type, p_arr, p_cap, p_count, p_default) do { \
		if ((p_count) > p_cap) { \
			p_cap = p_cap ? p_cap * 2 : p_default; \
		} \
		p_arr = (p_type*)realloc(p_arr, p_cap * sizeof(p_type)); \
		im_arg__assert(p_arr != NULL); \
	} while (0)

static void im_arg__errorf(im_arg_context *ctx, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	va_end(args);
	fprintf(stderr, "\n");
	exit(1);
}

static uint16_t im_arg__hash(const char *str, size_t length)
{
	uint32_t hash = 0x811c9dc5;
	for (size_t i = 0; i < length; i++) {
		hash = (hash ^ (uint32_t)(uint8_t)str[i]) * 0x01000193;
	}
	return (uint16_t)(hash >> 16u);
}

static bool im_arg__parse_count(im_arg__count *count, const char **fmt)
{
	switch (**fmt) {
	case '*': *count = IM_ARG__COUNT_ANY; break;
	case '?': *count = IM_ARG__COUNT_ZERO_OR_ONE; break;
	case '!': *count = IM_ARG__COUNT_ONE; break;
	case '+': *count = IM_ARG__COUNT_ONE_OR_MORE; break;
	default: *count = IM_ARG__COUNT_UNSPECIFIED; return false;
	}
	++*fmt;
	return true;
}

static void im_arg__parse_opt(im_arg__opt *opt, const char **fmt)
{
	const char *p_begin = *fmt, *p = p_begin;
	if (*p == '-') {
		const char *begin = p;
		for (;;) {
			if (*p == '/' || *p == ' ' || *p == '\0') {
				size_t length = (size_t)(p - begin);

				im_arg__assert(opt->alias_count < IM_ARG_MAX_ALIAS);
				size_t index = opt->alias_count++;
				opt->alias_offset[index] = (uint16_t)(size_t)(begin - p_begin);
				opt->alias_length[index] = (uint16_t)length;
				opt->alias_hash[index] = im_arg__hash(begin, length);

				if (*p == '/') {
					begin = ++p;
				} else if (*p == ' ') {
					++p;
					break;
				} else {
					break;
				}
			} else {
				p++;
			}
		}
	} else {
		const char *begin = p;
		while (*p && *p != '*' && *p != '?' && *p != '!' && *p != '+') {
			p++;
		}
		opt->positional_length = (uint16_t)(size_t)(p - begin);
	}

	*fmt = p;
}

static void im_arg__parse_arg(im_arg__arg *arg, const char **fmt, size_t offset)
{
	const char *p_begin = *fmt, *p = p_begin;

	if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z')) {
		const char *begin = p;
		while (*p != ' ' && *p != '?' && *p != '!' && *p != '*' && *p != '+' && *p != '\0') {
			p++;
		}

		size_t length = (size_t)(p - begin);
		arg->offset = (uint16_t)offset;
		arg->length = (uint16_t)length;
		im_arg__parse_count(&arg->count, &p);

		if (*p == ' ') {
			p++;
		}

		*fmt = p;
	} else {
		im_arg__assert(0);
	}
}

static im_arg__str_span im_arg__opt_name(im_arg_context *ctx, im_arg__opt *opt, const char *fmt)
{
	im_arg__str_span res;
	if (opt->alias_count > 0) {
		size_t last = opt->alias_count - 1;
		res.ptr = fmt;
		res.len = (int)opt->alias_offset[last] + (int)opt->alias_length[last];
	} else if (opt->arg_count > 0) {
		im_arg__arg *arg = &ctx->args[opt->arg_offset];
		res.ptr = fmt + arg->offset;
		res.len = (int)arg->length;
	} else {
		res.ptr = fmt;
		res.len = opt->positional_length;
	}
	return res;
}

im_arg_context im_arg_global;

#if defined(__cplusplus)
extern "C" {
#endif

void im_arg_begin_ctx(im_arg_context *ctx, const char *exe, size_t argc, const char *const *argv)
{
	ctx->argc = argc;
	ctx->argv = argv;
	ctx->max_width = 40;
	ctx->argv_handled = (bool*)calloc(argc, 1);
	im_arg__assert(ctx->argv_handled);
}

void im_arg_begin_c_ctx(im_arg_context *ctx, int argc, char **argv)
{
	im_arg_begin_ctx(ctx, argv[0], (size_t)argc - 1, (const char *const *)argv + 1);
}

void im_arg_begin(const char *exe, size_t argc, const char *const *argv)
{
	im_arg_begin_ctx(&im_arg_global, exe, argc, argv);
}

void im_arg_begin_c(int argc, char **argv)
{
	im_arg_begin_c_ctx(&im_arg_global, argc, argv);
}

bool im_arg_help(const char *help, const char *description)
{
	return im_arg_help_ctx(&im_arg_global, help, description);
}

void im_arg_helpf(const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	im_arg_vhelpf_ctx(&im_arg_global, fmt, args);
	va_end(args);
}

void im_arg_vhelpf(const char *fmt, va_list args)
{
	im_arg_vhelpf_ctx(&im_arg_global, fmt, args);
}

const char *im_arg_str(size_t index)
{
	return im_arg_str_ctx(&im_arg_global, index);
}

int im_arg_int(size_t index)
{
	return im_arg_int_ctx(&im_arg_global, index);
}

const char *im_arg_str_ctx(im_arg_context *ctx, size_t index)
{
	im_arg__assert(index < ctx->arg_count);
	return ctx->argv[ctx->arg_base + index];
}

size_t im_arg_count_ctx(im_arg_context *ctx)
{
	return ctx->arg_count;
}

int im_arg_int_ctx(im_arg_context *ctx, size_t index)
{
	const char *arg = im_arg_str_ctx(ctx, index);
	char *end = NULL;
	long value = strtol(arg, &end, 10);
	if (!end || *end) {
		im_arg__errorf(ctx, "expected integer for '%s' argument %zu, got '%s'", ctx->opt, index, arg);
	}
	return (int)value;
}

bool im_arg_help_ctx(im_arg_context *ctx, const char *help, const char *description)
{
	if (im_arg_ctx(ctx, help, description)) {
		im_arg_show_help_ctx(ctx);
		return true;
	} else {
		return false;
	}
}

void im_arg_helpf_ctx(im_arg_context *ctx, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);
	im_arg_vhelpf_ctx(ctx, fmt, args);
	va_end(args);
}

void im_arg_vhelpf_ctx(im_arg_context *ctx, const char *fmt, va_list args)
{
	if (ctx->state == IM_ARG__STATE_HELP) {
		vprintf(fmt, args);
	}
}

void im_arg_category(const char *description)
{
	im_arg_category_ctx(&im_arg_global, description);
}

bool im_arg(const char *fmt, const char *description)
{
	return im_arg_ctx(&im_arg_global, fmt, description);
}

bool im_arg_unknown()
{
	return im_arg_unknown_ctx(&im_arg_global);
}

static void im_arg_printf(im_arg_context *ctx, const char *fmt, ...)
{
}

void im_arg_show_help()
{
	im_arg_show_help_ctx(&im_arg_global);
}

void im_arg_show_help_ctx(im_arg_context *ctx)
{
	ctx->state = IM_ARG__STATE_PRE_HELP;
}

static void im_arg__free_ctx(im_arg_context *ctx)
{
	free(ctx->opts);
	free(ctx->args);
	free(ctx->cats);
	free(ctx->argv_handled);
}

bool im_arg_next()
{
	return im_arg_next_ctx(&im_arg_global);
}

bool im_arg_next_ctx(im_arg_context *ctx)
{
	ctx->opt_index = 0;
	ctx->cat_index = 0;
	ctx->cat = &ctx->default_cat;
	if (ctx->state == IM_ARG__STATE_INITIAL) {
		ctx->state = IM_ARG__STATE_PARSE;
		return true;
	} else if (ctx->state == IM_ARG__STATE_PARSE) {
		ctx->state = IM_ARG__STATE_MAIN_OPT;
		return im_arg_next_ctx(ctx);
	} else if (ctx->state == IM_ARG__STATE_MAIN_OPT || ctx->state == IM_ARG__STATE_MAIN_ARG) {
		if (ctx->index > 0 && !ctx->handled) {
			if (ctx->state == IM_ARG__STATE_MAIN_ARG) {
				if (ctx->opt[0] == '-') {
					im_arg__errorf(ctx, "unknown option '%s'", ctx->opt);
				} else {
					im_arg__errorf(ctx, "unexpected argument '%s'", ctx->opt);
				}
			}
		}

		if (ctx->handled) {
			for (size_t i = ctx->start_index; i < ctx->index; i++) {
				ctx->argv_handled[i] = true;
			}
		}

		ctx->handled = false;
		while (ctx->index < ctx->argc && ctx->argv_handled[ctx->index]) {
			ctx->index++;
		}

		if (ctx->index < ctx->argc) {
			ctx->opt = ctx->argv[ctx->index];
			ctx->opt_length = strlen(ctx->opt);
			ctx->opt_hash = im_arg__hash(ctx->opt, ctx->opt_length);
			ctx->start_index = ctx->index;
			ctx->index++;
			return true;
		} else {
			if (ctx->state == IM_ARG__STATE_MAIN_OPT) {
				ctx->state = IM_ARG__STATE_MAIN_ARG;
				ctx->index = 0;
				ctx->start_index = 0;
				return im_arg_next_ctx(ctx);
			} else {
				ctx->state = IM_ARG__STATE_CHECK;
				return true;
			}
		}
	} else if (ctx->state == IM_ARG__STATE_CHECK) {
		im_arg__free_ctx(ctx);
		return false;
	} else if (ctx->state == IM_ARG__STATE_PRE_HELP) {
		ctx->state = IM_ARG__STATE_HELP_MEASURE;
		return true;
	} else if (ctx->state == IM_ARG__STATE_HELP_MEASURE) {
		ctx->state = IM_ARG__STATE_HELP;
		return true;
	} else if (ctx->state == IM_ARG__STATE_HELP) {
		im_arg__free_ctx(ctx);
		exit(0);
		return false;
	} else {
		im_arg__assert(0);
		return false;
	}
}

void im_arg_category_ctx(im_arg_context *ctx, const char *description)
{
	if (ctx->state == IM_ARG__STATE_PARSE) {
		im_arg__grow(im_arg__cat, ctx->cats, ctx->cap_cats, ctx->num_cats + 1, 16);
		im_arg__cat *cat = &ctx->cats[ctx->num_cats++];
		memset(cat, 0, sizeof(im_arg__cat));
		ctx->cat = cat;
	} else {
		im_arg__assert(ctx->cat_index < ctx->num_cats);
		ctx->cat = &ctx->cats[ctx->cat_index++];
	}

	if (ctx->state == IM_ARG__STATE_HELP) {
		printf("%s\n", description);
	}
}

bool im_arg_ctx(im_arg_context *ctx, const char *fmt, const char *description)
{
	im_arg__opt *opt = ctx->opts ? &ctx->opts[ctx->opt_index++] : NULL;

	if (ctx->state == IM_ARG__STATE_PARSE) {
		im_arg__grow(im_arg__opt, ctx->opts, ctx->cap_opts, ctx->num_opts + 1, 64);
		im_arg__opt *opt = &ctx->opts[ctx->num_opts++];
		memset(opt, 0, sizeof(im_arg__opt));
		opt->arg_offset = (uint32_t)ctx->num_args;

		const char *p = fmt;
		im_arg__parse_opt(opt, &p);
		while (*p) {
			if (im_arg__parse_count(&opt->count, &p)) {
				im_arg__assert(*p == '\0');
				break;
			}

			im_arg__grow(im_arg__arg, ctx->args, ctx->cap_args, ctx->num_args + 1, 64);
			im_arg__arg *arg = &ctx->args[ctx->num_args++];
			memset(arg, 0, sizeof(im_arg__arg));
			opt->arg_count++;

			im_arg__parse_arg(arg, &p, (size_t)(p - fmt));
		}

		if (opt->count == IM_ARG__COUNT_UNSPECIFIED) {
			opt->count = opt->alias_count > 0 ? IM_ARG__COUNT_ZERO_OR_ONE : IM_ARG__COUNT_ONE;
		}

		return false;
	} else if (ctx->state == IM_ARG__STATE_MAIN_OPT || ctx->state == IM_ARG__STATE_MAIN_ARG) {
		if (ctx->handled) return false;

		bool match = false;
		if (opt->alias_count > 0) {
			if (ctx->state == IM_ARG__STATE_MAIN_ARG) return false;
			for (size_t i = 0; i < opt->alias_count; i++) {
				size_t length = opt->alias_length[i];
				if (opt->alias_hash[i] == ctx->opt_hash && length == ctx->opt_length) {
					const char *alias = fmt + opt->alias_offset[i];
					if (!memcmp(alias, ctx->opt, length)) {
						match = true;
						break;
					}
				}
			}
		} else {
			if (ctx->state == IM_ARG__STATE_MAIN_ARG) {
				match = true;
			}
		}

		if (match) {
			if (++opt->seen_count > 1) {
				if (opt->count == IM_ARG__COUNT_ONE) {
					if (opt->alias_count == 0) return false;
					im_arg__errorf(ctx, "expected only once");
				} else if (opt->count == IM_ARG__COUNT_ZERO_OR_ONE) {
					if (opt->alias_count == 0) return false;
					im_arg__errorf(ctx, "expected at most once");
				}
			}

			if (opt->alias_count == 0) {
				ctx->arg_base = ctx->index - 1;
			}

			for (size_t i = 0; i < opt->arg_count; i++) {
				im_arg__arg *arg = &ctx->args[opt->arg_offset + i];
				size_t min_count = 1, max_count = 1;
				im_arg__expand_count(&min_count, &max_count, arg->count);

				const char *name = fmt + arg->offset;

				ctx->arg_base = ctx->index;

				size_t count;
				for (count = 0; count < max_count; count++) {
					if (ctx->index >= ctx->argc || ctx->argv_handled[ctx->index]) break;
					const char *arg = ctx->argv[ctx->index];
					if (count >= min_count && arg[0] == '-') break;
					ctx->index++;
				}

				ctx->arg_count = count;
				if (count < min_count) {
					if (arg->count == IM_ARG__COUNT_ONE) {
						im_arg__errorf(ctx, "expected one argument for '%.*s'", (int)arg->length, name);
					} else if (arg->count == IM_ARG__COUNT_ONE_OR_MORE) {
						im_arg__errorf(ctx, "expected one or more arguments for '%.*s'", (int)arg->length, name);
					}
				}
			}

			ctx->handled = true;
			return true;
		} else {
			return false;
		}

	} else if (ctx->state == IM_ARG__STATE_CHECK) {
		if (opt->seen_count == 0) {
			im_arg__str_span name = im_arg__opt_name(ctx, opt, fmt);
			if (opt->count == IM_ARG__COUNT_ONE || opt->count == IM_ARG__COUNT_ONE_OR_MORE) {
				im_arg__errorf(ctx, "missing argument '%.*s'", name.len, name.ptr);
			}
		}
		return false;
	} else if (ctx->state == IM_ARG__STATE_PRE_HELP) {
		return false;
	} else if (ctx->state == IM_ARG__STATE_HELP || ctx->state == IM_ARG__STATE_HELP_MEASURE) {
		if (!description || !*description) return false;

		const char *name = "";
		int name_length = 0;

		bool do_print = ctx->state == IM_ARG__STATE_HELP;

		if (do_print) {
			printf("  ");
		}
		size_t width = 0;
		if (opt->alias_count > 0) {
			im_arg__str_span name = im_arg__opt_name(ctx, opt, fmt);
			if (do_print) {
				printf("%.*s", (int)name.len, name.ptr);
			}
			width += (size_t)name.len;
		} else if (opt->positional_length > 0) {
			im_arg__str_span name;
			name.ptr = fmt;
			name.len = opt->positional_length;
			if (do_print) {
				printf("%.*s", (int)name.len, name.ptr);
			}
			width += (size_t)name.len;
		}

		for (size_t i = 0; i < opt->arg_count; i++) {
			im_arg__arg *arg = &ctx->args[opt->arg_offset + i];
			const char *prefix = i > 0 || opt->alias_count > 0 ? " " : "";
			const char *suffix = "";
			switch (arg->count) {
			case IM_ARG__COUNT_ANY: suffix = "*"; break;
			case IM_ARG__COUNT_ZERO_OR_ONE: suffix = "?"; break;
			case IM_ARG__COUNT_ONE_OR_MORE: suffix = "+"; break;
			}
			if (do_print) {
				printf("%s%.*s%s", prefix, (int)arg->length, fmt + arg->offset, suffix);
			}
			width += strlen(prefix) + (size_t)arg->length + strlen(suffix);
		}

		if (!do_print && width < ctx->max_width) {
			if (width > ctx->cat->option_width) {
				ctx->cat->option_width = width;
			}
		}

		if (do_print) {
			int left = (int)(ctx->cat->option_width - width);
			while (left > 0) {
				printf("%.*s", left <= 16 ? left : 16, "                ");
				left -= 16;
			}
			printf("  %s\n", description);
		}

		return false;
	} else {
		im_arg__assert(0);
		return false;
	}
}

bool im_arg_unknown_ctx(im_arg_context *ctx)
{
	if (ctx->state == IM_ARG__STATE_MAIN_ARG && !ctx->handled) {
		ctx->handled = true;
		return true;
	} else {
		return false;
	}
}

#if defined(__cplusplus)
}
#endif

#endif
#endif
