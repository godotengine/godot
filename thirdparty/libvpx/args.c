/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "args.h"

#include "vpx/vpx_integer.h"

#if defined(__GNUC__)
__attribute__((noreturn)) extern void die(const char *fmt, ...);
#elif defined(_MSC_VER)
__declspec(noreturn) extern void die(const char *fmt, ...);
#else
extern void die(const char *fmt, ...);
#endif

static struct arg arg_init(char **argv) {
  struct arg a;

  a.argv = argv;
  a.argv_step = 1;
  a.name = NULL;
  a.val = NULL;
  a.def = NULL;
  return a;
}

int arg_match(struct arg *arg_, const struct arg_def *def, char **argv) {
  struct arg arg;

  if (!argv[0] || argv[0][0] != '-') return 0;

  arg = arg_init(argv);

  if (def->short_name && strlen(arg.argv[0]) == strlen(def->short_name) + 1 &&
      !strcmp(arg.argv[0] + 1, def->short_name)) {
    arg.name = arg.argv[0] + 1;
    arg.val = def->has_val ? arg.argv[1] : NULL;
    arg.argv_step = def->has_val ? 2 : 1;
  } else if (def->long_name) {
    const size_t name_len = strlen(def->long_name);

    if (strlen(arg.argv[0]) >= name_len + 2 && arg.argv[0][1] == '-' &&
        !strncmp(arg.argv[0] + 2, def->long_name, name_len) &&
        (arg.argv[0][name_len + 2] == '=' ||
         arg.argv[0][name_len + 2] == '\0')) {
      arg.name = arg.argv[0] + 2;
      arg.val = arg.name[name_len] == '=' ? arg.name + name_len + 1 : NULL;
      arg.argv_step = 1;
    }
  }

  if (arg.name && !arg.val && def->has_val)
    die("Error: option %s requires argument.\n", arg.name);

  if (arg.name && arg.val && !def->has_val)
    die("Error: option %s requires no argument.\n", arg.name);

  if (arg.name && (arg.val || !def->has_val)) {
    arg.def = def;
    *arg_ = arg;
    return 1;
  }

  return 0;
}

const char *arg_next(struct arg *arg) {
  if (arg->argv[0]) arg->argv += arg->argv_step;

  return *arg->argv;
}

char **argv_dup(int argc, const char **argv) {
  char **new_argv = malloc((argc + 1) * sizeof(*argv));
  if (!new_argv) return NULL;

  memcpy(new_argv, argv, argc * sizeof(*argv));
  new_argv[argc] = NULL;
  return new_argv;
}

void arg_show_usage(FILE *fp, const struct arg_def *const *defs) {
  for (; *defs; defs++) {
    const struct arg_def *def = *defs;
    char *short_val = def->has_val ? " <arg>" : "";
    char *long_val = def->has_val ? "=<arg>" : "";
    int n = 0;

    // Short options are indented with two spaces. Long options are indented
    // with 12 spaces.
    if (def->short_name && def->long_name) {
      char *comma = def->has_val ? "," : ",      ";

      n = fprintf(fp, "  -%s%s%s --%s%s", def->short_name, short_val, comma,
                  def->long_name, long_val);
    } else if (def->short_name)
      n = fprintf(fp, "  -%s%s", def->short_name, short_val);
    else if (def->long_name)
      n = fprintf(fp, "            --%s%s", def->long_name, long_val);

    // Descriptions are indented with 40 spaces. If an option is 40 characters
    // or longer, its description starts on the next line.
    if (n < 40)
      for (int i = 0; i < 40 - n; i++) fputc(' ', fp);
    else
      fputs("\n                                        ", fp);
    fprintf(fp, "%s\n", def->desc);

    if (def->enums) {
      const struct arg_enum_list *listptr;

      fprintf(fp, "  %-37s\t  ", "");

      for (listptr = def->enums; listptr->name; listptr++)
        fprintf(fp, "%s%s", listptr->name, listptr[1].name ? ", " : "\n");
    }
  }
}

unsigned int arg_parse_uint(const struct arg *arg) {
  uint32_t rawval;
  char *endptr;

  rawval = (uint32_t)strtoul(arg->val, &endptr, 10);

  if (arg->val[0] != '\0' && endptr[0] == '\0') {
    if (rawval <= UINT_MAX) return rawval;

    die("Option %s: Value %ld out of range for unsigned int\n", arg->name,
        rawval);
  }

  die("Option %s: Invalid character '%c'\n", arg->name, *endptr);
}

int arg_parse_int(const struct arg *arg) {
  int32_t rawval;
  char *endptr;

  rawval = (int32_t)strtol(arg->val, &endptr, 10);

  if (arg->val[0] != '\0' && endptr[0] == '\0') {
    if (rawval >= INT_MIN && rawval <= INT_MAX) return (int)rawval;

    die("Option %s: Value %ld out of range for signed int\n", arg->name,
        rawval);
  }

  die("Option %s: Invalid character '%c'\n", arg->name, *endptr);
}

struct vpx_rational {
  int num; /**< fraction numerator */
  int den; /**< fraction denominator */
};
struct vpx_rational arg_parse_rational(const struct arg *arg) {
  long int rawval;
  char *endptr;
  struct vpx_rational rat;

  /* parse numerator */
  rawval = strtol(arg->val, &endptr, 10);

  if (arg->val[0] != '\0' && endptr[0] == '/') {
    if (rawval >= INT_MIN && rawval <= INT_MAX)
      rat.num = (int)rawval;
    else
      die("Option %s: Value %ld out of range for signed int\n", arg->name,
          rawval);
  } else
    die("Option %s: Expected / at '%c'\n", arg->name, *endptr);

  /* parse denominator */
  rawval = strtol(endptr + 1, &endptr, 10);

  if (arg->val[0] != '\0' && endptr[0] == '\0') {
    if (rawval >= INT_MIN && rawval <= INT_MAX)
      rat.den = (int)rawval;
    else
      die("Option %s: Value %ld out of range for signed int\n", arg->name,
          rawval);
  } else
    die("Option %s: Invalid character '%c'\n", arg->name, *endptr);

  return rat;
}

int arg_parse_enum(const struct arg *arg) {
  const struct arg_enum_list *listptr;
  long int rawval;
  char *endptr;

  /* First see if the value can be parsed as a raw value */
  rawval = strtol(arg->val, &endptr, 10);
  if (arg->val[0] != '\0' && endptr[0] == '\0') {
    /* Got a raw value, make sure it's valid */
    for (listptr = arg->def->enums; listptr->name; listptr++)
      if (listptr->val == rawval) return (int)rawval;
  }

  /* Next see if it can be parsed as a string */
  for (listptr = arg->def->enums; listptr->name; listptr++)
    if (!strcmp(arg->val, listptr->name)) return listptr->val;

  die("Option %s: Invalid value '%s'\n", arg->name, arg->val);
}

int arg_parse_enum_or_int(const struct arg *arg) {
  if (arg->def->enums) return arg_parse_enum(arg);
  return arg_parse_int(arg);
}
