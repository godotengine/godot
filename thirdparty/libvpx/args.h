/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_ARGS_H_
#define VPX_ARGS_H_
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

struct arg {
  char **argv;
  const char *name;
  const char *val;
  unsigned int argv_step;
  const struct arg_def *def;
};

struct arg_enum_list {
  const char *name;
  int val;
};
#define ARG_ENUM_LIST_END { 0 }

typedef struct arg_def {
  const char *short_name;
  const char *long_name;
  int has_val;
  const char *desc;
  const struct arg_enum_list *enums;
} arg_def_t;
#define ARG_DEF(s, l, v, d) { s, l, v, d, NULL }
#define ARG_DEF_ENUM(s, l, v, d, e) { s, l, v, d, e }
#define ARG_DEF_LIST_END { 0 }

int arg_match(struct arg *arg_, const struct arg_def *def, char **argv);
const char *arg_next(struct arg *arg);
void arg_show_usage(FILE *fp, const struct arg_def *const *defs);
char **argv_dup(int argc, const char **argv);

// Note: arg_match() must be called before invoking these functions.
unsigned int arg_parse_uint(const struct arg *arg);
int arg_parse_int(const struct arg *arg);
struct vpx_rational arg_parse_rational(const struct arg *arg);
int arg_parse_enum(const struct arg *arg);
int arg_parse_enum_or_int(const struct arg *arg);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_ARGS_H_
