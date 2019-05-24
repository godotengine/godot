# Copyright 2009 The RE2 Authors.  All Rights Reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Bazel (http://bazel.io/) BUILD file for RE2.

licenses(["notice"])

exports_files(["LICENSE"])

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)

config_setting(
    name = "windows_msvc",
    values = {"cpu": "x64_windows_msvc"},
)

cc_library(
    name = "re2",
    srcs = [
        "re2/bitmap256.h",
        "re2/bitstate.cc",
        "re2/compile.cc",
        "re2/dfa.cc",
        "re2/filtered_re2.cc",
        "re2/mimics_pcre.cc",
        "re2/nfa.cc",
        "re2/onepass.cc",
        "re2/parse.cc",
        "re2/perl_groups.cc",
        "re2/prefilter.cc",
        "re2/prefilter.h",
        "re2/prefilter_tree.cc",
        "re2/prefilter_tree.h",
        "re2/prog.cc",
        "re2/prog.h",
        "re2/re2.cc",
        "re2/regexp.cc",
        "re2/regexp.h",
        "re2/set.cc",
        "re2/simplify.cc",
        "re2/stringpiece.cc",
        "re2/tostring.cc",
        "re2/unicode_casefold.cc",
        "re2/unicode_casefold.h",
        "re2/unicode_groups.cc",
        "re2/unicode_groups.h",
        "re2/walker-inl.h",
        "util/flags.h",
        "util/logging.h",
        "util/mix.h",
        "util/mutex.h",
        "util/pod_array.h",
        "util/rune.cc",
        "util/sparse_array.h",
        "util/sparse_set.h",
        "util/strutil.cc",
        "util/strutil.h",
        "util/utf.h",
        "util/util.h",
    ],
    hdrs = [
        "re2/filtered_re2.h",
        "re2/re2.h",
        "re2/set.h",
        "re2/stringpiece.h",
    ],
    copts = select({
        ":windows": [],
        ":windows_msvc": [],
        "//conditions:default": ["-pthread"],
    }),
    linkopts = select({
        # Darwin doesn't need `-pthread' when linking and it appears that
        # older versions of Clang will warn about the unused command line
        # argument, so just don't pass it.
        ":darwin": [],
        ":windows": [],
        ":windows_msvc": [],
        "//conditions:default": ["-pthread"],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "testing",
    testonly = 1,
    srcs = [
        "re2/testing/backtrack.cc",
        "re2/testing/dump.cc",
        "re2/testing/exhaustive_tester.cc",
        "re2/testing/null_walker.cc",
        "re2/testing/regexp_generator.cc",
        "re2/testing/string_generator.cc",
        "re2/testing/tester.cc",
        "util/pcre.cc",
    ],
    hdrs = [
        "re2/testing/exhaustive_tester.h",
        "re2/testing/regexp_generator.h",
        "re2/testing/string_generator.h",
        "re2/testing/tester.h",
        "util/benchmark.h",
        "util/pcre.h",
        "util/test.h",
    ],
    deps = [":re2"],
)

cc_library(
    name = "test",
    testonly = 1,
    srcs = ["util/test.cc"],
    deps = [":testing"],
)

load(":re2_test.bzl", "re2_test")

re2_test(
    "charclass_test",
    size = "small",
)

re2_test(
    "compile_test",
    size = "small",
)

re2_test(
    "filtered_re2_test",
    size = "small",
)

re2_test(
    "mimics_pcre_test",
    size = "small",
)

re2_test(
    "parse_test",
    size = "small",
)

re2_test(
    "possible_match_test",
    size = "small",
)

re2_test(
    "re2_arg_test",
    size = "small",
)

re2_test(
    "re2_test",
    size = "small",
)

re2_test(
    "regexp_test",
    size = "small",
)

re2_test(
    "required_prefix_test",
    size = "small",
)

re2_test(
    "search_test",
    size = "small",
)

re2_test(
    "set_test",
    size = "small",
)

re2_test(
    "simplify_test",
    size = "small",
)

re2_test(
    "string_generator_test",
    size = "small",
)

re2_test(
    "dfa_test",
    size = "large",
)

re2_test(
    "exhaustive1_test",
    size = "large",
)

re2_test(
    "exhaustive2_test",
    size = "large",
)

re2_test(
    "exhaustive3_test",
    size = "large",
)

re2_test(
    "exhaustive_test",
    size = "large",
)

re2_test(
    "random_test",
    size = "large",
)

cc_library(
    name = "benchmark",
    testonly = 1,
    srcs = ["util/benchmark.cc"],
    deps = [":testing"],
)

cc_binary(
    name = "regexp_benchmark",
    testonly = 1,
    srcs = ["re2/testing/regexp_benchmark.cc"],
    deps = [":benchmark"],
)
