
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
/** \file
 * This is a convenience header for Catch2. It includes **all** of Catch2 headers.
 *
 * Generally the Catch2 users should use specific includes they need,
 * but this header can be used instead for ease-of-experimentation, or
 * just plain convenience, at the cost of (significantly) increased
 * compilation times.
 *
 * When a new header is added to either the top level folder, or to the
 * corresponding internal subfolder, it should be added here. Headers
 * added to the various subparts (e.g. matchers, generators, etc...),
 * should go their respective catch-all headers.
 */

#ifndef CATCH_ALL_HPP_INCLUDED
#define CATCH_ALL_HPP_INCLUDED

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_assertion_info.hpp>
#include <catch2/catch_assertion_result.hpp>
#include <catch2/catch_case_sensitive.hpp>
#include <catch2/catch_config.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_section_info.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_tag_alias.hpp>
#include <catch2/catch_tag_alias_autoregistrar.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_test_run_info.hpp>
#include <catch2/catch_test_spec.hpp>
#include <catch2/catch_timer.hpp>
#include <catch2/catch_tostring.hpp>
#include <catch2/catch_totals.hpp>
#include <catch2/catch_translate_exception.hpp>
#include <catch2/catch_version.hpp>
#include <catch2/catch_version_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/interfaces/catch_interfaces_all.hpp>
#include <catch2/internal/catch_assertion_handler.hpp>
#include <catch2/internal/catch_case_insensitive_comparisons.hpp>
#include <catch2/internal/catch_clara.hpp>
#include <catch2/internal/catch_commandline.hpp>
#include <catch2/internal/catch_compare_traits.hpp>
#include <catch2/internal/catch_compiler_capabilities.hpp>
#include <catch2/internal/catch_config_android_logwrite.hpp>
#include <catch2/internal/catch_config_counter.hpp>
#include <catch2/internal/catch_config_prefix_messages.hpp>
#include <catch2/internal/catch_config_static_analysis_support.hpp>
#include <catch2/internal/catch_config_uncaught_exceptions.hpp>
#include <catch2/internal/catch_config_wchar.hpp>
#include <catch2/internal/catch_console_colour.hpp>
#include <catch2/internal/catch_console_width.hpp>
#include <catch2/internal/catch_container_nonmembers.hpp>
#include <catch2/internal/catch_context.hpp>
#include <catch2/internal/catch_debug_console.hpp>
#include <catch2/internal/catch_debugger.hpp>
#include <catch2/internal/catch_decomposer.hpp>
#include <catch2/internal/catch_deprecation_macro.hpp>
#include <catch2/internal/catch_enforce.hpp>
#include <catch2/internal/catch_enum_values_registry.hpp>
#include <catch2/internal/catch_errno_guard.hpp>
#include <catch2/internal/catch_exception_translator_registry.hpp>
#include <catch2/internal/catch_fatal_condition_handler.hpp>
#include <catch2/internal/catch_floating_point_helpers.hpp>
#include <catch2/internal/catch_getenv.hpp>
#include <catch2/internal/catch_is_permutation.hpp>
#include <catch2/internal/catch_istream.hpp>
#include <catch2/internal/catch_jsonwriter.hpp>
#include <catch2/internal/catch_lazy_expr.hpp>
#include <catch2/internal/catch_leak_detector.hpp>
#include <catch2/internal/catch_list.hpp>
#include <catch2/internal/catch_logical_traits.hpp>
#include <catch2/internal/catch_message_info.hpp>
#include <catch2/internal/catch_meta.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>
#include <catch2/internal/catch_noncopyable.hpp>
#include <catch2/internal/catch_optional.hpp>
#include <catch2/internal/catch_output_redirect.hpp>
#include <catch2/internal/catch_parse_numbers.hpp>
#include <catch2/internal/catch_platform.hpp>
#include <catch2/internal/catch_polyfills.hpp>
#include <catch2/internal/catch_preprocessor.hpp>
#include <catch2/internal/catch_preprocessor_internal_stringify.hpp>
#include <catch2/internal/catch_preprocessor_remove_parens.hpp>
#include <catch2/internal/catch_random_floating_point_helpers.hpp>
#include <catch2/internal/catch_random_integer_helpers.hpp>
#include <catch2/internal/catch_random_number_generator.hpp>
#include <catch2/internal/catch_random_seed_generation.hpp>
#include <catch2/internal/catch_reporter_registry.hpp>
#include <catch2/internal/catch_reporter_spec_parser.hpp>
#include <catch2/internal/catch_result_type.hpp>
#include <catch2/internal/catch_reusable_string_stream.hpp>
#include <catch2/internal/catch_run_context.hpp>
#include <catch2/internal/catch_section.hpp>
#include <catch2/internal/catch_sharding.hpp>
#include <catch2/internal/catch_singletons.hpp>
#include <catch2/internal/catch_source_line_info.hpp>
#include <catch2/internal/catch_startup_exception_registry.hpp>
#include <catch2/internal/catch_stdstreams.hpp>
#include <catch2/internal/catch_stream_end_stop.hpp>
#include <catch2/internal/catch_string_manip.hpp>
#include <catch2/internal/catch_stringref.hpp>
#include <catch2/internal/catch_tag_alias_registry.hpp>
#include <catch2/internal/catch_template_test_registry.hpp>
#include <catch2/internal/catch_test_case_info_hasher.hpp>
#include <catch2/internal/catch_test_case_registry_impl.hpp>
#include <catch2/internal/catch_test_case_tracker.hpp>
#include <catch2/internal/catch_test_failure_exception.hpp>
#include <catch2/internal/catch_test_macro_impl.hpp>
#include <catch2/internal/catch_test_registry.hpp>
#include <catch2/internal/catch_test_spec_parser.hpp>
#include <catch2/internal/catch_textflow.hpp>
#include <catch2/internal/catch_thread_support.hpp>
#include <catch2/internal/catch_to_string.hpp>
#include <catch2/internal/catch_uncaught_exceptions.hpp>
#include <catch2/internal/catch_uniform_floating_point_distribution.hpp>
#include <catch2/internal/catch_uniform_integer_distribution.hpp>
#include <catch2/internal/catch_unique_name.hpp>
#include <catch2/internal/catch_unique_ptr.hpp>
#include <catch2/internal/catch_unreachable.hpp>
#include <catch2/internal/catch_void_type.hpp>
#include <catch2/internal/catch_wildcard_pattern.hpp>
#include <catch2/internal/catch_xmlwriter.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <catch2/reporters/catch_reporters_all.hpp>

#endif // CATCH_ALL_HPP_INCLUDED
