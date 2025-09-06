#              Copyright Catch2 Authors
# Distributed under the Boost Software License, Version 1.0.
#   (See accompanying file LICENSE.txt or copy at
#        https://www.boost.org/LICENSE_1_0.txt)

# SPDX-License-Identifier: BSL-1.0
pkg = import('pkgconfig')

conf_data = configuration_data()
conf_data.set('CATCH_CONFIG_DEFAULT_REPORTER', 'console')
conf_data.set('CATCH_CONFIG_CONSOLE_WIDTH', '80')

configure_file(
  input: 'catch_user_config.hpp.in',
  output: 'catch_user_config.hpp',
  format: 'cmake@',
  install_dir: get_option('includedir') / 'catch2',
  configuration: conf_data,
  install: get_option('install')
)

fs = import('fs')

benchmark_headers = [
  'benchmark/catch_benchmark.hpp',
  'benchmark/catch_benchmark_all.hpp',
  'benchmark/catch_chronometer.hpp',
  'benchmark/catch_clock.hpp',
  'benchmark/catch_constructor.hpp',
  'benchmark/catch_environment.hpp',
  'benchmark/catch_estimate.hpp',
  'benchmark/catch_execution_plan.hpp',
  'benchmark/catch_optimizer.hpp',
  'benchmark/catch_outlier_classification.hpp',
  'benchmark/catch_sample_analysis.hpp',
  'benchmark/detail/catch_analyse.hpp',
  'benchmark/detail/catch_benchmark_function.hpp',
  'benchmark/detail/catch_benchmark_stats.hpp',
  'benchmark/detail/catch_benchmark_stats_fwd.hpp',
  'benchmark/detail/catch_complete_invoke.hpp',
  'benchmark/detail/catch_estimate_clock.hpp',
  'benchmark/detail/catch_measure.hpp',
  'benchmark/detail/catch_repeat.hpp',
  'benchmark/detail/catch_run_for_at_least.hpp',
  'benchmark/detail/catch_stats.hpp',
  'benchmark/detail/catch_timing.hpp',
]

benchmark_sources = files(
  'benchmark/catch_chronometer.cpp',
  'benchmark/detail/catch_analyse.cpp',
  'benchmark/detail/catch_benchmark_function.cpp',
  'benchmark/detail/catch_run_for_at_least.cpp',
  'benchmark/detail/catch_stats.cpp',
)

internal_headers = [
  'generators/catch_generator_exception.hpp',
  'generators/catch_generators.hpp',
  'generators/catch_generators_adapters.hpp',
  'generators/catch_generators_all.hpp',
  'generators/catch_generators_random.hpp',
  'generators/catch_generators_range.hpp',
  'interfaces/catch_interfaces_all.hpp',
  'interfaces/catch_interfaces_capture.hpp',
  'interfaces/catch_interfaces_config.hpp',
  'interfaces/catch_interfaces_enum_values_registry.hpp',
  'interfaces/catch_interfaces_exception.hpp',
  'interfaces/catch_interfaces_generatortracker.hpp',
  'interfaces/catch_interfaces_registry_hub.hpp',
  'interfaces/catch_interfaces_reporter.hpp',
  'interfaces/catch_interfaces_reporter_factory.hpp',
  'interfaces/catch_interfaces_tag_alias_registry.hpp',
  'interfaces/catch_interfaces_test_invoker.hpp',
  'interfaces/catch_interfaces_testcase.hpp',
  'internal/catch_assertion_handler.hpp',
  'internal/catch_case_insensitive_comparisons.hpp',
  'internal/catch_clara.hpp',
  'internal/catch_commandline.hpp',
  'internal/catch_compare_traits.hpp',
  'internal/catch_compiler_capabilities.hpp',
  'internal/catch_config_android_logwrite.hpp',
  'internal/catch_config_counter.hpp',
  'internal/catch_config_prefix_messages.hpp',
  'internal/catch_config_static_analysis_support.hpp',
  'internal/catch_config_uncaught_exceptions.hpp',
  'internal/catch_config_wchar.hpp',
  'internal/catch_console_colour.hpp',
  'internal/catch_console_width.hpp',
  'internal/catch_container_nonmembers.hpp',
  'internal/catch_context.hpp',
  'internal/catch_debug_console.hpp',
  'internal/catch_debugger.hpp',
  'internal/catch_decomposer.hpp',
  'internal/catch_deprecation_macro.hpp',
  'internal/catch_enforce.hpp',
  'internal/catch_enum_values_registry.hpp',
  'internal/catch_errno_guard.hpp',
  'internal/catch_exception_translator_registry.hpp',
  'internal/catch_fatal_condition_handler.hpp',
  'internal/catch_floating_point_helpers.hpp',
  'internal/catch_getenv.hpp',
  'internal/catch_istream.hpp',
  'internal/catch_is_permutation.hpp',
  'internal/catch_jsonwriter.hpp',
  'internal/catch_lazy_expr.hpp',
  'internal/catch_leak_detector.hpp',
  'internal/catch_list.hpp',
  'internal/catch_logical_traits.hpp',
  'internal/catch_message_info.hpp',
  'internal/catch_meta.hpp',
  'internal/catch_move_and_forward.hpp',
  'internal/catch_noncopyable.hpp',
  'internal/catch_optional.hpp',
  'internal/catch_output_redirect.hpp',
  'internal/catch_parse_numbers.hpp',
  'internal/catch_platform.hpp',
  'internal/catch_polyfills.hpp',
  'internal/catch_preprocessor.hpp',
  'internal/catch_preprocessor_internal_stringify.hpp',
  'internal/catch_preprocessor_remove_parens.hpp',
  'internal/catch_random_floating_point_helpers.hpp',
  'internal/catch_random_integer_helpers.hpp',
  'internal/catch_random_number_generator.hpp',
  'internal/catch_random_seed_generation.hpp',
  'internal/catch_reporter_registry.hpp',
  'internal/catch_reporter_spec_parser.hpp',
  'internal/catch_result_type.hpp',
  'internal/catch_reusable_string_stream.hpp',
  'internal/catch_run_context.hpp',
  'internal/catch_section.hpp',
  'internal/catch_sharding.hpp',
  'internal/catch_singletons.hpp',
  'internal/catch_source_line_info.hpp',
  'internal/catch_startup_exception_registry.hpp',
  'internal/catch_stdstreams.hpp',
  'internal/catch_stream_end_stop.hpp',
  'internal/catch_string_manip.hpp',
  'internal/catch_stringref.hpp',
  'internal/catch_tag_alias_registry.hpp',
  'internal/catch_template_test_registry.hpp',
  'internal/catch_test_case_info_hasher.hpp',
  'internal/catch_test_case_registry_impl.hpp',
  'internal/catch_test_case_tracker.hpp',
  'internal/catch_test_failure_exception.hpp',
  'internal/catch_test_macro_impl.hpp',
  'internal/catch_test_registry.hpp',
  'internal/catch_test_spec_parser.hpp',
  'internal/catch_textflow.hpp',
  'internal/catch_thread_support.hpp',
  'internal/catch_to_string.hpp',
  'internal/catch_uncaught_exceptions.hpp',
  'internal/catch_uniform_floating_point_distribution.hpp',
  'internal/catch_uniform_integer_distribution.hpp',
  'internal/catch_unique_name.hpp',
  'internal/catch_unique_ptr.hpp',
  'internal/catch_unreachable.hpp',
  'internal/catch_void_type.hpp',
  'internal/catch_wildcard_pattern.hpp',
  'internal/catch_windows_h_proxy.hpp',
  'internal/catch_xmlwriter.hpp',
  'matchers/catch_matchers.hpp',
  'matchers/catch_matchers_all.hpp',
  'matchers/catch_matchers_container_properties.hpp',
  'matchers/catch_matchers_contains.hpp',
  'matchers/catch_matchers_exception.hpp',
  'matchers/catch_matchers_floating_point.hpp',
  'matchers/catch_matchers_predicate.hpp',
  'matchers/catch_matchers_quantifiers.hpp',
  'matchers/catch_matchers_range_equals.hpp',
  'matchers/catch_matchers_string.hpp',
  'matchers/catch_matchers_templated.hpp',
  'matchers/catch_matchers_vector.hpp',
  'matchers/internal/catch_matchers_impl.hpp',
  'catch_all.hpp',
  'catch_approx.hpp',
  'catch_assertion_info.hpp',
  'catch_assertion_result.hpp',
  'catch_case_sensitive.hpp',
  'catch_config.hpp',
  'catch_get_random_seed.hpp',
  'catch_message.hpp',
  'catch_section_info.hpp',
  'catch_session.hpp',
  'catch_tag_alias.hpp',
  'catch_tag_alias_autoregistrar.hpp',
  'catch_template_test_macros.hpp',
  'catch_test_case_info.hpp',
  'catch_test_macros.hpp',
  'catch_test_run_info.hpp',
  'catch_test_spec.hpp',
  'catch_timer.hpp',
  'catch_tostring.hpp',
  'catch_totals.hpp',
  'catch_translate_exception.hpp',
  'catch_version.hpp',
  'catch_version_macros.hpp',
]

internal_sources = files(
  'generators/catch_generator_exception.cpp',
  'generators/catch_generators.cpp',
  'generators/catch_generators_random.cpp',
  'interfaces/catch_interfaces_capture.cpp',
  'interfaces/catch_interfaces_config.cpp',
  'interfaces/catch_interfaces_exception.cpp',
  'interfaces/catch_interfaces_generatortracker.cpp',
  'interfaces/catch_interfaces_registry_hub.cpp',
  'interfaces/catch_interfaces_reporter.cpp',
  'interfaces/catch_interfaces_reporter_factory.cpp',
  'interfaces/catch_interfaces_testcase.cpp',
  'internal/catch_assertion_handler.cpp',
  'internal/catch_case_insensitive_comparisons.cpp',
  'internal/catch_clara.cpp',
  'internal/catch_commandline.cpp',
  'internal/catch_console_colour.cpp',
  'internal/catch_context.cpp',
  'internal/catch_debug_console.cpp',
  'internal/catch_debugger.cpp',
  'internal/catch_decomposer.cpp',
  'internal/catch_enforce.cpp',
  'internal/catch_enum_values_registry.cpp',
  'internal/catch_errno_guard.cpp',
  'internal/catch_exception_translator_registry.cpp',
  'internal/catch_fatal_condition_handler.cpp',
  'internal/catch_floating_point_helpers.cpp',
  'internal/catch_getenv.cpp',
  'internal/catch_istream.cpp',
  'internal/catch_jsonwriter.cpp',
  'internal/catch_lazy_expr.cpp',
  'internal/catch_leak_detector.cpp',
  'internal/catch_list.cpp',
  'internal/catch_message_info.cpp',
  'internal/catch_output_redirect.cpp',
  'internal/catch_parse_numbers.cpp',
  'internal/catch_polyfills.cpp',
  'internal/catch_random_number_generator.cpp',
  'internal/catch_random_seed_generation.cpp',
  'internal/catch_reporter_registry.cpp',
  'internal/catch_reporter_spec_parser.cpp',
  'internal/catch_reusable_string_stream.cpp',
  'internal/catch_run_context.cpp',
  'internal/catch_section.cpp',
  'internal/catch_singletons.cpp',
  'internal/catch_source_line_info.cpp',
  'internal/catch_startup_exception_registry.cpp',
  'internal/catch_stdstreams.cpp',
  'internal/catch_string_manip.cpp',
  'internal/catch_stringref.cpp',
  'internal/catch_tag_alias_registry.cpp',
  'internal/catch_test_case_info_hasher.cpp',
  'internal/catch_test_case_registry_impl.cpp',
  'internal/catch_test_case_tracker.cpp',
  'internal/catch_test_failure_exception.cpp',
  'internal/catch_test_registry.cpp',
  'internal/catch_test_spec_parser.cpp',
  'internal/catch_textflow.cpp',
  'internal/catch_uncaught_exceptions.cpp',
  'internal/catch_wildcard_pattern.cpp',
  'internal/catch_xmlwriter.cpp',
  'matchers/catch_matchers.cpp',
  'matchers/catch_matchers_container_properties.cpp',
  'matchers/catch_matchers_exception.cpp',
  'matchers/catch_matchers_floating_point.cpp',
  'matchers/catch_matchers_predicate.cpp',
  'matchers/catch_matchers_quantifiers.cpp',
  'matchers/catch_matchers_string.cpp',
  'matchers/catch_matchers_templated.cpp',
  'matchers/internal/catch_matchers_impl.cpp',
  'catch_approx.cpp',
  'catch_assertion_result.cpp',
  'catch_config.cpp',
  'catch_get_random_seed.cpp',
  'catch_message.cpp',
  'catch_registry_hub.cpp',
  'catch_session.cpp',
  'catch_tag_alias_autoregistrar.cpp',
  'catch_test_case_info.cpp',
  'catch_test_spec.cpp',
  'catch_timer.cpp',
  'catch_tostring.cpp',
  'catch_totals.cpp',
  'catch_translate_exception.cpp',
  'catch_version.cpp',
)

reporter_headers = [
  'reporters/catch_reporter_automake.hpp',
  'reporters/catch_reporter_common_base.hpp',
  'reporters/catch_reporter_compact.hpp',
  'reporters/catch_reporter_console.hpp',
  'reporters/catch_reporter_cumulative_base.hpp',
  'reporters/catch_reporter_event_listener.hpp',
  'reporters/catch_reporter_helpers.hpp',
  'reporters/catch_reporter_json.hpp',
  'reporters/catch_reporter_junit.hpp',
  'reporters/catch_reporter_multi.hpp',
  'reporters/catch_reporter_registrars.hpp',
  'reporters/catch_reporter_sonarqube.hpp',
  'reporters/catch_reporter_streaming_base.hpp',
  'reporters/catch_reporter_tap.hpp',
  'reporters/catch_reporter_teamcity.hpp',
  'reporters/catch_reporter_xml.hpp',
  'reporters/catch_reporters_all.hpp',
]

reporter_sources = files(
  'reporters/catch_reporter_automake.cpp',
  'reporters/catch_reporter_common_base.cpp',
  'reporters/catch_reporter_compact.cpp',
  'reporters/catch_reporter_console.cpp',
  'reporters/catch_reporter_cumulative_base.cpp',
  'reporters/catch_reporter_event_listener.cpp',
  'reporters/catch_reporter_helpers.cpp',
  'reporters/catch_reporter_json.cpp',
  'reporters/catch_reporter_junit.cpp',
  'reporters/catch_reporter_multi.cpp',
  'reporters/catch_reporter_registrars.cpp',
  'reporters/catch_reporter_sonarqube.cpp',
  'reporters/catch_reporter_streaming_base.cpp',
  'reporters/catch_reporter_tap.cpp',
  'reporters/catch_reporter_teamcity.cpp',
  'reporters/catch_reporter_xml.cpp',
)

headers = benchmark_headers + internal_headers + reporter_headers
sources = benchmark_sources + internal_sources + reporter_sources

# The headers must be installed with their full paths, which meson
# v0.63 supports using `preserve_path` for `install_headers`. We'd
# like to be compatible with Debian 11 (current stable) which has
# meson v0.56.2. Instead, let's use the technique from
# https://github.com/mesonbuild/meson/issues/14 with some tweaks to
# make it compatible with newer meson.
include_subdir = 'catch2'
foreach file : headers
  file_path = file.split('/')

  folder = ''
  foreach path : file_path
    if path != file_path[-1]
      folder = folder / path
    endif
  endforeach

  if get_option('install')
    install_headers(file, subdir: join_paths(include_subdir, folder))
  endif
endforeach

catch2_dependencies = []
# Check if this is an Android NDK build.
if ((host_machine.system() == 'android') or
  # Check if this is an Android Termux build.
  (host_machine.system() == 'linux' and fs.is_dir('/data/data/com.termux')))
  log_dep = meson.get_compiler('cpp').find_library('log')
  catch2_dependencies += log_dep
endif

catch2 = static_library(
  'Catch2',
  sources,
  dependencies: catch2_dependencies,
  include_directories: '..',
  install: get_option('install'),
)

catch2_dep = declare_dependency(
  link_with: catch2,
  include_directories: '..',
)

if get_option('install')
  pkg.generate(
    catch2,
    filebase: 'catch2',
    description: 'A modern, C++-native, test framework for C++14 and above',
    url: 'https://github.com/catchorg/Catch2',
  )
endif

catch2_with_main = static_library(
  'Catch2Main',
  'internal/catch_main.cpp',
  link_with: catch2,
  include_directories: '..',
  install: get_option('install'),
)

catch2_with_main_dep = declare_dependency(
  link_with: [catch2, catch2_with_main],
  include_directories: '..',
)

if get_option('install')
  pkg.generate(
    catch2_with_main,
    filebase: 'catch2-with-main',
    description: 'A modern, C++-native, test framework for C++14 and above (links in default main)',
    requires: 'catch2 = ' + meson.project_version(),
  )
endif
