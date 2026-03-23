var tests_requiring_unicode_security := [
	# these require TextServer.FEATURE_UNICODE_SECURITY for identifier spoof checking and confusability warnings
	"analyzer/features/warning_ignore_targets.gd",
	"parser/errors/identifier_similar_to_keyword.gd",
	"parser/warnings/confusable_identifier.gd"
]

var tests_requiring_debug := [
	# crashes when enabled in release
	"runtime/errors/callable_call_after_free_object.gd"
]

func should_run_test(path: String) -> bool:
	if path in tests_requiring_unicode_security:
		# ICU_STATIC_DATA is not defined by default in non-editor builds, and for some reason this line:
		# TextServerManager.get_primary_interface().has_feature(TextServer.FEATURE_UNICODE_SECURITY)
		# returns true even if the feature doesn't work, so:
		return not OS.has_feature("template")
	if path in tests_requiring_debug:
		return OS.is_debug_build()

	return true
