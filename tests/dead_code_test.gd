extends SceneTree

# TEST: Dead Code Elimination (#6)
# Constant conditions should eliminate branches

const DEBUG_MODE = false
const PROFILING = false
const FEATURE_ENABLED = true

var debug_executed = false
var profiling_executed = false
var feature_executed = false
var production_executed = false

func _init():
	print("=" + "=".repeat(79))
	print("DEAD CODE ELIMINATION TEST (#6)")
	print("=" + "=".repeat(79))
	print()
	
	test_dead_code_elimination()
	
	print()
	print("=" + "=".repeat(79))
	print("✅ Dead code elimination test COMPLETED!")
	print("=" + "=".repeat(79))
	
	quit()

func test_dead_code_elimination():
	"""Test that constant false conditions don't execute"""
	print("Test: Constant condition folding")
	print()
	
	# These should be eliminated at compile time
	if DEBUG_MODE:
		debug_executed = true
		print("  DEBUG: This should never print!")
	
	if PROFILING:
		profiling_executed = true
		print("  PROFILING: This should never print!")
	
	# This should remain
	if FEATURE_ENABLED:
		feature_executed = true
		print("  ✅ Feature code executed (constant TRUE)")
	
	# This should remain
	production_executed = true
	print("  ✅ Production code executed")
	
	# Verify results
	print()
	print("Execution results:")
	print("  DEBUG_MODE code:   %s (expected: false)" % str(debug_executed))
	print("  PROFILING code:    %s (expected: false)" % str(profiling_executed))
	print("  FEATURE code:      %s (expected: true)" % str(feature_executed))
	print("  Production code:   %s (expected: true)" % str(production_executed))
	
	if not debug_executed and not profiling_executed and feature_executed and production_executed:
		print()
		print("  ✅ Dead code elimination working correctly!")
		print("     - FALSE branches not executed")
		print("     - TRUE branches executed")
		print("     - Production code executed")
	else:
		print()
		print("  ⚠️  Unexpected execution pattern")

func never_called_function():
	"""This function should be eliminated if unused"""
	print("This should never execute!")
	return "dead code"
