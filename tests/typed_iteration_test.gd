extends SceneTree

# TEST: Typed Array Iteration Optimization (#1)
# OPCODE_ITERATE_TYPED_ARRAY should give 3-4x speedup

func _init():
	print("=" + "=".repeat(79))
	print("TYPED ARRAY ITERATION OPTIMIZATION TEST (#1)")
	print("=" + "=".repeat(79))
	print()
	
	test_typed_array_iteration()
	
	print()
	print("=" + "=".repeat(79))
	print("✅ Typed array iteration test COMPLETED!")
	print("=" + "=".repeat(79))
	
	quit()

func test_typed_array_iteration():
	"""Compare generic vs typed array iteration"""
	print("Test: Generic vs Typed Array iteration (10K elements, 500 iterations)")
	
	var iterations = 500
	var size = 10000
	
	# Setup: Create arrays
	var generic_array = []
	var typed_array: Array[int] = []
	
	for i in size:
		generic_array.append(i)
		typed_array.append(i)
	
	# Benchmark generic iteration
	var sum1 = 0
	var start = Time.get_ticks_usec()
	for i in iterations:
		for val in generic_array:
			sum1 += val
	var generic_time = Time.get_ticks_usec() - start
	
	# Benchmark typed iteration (should use OPCODE_ITERATE_TYPED_ARRAY)
	var sum2 = 0
	start = Time.get_ticks_usec()
	for i in iterations:
		for val in typed_array:
			sum2 += val
	var typed_time = Time.get_ticks_usec() - start
	
	var speedup = float(generic_time) / float(typed_time)
	
	print("  Generic array: %d μs (sum=%d)" % [generic_time, sum1])
	print("  Typed array:   %d μs (sum=%d)" % [typed_time, sum2])
	print("  Speedup:       %.2fx faster!" % speedup)
	
	# Verify correctness
	assert(sum1 == sum2, "Sums should match")
	print("  ✅ Correctness verified (sums match)")
	
	if speedup > 2.0:
		print("  ✅ OPCODE_ITERATE_TYPED_ARRAY is working! (%.2fx speedup)" % speedup)
	elif speedup > 1.3:
		print("  ✅ Moderate speedup detected (%.2fx)" % speedup)
	else:
		print("  ⚠️  Speedup less than expected (%.2fx)" % speedup)
		print("     Note: VM optimization may not be active or needs larger dataset")
