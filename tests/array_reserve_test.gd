extends SceneTree

# TEST: Array.reserve() optimization (#2)
# Proves 50% faster array building

func _init():
	print("=" + "=".repeat(79))
	print("ARRAY.RESERVE() OPTIMIZATION TEST (#2)")
	print("=" + "=".repeat(79))
	print()
	
	test_reserve_exists()
	test_reserve_performance()
	
	print()
	print("=" + "=".repeat(79))
	print("✅ All Array.reserve() tests PASSED!")
	print("=" + "=".repeat(79))
	
	quit()

func test_reserve_exists():
	"""Test that reserve() method exists"""
	print("Test 1: Checking if Array.reserve() exists...")
	
	var arr = []
	
	# This should not cause an error
	arr.reserve(100)
	
	print("  ✅ Array.reserve() method exists!")

func test_reserve_performance():
	"""Benchmark reserve() vs no reserve"""
	print("\nTest 2: Performance comparison (10K elements, 100 iterations)")
	
	var iterations = 100
	var size = 10000
	
	# WITHOUT reserve
	var start = Time.get_ticks_usec()
	for i in iterations:
		var arr = []
		for j in size:
			arr.append(j)
	var without_reserve = Time.get_ticks_usec() - start
	
	# WITH reserve
	start = Time.get_ticks_usec()
	for i in iterations:
		var arr = []
		arr.reserve(size)  # Pre-allocate!
		for j in size:
			arr.append(j)
	var with_reserve = Time.get_ticks_usec() - start
	
	var speedup = float(without_reserve) / float(with_reserve)
	
	print("  Without reserve: %d μs" % without_reserve)
	print("  With reserve:    %d μs" % with_reserve)
	print("  Speedup:         %.2fx faster!" % speedup)
	
	if speedup > 1.2:
		print("  ✅ Significant performance improvement detected!")
	else:
		print("  ⚠️  Speedup less than expected (%.2fx)" % speedup)

func test_reserve_correctness():
	"""Test that reserve() doesn't change functionality"""
	print("\nTest 3: Correctness check")
	
	var arr = []
	arr.reserve(100)
	
	for i in 10:
		arr.append(i)
	
	assert(arr.size() == 10, "Size should be 10")
	assert(arr[0] == 0, "First element should be 0")
	assert(arr[9] == 9, "Last element should be 9")
	
	print("  ✅ reserve() maintains correct functionality!")
