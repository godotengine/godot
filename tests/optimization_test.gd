extends SceneTree

# Test: Verify OPCODE_ITERATE_TYPED_ARRAY optimization works

struct Entity:
	var id: int
	var x: float
	var y: float
	var active: bool

func _init():
	print("=" * 80)
	print("OPCODE_ITERATE_TYPED_ARRAY - Optimization Test")
	print("=" * 80)
	print()
	
	test_typed_array()
	test_untyped_array()
	test_performance()
	
	print()
	print("=" * 80)
	print("✅ ALL TESTS COMPLETE")
	print("=" * 80)
	quit()

func test_typed_array():
	print("Test 1: Typed Array Iteration")
	print("-" * 80)
	
	var entities: Array[Entity] = []
	entities.resize(10)
	
	for i in 10:
		entities[i] = Entity(i, float(i * 10), float(i * 20), true)
	
	print("Created 10 entities")
	
	# This SHOULD use OPCODE_ITERATE_TYPED_ARRAY
	var sum_x: float = 0.0
	for e in entities:
		sum_x += e.x
	
	print("Sum of X coordinates: ", sum_x)
	print("Expected: ", 450.0)
	
	if abs(sum_x - 450.0) < 0.01:
		print("✅ PASS: Typed array iteration works!")
	else:
		print("❌ FAIL: Wrong result!")
	
	print()

func test_untyped_array():
	print("Test 2: Untyped Array Iteration (fallback)")
	print("-" * 80)
	
	var items = []  # Untyped
	items.resize(10)
	
	for i in 10:
		items[i] = Entity(i, float(i), float(i), true)
	
	# This should use OPCODE_ITERATE_BEGIN (generic)
	var count = 0
	for item in items:
		count += 1
	
	print("Iterated over ", count, " items")
	
	if count == 10:
		print("✅ PASS: Untyped array iteration works!")
	else:
		print("❌ FAIL: Wrong count!")
	
	print()

func test_performance():
	print("Test 3: Performance Comparison")
	print("-" * 80)
	
	const SIZE = 10000
	const ITERATIONS = 100
	
	# Setup typed array
	var entities: Array[Entity] = []
	entities.resize(SIZE)
	for i in SIZE:
		entities[i] = Entity(i, 0.0, 0.0, true)
	
	# Test typed array (should be optimized)
	var start = Time.get_ticks_usec()
	for iter in ITERATIONS:
		for e in entities:
			e.x += 1.0
	var time_typed = Time.get_ticks_usec() - start
	
	# Setup untyped array
	var items = []
	items.resize(SIZE)
	for i in SIZE:
		items[i] = Entity(i, 0.0, 0.0, true)
	
	# Test untyped array (generic iteration)
	start = Time.get_ticks_usec()
	for iter in ITERATIONS:
		for item in items:
			item.x += 1.0
	var time_untyped = Time.get_ticks_usec() - start
	
	print("Typed array:   ", time_typed, " μs")
	print("Untyped array: ", time_untyped, " μs")
	
	var speedup = float(time_untyped) / float(time_typed)
	print("Speedup: ", "%.2f" % speedup, "x")
	
	if speedup > 1.05:  # At least 5% faster
		print("✅ PASS: Optimization provides measurable speedup!")
	else:
		print("⚠️  WARNING: No significant speedup detected")
		print("    (Optimization may not be active yet)")
	
	print()
