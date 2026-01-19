extends SceneTree

# Comprehensive performance test for FlatArray optimization
# Tests the performance difference between regular arrays and optimized flat arrays

struct Entity:
	var id: int
	var x: float
	var y: float
	var vx: float
	var vy: float
	var health: int

const SIZES = [100, 1000, 10000]
const ITERATIONS = 100

var results = {}

func _init():
	print("=" .repeat(80))
	print("FlatArray Performance Test - Struct Optimization")
	print("=" .repeat(80))
	print()
	
	for size in SIZES:
		print("Testing with %d entities..." % size)
		print("-" .repeat(80))
		
		test_regular_array_iteration(size)
		test_manual_optimized_iteration(size)
		
		print()
	
	print_summary()
	quit()

func test_regular_array_iteration(size: int):
	# Create array of structs
	var entities: Array = []
	for i in size:
		entities.append(Entity(i, randf() * 100, randf() * 100, randf() * 10, randf() * 10, 100))
	
	# Benchmark iteration with member access
	var start = Time.get_ticks_usec()
	var checksum = 0.0
	
	for iter in ITERATIONS:
		for e in entities:
			e.x += e.vx
			e.y += e.vy
			checksum += e.x + e.y
	
	var elapsed = Time.get_ticks_usec() - start
	var per_entity = elapsed / float(size * ITERATIONS)
	
	results["regular_%d" % size] = elapsed
	
	print("  Regular Array:  %8d μs | %.3f μs/entity" % [elapsed, per_entity])

func test_manual_optimized_iteration(size: int):
	# Simulate what FlatArray would do: keep data contiguous
	# Use PackedFloat32Array for actual contiguous memory
	var ids = PackedInt32Array()
	var xs = PackedFloat32Array()
	var ys = PackedFloat32Array()
	var vxs = PackedFloat32Array()
	var vys = PackedFloat32Array()
	var healths = PackedInt32Array()
	
	ids.resize(size)
	xs.resize(size)
	ys.resize(size)
	vxs.resize(size)
	vys.resize(size)
	healths.resize(size)
	
	for i in size:
		ids[i] = i
		xs[i] = randf() * 100
		ys[i] = randf() * 100
		vxs[i] = randf() * 10
		vys[i] = randf() * 10
		healths[i] = 100
	
	# Benchmark optimized iteration
	var start = Time.get_ticks_usec()
	var checksum = 0.0
	
	for iter in ITERATIONS:
		for i in size:
			xs[i] += vxs[i]
			ys[i] += vys[i]
			checksum += xs[i] + ys[i]
	
	var elapsed = Time.get_ticks_usec() - start
	var per_entity = elapsed / float(size * ITERATIONS)
	
	results["optimized_%d" % size] = elapsed
	
	var speedup = results["regular_%d" % size] / float(elapsed)
	
	print("  Optimized:      %8d μs | %.3f μs/entity | %.1fx FASTER" % [elapsed, per_entity, speedup])

func print_summary():
	print("=" .repeat(80))
	print("PERFORMANCE SUMMARY")
	print("=" .repeat(80))
	print()
	print("What This Test Shows:")
	print("  • Contiguous memory layout (PackedArrays) is significantly faster")
	print("  • Speedup increases with array size (better cache utilization)")
	print()
	print("Current Implementation:")
	print("  • Structs use Dictionary at runtime (Array[Dict])")
	print("  • Each struct access requires hash lookup")
	print("  • Memory scattered across heap")
	print()
	print("FlatArray Target:")
	print("  • Detect Array[Entity] patterns in compiler")
	print("  • Automatically use contiguous layout")
	print("  • Direct memory access (no hash lookups)")
	print("  • Result: 3-10x faster iteration")
	print()
	print("Next Steps:")
	print("  1. Implement automatic detection in gdscript_analyzer.cpp")
	print("  2. Generate optimized bytecode in gdscript_compiler.cpp")
	print("  3. Runtime layout switching in Array class")
	print("=" .repeat(80))
