extends SceneTree

# Performance benchmark for GDScript structs

struct Entity:
	var id: int
	var pos_x: float
	var pos_y: float
	var vel_x: float
	var vel_y: float
	var health: int

class EntityClass:
	var id: int
	var pos_x: float
	var pos_y: float
	var vel_x: float
	var vel_y: float
	var health: int
	
	func _init(p_id: int, p_px: float, p_py: float, p_vx: float, p_vy: float, p_health: int):
		id = p_id
		pos_x = p_px
		pos_y = p_py
		vel_x = p_vx
		vel_y = p_vy
		health = p_health

const ENTITY_COUNTS = [100, 1000, 10000]
const ITERATIONS = 1000

func _init():
	print("=".repeat(80))
	print("GDScript Struct Performance Benchmark")
	print("=".repeat(80))
	print()
	
	for count in ENTITY_COUNTS:
		print("Testing with %d entities, %d iterations" % [count, ITERATIONS])
		print("-".repeat(80))
		
		benchmark_struct_creation(count)
		benchmark_dict_creation(count)
		benchmark_class_creation(count)
		print()
		
		benchmark_struct_iteration(count)
		benchmark_dict_iteration(count)
		benchmark_class_iteration(count)
		print()
		
		benchmark_struct_member_access(count)
		benchmark_dict_member_access(count)
		benchmark_class_member_access(count)
		print()
		
		benchmark_struct_update(count)
		benchmark_dict_update(count)
		benchmark_class_update(count)
		print()
		print()
	
	print_summary()
	quit()

func benchmark_struct_creation(count: int):
	var start = Time.get_ticks_usec()
	for i in ITERATIONS:
		var entities = []
		for j in count:
			var e = Entity(j, randf() * 100, randf() * 100, randf() * 10, randf() * 10, 100)
			entities.append(e)
	var elapsed = Time.get_ticks_usec() - start
	var per_entity = elapsed / float(count * ITERATIONS)
	print("  Struct creation:     %8d μs total | %.3f μs/entity" % [elapsed, per_entity])

func benchmark_dict_creation(count: int):
	var start = Time.get_ticks_usec()
	for i in ITERATIONS:
		var entities = []
		for j in count:
			var e = {"id": j, "pos_x": randf() * 100, "pos_y": randf() * 100, "vel_x": randf() * 10, "vel_y": randf() * 10, "health": 100}
			entities.append(e)
	var elapsed = Time.get_ticks_usec() - start
	var per_entity = elapsed / float(count * ITERATIONS)
	print("  Dict creation:       %8d μs total | %.3f μs/entity" % [elapsed, per_entity])

func benchmark_class_creation(count: int):
	var start = Time.get_ticks_usec()
	for i in ITERATIONS:
		var entities = []
		for j in count:
			var e = EntityClass.new(j, randf() * 100, randf() * 100, randf() * 10, randf() * 10, 100)
			entities.append(e)
	var elapsed = Time.get_ticks_usec() - start
	var per_entity = elapsed / float(count * ITERATIONS)
	print("  Class creation:      %8d μs total | %.3f μs/entity" % [elapsed, per_entity])

func benchmark_struct_iteration(count: int):
	var entities = []
	for j in count:
		entities.append(Entity(j, randf() * 100, randf() * 100, randf() * 10, randf() * 10, 100))
	var start = Time.get_ticks_usec()
	var sum = 0.0
	for i in ITERATIONS:
		for e in entities:
			sum += e.pos_x + e.pos_y
	var elapsed = Time.get_ticks_usec() - start
	var per_iter = elapsed / float(count * ITERATIONS)
	print("  Struct iteration:    %8d μs total | %.3f μs/entity | sum=%.0f" % [elapsed, per_iter, sum])

func benchmark_dict_iteration(count: int):
	var entities = []
	for j in count:
		entities.append({"id": j, "pos_x": randf() * 100, "pos_y": randf() * 100, "vel_x": randf() * 10, "vel_y": randf() * 10, "health": 100})
	var start = Time.get_ticks_usec()
	var sum = 0.0
	for i in ITERATIONS:
		for e in entities:
			sum += e["pos_x"] + e["pos_y"]
	var elapsed = Time.get_ticks_usec() - start
	var per_iter = elapsed / float(count * ITERATIONS)
	print("  Dict iteration:      %8d μs total | %.3f μs/entity | sum=%.0f" % [elapsed, per_iter, sum])

func benchmark_class_iteration(count: int):
	var entities = []
	for j in count:
		entities.append(EntityClass.new(j, randf() * 100, randf() * 100, randf() * 10, randf() * 10, 100))
	var start = Time.get_ticks_usec()
	var sum = 0.0
	for i in ITERATIONS:
		for e in entities:
			sum += e.pos_x + e.pos_y
	var elapsed = Time.get_ticks_usec() - start
	var per_iter = elapsed / float(count * ITERATIONS)
	print("  Class iteration:     %8d μs total | %.3f μs/entity | sum=%.0f" % [elapsed, per_iter, sum])

func benchmark_struct_member_access(count: int):
	var entities = []
	for j in count:
		entities.append(Entity(j, randf() * 100, randf() * 100, randf() * 10, randf() * 10, 100))
	var start = Time.get_ticks_usec()
	var checksum = 0
	for i in ITERATIONS:
		for e in entities:
			checksum += e.id
			checksum += int(e.pos_x)
			checksum += int(e.health)
	var elapsed = Time.get_ticks_usec() - start
	var per_access = elapsed / float(count * ITERATIONS * 3)
	print("  Struct mem access:   %8d μs total | %.3f μs/access | sum=%d" % [elapsed, per_access, checksum])

func benchmark_dict_member_access(count: int):
	var entities = []
	for j in count:
		entities.append({"id": j, "pos_x": randf() * 100, "pos_y": randf() * 100, "vel_x": randf() * 10, "vel_y": randf() * 10, "health": 100})
	var start = Time.get_ticks_usec()
	var checksum = 0
	for i in ITERATIONS:
		for e in entities:
			checksum += e["id"]
			checksum += int(e["pos_x"])
			checksum += int(e["health"])
	var elapsed = Time.get_ticks_usec() - start
	var per_access = elapsed / float(count * ITERATIONS * 3)
	print("  Dict mem access:     %8d μs total | %.3f μs/access | sum=%d" % [elapsed, per_access, checksum])

func benchmark_class_member_access(count: int):
	var entities = []
	for j in count:
		entities.append(EntityClass.new(j, randf() * 100, randf() * 100, randf() * 10, randf() * 10, 100))
	var start = Time.get_ticks_usec()
	var checksum = 0
	for i in ITERATIONS:
		for e in entities:
			checksum += e.id
			checksum += int(e.pos_x)
			checksum += int(e.health)
	var elapsed = Time.get_ticks_usec() - start
	var per_access = elapsed / float(count * ITERATIONS * 3)
	print("  Class mem access:    %8d μs total | %.3f μs/access | sum=%d" % [elapsed, per_access, checksum])

func benchmark_struct_update(count: int):
	var entities = []
	for j in count:
		entities.append(Entity(j, 0, 0, 1, 1, 100))
	var start = Time.get_ticks_usec()
	for i in ITERATIONS:
		for e in entities:
			e.pos_x += e.vel_x
			e.pos_y += e.vel_y
			e.health -= 1
	var elapsed = Time.get_ticks_usec() - start
	var per_update = elapsed / float(count * ITERATIONS * 3)
	print("  Struct updates:      %8d μs total | %.3f μs/update" % [elapsed, per_update])

func benchmark_dict_update(count: int):
	var entities = []
	for j in count:
		entities.append({"id": j, "pos_x": 0.0, "pos_y": 0.0, "vel_x": 1.0, "vel_y": 1.0, "health": 100})
	var start = Time.get_ticks_usec()
	for i in ITERATIONS:
		for e in entities:
			e["pos_x"] += e["vel_x"]
			e["pos_y"] += e["vel_y"]
			e["health"] -= 1
	var elapsed = Time.get_ticks_usec() - start
	var per_update = elapsed / float(count * ITERATIONS * 3)
	print("  Dict updates:        %8d μs total | %.3f μs/update" % [elapsed, per_update])

func benchmark_class_update(count: int):
	var entities = []
	for j in count:
		entities.append(EntityClass.new(j, 0, 0, 1, 1, 100))
	var start = Time.get_ticks_usec()
	for i in ITERATIONS:
		for e in entities:
			e.pos_x += e.vel_x
			e.pos_y += e.vel_y
			e.health -= 1
	var elapsed = Time.get_ticks_usec() - start
	var per_update = elapsed / float(count * ITERATIONS * 3)
	print("  Class updates:       %8d μs total | %.3f μs/update" % [elapsed, per_update])

func print_summary():
	print("=".repeat(80))
	print("BENCHMARK SUMMARY")
	print("=".repeat(80))
	print()
	print("Current Implementation Status:")
	print("  ✅ Structs have compile-time type safety")
	print("  ✅ Clean syntax with dot notation")
	print("  ⚠️  Runtime performance = Dictionary (no optimization yet)")
	print()
	print("Expected Results:")
	print("  • Struct ≈ Dict performance (both use Dictionary internally)")
	print("  • Class ~2-10x slower (RefCounted overhead)")
	print()
	print("Future Optimizations (with FlatArray):")
	print("  • 10x faster iteration (cache-friendly layout)")
	print("  • 500x less memory (no RefCounted overhead)")
	print("  • SIMD batch operations")
	print()
	print("To unlock performance gains:")
	print("  1. Implement FlatArray (Phase 3)")
	print("  2. Add Array.to_flat() conversion")
	print("  3. Use: var entities_flat: FlatArray[Entity] = entities.to_flat()")
	print()
	print("Current Benefits:")
	print("  ✅ Type safety catches bugs at compile time")
	print("  ✅ Better code completion and hints")
	print("  ✅ Self-documenting data structures")
	print("  ✅ Foundation for future performance")
	print("=".repeat(80))
