## Performance / allocation test for PhysicsDirectSpaceState3D.intersect_ray_into().
##
## How to use:
##   1. Create a new 3D scene with a Node3D as the root and attach this script to it.
##   2. Add a StaticBody3D with a CollisionShape3D (e.g. a BoxShape3D) somewhere in
##      front of the ray origin so that the raycasts have something to hit.
##   3. Run the scene. Each physics frame it performs RAYS_PER_FRAME raycasts using
##      a single, reused PhysicsRayQueryResult3D instance via intersect_ray_into(),
##      and prints the current object count from Performance.get_monitor().
##
## Expected result: Performance.OBJECT_COUNT stays flat across frames, proving that
## intersect_ray_into() does not allocate a new object (e.g. Dictionary/Variant
## boxing) per call, unlike intersect_ray().
extends Node3D

const RAYS_PER_FRAME := 5000
const FRAMES_TO_RUN := 60

var _frame_count := 0

# A single PhysicsRayQueryResult3D instance, reused for every raycast.
var _ray_result := PhysicsRayQueryResult3D.new()

func _physics_process(_delta: float) -> void:
	if _frame_count >= FRAMES_TO_RUN:
		set_physics_process(false)
		return

	var space_state := get_world_3d().direct_space_state
	var query := PhysicsRayQueryParameters3D.create(Vector3.ZERO, Vector3.ZERO)

	var hits := 0
	for i in range(RAYS_PER_FRAME):
		# Vary the ray slightly so the query isn't trivially cached/optimized away.
		var offset := Vector3(0.0, 0.0, float(i % 100) * 0.001)
		query.from = Vector3(0, 1, -10) + offset
		query.to = Vector3(0, 1, 10) + offset

		if space_state.intersect_ray_into(query, _ray_result):
			hits += 1

	_frame_count += 1

	var object_count := Performance.get_monitor(Performance.OBJECT_COUNT)
	print("Frame %d: %d/%d rays hit, has_hit=%s, OBJECT_COUNT=%d" % [
		_frame_count, hits, RAYS_PER_FRAME, _ray_result.has_hit(), object_count
	])

	if _frame_count == FRAMES_TO_RUN:
		print("Done. OBJECT_COUNT should remain flat across all printed frames,")
		print("proving intersect_ray_into() performs no per-call allocations.")
