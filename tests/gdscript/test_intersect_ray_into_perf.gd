## test_intersect_ray_into_perf.gd
##
## Performance test for the allocation-free intersect_ray_into() API.
##
## Drop this script on a Node3D in a scene that has PhysicsBody3D objects
## present so raycasts have something to hit.  The test fires 5000 raycasts
## per _physics_process() frame using a **single reused** PhysicsRayQueryResult3D
## instance and prints Performance monitor object counts to the console so you
## can verify the count stays completely flat while the raycasts run.
##
## Expected console output (approximate):
##   [frame 1] object_count=<N>  orphan_nodes=0  mem_static=<M>
##   [frame 2] object_count=<N>  orphan_nodes=0  mem_static=<M>
##   ...
## The object_count must NOT grow across frames.  If it climbs, objects are
## leaking — which is exactly what the old Dictionary-returning API would cause.

extends Node3D

const RAYCASTS_PER_FRAME := 5000
const FRAMES_TO_MEASURE  := 10        # stop after this many frames
const RAY_LENGTH         := 100.0

## A single result object allocated once and reused forever.
var _result: PhysicsRayQueryResult3D

## Reused query parameters object — avoids allocating a new one each frame.
var _params: PhysicsRayQueryParameters3D

var _frame_count := 0

func _ready() -> void:
	# Allocate exactly once.
	_result = PhysicsRayQueryResult3D.new()
	_params  = PhysicsRayQueryParameters3D.new()
	set_physics_process(true)
	print("[intersect_ray_into perf] starting — %d raycasts x %d frames" %
			[RAYCASTS_PER_FRAME, FRAMES_TO_MEASURE])

func _physics_process(_delta: float) -> void:
	if _frame_count >= FRAMES_TO_MEASURE:
		set_physics_process(false)
		print("[intersect_ray_into perf] done.")
		return

	var space: PhysicsDirectSpaceState3D = get_world_3d().direct_space_state

	# Vary ray direction each iteration so the compiler can't eliminate the calls.
	for i in RAYCASTS_PER_FRAME:
		var angle := float(i) * 0.001   # cheap deterministic spread
		_params.from = global_position
		_params.to   = global_position + Vector3(sin(angle), 0.0, cos(angle)) * RAY_LENGTH

		# intersect_ray_into() writes directly into _result — no allocation.
		var hit: bool = space.intersect_ray_into(_params, _result)

		# Access result fields to prevent the loop from being optimised away.
		if hit:
			# Use has_hit() as a guard; redundant here (hit == _result.has_hit())
			# but demonstrates the API.
			var _pos := _result.get_position()
			var _nrm := _result.get_normal()
		# When there is no hit every field is safely reset to defaults.

	# -----------------------------------------------------------------------
	# Performance snapshot — the critical part of this test.
	# Performance.OBJECT_COUNT should be perfectly flat across all frames.
	# -----------------------------------------------------------------------
	var obj_count    := int(Performance.get_monitor(Performance.OBJECT_COUNT))
	var orphan_nodes := int(Performance.get_monitor(Performance.OBJECT_ORPHAN_NODE_COUNT))
	var mem_static   := int(Performance.get_monitor(Performance.MEMORY_STATIC))

	print("[intersect_ray_into perf] frame=%d  object_count=%d  orphan_nodes=%d  mem_static=%d" %
			[_frame_count, obj_count, orphan_nodes, mem_static])

	_frame_count += 1
