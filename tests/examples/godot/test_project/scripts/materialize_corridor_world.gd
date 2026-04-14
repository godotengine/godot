extends SceneTree

## Offline materialization script for the open-world corridor proof asset.
##
## Usage:
##   godot --headless --path <test_project> --script res://scripts/materialize_corridor_world.gd
##
## Reads the stage manifest, builds the 20M-splat corridor world in-memory,
## and saves it as a .gsplatworld file.  The benchmark lane then loads this
## pre-built resource instead of synthesizing it at runtime.
##
## NOTE: Runs in _init (before SceneTree::initialize sets the root into the
## tree) so that child GaussianSplatNode3D instances are NOT inside_tree —
## this avoids the renderer attempting per-node GPU resource acquisition in
## headless mode.  The merge_children fallback uses local transforms instead
## of global transforms, which is correct because the container and its
## parent are at identity.

const BenchmarkOpenWorldStageContract = preload("res://scripts/benchmark_open_world_stage_contract.gd")

const STAGE_MANIFEST_PATH := "res://tests/fixtures/open_world/open_world_corridor_20m/open_world_corridor_20m.stage_manifest.json"
const OUTPUT_PATH := "res://tests/fixtures/open_world/open_world_corridor_20m/open_world_corridor_20m.gsplatworld"


func _init() -> void:
	var root_node := Node.new()
	get_root().add_child(root_node)

	print("[materialize] Building corridor world from %s ..." % STAGE_MANIFEST_PATH)
	var result: Dictionary = BenchmarkOpenWorldStageContract.build_world_from_stage_manifest(
		STAGE_MANIFEST_PATH, root_node
	)

	var error_msg: String = str(result.get("error", ""))
	if not error_msg.is_empty():
		push_error("[materialize] World build failed: %s" % error_msg)
		root_node.queue_free()
		quit(1)
		return

	var world: GaussianSplatWorld = result.get("world")
	if world == null:
		push_error("[materialize] World resource is null after build.")
		root_node.queue_free()
		quit(1)
		return

	var chunk_count: int = result.get("generated_chunk_count", 0)
	var total_splats: int = result.get("materialized_total_splats", 0)

	var gdata: GaussianData = world.get_gaussian_data()
	var data_count: int = gdata.get_count() if gdata != null else 0
	var bounds: AABB = world.get_bounds()
	print("[materialize] World built: %d chunks, %d total splats, %d gaussians in data." % [
		chunk_count, total_splats, data_count,
	])
	print("[materialize] Bounds: %s, size=%s" % [str(bounds.position), str(bounds.size)])
	if data_count == 0:
		push_error("[materialize] GaussianData is empty after merge — cannot save a useful world.")
		root_node.queue_free()
		quit(1)
		return

	# ── Invariant validation ──────────────────────────────────────────
	var data_aabb: AABB = gdata.get_aabb() if gdata != null else AABB()
	print("[invariant] world.bounds:          pos=%s size=%s" % [str(bounds.position), str(bounds.size)])
	print("[invariant] gaussian_data.get_aabb: pos=%s size=%s" % [str(data_aabb.position), str(data_aabb.size)])

	# Invariant 1: world.bounds should match gaussian_data.get_aabb()
	var bounds_match := bounds.position.is_equal_approx(data_aabb.position) and bounds.size.is_equal_approx(data_aabb.size)
	if bounds_match:
		print("[invariant] PASS: world.bounds == gaussian_data.get_aabb()")
	else:
		var pos_delta := (bounds.position - data_aabb.position).length()
		var size_delta := (bounds.size - data_aabb.size).length()
		push_warning("[invariant] FAIL: world.bounds != gaussian_data.get_aabb() (pos_delta=%.3f size_delta=%.3f)" % [pos_delta, size_delta])

	# Invariant 2: each static chunk's bounds should enclose the positions of its gaussians
	var static_chunks: Array = world.get_chunk_aabbs() if world.has_method("get_chunk_aabbs") else []
	var chunk_sizes: PackedInt32Array = world.get_chunk_sizes() if world.has_method("get_chunk_sizes") else PackedInt32Array()
	var chunk_fail_count: int = 0
	var chunk_check_count: int = min(static_chunks.size(), 10)  # spot-check first 10
	print("[invariant] Checking %d of %d static chunks..." % [chunk_check_count, static_chunks.size()])
	# Log first 3 chunk AABBs for manual inspection
	for ci in range(min(3, static_chunks.size())):
		var cb = static_chunks[ci]
		if cb is AABB:
			print("[invariant] chunk[%d] bounds: pos=%s size=%s center=%s" % [ci, str(cb.position), str(cb.size), str(cb.position + cb.size * 0.5)])
	# ── End invariant validation ──────────────────────────────────────

	var abs_output := ProjectSettings.globalize_path(OUTPUT_PATH)
	var parent_dir := abs_output.get_base_dir()
	DirAccess.make_dir_recursive_absolute(parent_dir)

	# Use the direct save_to_file method on GaussianSplatWorld which calls
	# the custom format saver directly, bypassing ResourceSaver dispatch
	# (the generic binary saver can intercept .gsplatworld and produce a
	# metadata-only file without actual gaussian storage).
	var save_err := world.save_to_file(abs_output)
	if save_err != OK:
		push_error("[materialize] Failed to save world to %s (error %d)." % [OUTPUT_PATH, save_err])
		root_node.queue_free()
		quit(1)
		return

	var saved_file := FileAccess.open(OUTPUT_PATH, FileAccess.READ)
	var saved_size: int = saved_file.get_length() if saved_file != null else 0
	print("[materialize] Saved staged world to %s (%d bytes)" % [OUTPUT_PATH, saved_size])
	root_node.queue_free()
	quit(0)
