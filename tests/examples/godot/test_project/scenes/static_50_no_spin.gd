extends Node3D

# 50 script-spawned instances, NO spinning - isolates count vs transform updates
const INSTANCE_COUNT := 50
const GRID_COLS := 10
const GRID_SPACING := 25.0

func _ready() -> void:
	var asset = load("res://tests/fixtures/test_splats.ply")
	if not asset:
		push_error("[TEST-50-NOSPIN] Failed to load synthetic fixture asset: res://tests/fixtures/test_splats.ply")
		get_tree().quit()
		return

	for i in INSTANCE_COUNT:
		var col := i % GRID_COLS
		var row := i / GRID_COLS
		var node := GaussianSplatNode3D.new()
		node.name = "Instance_%02d" % i
		node.transform.origin = Vector3(col * GRID_SPACING, 0, row * GRID_SPACING)
		add_child(node)
		node.splat_asset = asset

	var cam := get_node_or_null("Camera3D")
	if cam:
		var center_x := (GRID_COLS - 1) * GRID_SPACING / 2.0
		var rows := (INSTANCE_COUNT - 1) / GRID_COLS
		var center_z := rows * GRID_SPACING / 2.0
		cam.look_at(Vector3(center_x, 0, center_z), Vector3.UP)

	print("[TEST-50-NOSPIN] Spawned %d instances, NO spinning" % INSTANCE_COUNT)
