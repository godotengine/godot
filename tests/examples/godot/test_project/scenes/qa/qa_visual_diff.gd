extends "res://scripts/qa_test_base.gd"
## Visual Diff Test: Compares world path vs instance path rendering.
## Passes if SSIM between both paths exceeds threshold.

@export var ssim_threshold: float = 0.95
@export var capture_delay_frames: int = 10

var world_path_node: Node3D
var instance_node: Node3D
var world_path_image: Image
var instance_image: Image

enum CapturePhase { NONE, WORLD_PATH, INSTANCE_PATH, COMPARE }
var capture_phase: CapturePhase = CapturePhase.NONE
var capture_frame_count: int = 0

func _ready():
	test_name = "World Path vs Instance Path Visual Diff"
	test_duration = 5.0
	super._ready()

	world_path_node = get_node_or_null("WorldPath")
	instance_node = get_node_or_null("InstancePath")

func _on_test_start():
	# Start with world path visible
	capture_phase = CapturePhase.WORLD_PATH
	capture_frame_count = 0
	_set_visibility(true, false)

func _on_test_frame(_delta: float):
	capture_frame_count += 1

	match capture_phase:
		CapturePhase.WORLD_PATH:
			if capture_frame_count >= capture_delay_frames:
				world_path_image = capture_viewport()
				print("[QA:%s] Captured world-path image (world_path.visible=%s, instance.visible=%s)" % [
					test_name, world_path_node.visible, instance_node.visible
				])
				# Switch visibility BEFORE waiting for next capture
				_set_visibility(false, true)
				capture_phase = CapturePhase.INSTANCE_PATH
				capture_frame_count = 0

		CapturePhase.INSTANCE_PATH:
			if capture_frame_count >= capture_delay_frames:
				instance_image = capture_viewport()
				print("[QA:%s] Captured instance-path image (world_path.visible=%s, instance.visible=%s)" % [
					test_name, world_path_node.visible, instance_node.visible
				])
				capture_phase = CapturePhase.COMPARE

		CapturePhase.COMPARE:
			# Let the test duration naturally complete
			pass

func _on_test_complete():
	if world_path_image == null or instance_image == null:
		_test_result = false
		_test_message = "Failed to capture images"
		return

	var ssim = calculate_ssim(world_path_image, instance_image)
	result_metrics["ssim"] = ssim
	result_metrics["ssim_threshold"] = ssim_threshold
	_test_result = ssim >= ssim_threshold
	_test_message = "SSIM: %.4f (threshold: %.2f)" % [ssim, ssim_threshold]

	# Always save images for inspection
	world_path_image.save_png("user://qa_visual_diff_world_path.png")
	instance_image.save_png("user://qa_visual_diff_instance_path.png")
	print("[QA:%s] Images saved to user://" % test_name)

func _set_visibility(world_path_visible: bool, instance_path_visible: bool):
	if world_path_node != null:
		world_path_node.visible = world_path_visible
	if instance_node != null:
		instance_node.visible = instance_path_visible
