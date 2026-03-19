extends "res://scripts/qa_test_base.gd"
## Sorting Tie-Breaker Test: Ensures stable ordering when depths are equal.

@export var capture_delay_frames: int = 8
@export var capture_interval_frames: int = 4
@export var capture_samples: int = 5
@export var ssim_stability_threshold: float = 0.98

var splat_node: GaussianSplatNode3D
var captured_images: Array[Image] = []
var _prev_tie_breaker = null

func _ready():
	test_name = "Sort Tie-Breaker"
	test_duration = 6.0
	warmup_frames = 5
	super._ready()

	splat_node = get_node_or_null("SplatNode")

func _on_test_start():
	_prev_tie_breaker = ProjectSettings.get_setting("rendering/gaussian_splatting/gpu_sorting/enable_tie_breaker")
	ProjectSettings.set_setting("rendering/gaussian_splatting/gpu_sorting/enable_tie_breaker", true)

	if splat_node == null:
		return

	var asset := GaussianSplatAsset.new()
	asset.set_splat_count(2)

	var positions := PackedFloat32Array([0.0, 0.0, -1.5, 0.0, 0.0, -1.5])
	var colors := PackedColorArray([Color(1.0, 0.0, 0.0, 1.0), Color(0.0, 1.0, 0.0, 1.0)])
	var scales := PackedFloat32Array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7])

	asset.set_positions(positions)
	asset.set_colors(colors)
	asset.set_scales(scales)

	splat_node.splat_asset = asset
	captured_images.clear()

func _on_test_frame(_delta: float):
	if frame_count < capture_delay_frames:
		return
	if captured_images.size() >= capture_samples:
		return
	if (frame_count - capture_delay_frames) % capture_interval_frames != 0:
		return

	var image = capture_viewport()
	if image != null:
		captured_images.append(image)
	if captured_images.size() >= capture_samples:
		_finish_test()

func _on_test_complete():
	if _prev_tie_breaker != null:
		ProjectSettings.set_setting("rendering/gaussian_splatting/gpu_sorting/enable_tie_breaker", _prev_tie_breaker)
	if captured_images.size() < 2:
		_test_result = false
		_test_message = "Insufficient captures"
		return

	var min_ssim = 1.0
	var sum_ssim = 0.0
	var comparisons = 0

	for i in range(1, captured_images.size()):
		var ssim = calculate_ssim(captured_images[i - 1], captured_images[i])
		min_ssim = min(min_ssim, ssim)
		sum_ssim += ssim
		comparisons += 1

	var avg_ssim = sum_ssim / float(comparisons)
	result_metrics["ssim_min"] = min_ssim
	result_metrics["ssim_avg"] = avg_ssim
	result_metrics["ssim_threshold"] = ssim_stability_threshold

	_test_result = min_ssim >= ssim_stability_threshold
	_test_message = "SSIM min=%.4f avg=%.4f" % [min_ssim, avg_ssim]
