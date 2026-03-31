class_name BenchmarkVisualMetrics
extends RefCounted

const SSIM_WINDOW_SIZE := 11
const SSIM_SIGMA := 1.5
const SSIM_K1 := 0.01
const SSIM_K2 := 0.03
const SSIM_MAX_DIM := 256

static func capture_viewport(viewport: Viewport) -> Image:
	if viewport == null:
		return null
	return viewport.get_texture().get_image()

static func ensure_parent_dir(path: String) -> bool:
	var parent_dir := path.get_base_dir()
	if parent_dir.is_empty():
		return true
	var absolute_dir := ProjectSettings.globalize_path(parent_dir)
	return DirAccess.make_dir_recursive_absolute(absolute_dir) == OK

static func save_png(image: Image, path: String) -> Error:
	if image == null:
		return ERR_INVALID_DATA
	if not ensure_parent_dir(path):
		return ERR_CANT_CREATE
	return image.save_png(path)

static func load_image(path: String) -> Image:
	if path.is_empty():
		return null
	var filesystem_path := ProjectSettings.globalize_path(path)
	var image := Image.new()
	var err := image.load(filesystem_path)
	if err != OK:
		return null
	return image

static func calculate_ssim(img_a: Image, img_b: Image) -> float:
	if img_a == null or img_b == null:
		return 0.0
	var a := _prepare_metric_image(img_a)
	var b := _prepare_metric_image(img_b)
	if a == null or b == null:
		return 0.0
	if a.get_size() != b.get_size():
		return 0.0

	var width := a.get_width()
	var height := a.get_height()
	if width < SSIM_WINDOW_SIZE or height < SSIM_WINDOW_SIZE:
		return 0.0

	var luma_a := _compute_luma_buffer(a)
	var luma_b := _compute_luma_buffer(b)
	var kernel := _build_ssim_kernel()
	var radius := int(SSIM_WINDOW_SIZE / 2)

	var c1 := pow(SSIM_K1, 2)
	var c2 := pow(SSIM_K2, 2)
	var ssim_sum := 0.0
	var count := 0

	for y in range(radius, height - radius):
		for x in range(radius, width - radius):
			var mean_a := 0.0
			var mean_b := 0.0
			var kernel_idx := 0
			for wy in range(-radius, radius + 1):
				var row := (y + wy) * width
				for wx in range(-radius, radius + 1):
					var weight := kernel[kernel_idx]
					kernel_idx += 1
					var sample_idx := row + x + wx
					mean_a += weight * luma_a[sample_idx]
					mean_b += weight * luma_b[sample_idx]

			var variance_a := 0.0
			var variance_b := 0.0
			var covariance := 0.0
			kernel_idx = 0
			for wy in range(-radius, radius + 1):
				var row := (y + wy) * width
				for wx in range(-radius, radius + 1):
					var weight := kernel[kernel_idx]
					kernel_idx += 1
					var sample_idx := row + x + wx
					var delta_a := luma_a[sample_idx] - mean_a
					var delta_b := luma_b[sample_idx] - mean_b
					variance_a += weight * delta_a * delta_a
					variance_b += weight * delta_b * delta_b
					covariance += weight * delta_a * delta_b

			var numerator := (2.0 * mean_a * mean_b + c1) * (2.0 * covariance + c2)
			var denominator := (mean_a * mean_a + mean_b * mean_b + c1) * (variance_a + variance_b + c2)
			if denominator > 0.0:
				ssim_sum += numerator / denominator
				count += 1

	if count == 0:
		return 0.0
	return ssim_sum / float(count)

static func calculate_psnr(img_a: Image, img_b: Image) -> float:
	if img_a == null or img_b == null:
		return 0.0
	var a := _prepare_metric_image(img_a)
	var b := _prepare_metric_image(img_b)
	if a == null or b == null:
		return 0.0
	if a.get_size() != b.get_size():
		return 0.0

	var width := a.get_width()
	var height := a.get_height()
	if width <= 0 or height <= 0:
		return 0.0

	var mse := 0.0
	for y in range(height):
		for x in range(width):
			var color_a := a.get_pixel(x, y)
			var color_b := b.get_pixel(x, y)
			mse += pow(color_a.r - color_b.r, 2)
			mse += pow(color_a.g - color_b.g, 2)
			mse += pow(color_a.b - color_b.b, 2)
	mse /= float(width * height * 3)
	if mse <= 0.0000001:
		return 100.0
	return 10.0 * log(1.0 / mse) / log(10.0)

static func _prepare_metric_image(image: Image) -> Image:
	var prepared := image.duplicate()
	if prepared == null:
		return null
	var max_dim := max(prepared.get_width(), prepared.get_height())
	if max_dim > SSIM_MAX_DIM:
		var scale := float(SSIM_MAX_DIM) / float(max_dim)
		var new_width := max(1, int(round(prepared.get_width() * scale)))
		var new_height := max(1, int(round(prepared.get_height() * scale)))
		prepared.resize(new_width, new_height, Image.INTERPOLATE_BILINEAR)
	prepared.convert(Image.FORMAT_RGB8)
	return prepared

static func _compute_luma_buffer(image: Image) -> PackedFloat32Array:
	var output := PackedFloat32Array()
	output.resize(image.get_width() * image.get_height())
	var index := 0
	for y in range(image.get_height()):
		for x in range(image.get_width()):
			var color := image.get_pixel(x, y)
			output[index] = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b
			index += 1
	return output

static func _build_ssim_kernel() -> PackedFloat32Array:
	var kernel := PackedFloat32Array()
	kernel.resize(SSIM_WINDOW_SIZE * SSIM_WINDOW_SIZE)
	var radius := int(SSIM_WINDOW_SIZE / 2)
	var total := 0.0
	var index := 0
	for y in range(-radius, radius + 1):
		for x in range(-radius, radius + 1):
			var weight := exp(-(float(x * x + y * y)) / (2.0 * SSIM_SIGMA * SSIM_SIGMA))
			kernel[index] = weight
			total += weight
			index += 1
	if total <= 0.0:
		return kernel
	for i in range(kernel.size()):
		kernel[i] /= total
	return kernel
