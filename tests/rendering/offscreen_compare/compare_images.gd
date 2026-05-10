extends SceneTree

func _initialize() -> void:
	var reference_path := _get_required_arg("--reference")
	var candidate_path := _get_required_arg("--candidate")
	var diff_path := _get_user_arg_value("--diff", "")
	var max_channel_delta := _get_user_arg_value("--max-channel-delta", "0").to_int()
	var max_different_pixels := _get_user_arg_value("--max-different-pixels", "0").to_int()

	var reference := Image.new()
	var candidate := Image.new()
	var err := reference.load(reference_path)
	if err != OK:
		_fail("Failed to load reference image: %s (%s)" % [reference_path, error_string(err)])
		return

	err = candidate.load(candidate_path)
	if err != OK:
		_fail("Failed to load candidate image: %s (%s)" % [candidate_path, error_string(err)])
		return

	if reference.get_width() != candidate.get_width() or reference.get_height() != candidate.get_height():
		_fail("Image sizes differ: reference=%dx%d candidate=%dx%d" % [reference.get_width(), reference.get_height(), candidate.get_width(), candidate.get_height()])
		return

	reference.convert(Image.FORMAT_RGBA8)
	candidate.convert(Image.FORMAT_RGBA8)

	var width := reference.get_width()
	var height := reference.get_height()
	var reference_data := reference.get_data()
	var candidate_data := candidate.get_data()
	var diff: Image = null
	if not diff_path.is_empty():
		diff = Image.create(width, height, false, Image.FORMAT_RGBA8)

	var different_pixels := 0
	var max_delta := 0
	var total_delta := 0
	var channel_count := width * height * 4
	for y in range(height):
		for x in range(width):
			var data_index := (y * width + x) * 4
			var pixel_delta := 0
			for channel in range(4):
				var delta: int = abs(reference_data[data_index + channel] - candidate_data[data_index + channel])
				pixel_delta = max(pixel_delta, delta)
				max_delta = max(max_delta, delta)
				total_delta += delta

			if pixel_delta > max_channel_delta:
				different_pixels += 1

			if diff != null:
				var diff_value := float(pixel_delta) / 255.0
				diff.set_pixel(x, y, Color(diff_value, diff_value, diff_value, 1.0))

	if diff != null:
		var diff_err := diff.save_png(diff_path)
		if diff_err != OK:
			_fail("Failed to save diff image: %s (%s)" % [diff_path, error_string(diff_err)])
			return

	var mean_delta := float(total_delta) / float(channel_count)
	print("IMAGE_COMPARE_SIZE=", width, "x", height)
	print("IMAGE_COMPARE_DIFFERENT_PIXELS=", different_pixels)
	print("IMAGE_COMPARE_MAX_DELTA=", max_delta)
	print("IMAGE_COMPARE_MEAN_DELTA=", mean_delta)

	if different_pixels > max_different_pixels:
		_fail("Images differ beyond tolerance: different_pixels=%d max_different_pixels=%d max_channel_delta=%d max_delta=%d" % [different_pixels, max_different_pixels, max_channel_delta, max_delta])
		return

	quit(0)

func _get_required_arg(name: String) -> String:
	var value := _get_user_arg_value(name, "")
	if value.is_empty():
		_fail("Missing required argument: %s" % name)
	return value

func _get_user_arg_value(name: String, default_value: String) -> String:
	var args := OS.get_cmdline_user_args()
	for i in range(args.size()):
		if args[i] == name and i + 1 < args.size():
			return args[i + 1]
	return default_value

func _fail(message: String) -> void:
	printerr(message)
	quit(1)
