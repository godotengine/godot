extends SceneTree
## Run with a real renderer; the C++ unit tests use a mock rendering server.

const ORIGIN := Vector2(350, 80)
const SMALL := Vector2(180, 36)
const LARGE := Vector2(800, 600)
const REGION := Rect2i(0, 0, 1200, 760)
var output: String
var results: Array[Dictionary] = []
var failures: Array[String] = []
var surface: SubViewport
var host: Control
var edit: RichTextEdit

func _initialize() -> void:
	output = ProjectSettings.globalize_path("res://../../../../bin/overflow_qa")
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--output="):
			output = argument.trim_prefix("--output=")
	DirAccess.make_dir_recursive_absolute(output)
	call_deferred("run")

func require(condition: bool, message: String) -> void:
	if not condition:
		failures.append(message)
		push_error(message)

## Both axes get the same mode; the axes themselves are covered by the C++ tests.
func set_overflow(target: RichTextEdit, mode: int) -> void:
	target.overflow_x = mode
	target.overflow_y = mode

func frame() -> void:
	await process_frame
	await process_frame
	await RenderingServer.frame_post_draw

func capture(name: String) -> Image:
	await frame()
	var image := surface.get_texture().get_image()
	image.save_png(output.path_join(name + ".png"))
	return image

func differences(a: Image, b: Image) -> int:
	var count := 0
	for y in range(REGION.position.y, REGION.end.y):
		for x in range(REGION.position.x, REGION.end.x):
			var ca := a.get_pixel(x, y)
			var cb := b.get_pixel(x, y)
			if absf(ca.r - cb.r) + absf(ca.g - cb.g) + absf(ca.b - cb.b) > 0.035:
				count += 1
	return count

func ink_outside(image: Image, rect: Rect2i) -> int:
	var count := 0
	for y in range(REGION.end.y):
		for x in range(REGION.end.x):
			if rect.has_point(Vector2i(x, y)):
				continue
			var color := image.get_pixel(x, y)
			if color.r + color.g + color.b < 2.8:
				count += 1
	return count

func set_content(bbcode: String, wrapped := false) -> void:
	edit.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY if wrapped else TextEdit.LINE_WRAPPING_NONE
	edit.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	edit.bbcode_text = bbcode
	edit.visible_characters = -1
	edit.set_caret_line(edit.get_line_count() - 1, false)
	edit.set_caret_column(edit.get_line(edit.get_line_count() - 1).length(), false)

func compare_reference(name: String, bbcode: String, wrapped := false, alignment := 0) -> void:
	host.clip_contents = false
	edit.clip_contents = false
	edit.editable = false
	set_overflow(edit, RichTextEdit.OVERFLOW_VISIBLE)
	edit.position = Vector2.ZERO
	edit.size = SMALL
	set_content(bbcode, wrapped)
	var actual := await capture(name + "_overflow")
	var caret := edit.get_caret_draw_pos()
	var last_line := edit.get_line_count() - 1
	var last_rect := edit.get_rect_at_line_column(last_line, edit.get_line(last_line).length())
	require(last_rect.position != Vector2i(-1, -1), name + ": offscreen caret rectangle missing")
	var content_height := edit.get_content_height()
	require(content_height == last_rect.end.y, name + ": measured height differs from the last rendered row")
	require(not edit.get_h_scroll_bar().visible and not edit.get_v_scroll_bar().visible,
			name + ": visible overflow acquired scrollbars")
	require(edit.size == SMALL, name + ": control resized")
	var outside := ink_outside(actual, Rect2i(ORIGIN, SMALL))
	require(outside > 0, name + ": fixture drew no overflow")
	# A tall control with the same wrap width, or a wide non-wrapping control,
	# is the ordinary native viewport reference. Align its origin to the same
	# authored center/right edge without changing either document.
	var reference_size := Vector2(SMALL.x, LARGE.y) if wrapped else LARGE
	var shift := 0.0 if wrapped else (SMALL.x - reference_size.x) * alignment / 2.0
	set_overflow(edit, RichTextEdit.OVERFLOW_AUTO)
	edit.position = Vector2(shift, 0)
	edit.size = reference_size
	var reference := await capture(name + "_reference")
	var reference_caret := edit.get_caret_draw_pos() + edit.position
	require(caret.distance_to(reference_caret) <= 1.1, name + ": caret differs from native reference")
	var changed := differences(actual, reference)
	require(changed == 0, name + ": pixels differ from native reference: " + str(changed))
	results.append({"case": name, "different_pixels": changed, "overflow_pixels": outside,
			"caret": str(caret), "last_rect": str(last_rect), "content_height": content_height})

func compare_mixed_alignment() -> void:
	var paragraphs := ["A long left paragraph outside its authored rectangle.",
			"[center]Short[/center]", "[right]Long right paragraph outside its rectangle.[/right]",
			"[center]Long centered paragraph outside its rectangle.[/center]", "[right]End[/right]"]
	var alignments := [0, 1, 2, 1, 2]
	set_overflow(edit, RichTextEdit.OVERFLOW_VISIBLE)
	edit.position = Vector2.ZERO
	edit.size = SMALL
	set_content("\n".join(paragraphs))
	var actual := await capture("mixed_alignment_overflow")
	# Each native reference paragraph keeps the original paragraph's alignment
	# anchor. A single wider reference would incorrectly move the short lines.
	var references: Array[RichTextEdit] = []
	for index in range(paragraphs.size()):
		var reference := edit.duplicate() as RichTextEdit
		set_overflow(reference, RichTextEdit.OVERFLOW_AUTO)
		reference.size = LARGE
		reference.bbcode_text = paragraphs[index]
		reference.position = Vector2((SMALL.x - LARGE.x) * alignments[index] / 2.0,
				edit.get_rect_at_line_column(index, 0).position.y)
		host.add_child(reference)
		references.append(reference)
	edit.hide()
	var reference_image := await capture("mixed_alignment_reference")
	var changed := differences(actual, reference_image)
	require(changed == 0, "mixed alignment pixels differ: " + str(changed))
	results.append({"case": "mixed_alignment", "different_pixels": changed})
	for reference in references:
		reference.free()
	edit.show()

func compare_default_clipping() -> void:
	var overflowing := "A long line that should be clipped by the native viewport\nSecond\nThird\nFourth"
	set_overflow(edit, RichTextEdit.OVERFLOW_AUTO)
	edit.position = Vector2.ZERO
	edit.size = SMALL
	edit.editable = true
	set_content(overflowing)
	var original := await capture("default_editable")
	# Overflow no longer follows the editable state in either direction.
	for editable in [false, true, false, true]:
		edit.editable = editable
		await frame()
		require(edit.get_h_scroll_bar().visible and edit.get_v_scroll_bar().visible,
				"auto overflow lost its scrollbars when editable=%s" % editable)
	var restored := await capture("restored_editable")
	require(differences(original, restored) == 0, "editing transition did not restore native pixels")
	for mode in [RichTextEdit.OVERFLOW_VISIBLE, RichTextEdit.OVERFLOW_HIDDEN]:
		for editable in [true, false]:
			set_overflow(edit, mode)
			edit.editable = editable
			await frame()
			require(not edit.get_h_scroll_bar().visible and not edit.get_v_scroll_bar().visible,
					"overflow %d retained scrollbars when editable=%s" % [mode, editable])
	# Hidden clips to the control without ever showing a scrollbar.
	set_overflow(edit, RichTextEdit.OVERFLOW_HIDDEN)
	edit.editable = false
	var hidden := await capture("hidden_read_only")
	require(ink_outside(hidden, Rect2i(ORIGIN, SMALL)) == 0, "hidden overflow leaked outside the control")
	# Scroll keeps both bars over content that fits, without shifting it.
	set_overflow(edit, RichTextEdit.OVERFLOW_SCROLL)
	set_content("Fits")
	await frame()
	require(edit.get_h_scroll_bar().visible and edit.get_v_scroll_bar().visible,
			"scroll overflow dropped its scrollbars over content that fits")
	require(edit.get_h_scroll() == 0 and edit.get_v_scroll() == 0, "forced scrollbars shifted the viewport")
	var fitted := await capture("scroll_fitted")
	require(ink_outside(fitted, Rect2i(ORIGIN, SMALL)) == 0, "scroll overflow leaked outside the control")
	set_overflow(edit, RichTextEdit.OVERFLOW_AUTO)
	set_content(overflowing)
	var read_only := await capture("default_read_only")
	require(ink_outside(read_only, Rect2i(ORIGIN, SMALL)) == 0, "default read-only viewport leaked")
	results.append({"case": "default_clipping_and_editing_transition", "different_pixels": differences(original, restored)})

func check_scene_round_trip() -> void:
	set_overflow(edit, RichTextEdit.OVERFLOW_VISIBLE)
	edit.size = SMALL
	set_content("A long scene-restored Article\nSecond\nThird")
	var scene := PackedScene.new()
	require(scene.pack(edit) == OK, "could not pack RichTextEdit scene")
	var restored := scene.instantiate() as RichTextEdit
	require(restored != null, "could not restore RichTextEdit scene")
	if restored == null:
		return
	host.add_child(restored)
	await frame()
	require(restored.overflow_x == RichTextEdit.OVERFLOW_VISIBLE and restored.overflow_y == RichTextEdit.OVERFLOW_VISIBLE and not restored.editable, "scene lost the overflow mode")
	require(restored.get_bbcode_text() == edit.get_bbcode_text(), "scene changed the document")
	require(restored.mouse_filter == Control.MOUSE_FILTER_IGNORE and restored.focus_mode == Control.FOCUS_NONE
			and not restored.caret_draw_when_editable_disabled, "scene changed passive input/caret settings")
	require(not restored.get_h_scroll_bar().visible and not restored.get_v_scroll_bar().visible,
			"restored overflow mode acquired scrollbars")
	restored.free()
	results.append({"case": "scene_round_trip"})

func run() -> void:
	if "--decorations-only" in OS.get_cmdline_user_args():
		await check_decoration_pixels()
		finish()
		return
	# Render into a fixed SubViewport so desktop scaling/window size cannot
	# change the pixel comparison or crop the overflow reference.
	surface = SubViewport.new()
	surface.size = Vector2i(1200, 800)
	surface.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	root.add_child(surface)
	host = Control.new()
	host.position = ORIGIN
	host.size = SMALL
	surface.add_child(host)
	edit = RichTextEdit.new()
	edit.bbcode_enabled = true
	edit.editable = false
	edit.mouse_filter = Control.MOUSE_FILTER_IGNORE
	edit.focus_mode = Control.FOCUS_NONE
	edit.caret_draw_when_editable_disabled = false
	edit.add_theme_stylebox_override("normal", StyleBoxEmpty.new())
	edit.add_theme_stylebox_override("read_only", StyleBoxEmpty.new())
	edit.add_theme_stylebox_override("focus", StyleBoxEmpty.new())
	edit.add_theme_constant_override("wrap_offset", 0)
	edit.add_theme_font_size_override("font_size", 24)
	edit.add_theme_color_override("font_color", Color.BLACK)
	edit.add_theme_color_override("font_readonly_color", Color.BLACK)
	host.add_child(edit)
	var identity := edit.get_instance_id()
	await compare_reference("nowrap", "A long Article extends beyond its small authored rectangle.")
	await compare_reference("multiline", "First line\nSecond line\nThird line\nFourth line")
	await compare_reference("wrapped", "one two three four five six seven eight nine ten eleven twelve", true)
	await compare_reference("center", "[center]Centered text wider than its Article.[/center]", false, 1)
	await compare_reference("right", "[right]Right aligned text wider than its Article.[/right]", false, 2)
	# Keep the antialiased line cap inside the native reference's left clip.
	# Overflow intentionally preserves that cap when it is outside the control.
	await compare_reference("decoration", " [bgcolor=#44aa88][outline_size=3][outline_color=#7722aa][u]Long decorated text beyond the rectangle[/u][/outline_color][/outline_size][/bgcolor]")
	await compare_reference("mixed_size", "[font_size=52]Big[/font_size] [b]bold[/b] [i]italic text[/i]\n[font_size=40]Second line[/font_size]")
	var inline_image := Image.create(64, 48, false, Image.FORMAT_RGBA8)
	inline_image.fill(Color(0.1, 0.6, 0.9))
	inline_image.save_png("user://overflow_inline.png")
	await compare_reference("inline_image", "An inline image outside: [img]user://overflow_inline.png[/img]\nLast line")
	await compare_reference("rtl", "[right]שלום עולם שלום עולם שלום עולם[/right]", false, 2)
	await compare_mixed_alignment()
	await compare_default_clipping()

	# Parent clipping remains authoritative, including glyph overhangs.
	set_overflow(edit, RichTextEdit.OVERFLOW_VISIBLE)
	edit.position = Vector2.ZERO
	edit.size = SMALL
	set_content("Text outside both bounds\nSecond line\nThird line")
	for parent_clip in [true, false]:
		host.clip_contents = parent_clip
		edit.clip_contents = not parent_clip
		var case_name := "parent_clip" if parent_clip else "control_clip"
		var clipped := await capture(case_name)
		var outside := ink_outside(clipped, Rect2i(ORIGIN, SMALL))
		require(outside == 0, case_name + ": clip leaked pixels")
		results.append({"case": case_name, "overflow_pixels": outside})
	host.clip_contents = false
	edit.clip_contents = false

	# Every revealed prefix uses the same instance and its native shaped caret.
	var document := "A😀\n中文 é\nlast line outside"
	set_content(document)
	for count in [0, 1, 2, 3, 5, 8, document.length()]:
		set_overflow(edit, RichTextEdit.OVERFLOW_VISIBLE)
		edit.size = SMALL
		edit.bbcode_text = document
		edit.visible_characters = count
		var prefix := document.substr(0, count)
		var lines := prefix.split("\n")
		edit.set_caret_line(lines.size() - 1, false)
		edit.set_caret_column(lines[-1].length(), false)
		var actual := await capture("prefix_" + str(count))
		var caret := edit.get_caret_draw_pos()
		var content_height := edit.get_content_height()
		set_overflow(edit, RichTextEdit.OVERFLOW_AUTO)
		edit.size = LARGE
		edit.bbcode_text = prefix
		edit.visible_characters = -1
		edit.set_caret_line(lines.size() - 1, false)
		edit.set_caret_column(lines[-1].length(), false)
		var reference := await capture("prefix_" + str(count) + "_reference")
		var changed := differences(actual, reference)
		require(changed == 0, "prefix " + str(count) + ": hidden content painted: " + str(changed))
		require(caret.distance_to(edit.get_caret_draw_pos()) <= 1.1, "prefix caret drift")
		# Height preserves the native document's logical rows, including empty
		# unrevealed rows. Compare the same document/reveal state, not a string
		# whose unrevealed line breaks have actually been deleted.
		edit.bbcode_text = document
		edit.visible_characters = count
		await frame()
		require(content_height == edit.get_content_height(), "prefix content height differs from native viewport")
		results.append({"case": "prefix_" + str(count), "different_pixels": changed, "content_height": content_height})
	require(edit.get_instance_id() == identity, "editor instance changed")
	await check_scene_round_trip()
	await check_decoration_pixels()
	finish()

func finish() -> void:
	var report := {"cases": results, "failures": failures}
	var file := FileAccess.open(output.path_join("report.json"), FileAccess.WRITE)
	file.store_string(JSON.stringify(report, "\t"))
	file.close()
	print("RICH TEXT EDIT OVERFLOW PIXELS: ", "PASS" if failures.is_empty() else "FAIL", "; report=", output)
	quit(0 if failures.is_empty() else 1)

func check_decoration_pixels() -> void:
	# Sample the spaces between glyphs: ink here must be a decoration, not
	# part of a letter. This is an independent presence/continuity oracle,
	# unlike comparing two images that both use the same broken renderer.
	var viewport := SubViewport.new()
	viewport.size = Vector2i(1200, 2000)
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	root.add_child(viewport)
	var parent := Node2D.new()
	viewport.add_child(parent)
	var probe := RichTextEdit.new()
	probe.size = Vector2(560, 1800)
	probe.bbcode_enabled = true
	probe.caret_draw_when_editable_disabled = false
	for style_name in ["normal", "read_only", "focus"]:
		probe.add_theme_stylebox_override(style_name, StyleBoxEmpty.new())
	parent.add_child(probe)
	var sizes := [12, 16, 24, 36, 48, 72]
	var scales := [1.0, 0.75, 0.5, 1.25, 1.5, 2.0]
	var heights := ["normal", "0.75", "1", "1.25", "1.5", "2", "125%", "150%", "96px"]
	var phases := [0.0, 0.25, 0.5, 0.75]
	if "--smoke" in OS.get_cmdline_user_args():
		sizes = [24]
		scales = [0.75]
		heights = ["1"]
		phases = [0.0, 0.5]
	var colors := ["#ff0000", "#00cc00", "#0000ff"]
	var failure_start := failures.size()
	var case_count := 0
	for font_size in sizes:
		for height in heights:
			var lines := PackedStringArray()
			for color in colors:
				lines.append("[line_height=%s][font_size=%d][color=%s][u][s][o]M      M[/o][/s][/u][/color][/font_size][/line_height]" % [height, font_size, color])
			probe.bbcode_text = "\n".join(lines)
			for scale_factor in scales:
				# Changing a canvas transform does not invalidate the text's layout.
				# Exercise this without manually calling queue_redraw().
				viewport.canvas_transform = Transform2D(0, Vector2(scale_factor, scale_factor), 0, Vector2(80, 80))
				for phase in phases:
					parent.position = Vector2(0, phase / scale_factor)
					await frame()
					await frame()
					var image := viewport.get_texture().get_image()
					var issues := decoration_issues(image, probe, 3)
					var name := "decorations_fs%d_h%s_s%s_y%s" % [font_size, height, scale_factor, phase]
					if not issues.is_empty():
						failures.append(name + ": " + "; ".join(issues))
					if (font_size == 24 and height == "1" and scale_factor == 0.75) or (not issues.is_empty() and failures.size() - failure_start <= 4):
						image.save_png(output.path_join(name + ".png"))
					results.append({"case": name, "failures": issues})
					case_count += 1
	if not "--smoke" in OS.get_cmdline_user_args():
		# Also exercise CanvasItem transforms and removal of individual styles
		# on the same instance, including the transition to undecorated text.
		viewport.canvas_transform = Transform2D(0, Vector2(80, 80))
		for font_size in [12, 24, 72]:
			for mask in [7, 1, 2, 4, 3, 5, 6, 0]:
				var open_tags := ""
				var close_tags := ""
				var expected := 0
				for kind in range(3):
					if mask & (1 << kind):
						var tag: String = ["u", "s", "o"][kind]
						open_tags += "[" + tag + "]"
						close_tags = "[/" + tag + "]" + close_tags
						expected += 1
				var lines := PackedStringArray()
				for color in colors:
					lines.append("[line_height=1.25][font_size=%d][color=%s]%sM      M%s[/color][/font_size][/line_height]" % [font_size, color, open_tags, close_tags])
				probe.bbcode_text = "\n".join(lines)
				for scale_factor in [0.5, 0.75, 1.5]:
					probe.scale = Vector2.ONE * scale_factor
					for phase in [0.0, 0.5]:
						parent.position = Vector2(0, phase)
						await frame()
						await frame()
						var image := viewport.get_texture().get_image()
						var issues := decoration_issues(image, probe, expected)
						var name := "decoration_mask%d_fs%d_node%s_y%s" % [mask, font_size, scale_factor, phase]
						if not issues.is_empty():
							failures.append(name + ": " + "; ".join(issues))
						if (font_size == 24 and scale_factor == 0.75 and phase == 0.0) or (not issues.is_empty() and failures.size() - failure_start <= 4):
							image.save_png(output.path_join(name + ".png"))
						results.append({"case": name, "failures": issues})
						case_count += 1
		probe.scale = Vector2(0.75, 0.75)
		for font_tags in [["[b]", "[/b]"], ["[i]", "[/i]"], ["[b][i]", "[/i][/b]"]]:
			var lines := PackedStringArray()
			for color in colors:
				lines.append("[font_size=24][color=%s]%s[u][s][o]M      M[/o][/s][/u]%s[/color][/font_size]" % [color, font_tags[0], font_tags[1]])
			probe.bbcode_text = "\n".join(lines)
			for phase in [0.0, 0.5]:
				parent.position = Vector2(0, phase)
				await frame()
				await frame()
				var image := viewport.get_texture().get_image()
				var issues := decoration_issues(image, probe, 3)
				var name := "decoration_font%s_y%s" % [font_tags[0], phase]
				if not issues.is_empty():
					failures.append(name + ": " + "; ".join(issues))
				image.save_png(output.path_join(name + ".png"))
				results.append({"case": name, "failures": issues})
				case_count += 1
	viewport.free()
	print("DECORATION PIXELS: ", case_count, " cases; ", failures.size() - failure_start, " failed")
	for i in range(failure_start, mini(failure_start + 5, failures.size())):
		print(failures[i])

func decoration_issues(image: Image, probe: RichTextEdit, expected: int) -> PackedStringArray:
	var transform := probe.get_global_transform_with_canvas()
	var from := transform * Vector2(probe.get_rect_at_line_column(0, 2).position)
	var to := transform * Vector2(probe.get_rect_at_line_column(0, 6).position)
	var left := ceili(minf(from.x, to.x)) + 1
	var right := floori(maxf(from.x, to.x)) - 1
	var middle := (left + right) / 2
	var issues := PackedStringArray()
	for color_index in range(3):
		var bands: Array[Vector2i] = []
		var start := -1
		for y in range(image.get_height()):
			var ink := decoration_ink(image.get_pixel(middle, y), color_index)
			if ink > 0.08 and start < 0:
				start = y
			elif ink <= 0.08 and start >= 0:
				bands.append(Vector2i(start, y))
				start = -1
		if start >= 0:
			bands.append(Vector2i(start, image.get_height()))
		if bands.size() != expected:
			issues.append("line %d: expected %d strokes, got %d" % [color_index + 1, expected, bands.size()])
		for band in bands:
			var lowest := 1.0
			var highest := 0.0
			for x in range(left, right + 1):
				var peak := 0.0
				for y in range(band.x, band.y):
					peak = maxf(peak, decoration_ink(image.get_pixel(x, y), color_index))
				lowest = minf(lowest, peak)
				highest = maxf(highest, peak)
			if lowest <= 0.08 or highest - lowest > 0.06:
				issues.append("line %d: broken/overlapping glyph joins" % [color_index + 1])
	return issues

func decoration_ink(color: Color, channel: int) -> float:
	match channel:
		0:
			return maxf(color.r - maxf(color.g, color.b), 0.0)
		1:
			return maxf(color.g - maxf(color.r, color.b), 0.0)
		_:
			return maxf(color.b - maxf(color.r, color.g), 0.0)
