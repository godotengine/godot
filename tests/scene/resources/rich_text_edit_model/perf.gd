extends SceneTree

# RichTextEdit editing cost on a long, heavily styled document. Run headless:
#   godot --headless --path tests/scene/resources/rich_text_edit_model --script perf.gd -- [--chars=500000] [--mode=scroll|canvas]
# "scroll" is a clipped 900x600 editor; "canvas" sizes the editor to its whole
# content with overflow visible and shows its middle, like an Article on a
# page. Each operation is timed with the two frames that draw its result;
# exits with 1 when one exceeds its budget.
const Doc = preload("res://doc.gd")

# Budgets in milliseconds, generous enough for slower machines.
const EDIT_BUDGET := 100.0
const LARGE_BUDGET := 600.0

var edit: RichTextEdit
var mode := "scroll"
var failures := PackedStringArray()


func frames(p_count := 2) -> void:
	for i in p_count:
		await process_frame


func measure(p_name: String, p_budget: float, p_body: Callable, p_iterations := 1) -> void:
	var samples := []
	for i in p_iterations:
		await frames()
		var start := Time.get_ticks_usec()
		p_body.call(i)
		await frames()
		samples.append((Time.get_ticks_usec() - start) / 1000.0)
	samples.sort()
	var median: float = samples[samples.size() / 2]
	var verdict := "ok" if median <= p_budget else "OVER %.0f" % p_budget
	print("%-22s median %8.2f ms  max %8.2f ms  %s" % [p_name, median, samples[-1], verdict])
	if median > p_budget:
		failures.append(p_name)


func place_middle() -> void:
	if mode != "canvas":
		return
	var height := edit.get_content_height()
	edit.size = Vector2(900, height)
	edit.position = Vector2(20, 300 - height / 2.0)


func _initialize() -> void:
	var chars := 500000
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--chars="):
			chars = argument.trim_prefix("--chars=").to_int()
		elif argument.begins_with("--mode="):
			mode = argument.trim_prefix("--mode=")
	var image := Image.create(64, 48, false, Image.FORMAT_RGBA8)
	image.fill(Color(0.2, 0.5, 0.8))
	image.save_png("user://perf_image.png")
	var doc := Doc.new()
	var bbcode := doc.make(chars)
	print("mode=%s characters=%d paragraphs=%d bbcode=%d" % [mode, chars, doc.line_count, bbcode.length()])

	edit = RichTextEdit.new()
	edit.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	edit.size = Vector2(900, 600)
	edit.position = Vector2(20, 20)
	if mode == "canvas":
		edit.overflow_x = RichTextEdit.OVERFLOW_VISIBLE
		edit.overflow_y = RichTextEdit.OVERFLOW_VISIBLE
		edit.content_height_changed.connect(place_middle)
	root.add_child(edit)
	await frames()
	edit.grab_focus()

	await measure("load", 2000.0, func(_i): edit.set_bbcode_text(bbcode))
	place_middle()
	await frames()
	var mid := edit.get_line_count() / 2

	await measure("type_char", EDIT_BUDGET, func(i):
		edit.deselect()
		edit.set_caret_line(mid, false)
		edit.set_caret_column(5 + i, false)
		edit.insert_text_at_caret("字"), 8)
	await measure("backspace", EDIT_BUDGET, func(_i):
		edit.set_caret_line(mid, false)
		edit.set_caret_column(12, false)
		edit.backspace(), 4)
	var chunk := ""
	for k in 300:
		chunk += Doc.paragraph(100000 + k) + "\n"
	await measure("paste_plain_18k", 300.0, func(_i):
		edit.set_caret_line(mid, false)
		edit.set_caret_column(0, false)
		edit.insert_text_at_caret(chunk), 2)
	await measure("rich_copy_200_lines", EDIT_BUDGET, func(_i):
		edit.select(mid - 300, 0, mid - 100, 0)
		edit.copy_rich_text(), 2)
	await measure("rich_paste_200_lines", 300.0, func(_i):
		edit.deselect()
		edit.set_caret_line(mid + 50, false)
		edit.set_caret_column(0, false)
		edit.paste_rich_text(), 2)
	await measure("insert_image", EDIT_BUDGET, func(_i):
		edit.deselect()
		edit.set_caret_line(mid, false)
		edit.set_caret_column(20, false)
		edit.insert_image("user://perf_image.png", 64, 48), 3)
	await measure("bold_word", EDIT_BUDGET, func(i):
		edit.select(mid + i, 2, mid + i, 9)
		edit.toggle_bold(), 6)
	await measure("color_word", EDIT_BUDGET, func(i):
		edit.select(mid + i, 2, mid + i, 9)
		edit.set_selection_color(Color(0.1, 0.6, 0.2)), 6)
	await measure("undo", EDIT_BUDGET, func(_i): edit.undo(), 6)
	await measure("redo", EDIT_BUDGET, func(_i): edit.redo(), 3)
	var quarter := edit.get_line_count() / 4
	await measure("color_quarter_doc", LARGE_BUDGET, func(_i):
		edit.select(quarter, 0, quarter * 2, 0)
		edit.set_selection_color(Color(0.3, 0.2, 0.7)), 2)
	await measure("italic_all", LARGE_BUDGET, func(_i):
		edit.select_all()
		edit.toggle_italic(), 1)
	await measure("undo_italic_all", LARGE_BUDGET, func(_i): edit.undo(), 1)
	await measure("center_line", EDIT_BUDGET, func(i):
		edit.deselect()
		edit.set_caret_line(mid + i, false)
		edit.set_alignment(HORIZONTAL_ALIGNMENT_CENTER), 3)
	await measure("list_10_lines", EDIT_BUDGET, func(_i):
		edit.select(mid, 0, mid + 10, 0)
		edit.toggle_ordered_list(), 2)
	await measure("hover_hit_test", EDIT_BUDGET, func(_i):
		for k in 20:
			edit.get_line_column_at_pos(Vector2i(200, 300 + k)), 3)
	await measure("caret_jump_end", EDIT_BUDGET, func(i):
		edit.deselect()
		edit.set_caret_line(edit.get_line_count() - 1 - i * 7)
		edit.adjust_viewport_to_caret(), 3)
	await measure("serialize_bbcode", 1000.0, func(_i):
		edit.toggle_bold()
		edit.get_bbcode_text(), 2)
	await measure("theme_font_size", LARGE_BUDGET, func(i):
		edit.add_theme_font_size_override("font_size", 17 + i % 2), 2)

	if failures.is_empty():
		print("RICH TEXT EDIT PERFORMANCE: PASS")
	else:
		print("RICH TEXT EDIT PERFORMANCE: OVER BUDGET: " + ", ".join(failures))
	quit(1 if not failures.is_empty() else 0)
