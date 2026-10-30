extends SceneTree

# RichTextEdit document model regression. Run headless from the repository:
#   godot --headless --path tests/scene/resources/rich_text_edit_model --script check.gd
# Exits with 1 when a check fails.
const Doc = preload("res://doc.gd")
const SMALL := "[b]Bold start[/b] plain [color=d9480f]red[/color]\n[ol]item one\nitem two[/ol]\n\n[quote]A quote[/quote]\n[line_height=1.6]Tall[/line_height] [url=https://a.b]link[/url] [i]it[/i]\n[center]centered[/center]\nlast line"
const WORDS := ["alpha", "你好", " ", "x", "a\nb", "\n", "two\nnew\nlines"]

var edit: RichTextEdit
var rng := RandomNumberGenerator.new()
var failures := 0


func fail(p_message: String) -> void:
	failures += 1
	push_error(p_message)


func settle() -> int:
	# Frames until background shaping has laid out every line.
	await process_frame
	for i in 5000:
		if not edit.has_pending_layout():
			break
		await process_frame
	await process_frame
	return edit.get_content_height()


func load_canonical(p_bbcode: String) -> void:
	edit.set_bbcode_text(p_bbcode)
	# An edit that changes nothing visible makes the BBCode canonical.
	edit.deselect()
	edit.set_caret_line(edit.get_line_count() - 1, false)
	edit.set_caret_column(edit.get_line(edit.get_line_count() - 1).length(), false)
	edit.insert_text_at_caret("#")
	edit.backspace()
	edit.clear_undo_history()


func rand_pos() -> Vector2i:
	var line := rng.randi_range(0, edit.get_line_count() - 1)
	return Vector2i(rng.randi_range(0, edit.get_line(line).length()), line)


func rand_select(p_max_lines := 0) -> void:
	var a := rand_pos()
	var b := rand_pos()
	if p_max_lines > 0:
		b.y = min(a.y + rng.randi_range(0, p_max_lines), edit.get_line_count() - 1)
		b.x = rng.randi_range(0, edit.get_line(b.y).length())
	if b.y < a.y or (b.y == a.y and b.x < a.x):
		var t := a
		a = b
		b = t
	if a == b:
		b = Vector2i(edit.get_line(a.y).length(), a.y)
	edit.select(a.y, a.x, b.y, b.x)


func place_caret() -> void:
	edit.deselect()
	var p := rand_pos()
	edit.set_caret_line(p.y, false)
	edit.set_caret_column(p.x, false)


func key(p_code: Key, p_shift := false) -> void:
	var event := InputEventKey.new()
	event.keycode = p_code
	event.pressed = true
	event.shift_pressed = p_shift
	edit.get_viewport().push_input(event)
	var release := event.duplicate()
	release.pressed = false
	edit.get_viewport().push_input(release)


func random_operation(p_large: bool) -> String:
	var span := 120 if p_large else 0
	match rng.randi_range(0, 19):
		0:
			place_caret()
			edit.insert_text_at_caret(WORDS[rng.randi_range(0, WORDS.size() - 1)])
			return "type"
		1:
			place_caret()
			key(KEY_ENTER, rng.randf() < 0.4)
			return "enter"
		2:
			place_caret()
			edit.backspace()
			return "backspace"
		3:
			rand_select(span)
			edit.delete_selection()
			return "delete"
		4:
			if p_large:
				edit.select_all()
			else:
				rand_select()
			edit.toggle_bold()
			return "bold"
		5:
			rand_select(span)
			edit.set_selection_color(Color(0.2, 0.4, 0.7))
			return "color"
		6:
			rand_select(span)
			edit.set_selection_font_size(rng.randi_range(12, 30))
			return "size"
		7:
			rand_select(span)
			edit.set_selection_url("https://example.com/x")
			return "url"
		8:
			rand_select(span)
			edit.set_alignment([HORIZONTAL_ALIGNMENT_CENTER, HORIZONTAL_ALIGNMENT_RIGHT, HORIZONTAL_ALIGNMENT_LEFT][rng.randi_range(0, 2)])
			return "align"
		9:
			rand_select(span)
			edit.toggle_ordered_list()
			return "ol"
		10:
			rand_select(span)
			edit.toggle_quote()
			return "quote"
		11:
			rand_select(span)
			edit.set_selection_line_height(["1.5", "24px", "120%"][rng.randi_range(0, 2)])
			return "line_height"
		12:
			rand_select(span)
			edit.copy_rich_text()
			place_caret()
			edit.paste_rich_text()
			return "copy_paste"
		13:
			rand_select(span)
			edit.increase_indent()
			return "indent"
		14:
			rand_select(span)
			edit.decrease_indent()
			return "outdent"
		15:
			place_caret()
			edit.insert_image("user://model_check.png", 20, 10)
			return "image"
		16:
			place_caret()
			edit.insert_horizontal_rule()
			return "hr"
		17:
			rand_select(span)
			edit.insert_text_at_caret(WORDS[rng.randi_range(0, WORDS.size() - 1)])
			return "replace_selection"
		18:
			rand_select(span)
			edit.clear_selection_line_height()
			return "clear_line_height"
		19:
			rand_select(span)
			edit.toggle_unordered_list()
			return "ul"
	return "noop"


# Undoing everything restores the document exactly; redoing restores the edit.
func check_history(p_bbcode: String, p_large: bool, p_trials: int, p_seed: int) -> void:
	rng.seed = p_seed
	for trial in p_trials:
		load_canonical(p_bbcode)
		var initial_height := await settle()
		var initial := edit.get_bbcode_text()
		var name := random_operation(p_large)
		var after_height := await settle()
		var after := edit.get_bbcode_text()
		var guard := 0
		while edit.has_undo() and guard < 20:
			edit.undo()
			guard += 1
		var undone_height := await settle()
		if edit.get_bbcode_text() != initial or undone_height != initial_height:
			fail("undo of %s (seed %d trial %d) did not restore the document" % [name, p_seed, trial])
		guard = 0
		while edit.has_redo() and guard < 20:
			edit.redo()
			guard += 1
		var redone_height := await settle()
		if edit.get_bbcode_text() != after or redone_height != after_height:
			fail("redo of %s (seed %d trial %d) did not restore the edit" % [name, p_seed, trial])


func expect(p_condition: bool, p_message: String) -> void:
	if not p_condition:
		fail(p_message + ": " + edit.get_bbcode_text().c_escape())


func check_semantics() -> void:
	# Toggling bold at a caret applies on top of the paragraph's formatting.
	edit.set_bbcode_text("[ol]item[/ol]\nnext")
	edit.set_caret_line(0, false)
	edit.set_caret_column(2, false)
	edit.toggle_bold()
	edit.insert_text_at_caret("Z")
	expect(edit.get_bbcode_text().begins_with("[ol]it[b]Z[/b]em[/ol]"), "typing style dropped the list")

	# Shift+Enter at the end of a list item starts a plain paragraph.
	edit.set_bbcode_text("[ol]a\nb[/ol]\nc")
	await process_frame
	edit.set_caret_line(1, false)
	edit.set_caret_column(1, false)
	key(KEY_ENTER, true)
	await process_frame
	expect(edit.get_bbcode_text() == "[ol]a\nb[/ol]\n\nc", "Shift+Enter kept the list")

	# A paragraph break typed at the start of a paragraph moves its line height down.
	edit.set_bbcode_text("[line_height=2]abc[/line_height]")
	await process_frame
	edit.set_caret_line(0, false)
	edit.set_caret_column(0, false)
	key(KEY_ENTER)
	await process_frame
	expect(edit.get_bbcode_text() == "\n[line_height=2]abc[/line_height]", "line height did not travel with its paragraph")

	# Rich copy and paste keeps the formatting.
	edit.set_bbcode_text("[b]bold[/b] plain")
	edit.select(0, 0, 0, 4)
	edit.copy_rich_text()
	edit.deselect()
	edit.set_caret_line(0, false)
	edit.set_caret_column(10, false)
	expect(edit.paste_rich_text(), "rich paste refused")
	expect(edit.get_bbcode_text() == "[b]bold[/b] plain[b]bold[/b]", "rich paste lost the formatting")

	# Inline images survive undo and redo.
	edit.set_bbcode_text("ab")
	edit.set_caret_line(0, false)
	edit.set_caret_column(1, false)
	edit.insert_image("user://model_check.png", 20, 10)
	var with_image := edit.get_bbcode_text()
	expect(with_image.contains("[img") and with_image.contains("model_check.png"), "image was not inserted")
	edit.undo()
	expect(edit.get_bbcode_text() == "ab", "undo kept the image")
	edit.redo()
	expect(edit.get_bbcode_text() == with_image, "redo lost the image")


# Deferred shaping converges on the layout a fully shaped document has.
func check_deferred_layout(p_bbcode: String) -> void:
	edit.set_bbcode_text(p_bbcode)
	var settled := await settle()
	var reference: RichTextEdit = RichTextEdit.new()
	reference.wrap_mode = edit.wrap_mode
	reference.size = edit.size
	root.add_child(reference)
	await process_frame
	reference.set_bbcode_text(p_bbcode)
	for line in reference.get_line_count():
		reference.get_line_wrap_count(line)
	if reference.get_content_height() != settled:
		fail("deferred layout settled at %d instead of %d" % [settled, reference.get_content_height()])
	reference.queue_free()


func _initialize() -> void:
	var image := Image.create(20, 10, false, Image.FORMAT_RGBA8)
	image.fill(Color(0.2, 0.5, 0.8))
	image.save_png("user://model_check.png")
	edit = RichTextEdit.new()
	edit.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	edit.size = Vector2(600, 400)
	root.add_child(edit)
	await process_frame
	edit.grab_focus()

	await check_semantics()
	for seed in [1, 2, 3]:
		await check_history(SMALL, false, 80, seed)
	var large: String = Doc.new().make(20000)
	await check_history(large, true, 12, 7)
	await check_deferred_layout(large)

	if failures == 0:
		print("RICH TEXT EDIT MODEL: PASS")
	else:
		print("RICH TEXT EDIT MODEL: %d FAILED" % failures)
	quit(1 if failures > 0 else 0)
