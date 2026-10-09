extends SceneTree

func _initialize() -> void:
	var source := RichTextEdit.new()
	root.add_child(source)
	source.bbcode_enabled = true
	source.bbcode_text = "[outline_size=3][outline_color=#ff0000][i][b]Alpha[/b][/i][/outline_color][/outline_size]"
	source.select(0, 0, 0, 5)
	var inline_style: PackedByteArray = source.get_selection_style_template()
	var target := RichTextEdit.new()
	root.add_child(target)
	target.bbcode_enabled = true
	target.bbcode_text = "[s][outline_size=1]hello world[/outline_size][/s]"
	target.select(0, 0, 0, 5)
	if not target.apply_style_template(inline_style):
		push_error("Could not apply the inline style")
		quit(1)
		return
	var result := target.bbcode_text
	if not ("outline_size=3" in result and "outline_color" in result and "[i]" in result and "[b]" in result):
		push_error("Inline style was lost: " + result)
		quit(1)
		return
	if "[s]hello" in result or "outline_size=1]hello" in result:
		push_error("Old inline style remains: " + result)
		quit(1)
		return

	source.bbcode_text = "[quote]Alpha[/quote]"
	source.select(0, 0, 0, 5)
	var quote_style: PackedByteArray = source.get_selection_style_template()
	target.bbcode_text = "hello world"
	target.select(0, 0, 0, 5)
	target.apply_style_template(quote_style)
	if "[quote" in target.bbcode_text:
		push_error("Quote applied to a partial line")
		quit(1)
		return
	target.select(0, 0, 0, 11)
	target.apply_style_template(quote_style)
	if not "[quote" in target.bbcode_text:
		push_error("Quote was not applied to a complete line: " + target.bbcode_text)
		quit(1)
		return
	for block_tag in ["indent", "ul", "ol", "right", "center", "fill"]:
		source.bbcode_text = "[" + block_tag + "]Alpha[/" + block_tag + "]"
		source.select(0, 0, 0, 5)
		var block_style: PackedByteArray = source.get_selection_style_template()
		target.bbcode_text = "hello world"
		target.select(0, 0, 0, 5)
		target.apply_style_template(block_style)
		if "[" + block_tag in target.bbcode_text:
			push_error(block_tag + " applied to a partial line")
			quit(1)
			return
		target.select(0, 0, 0, 11)
		target.apply_style_template(block_style)
		if not "[" + block_tag in target.bbcode_text:
			push_error(block_tag + " was not applied to a complete line: " + target.bbcode_text)
			quit(1)
			return
	source.bbcode_text = "[line_height=26px]Alpha[/line_height]"
	source.select(0, 0, 0, 5)
	var line_style: PackedByteArray = source.get_selection_style_template()
	target.bbcode_text = "hello world"
	target.select(0, 0, 0, 5)
	target.apply_style_template(line_style)
	if "line_height" in target.bbcode_text:
		push_error("Line height applied to a partial line")
		quit(1)
		return
	target.select(0, 0, 0, 11)
	target.apply_style_template(line_style)
	if not "line_height=26px" in target.bbcode_text:
		push_error("Line height was not applied to a complete line: " + target.bbcode_text)
		quit(1)
		return
	print("RICH TEXT STYLE PROBE: PASS")
	quit()
