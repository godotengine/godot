extends Window

@onready var rtl = $TextDisplay/RichTextLabel

func _get_file_as_text(path):
	var to_return = null
	var f = FileAccess.open(path, FileAccess.READ)
	if(f != null):
		to_return = f.get_as_text()
	else:
		to_return = str('ERROR:  Could not open file.  Error code ', FileAccess.get_open_error())
	return to_return

func _ready():
	rtl.clear()

func _on_OpenFile_pressed():
	$FileDialog.popup_centered()

func _on_FileDialog_file_selected(path):
	show_file(path)

func _on_Close_pressed():
	self.hide()

func show_file(path):
	var text = _get_file_as_text(path)
	if(text == ''):
		text = '<Empty File>'
	rtl.set_text(text)
	self.window_title = path

func show_open():
	self.popup_centered()
	$FileDialog.popup_centered()

func get_rich_text_label():
	return $TextDisplay/RichTextLabel

func _on_Home_pressed():
	rtl.scroll_to_line(0)

func _on_End_pressed():
	rtl.scroll_to_line(rtl.get_line_count() -1)

func _on_Copy_pressed():
	return
	# OS.clipboard = rtl.text

func _on_file_dialog_visibility_changed():
	if rtl.text.length() == 0 and not $FileDialog.visible:
		self.hide()
