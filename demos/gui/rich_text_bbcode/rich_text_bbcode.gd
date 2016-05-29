
extends Panel


func _on_RichTextLabel_meta_clicked(meta):
	OS.shell_open(meta)
