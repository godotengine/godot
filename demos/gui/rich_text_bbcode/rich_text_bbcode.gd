
extends Panel

# member variables here, example:
# var a=2
# var b="textvar"

func _ready():
	# Initialization here
	pass




func _on_RichTextLabel_meta_clicked( meta ):
	OS.shell_open(meta)
	pass # replace with function body
