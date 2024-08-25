@tool
extends Window
signal confirmed
func _ready():
	find_child("delete_layer").pressed.connect(func(): confirmed.emit(false))
	find_child("delete_layer_and_image").pressed.connect(func(): confirmed.emit(true))
	find_child("delete_layer").pressed.connect(queue_free)
	find_child("delete_layer_and_image").pressed.connect(queue_free)
	find_child("cancel_button").pressed.connect(queue_free)
	close_requested.connect(queue_free)

func set_shared_uniform_label(list:Array):
	if list == []:		
		find_child("shared_uniform_label").visible = false
	else:		
		var names = ""
		for i in list:
			names += str(i, ", ")
		find_child("shared_uniform_label").text = "this image is shared with: " + names
