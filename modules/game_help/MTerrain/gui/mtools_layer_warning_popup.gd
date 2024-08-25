@tool
extends Window
signal confirmed
func _ready():
	find_child("confirm_button").pressed.connect(func(): confirmed.emit())
	find_child("confirm_button").pressed.connect(queue_free)
	find_child("cancel_button").pressed.connect(queue_free)
	close_requested.connect(queue_free)
