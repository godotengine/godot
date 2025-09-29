# Message Container Helper
# Simple helper script for chat message containers
extends HBoxContainer

var message_type: String = ""

func set_message_type(type: String):
	message_type = type

func get_message_type() -> String:
	return message_type