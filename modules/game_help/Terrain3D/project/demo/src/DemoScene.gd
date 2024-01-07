extends Node


func _ready():
	if has_node("UI"):
		$UI.player = $Player
