extends Node2D

var player: Player

func _ready() -> void:
	player = Player.new()
	add_child(player)
