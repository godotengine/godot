extends Node2D

func get_sword_ds():
	return $Player1/Sword

func get_sword():
	return get_node('Player1/Sword')

func get_player1_ds():
	return $Player1

func get_player1():
	return get_node('Player1')
