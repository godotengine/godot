extends Node

# Starting point of the game
# It loads the 1st scene and initialize all needed parameters, including the background music to play.

func _ready():
	get_node("/root/global").goto_cinematic_scene("res://maps/map1/intro.scn")
	get_node("/root/global").fade_in()
	get_node("HUDLayer/HUDroot").get_node("lifeBar").set_life(6)
	get_node("/root/global").play_map_track()
	