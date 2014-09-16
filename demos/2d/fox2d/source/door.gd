extends Sprite

# class for doors, which allows the player to teleport from a level/scene to another and go back.
# 

# Parameters --------------------------------------------------------------
# relative path of the scene to load, the root path being "res://maps/"
export var destination="" 

# name of the entry point in the destination scene. Recommanded to use a Position2D, 
# but as long as it has a position and it's in the group "entries", it's fine.
# it can be "", in which case the player is not moved and starts as defined in the editor.
export var dest_point=""  

# Constants ---------------------------------------------------------------
const PlayerClass = preload("res://player.gd")

# Functions ---------------------------------------------------------------

# Initializer
func _ready():
	pass

# when the player enters the door area, tell the player to use this door if he presses the key to enter the door
func _on_Area2D_body_enter( body ):
	if(body extends PlayerClass):
		body.setDoor(self)

# when the player leaves the door, tell the player to not use the door anymore if he presses the key to enter the door
func _on_Area2D_body_exit( body ):
	if(body extends PlayerClass):
		body.setDoor(null)
