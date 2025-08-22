extends Node2D
## This is just a simple script for a simple scene.

@export var this_is_different_in_scene := "default"
@export var exported_string := 'This is an exported string'
var public_string := 'this is a public string'


func foo():
	return "bar"

func bar():
	return "foo"
