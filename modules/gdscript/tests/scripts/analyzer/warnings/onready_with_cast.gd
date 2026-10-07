extends Node

@onready var untyped_cast = $Node as Node2D
@onready var inferred_cast := $Node as Node2D
@onready var explicit_cast: Node2D = $Node as Node2D

@onready var untyped_direct = $Node
@onready var inferred_direct := $Node
@onready var explicit_direct: Node2D = $Node

func test():
	pass
