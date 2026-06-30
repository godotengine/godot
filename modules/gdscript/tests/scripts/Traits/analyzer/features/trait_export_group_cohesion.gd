extends Node
uses TraitA, TraitB

trait TraitA:
	@export_group("Group A")
	@export var value_a := 1

trait TraitB:
	@export_group("Group B")
	@export var value_b := 2

func test():
	print("ok")
