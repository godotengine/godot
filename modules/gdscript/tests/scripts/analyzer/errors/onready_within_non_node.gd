extends RefCounted

@onready var nope := 0

func test():
	print("Cannot use @onready without a Node base")
