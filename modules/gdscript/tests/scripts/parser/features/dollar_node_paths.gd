extends Node


func test():
	# Create the required node structure.
	var hello = Node.new()
	hello.name = "Hello"
	add_child(hello)
	var world = Node.new()
	world.name = "World"
	hello.add_child(world)

	# All the ways of writing node paths below with the `$` operator are valid.
	# Results are assigned to variables to avoid warnings.
	var __ = $Hello
	__ = $"Hello"
	__ = $Hello/World
	__ = $"Hello/World"
	__ = $"Hello/.."
	__ = $"Hello/../Hello/World"
