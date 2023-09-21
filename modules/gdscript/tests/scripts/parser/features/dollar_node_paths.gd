extends Node


func test():
	# Create the required node structure.
	var hello = Node.new()
	hello.name = "Hello"
	@warning_ignore("unsafe_call_argument")
	add_child(hello)
	var world = Node.new()
	world.name = "World"
	@warning_ignore("unsafe_call_argument")
	hello.add_child(world)

	# All the ways of writing node paths below with the `$` operator are valid.
	# Results are assigned to variables to avoid warnings.
	var __ = $Hello
	__ = $"Hello"
	__ = $Hello/World
	__ = $"Hello/World"
	__ = $"Hello/.."
	__ = $"Hello/../Hello/World"
