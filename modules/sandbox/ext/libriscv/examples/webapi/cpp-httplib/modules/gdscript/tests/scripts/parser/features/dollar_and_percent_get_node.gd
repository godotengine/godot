extends Node

func test():
	var child = Node.new()
	child.name = "Child"
	@warning_ignore("unsafe_call_argument")
	add_child(child)
	child.owner = self

	var hey = Node.new()
	hey.name = "Hey"
	@warning_ignore("unsafe_call_argument")
	child.add_child(hey)
	hey.owner = self
	hey.unique_name_in_owner = true

	var fake_hey = Node.new()
	fake_hey.name = "Hey"
	@warning_ignore("unsafe_call_argument")
	add_child(fake_hey)
	fake_hey.owner = self

	var sub_child = Node.new()
	sub_child.name = "SubChild"
	@warning_ignore("unsafe_call_argument")
	hey.add_child(sub_child)
	sub_child.owner = self

	var howdy = Node.new()
	howdy.name = "Howdy"
	@warning_ignore("unsafe_call_argument")
	sub_child.add_child(howdy)
	howdy.owner = self
	howdy.unique_name_in_owner = true

	print(hey == $Child/Hey)
	print(howdy == $Child/Hey/SubChild/Howdy)

	print(%Hey == hey)
	print($%Hey == hey)
	print(%"Hey" == hey)
	print($"%Hey" == hey)
	print($%"Hey" == hey)
	print(%Hey/%Howdy == howdy)
	print($%Hey/%Howdy == howdy)
	print($"%Hey/%Howdy" == howdy)
	print($"%Hey"/"%Howdy" == howdy)
	print(%"Hey"/"%Howdy" == howdy)
	print($%"Hey"/"%Howdy" == howdy)
	print($"%Hey"/%"Howdy" == howdy)
	print(%"Hey"/%"Howdy" == howdy)
	print($%"Hey"/%"Howdy" == howdy)
	print(%"Hey/%Howdy" == howdy)
	print($%"Hey/%Howdy" == howdy)
