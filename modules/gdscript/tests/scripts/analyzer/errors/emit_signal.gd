extends Node
signal  my_signal(arg1: int)

func _ready():

	var lc = LocalClass.new()
	# args passed can not be > than signal's params
	my_signal.emit(1, "placeholder")
	emit_signal("my_signal",1,"placeholder")
	child_exiting_tree.emit(Node.new(), "placeholder") # native signal
	emit_signal("child_exiting_tree",Node.new(), "placeholder") # native signal
	lc.another_signal.emit(1, "placeholder")
	lc.emit_signal("another_signal", 1, "placeholder")
	lc.child_exiting_tree.emit(Node.new(), "placeholder") # native signal
	lc.emit_signal("child_exiting_tree", Node.new(), "placeholder") # native signal

	# arg type must match signal's param type
	my_signal.emit(0.1)
	emit_signal("my_signal", 0.1)
	child_exiting_tree.emit(Object.new())  # native signal
	emit_signal("child_exiting_tree", Object .new())
	lc.another_signal.emit(0.1)
	lc.emit_signal("another_signal", 0.1)
	lc.child_exiting_tree.emit(Object.new())  # native signal
	lc.emit_signal("child_exiting_tree", Object .new())# native signal

	lc.free()

class LocalClass extends Node:
	signal another_signal(arg1: int)
	func _ready():
		another_signal.emit(1)
