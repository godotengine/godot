class Say:
	var prefix = "S"

	func greet():
		prefix = "S Greeted"
		print("hello")

	func say(name):
		print(prefix, " say something ", name)


class SayAnotherThing extends Say:
	# This currently crashes the engine.
	#var prefix = "SAT"

	func greet():
		prefix = "SAT Greeted"
		print("hi")

	func say(name):
		print(prefix, " say another thing ", name)


class SayNothing extends Say:
	# This currently crashes the engine.
	#var prefix = "SN"

	func greet():
		super()
		prefix = "SN Greeted"
		print("howdy, see above")

	func greet_prefix_before_super():
		prefix = "SN Greeted"
		super.greet()
		print("howdy, see above")

	func say(name):
		@warning_ignore("unsafe_call_argument")
		super(name + " super'd")
		print(prefix, " say nothing... or not? ", name)


func test():
	var say = Say.new()
	say.greet()
	say.say("foo")
	print()

	var say_another_thing = SayAnotherThing.new()
	say_another_thing.greet()
	say_another_thing.say("bar")
	print()

	var say_nothing = SayNothing.new()
	say_nothing.greet()
	print(say_nothing.prefix)
	say_nothing.greet_prefix_before_super()
	print(say_nothing.prefix)
	# This currently triggers a compiler bug: "compiler bug, function name not found"
	#say_nothing.say("baz")
