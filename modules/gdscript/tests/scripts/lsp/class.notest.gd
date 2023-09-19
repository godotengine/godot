extends Node

class Inner1 extends Node:
#     ^^^^^^ class1 -> class1
	var member1 := 42
	#   ^^^^^^^ class1:member1 -> class1:member1
	var member2 : int = 13
	#   ^^^^^^^ class1:member2 -> class1:member2
	var member3 = 1337
	#   ^^^^^^^ class1:member3 -> class1:member3

	signal changed(old, new)
	#      ^^^^^^^ class1:signal -> class1:signal
	func my_func(arg1: int, arg2: String, arg3):
	#    |     | |  |       |  |          ^^^^ class1:func:arg3 -> class1:func:arg3
	#    |     | |  |       ^^^^ class1:func:arg2 -> class1:func:arg2
	#    |     | ^^^^ class1:func:arg1 -> class1:func:arg1
	#    ^^^^^^^ class1:func -> class1:func
		print(arg1, arg2, arg3)
		#     |  |  |  |  ^^^^ -> class1:func:arg3
		#     |  |  ^^^^ -> class1:func:arg2
		#     ^^^^ -> class1:func:arg1
		changed.emit(arg1, arg3)
		#     |      |  |  ^^^^ -> class1:func:arg3
		#     |      ^^^^ -> class1:func:arg1
		#<^^^^^ -> class1:signal
		return arg1 + arg2.length() + arg3
		#      |  |   |  |            ^^^^ -> class1:func:arg3
		#      |  |   ^^^^ -> class1:func:arg2
		#      ^^^^ -> class1:func:arg1

class Inner2:
#     ^^^^^^ class2 -> class2
	var member1 := 42
	#   ^^^^^^^ class2:member1 -> class2:member1
	var member2 : int = 13
	#   ^^^^^^^ class2:member2 -> class2:member2
	var member3 = 1337
	#   ^^^^^^^ class2:member3 -> class2:member3

	signal changed(old, new)
	#      ^^^^^^^ class2:signal -> class2:signal
	func my_func(arg1: int, arg2: String, arg3):
	#    |     | |  |       |  |          ^^^^ class2:func:arg3 -> class2:func:arg3
	#    |     | |  |       ^^^^ class2:func:arg2 -> class2:func:arg2
	#    |     | ^^^^ class2:func:arg1 -> class2:func:arg1
	#    ^^^^^^^ class2:func -> class2:func
		print(arg1, arg2, arg3)
		#     |  |  |  |  ^^^^ -> class2:func:arg3
		#     |  |  ^^^^ -> class2:func:arg2
		#     ^^^^ -> class2:func:arg1
		changed.emit(arg1, arg3)
		#     |      |  |  ^^^^ -> class2:func:arg3
		#     |      ^^^^ -> class2:func:arg1
		#<^^^^^ -> class2:signal
		return arg1 + arg2.length() + arg3
		#      |  |   |  |            ^^^^ -> class2:func:arg3
		#      |  |   ^^^^ -> class2:func:arg2
		#      ^^^^ -> class2:func:arg1

class Inner3 extends Inner2:
#     |    |         ^^^^^^ -> class2
#     ^^^^^^ class3 -> class3
	var whatever = "foo"
	#   ^^^^^^^^ class3:whatever -> class3:whatever

	func _init():
	#    ^^^^^ class3:init
	# Note: no self-ref check here: resolves to `Object._init`.
	#       usages of `Inner3.new()` DO resolve to this `_init`
		pass

	class NestedInInner3:
	#     ^^^^^^^^^^^^^^ class3:nested1 -> class3:nested1
		var some_value := 42
		#   ^^^^^^^^^^ class3:nested1:some_value -> class3:nested1:some_value

	class AnotherNestedInInner3 extends NestedInInner3:
	#!    |                   |         ^^^^^^^^^^^^^^ -> class3:nested1
	#     ^^^^^^^^^^^^^^^^^^^^^ class3:nested2 -> class3:nested2
		var another_value := 13
		#   ^^^^^^^^^^^^^ class3:nested2:another_value -> class3:nested2:another_value

func _ready():
	var inner1 = Inner1.new()
	#   |    |   ^^^^^^ -> class1
	#   ^^^^^^ func:class1 -> func:class1
	var value1 = inner1.my_func(1,"",3)
	#   |    |   |    | ^^^^^^^ -> class1:func
	#   |    |   ^^^^^^ -> func:class1
	#   ^^^^^^ func:class1:value1 -> func:class1:value1
	var value2 = inner1.member3
	#   |    |   |    | ^^^^^^^ -> class1:member3
	#   |    |   ^^^^^^ -> func:class1
	#   ^^^^^^ func:class1:value2 -> func:class1:value2
	print(value1, value2)
	#     |    |  ^^^^^^ -> func:class1:value2
	#     ^^^^^^ -> func:class1:value1

	var inner3 = Inner3.new()
	#   |    |   |    | ^^^ -> class3:init
	#   |    |   ^^^^^^ -> class3
	#   ^^^^^^ func:class3 -> func:class3
	print(inner3)
	#     ^^^^^^ -> func:class3

	var nested1 = Inner3.NestedInInner3.new()
	#   |     |   |    | ^^^^^^^^^^^^^^ -> class3:nested1
	#   |     |   ^^^^^^ -> class3
	#   ^^^^^^^ func:class3:nested1 -> func:class3:nested1
	var value_nested1 = nested1.some_value
	#   |           |   |     | ^^^^^^^^^^ -> class3:nested1:some_value
	#   |           |   ^^^^^^^ -> func:class3:nested1
	#   ^^^^^^^^^^^^^ func:class3:nested1:value
	print(value_nested1)
	#     ^^^^^^^^^^^^^ -> func:class3:nested1:value

	var nested2 = Inner3.AnotherNestedInInner3.new()
	#   |     |   |    | ^^^^^^^^^^^^^^^^^^^^^ -> class3:nested2
	#   |     |   ^^^^^^ -> class3
	#   ^^^^^^^ func:class3:nested2 -> func:class3:nested2
	var value_nested2 = nested2.some_value
	#   |           |   |     | ^^^^^^^^^^ -> class3:nested1:some_value
	#   |           |   ^^^^^^^ -> func:class3:nested2
	#   ^^^^^^^^^^^^^ func:class3:nested2:value
	var another_value_nested2 = nested2.another_value
	#   |                   |   |     | ^^^^^^^^^^^^^ -> class3:nested2:another_value
	#   |                   |   ^^^^^^^ -> func:class3:nested2
	#   ^^^^^^^^^^^^^^^^^^^^^ func:class3:nested2:another_value_nested
	print(value_nested2, another_value_nested2)
	#     |           |  ^^^^^^^^^^^^^^^^^^^^^ -> func:class3:nested2:another_value_nested
	#     ^^^^^^^^^^^^^ -> func:class3:nested2:value
