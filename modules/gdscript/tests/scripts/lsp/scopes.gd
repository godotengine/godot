extends Node

var member := 2
#   ^^^^^^ public -> public

signal some_changed(new_value)
#      |          | ^^^^^^^^^ signal:parameter -> signal:parameter
#      ^^^^^^^^^^^^ signal -> signal
var some_value := 42:
#   ^^^^^^^^^^ property -> property
	get:
		return some_value
		#      ^^^^^^^^^^ -> property
	set(value):
	#   ^^^^^ property:set:value -> property:set:value
		some_changed.emit(value)
		#          |      ^^^^^ -> property:set:value
		#<^^^^^^^^^^ -> signal
		some_value = value
		#        |   ^^^^^ -> property:set:value
		#<^^^^^^^^ -> property

func v():
	var value := member + 2
	#   |   |    ^^^^^^ -> public
	#   ^^^^^ v:value -> v:value
	print(value)
	#     ^^^^^ -> v:value
	if value > 0:
	#  ^^^^^ -> v:value
		var beta := value + 2
		#   |  |    ^^^^^ -> v:value
		#   ^^^^ v:if:beta -> v:if:beta
		print(beta)
		#     ^^^^ -> v:if:beta

		for counter in beta:
		#   |     |    ^^^^ -> v:if:beta
		#   ^^^^^^^ v:if:counter -> v:if:counter
			print (counter)
			#      ^^^^^^^ -> v:if:counter

	else:
		for counter in value:
		#   |     |    ^^^^^ -> v:value
		#   ^^^^^^^ v:else:counter -> v:else:counter
			print(counter)
			#     ^^^^^^^ -> v:else:counter

func f():
	var func1 = func(value): print(value + 13)
	#   |   |        |   |         ^^^^^ -> f:func1:value
	#   |   |        ^^^^^ f:func1:value -> f:func1:value
	#   ^^^^^ f:func1 -> f:func1
	var func2 = func(value): print(value + 42)
	#   |   |        |   |         ^^^^^ -> f:func2:value
	#   |   |        ^^^^^ f:func2:value -> f:func2:value
	#   ^^^^^ f:func2 -> f:func2

	func1.call(1)
	#<^^^ -> f:func1
	func2.call(2)
	#<^^^ -> f:func2

func m():
	var value = 42
	#   ^^^^^ m:value -> m:value

	match value:
	#     ^^^^^ -> m:value
		13: 
			print(value)
			#     ^^^^^ -> m:value
		[var start, _, var end]: 
		#    |   |         ^^^ m:match:array:end -> m:match:array:end
		#    ^^^^^  m:match:array:start -> m:match:array:start
			print(start + end)
			#     |   |   ^^^ -> m:match:array:end
			#     ^^^^^ -> m:match:array:start
		{ "name": var name }: 
		#             ^^^^ m:match:dict:var -> m:match:dict:var
			print(name)
			#     ^^^^ -> m:match:dict:var
		var whatever:
		#   ^^^^^^^^ m:match:var -> m:match:var
			print(whatever)
			#     ^^^^^^^^ -> m:match:var

func m2():
	var value = 42 
	#   ^^^^^ m2:value -> m2:value

	match value: 
	#     ^^^^^ -> m2:value
		{ "name": var name }:
		#             ^^^^ m2:match:dict:var -> m2:match:dict:var
			print(name)
			#     ^^^^ -> m2:match:dict:var
		[var name, ..]:
		#    ^^^^ m2:match:array:var -> m2:match:array:var
			print(name)
			#     ^^^^ -> m2:match:array:var
		var name:
		#   ^^^^ m2:match:var -> m2:match:var
			print(name)
			#     ^^^^ -> m2:match:var
