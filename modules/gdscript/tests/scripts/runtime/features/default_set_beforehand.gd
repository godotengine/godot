extends Node

@onready var later_inferred := [1]
@onready var later_static: Array
@onready var later_static_with_init: Array = [1]
@onready var later_untyped = [1]

func test():
	Utils.check(typeof(later_inferred) == TYPE_ARRAY)
	Utils.check(later_inferred.size() == 0)

	Utils.check(typeof(later_static) == TYPE_ARRAY)
	Utils.check(later_static.size() == 0)

	Utils.check(typeof(later_static_with_init) == TYPE_ARRAY)
	Utils.check(later_static_with_init.size() == 0)

	Utils.check(typeof(later_untyped) == TYPE_NIL)

	print("ok")
