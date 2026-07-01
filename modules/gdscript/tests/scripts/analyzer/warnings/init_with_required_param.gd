
# -- Extending non-Node/Resource

class Ref_RequiredOnly extends RefCounted:
	func _init(_a: int):
		pass

class Ref_AnyRequired extends RefCounted:
	func _init(_a: int, _b: int = 3):
		pass

@abstract
class Ref_RequiredAbstract extends RefCounted:
	func _init(_a: int):
		pass

class Child_Ref_RequiredAbstract extends Ref_RequiredAbstract:
	pass

class Ref_RequiredOnly_RequiredOnly extends Ref_RequiredOnly:
	func _init(_a: int):
		pass

class Ref_RequiredOnly_AnyRequired extends Ref_RequiredOnly:
	func _init(_a: int, _b: int = 3):
		pass

# -- Extending Node/Resource

class Node_NoInit extends Node:
	pass
class Resource_NoInit extends Resource:
	pass

class Node_NoParams extends Node:
	func _init():
		pass
class Resource_NoParams extends Resource:
	func _init():
		pass

class Node_DefaultOnly extends Node:
	func _init(_a: int = 3):
		pass
class Resource_DefaultOnly extends Resource:
	func _init(_a: int = 3):
		pass

class Node_RequiredOnly extends Node:
	func _init(_a: int): # (warn)
		pass
class Resource_RequiredOnly extends Resource:
	func _init(_a: int): # (warn)
		pass

class Node_AnyRequired extends Node:
	func _init(_a: int, _b: int = 3): # (warn)
		pass
class Resource_AnyRequired extends Resource:
	func _init(_a: int, _b: int = 3): # (warn)
		pass

@abstract
class Node_RequiredAbstract extends Node:
	func _init(_a: int):
		pass
@abstract
class Resource_RequiredAbstract extends Resource:
	func _init(_a: int):
		pass


# -- Extending in-built sub-class of Node/Resource

class SubNode_NoInit extends Node3D:
	pass
class SubResource_NoInit extends Theme:
	pass

class SubNode_NoParams extends Node3D:
	func _init():
		pass
class SubResource_NoParams extends Theme:
	func _init():
		pass

class SubNode_AnyRequired extends Node3D:
	func _init(_a: int, _b: int = 3): # (warn)
		pass
class SubResource_AnyRequired extends Theme:
	func _init(_a: int, _b: int = 3): # (warn)
		pass

@abstract
class SubNode_RequiredAbstract extends Node3D:
	func _init(_a: int):
		pass
@abstract
class SubResource_RequiredAbstract extends Theme:
	func _init(_a: int):
		pass


# -- Extending non-req scripts of Node/Resource

class Node_NoParams_NoInit extends Node_NoParams:
	pass
class Resource_NoParams_NoInit extends Resource_NoParams:
	pass

class Node_NoParams_NoParams extends Node_NoParams:
	func _init():
		pass
class Resource_NoParams_NoParams extends Resource_NoParams:
	func _init():
		pass

class Node_NoParams_DefaultOnly extends Node_NoParams:
	func _init(_a: int = 3):
		pass
class Resource_NoParams_DefaultOnly extends Resource_NoParams:
	func _init(_a: int = 3):
		pass

class Node_NoParams_AnyRequired extends Node_NoParams:
	func _init(_a: int, _b: int = 3): # (warn)
		pass
class Resource_NoParams_AnyRequired extends Resource_NoParams:
	func _init(_a: int, _b: int = 3): # (warn)
		pass

@abstract
class Node_NoParams_RequiredAbstract extends Node_NoParams:
	func _init(_a: int, _b: int = 3):
		pass
@abstract
class Resource_NoParams_RequiredAbstract extends Resource_NoParams:
	func _init(_a: int, _b: int = 3):
		pass


# -- Extending req scripts of Node/Resource

class Node_RequiredOnly_NoInit extends Node_RequiredOnly: # (warn)
	pass
class Resource_RequiredOnly_NoInit extends Resource_RequiredOnly: # (warn)
	pass

@abstract
class Node_RequiredOnly_AbstractNoInit extends Node_RequiredOnly:
	pass
@abstract
class Resource_RequiredOnly_AbstractNoInit extends Resource_RequiredOnly:
	pass

class Node_RequiredOnly_NoParams extends Node_RequiredOnly:
	func _init():
		pass
class Resource_RequiredOnly_NoParams extends Resource_RequiredOnly:
	func _init():
		pass

class Node_RequiredOnly_AnyRequired extends Node_RequiredOnly:
	func _init(_a: int, _b: int = 3): # (warn)
		pass
class Resource_RequiredOnly_AnyRequired extends Resource_RequiredOnly:
	func _init(_a: int, _b: int = 3): # (warn)
		pass

@abstract
class Node_RequiredOnly_RequiredAbstract extends Node_RequiredOnly:
	func _init(_a: int, _b: int = 3):
		pass
@abstract
class Resource_RequiredOnly_RequiredAbstract extends Resource_RequiredOnly:
	func _init(_a: int, _b: int = 3):
		pass


# -- Extending abstract scripts of Node/Resource

class Node_RequiredAbstract_NoInit extends Node_RequiredAbstract: # (warn)
	pass
class Resource_RequiredAbstract_NoInit extends Resource_RequiredAbstract: # (warn)
	pass

@abstract
class Node_RequiredAbstract_AbstractNoInit extends Node_RequiredAbstract:
	pass
@abstract
class Resource_RequiredAbstract_AbstractNoInit extends Resource_RequiredAbstract:
	pass

class Node_RequiredAbstract_NoParams extends Node_RequiredAbstract:
	func _init():
		pass
class Resource_RequiredAbstract_NoParams extends Resource_RequiredAbstract:
	func _init():
		pass

class Node_RequiredAbstract_AnyRequired extends Node_RequiredAbstract:
	func _init(_a: int, _b: int = 3): # (warn)
		pass
class Resource_RequiredAbstract_AnyRequired extends Resource_RequiredAbstract:
	func _init(_a: int, _b: int = 3): # (warn)
		pass

@abstract
class Node_RequiredAbstract_RequiredAbstract extends Node_RequiredAbstract:
	func _init(_a: int, _b: int = 3):
		pass
@abstract
class Resource_RequiredAbstract_RequiredAbstract extends Resource_RequiredAbstract:
	func _init(_a: int, _b: int = 3):
		pass


# -- Extending scripts of in-built sub-class of Node/Resource

class SubNode_NoParams_NoInit extends SubNode_NoParams:
	pass
class SubResource_NoParams_NoInit extends SubResource_NoParams:
	pass

class SubNode_NoParams_NoParams extends SubNode_NoParams:
	func _init():
		pass
class SubResource_NoParams_NoParams extends SubResource_NoParams:
	func _init():
		pass

class SubNode_NoParams_AnyRequired extends SubNode_NoParams:
	func _init(_a: int, _b: int = 3): # (warn)
		pass
class SubResource_NoParams_AnyRequired extends SubResource_NoParams:
	func _init(_a: int, _b: int = 3): # (warn)
		pass

@abstract
class SubNode_NoParams_RequiredAbstract extends SubNode_NoParams:
	func _init(_a: int, _b: int = 3):
		pass
@abstract
class SubResource_NoParams_RequiredAbstract extends SubResource_NoParams:
	func _init(_a: int, _b: int = 3):
		pass


# -- Extending req scripts of in-built sub-class of Node/Resource

class SubNode_AnyRequired_NoInit extends SubNode_AnyRequired: # (warn)
	pass
class SubResource_AnyRequired_NoInit extends SubResource_AnyRequired: # (warn)
	pass

@abstract
class SubNode_AnyRequired_AbstractNoInit extends SubNode_AnyRequired:
	pass
@abstract
class SubResource_AnyRequired_AbstractNoInit extends SubResource_AnyRequired:
	pass

class SubNode_AnyRequired_NoParams extends SubNode_AnyRequired:
	func _init():
		pass
class SubResource_AnyRequired_NoParams extends SubResource_AnyRequired:
	func _init():
		pass


# -- Extending double abstract scripts of Node/Resource

class Node_RequiredOnly_AbstractNoInit_NoInit extends Node_RequiredOnly_AbstractNoInit: # (warn)
	pass
class Resource_RequiredOnly_AbstractNoInit_NoInit extends Resource_RequiredOnly_AbstractNoInit: # (warn)
	pass

class Node_RequiredOnly_AbstractNoInit_NoParams extends Node_RequiredOnly_AbstractNoInit:
	func _init():
		pass
class Resource_RequiredOnly_AbstractNoInit_NoParams extends Resource_RequiredOnly_AbstractNoInit:
	func _init():
		pass

class Node_RequiredAbstract_AbstractNoInit_NoInit extends Node_RequiredAbstract_AbstractNoInit: # (warn)
	pass
class Resource_RequiredAbstract_AbstractNoInit_NoInit extends Resource_RequiredAbstract_AbstractNoInit: # (warn)
	pass

class Node_RequiredAbstract_AbstractNoInit_NoParams extends Node_RequiredAbstract_AbstractNoInit:
	func _init():
		pass
class Resource_RequiredAbstract_AbstractNoInit_NoParams extends Resource_RequiredAbstract_AbstractNoInit:
	func _init():
		pass

class SubNode_NoParams_RequiredAbstract_NoInit extends SubNode_NoParams_RequiredAbstract: # (warn)
	pass
class SubResource_NoParams_RequiredAbstract_NoInit extends SubResource_NoParams_RequiredAbstract: # (warn)
	pass

class SubNode_NoParams_RequiredAbstract_NoParams extends SubNode_NoParams_RequiredAbstract:
	func _init():
		pass
class SubResource_NoParams_RequiredAbstract_NoParams extends SubResource_NoParams_RequiredAbstract:
	func _init():
		pass

class SubNode_AnyRequired_AbstractNoInit_NoInit extends SubNode_AnyRequired_AbstractNoInit: # (warn)
	pass
class SubResource_AnyRequired_AbstractNoInit_NoInit extends SubResource_AnyRequired_AbstractNoInit: # (warn)
	pass

class SubNode_AnyRequired_AbstractNoInit_NoParams extends SubNode_AnyRequired_AbstractNoInit:
	func _init():
		pass
class SubResource_AnyRequired_AbstractNoInit_NoParams extends SubResource_AnyRequired_AbstractNoInit:
	func _init():
		pass


func test():
	print("warn")
	print("(this message should be on line 31)")
