class ImmutableTestObject:
	let immutable_member: float = 0.0 # Case: immutable member
	static let immutable_static: float = 0.0 # Case: static immutable member
	# let immutable_unassigned: float # Case: unassigned immutable member													NOT ALLOWED
	let immutable_array: Array[float] = [0.0, 1.0, 2.0, 3.0]

	func _init():
		immutable_member = 1.0 # Case: assign immutable member															NOT ALLOWED
		immutable_static = 1.0 # Case: assign static immutable member													NOT ALLOWED
		immutable_array = [0.0] # Case: assign immutable pass-by-ref var												NOT ALLOWED
		immutable_array[0] = 1.0 # Case: modify immutable pass-by-ref var's state
		param_test(0.0) # Case call function with input for immutable parameter

	# Case: function with immutable parameter & immutable parameter with default
	func param_test(let immutable_parameter: float, let immutable_default: float = 0.0):
		immutable_parameter = 1.0 # Case: assign immutable parameter													NOT ALLOWED
		immutable_default = 1.0 # Case: assign immutable default parameter												NOT ALLOWED

class ImmutableExtends extends ImmutableTestObject:
	func _init():
		immutable_member = 1.0 # Case: assign base class immutable member												NOT ALLOWED

func test():
	let immutable_local: float = 0.0 # Case: immutable local
	immutable_local = 1.0 # Case: assign immutable local																NOT ALLOWED

	# let immutable_local_unassigned: float # Case: unassigned imutable local											NOT ALLOWED

	var lambda = func(let delta: float): # Case: lambda immutable parameter
		delta = 0.0 # Case: assign immutable lambda parameter															NOT ALLOWED
	lambda.call(0.0) # Case: call lambda with input for immutable parameter

	var test:= ImmutableTestObject.new()
	test.immutable_member = 1.0 # Case: assign object immutable member													NOT ALLOWED
	test.set("immutable_member", 1.0) # Case: assign object immutable member via .set()									NOT ALLOWED
