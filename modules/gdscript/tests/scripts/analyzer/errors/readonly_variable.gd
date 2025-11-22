class A:
	@read_only var a: bool = true # Case: initialized inline
	@read_only var b: bool # Case: initialized in constructor
	@read_only var c: bool # Case: not initialized	 														NOT ALLOWED
	@read_only var d: bool # Case: initialized in constructor by non-pure initializer
	var e: bool = true # Control case (regression check)
	var f: bool # Control case
	@read_only var g: bool = true: # Case: has setter		 												NOT ALLOWED
		set(value):
			g = value
	@read_only var h: bool = true: # Case: has getter
		get():
			return h
	var i: bool = b # Case: accessing before initialized													NOT ALLOWED
	@read_only var j: bool # Case: initialized in all branches in constructor
	@read_only var k: bool # Case: initialized in not all branches in constructor							NOT ALLOWED

	func _init(d_val: bool) -> void:
		print(a) # Case: accessing inline initialized

		a = false # Case: reinitializing inline initialized 												NOT ALLOWED

		print(b) # Case: accessing beore initializing														NOT ALLOWED
		print(self.b) # Case: same as bove but with self.													NOT ALLOWED

		b = true # Case: initializing in constructor
		b = false # Case: reinitializing constructor initialized											NOT ALLOWED

		d = d_val # Case: non-pure initializing in constructor

		if a:
			j = true
			k = true
		else:
			match b:
				true:
					j = true
				_:
					j = false

	func test() -> void:
		a = false # Case: give value to read-only															NOT ALLOWED
		print(b) # Case: access after initialization
		e = false # Control case

class B extends A:
	func _init() -> void:
		b = false # Case: initializing in NOT base constructor												NOT ALLOWED
		self.b = true  # Case: same as above but with self.													NOT ALLOWED

		super._init(true) # Case: initializing through constructor parameter

class C:
	var a:= A.new(true)
	@read_only var b:= B.new()
	@read_only var c:= [1,2,3]
	@read_only var d:= Vector2.ZERO
	@read_only var e: bool # Case: not initialized, no constructor											NOT ALLOWED

class D:
	var c:= C.new()
	@read_only var c2:= C.new()

class E:
	var d:= D.new()

func test():
	var a:= A.new(true)
	a.a = false # Case: give value to read-only in subscript												NOT ALLOWED
	a.set("a", false) # Case: set read-only via .set()														NOT ALLOWED
	a.e = false # Control case

	var c := C.new()
	c.a.a = false # Case: give value to read-only in nested subscript										NOT ALLOWED
	c.c = [3,2,1] # Case: give value to pass-by-ref read-only												NOT ALLOWED
	c.c[0] = 0 # Case: Modifying array value (pass by ref)
	c.d.x = 1 # Case: Modifying vector value (pass by value)												NOT ALLOWED

	var e := E.new()
	e.d.c.d.x = 1 # Case: same as above, deep																NOT ALLOWED
	e.d.c2.a.e = false # Case: read-only pass-by-ref in chain but not last
