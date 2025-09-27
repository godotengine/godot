class A:
	@read_only var a: bool = true
	@read_only var b: bool
	@read_only var c: bool # Not defined anywhere, not allowed
	@read_only var d: bool
	var e: bool = true
	@read_only var f: bool: # Not allowed, has setter/getter
		set(value):
			f = value

	func _init(d_val: bool) -> void:
		a = false # Not allowed since done inline
		b = true
		b = false # Not allowed, more than one assignment
		d = d_val

	func test() -> void:
		a = false # Not allowed
		e = false

class B extends A:
	func _init() -> void:
		a = false # Not allowed, not base constructor
		self.b = true  # Not allowed, not base constructor
		super._init(true)

func test():
	var a:= A.new(true)
	a.a = false # Not allowed
	a.e = false

	@read_only var c: bool = true
	c = false # Not allowed
