var a: Array = [1]:
	set(v):
		prints("set a", v)
		a = v
	get:
		prints("get a")
		return a

@warning_ignore("implicit_conversion_causes_copy")
var b: PackedByteArray = [1]:
	set(v):
		prints("set b", v)
		b = v
	get:
		prints("get b")
		return b

@warning_ignore("implicit_conversion_causes_copy")
var c: PackedVector2Array = [Vector2.ONE]:
	set(v):
		prints("set c", v)
		c = v
	get:
		prints("get c")
		return c

func test():
	a[0] = 2
	print(a)
	b[0] = 2
	print(b)
	c[0].x = 2
	print(c)
