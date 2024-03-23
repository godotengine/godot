var Value:int = 8:
	get:
		return Value
	set(v):
		Value = v

func test():
	var f:float = Value
	print(int(f))
