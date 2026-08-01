func print_enum(e: Mesh.ArrayType) -> Mesh.ArrayType:
	print(e)
	return e

func test():
	var v: Mesh.ArrayType
	v = Mesh.ARRAY_VERTEX
	v = print_enum(v)
	v = print_enum(Mesh.ARRAY_VERTEX)
	v = Mesh.ArrayType.ARRAY_VERTEX
	v = print_enum(v)
	v = print_enum(Mesh.ArrayType.ARRAY_VERTEX)

	v = Mesh.ARRAY_NORMAL
	v = print_enum(v)
	v = print_enum(Mesh.ARRAY_NORMAL)
	v = Mesh.ArrayType.ARRAY_NORMAL
	v = print_enum(v)
	v = print_enum(Mesh.ArrayType.ARRAY_NORMAL)
