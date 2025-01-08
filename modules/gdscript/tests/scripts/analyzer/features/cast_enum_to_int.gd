# GH-85882

enum Foo { A, B, C }

func test():
	var a := Foo.A
	var b := a as int + 1
	print(b)
