const A := 1
enum { B }
enum NamedEnum { C }

class Parent:
	const D := 2
	enum { E }
	enum NamedEnum2 { F }

class Child extends Parent:
	enum TestEnum { A, B, C, D, E, F, Node, Object, Child, Parent}

func test():
	print(A, B, NamedEnum.C, Parent.D, Parent.E, Parent.NamedEnum2.F)
	print(Child.TestEnum.A, Child.TestEnum.B, Child.TestEnum.C, Child.TestEnum.D, Child.TestEnum.E, Child.TestEnum.F)
	print(Child.TestEnum.Node, Child.TestEnum.Object, Child.TestEnum.Child, Child.TestEnum.Parent)
