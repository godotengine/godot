class_name EnumAccessOuterClass

class InnerClass:
	enum MyEnum { V0, V2, V1 }

	static func print_enums():
		print("Inner - Inner")
		print(MyEnum.V0, MyEnum.V1, MyEnum.V2)
		print(InnerClass.MyEnum.V0, InnerClass.MyEnum.V1, InnerClass.MyEnum.V2)
		print(EnumAccessOuterClass.InnerClass.MyEnum.V0, EnumAccessOuterClass.InnerClass.MyEnum.V1, EnumAccessOuterClass.InnerClass.MyEnum.V2)

		print("Inner - Outer")
		print(EnumAccessOuterClass.MyEnum.V0, EnumAccessOuterClass.MyEnum.V1, EnumAccessOuterClass.MyEnum.V2)


enum MyEnum { V0, V1, V2 }

func print_enums():
	print("Outer - Outer")
	print(MyEnum.V0, MyEnum.V1, MyEnum.V2)
	print(EnumAccessOuterClass.MyEnum.V0, EnumAccessOuterClass.MyEnum.V1, EnumAccessOuterClass.MyEnum.V2)

	print("Outer - Inner")
	print(InnerClass.MyEnum.V0, InnerClass.MyEnum.V1, InnerClass.MyEnum.V2)
	print(EnumAccessOuterClass.InnerClass.MyEnum.V0, EnumAccessOuterClass.InnerClass.MyEnum.V1, EnumAccessOuterClass.InnerClass.MyEnum.V2)

func test():
	print_enums()
	InnerClass.print_enums()
