class_name EnumSizeDoesNotCrash

enum MyEnum {}

func test():
	print(MyEnum.size())
	print(EnumSizeDoesNotCrash.MyEnum.size())
