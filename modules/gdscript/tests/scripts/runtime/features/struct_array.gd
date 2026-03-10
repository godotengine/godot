struct Data:
	var id: int
	var text: String

func test():
	var array: Array[Data] =[]
	array.push_back(Data(1, "Hello"))
	array.push_back(Data(2, "World"))

	print(array.size())
	print(array[0].id)
	print(array[0].text)
	print(array[1].id)
	print(array[1].text)
