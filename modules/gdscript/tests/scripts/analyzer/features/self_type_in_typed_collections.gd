var list: Array[Self] = []
var dict: Dictionary[String, Self] = {}

func test():
	list.append(self)
	print(list.size() == 1 and list.back() is Self)
	list.clear()

	dict["self"] = self
	print(dict.has("self") and dict["self"] is Self)
	dict.clear()
