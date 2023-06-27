class Inner:
	var prop = "Inner"


var array: Array[Inner] = [Inner.new()]


func test():
	var element: Inner = array[0]
	print(element.prop)
