class Inner:
	var prop = "Inner"

var dict: Dictionary[int, Inner] = { 0: Inner.new() }


func test():
	var element: Inner = dict[0]
	print(element.prop)
