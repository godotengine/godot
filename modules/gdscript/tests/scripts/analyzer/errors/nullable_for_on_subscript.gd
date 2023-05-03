class A:
    var arr: Array? = null
func test():
    var instance := A.new()
    for element in instance.arr: # Should error
        print(element)