var array_var: Array = ["one", "two", "three", "four"]
const array_const: Array = ["one", "two", "three", "four"]

var array_nested_var: Array = [["one"], ["two"], ["three"], ["four"]]
const array_nested_const: Array = [["one"], ["two"], ["three"], ["four"]]


func test():
    Utils.check(array_const.is_read_only() == true)
    Utils.check(array_nested_const.is_read_only() == true)

    print("TEST Callable::callv")
    print_four_variants.callv(array_var)
    print_four_variants.callv(array_const)
    print_four_variants.callv(array_nested_var)
    print_four_variants.callv(array_nested_const)

    print("TEST Object::callv")
    self.callv("print_four_variants", array_var)
    self.callv("print_four_variants", array_const)
    self.callv("print_four_variants", array_nested_var)
    self.callv("print_four_variants", array_nested_const)


func print_four_variants(v1, v2, v3, v4):
    print("%s %s %s %s" % [v1, v2, v3, v4])
