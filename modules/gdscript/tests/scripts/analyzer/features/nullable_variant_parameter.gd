## Ensure we can pass nullable arguments to functions which expect a Variant
func test():
    var foo: String? = null
    print(is_instance_valid(foo))