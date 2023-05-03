func test():
    var nullable_str: String? = null
    print((nullable_str ?? "123").begins_with("123"))
