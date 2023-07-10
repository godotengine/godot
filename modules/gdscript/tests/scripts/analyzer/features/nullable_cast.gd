func test():
    var nullable_string: String? = "123"
    print((nullable_string as String).begins_with("1"))
    nullable_string = null
    print((nullable_string as String?)?.begins_with("1"))
