func test():
    var nullable_string: String? = "123"
    if (nullable_string as String) == "123":
        print(nullable_string.begins_with("123")) # Should not error