func test():
    var nullable_string: String? = null
    match nullable_string:
        "123":
            print(nullable_string.begins_with("1")) # This is safe
        null:
            print(nullable_string.begins_with("asd")) # This is not