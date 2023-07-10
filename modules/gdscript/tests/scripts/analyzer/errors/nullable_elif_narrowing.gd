func test():
    var nullable_string: String? = null
    if nullable_string == "2134":
        print(nullable_string.begins_with("asd")) # safe
    elif nullable_string == null:
        print(nullable_string.begins_with("asd")) # Not safe
    else:
        print(nullable_string.begins_with("asd")) # safe