func test():
    var nullable_string: String? = null
    match nullable_string:
        "123", "13":
            print(nullable_string.begins_with("1"))
        {}:
            print(nullable_string.begins_with("1"))
        []:
            print(nullable_string.begins_with("1"))
        null:
            prints("Nullable is null! Proof: ", nullable_string)
        var nullable:
            prints("Nullable is null! Proof: ", nullable)