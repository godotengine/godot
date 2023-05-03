func test():
    var nullable_string: String? = null
    (nullable_string as String).begins_with("") # Should not error
    (nullable_string as String?).begins_with("") # Should error
