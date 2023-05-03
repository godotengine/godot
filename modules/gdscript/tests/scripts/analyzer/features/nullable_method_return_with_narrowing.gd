func test():
    print(_get_result())


func _get_result() -> String:
    var might_be_null: String? = null
    if might_be_null:
        return might_be_null
    return "123"
