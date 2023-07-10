func test():
    # OK
    var object: Object = null
    # Potentially unsafe to assign the nullable return of "nullable_object_func()" to non-nullable "object".
    object = nullable_object_func()
    # Value of type "Object" cannot be assigned to a variable of type "Object".
    object = nullable_object_func() ?? null
    print(object)
func nullable_object_func() -> Object?:
    return null