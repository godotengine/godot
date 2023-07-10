func test():
    non_nullable_return()

func non_nullable_return() -> Vector2:
    return nullable_return() # Should error

func nullable_return() -> Vector2?:
    return null