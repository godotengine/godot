func test():
    for element in nullable_array(): # Should error
        print(element)

func nullable_array() -> Array?:
    return null
