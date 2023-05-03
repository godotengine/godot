func test():
    var bar := foo()
    print(bar.begins_with("")) # Should error


func foo() -> String?:
    return null