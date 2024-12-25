func use_lambda(lambda: Callable) -> void:
    lambda.call()

func test():
    use_lambda(func(): print("extra line before EOF not required!"))