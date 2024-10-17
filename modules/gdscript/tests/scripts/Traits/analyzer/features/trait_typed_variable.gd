trait SomeTrait:
    pass

class SomeClass:
    uses SomeTrait

func test():
    var trait_variable: SomeTrait = SomeClass.new()
    print(trait_variable != null)
    print("ok")
