#GDTEST_OK

func test(): C.new().test("Ok"); test2()

func test2(): print("Ok 2")

class A: pass

class B extends RefCounted: pass

class C extends RefCounted: func test(x): print(x)
