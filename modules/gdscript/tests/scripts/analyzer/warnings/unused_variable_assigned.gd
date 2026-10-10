class Test:
    var member := 0

func param_test(param):
    param = 1

func param_member_test(param):
    param.member = 1

func test():
    var a = 1
    a = 2

    var b = 1
    b += 1

    var c = Test.new()
    c.member = 2
