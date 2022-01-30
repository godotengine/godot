enum Named { VALUE_A, VALUE_B }

func test():
    var typed_variable_A : Named = Named.VALUE_A
    prints(typed_variable_A == Named.VALUE_A)
    prints(typed_variable_A == Named.VALUE_B)
    