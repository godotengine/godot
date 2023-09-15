func test():
    var user_id_key := "userId"
    var {
        user_id_key: var user_id,
        name = var name,
        arr = [_, { foo = var foo }, var ..everything_else]
    } = {
        userId = 123,
        name = "josaid",
        arr = ["ignore me", { foo = "this is foo" }, "everything", "else"]
    }
    prints(user_id, name, foo, everything_else)
