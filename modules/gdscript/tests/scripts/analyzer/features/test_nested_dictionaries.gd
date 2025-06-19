# Test nested dictionary types

func test():
    # Strong typing with nested dictionaries
    var nested_dict : Dictionary[String, Dictionary[String, int]] = {
        "user1" : { "age" : 25, "score" : 100 },
        "user2" : { "age" : 30, "score" : 150 }
    }
    Utils.check(nested_dict["user1"]["age"] == 25)

    # Accessing and modifying nested values
    nested_dict["user1"]["age"] = 26
    Utils.check(nested_dict["user1"]["age"] == 26)
    nested_dict["user3"] = { "age" : 35, "score" : 200 }
    Utils.check(nested_dict["user3"]["score"] == 200)

    # Inferred typing
    var inferred_nested : = {
        "a" : { "x" : 1, "y" : 2 },
        "b" : { "x" : 3, "y" : 4 }
    }
    Utils.check(inferred_nested["a"]["x"] == 1)

    # Triple nested dictionary
    var triple_nested : Dictionary[String, Dictionary[String, Dictionary[String, int]]] = {
        "level1" : {
            "level2" : {
                "level3" : 42
            }
        }
    }
    Utils.check(triple_nested["level1"]["level2"]["level3"] == 42)
    triple_nested["level1"]["level2"]["level3"] = 100
    Utils.check(triple_nested["level1"]["level2"]["level3"] == 100)

    # Dictionary with array values
    var dict_of_arrays : Dictionary[String, Array[int]] = {
        "numbers" : [ 1, 2, 3 ],
        "more_numbers" : [ 4, 5, 6 ]
    }
    Utils.check(dict_of_arrays["numbers"][0] == 1)
    dict_of_arrays["numbers"][0] = 10
    Utils.check(dict_of_arrays["numbers"][0] == 10)

    # Weak assignment
    var weak_dict = { "a" : { "b" : 1 } }
    Utils.check(typeof(weak_dict) == TYPE_DICTIONARY)
    weak_dict = { "x" : { "y" : "string" } } # Should work - weak typing
    Utils.check(weak_dict["x"]["y"] == "string")

    print("ok")
