# Test that type information is preserved correctly for different assignment types

func test():
    # Test 1: Weak assignment should allow type changes
    var weak_array = [[ 1, 2 ]]
    Utils.check(weak_array is Array)
    weak_array = "now a string"
    Utils.check(typeof(weak_array) == TYPE_STRING)

    # Test 2: Strong inference should maintain type
    var strong_array : = [[ 1, 2 ]]
    Utils.check(typeof(strong_array) == TYPE_ARRAY)
    # Check that elements are properly typed
    var strong_element = strong_array[0]
    Utils.check(typeof(strong_element) == TYPE_ARRAY)

    # Test 3: Explicit typing
    var explicit : Array[Array[int]] = [[ 1, 2 ]]
    Utils.check(typeof(explicit) == TYPE_ARRAY)
    Utils.check(explicit.is_typed() == true) # Explicitly typed arrays should be typed
    # Verify nested structure
    var explicit_row = explicit[0]
    Utils.check(typeof(explicit_row) == TYPE_ARRAY)
    Utils.check(explicit_row.is_typed() == true) # Inner arrays should also be typed
    Utils.check(explicit_row[0] == 1)

    # Test 4: Nested dictionary variations
    var weak_dict = { "a" : { "b" : 1 } }
    Utils.check(typeof(weak_dict) == TYPE_DICTIONARY)
    Utils.check(weak_dict["a"]["b"] == 1)

    var strong_dict : = { "a" : { "b" : 1 } }
    Utils.check(typeof(strong_dict) == TYPE_DICTIONARY)
    var inner_dict = strong_dict["a"]
    Utils.check(typeof(inner_dict) == TYPE_DICTIONARY)

    var explicit_dict : Dictionary[String, Dictionary[String, int]] = { "a" : { "b" : 1 } }
    Utils.check(typeof(explicit_dict) == TYPE_DICTIONARY)
    Utils.check(explicit_dict["a"]["b"] == 1)

    # Test 5: Export behavior differences
    var export_simulation_weak = [[ 1, 2 ]]
    Utils.check(typeof(export_simulation_weak) == TYPE_ARRAY)
    export_simulation_weak = "can change type"
    Utils.check(typeof(export_simulation_weak) == TYPE_STRING)

    var export_simulation_strong : Array[Array[int]] = [[ 3, 4 ]]
    Utils.check(typeof(export_simulation_strong) == TYPE_ARRAY)
    Utils.check(export_simulation_strong.is_typed() == true) # Should be typed
    Utils.check(export_simulation_strong[0][0] == 3)

    # Test 6: Complex nested with mixed types
    var complex_weak = [ { "a" : [ 1, 2 ] }, { "b" : [ 3, 4 ] } ]
    Utils.check(typeof(complex_weak) == TYPE_ARRAY)
    Utils.check(complex_weak[0]["a"][0] == 1)

    var complex_strong : Array[Dictionary[String, Array[int]]] = [ { "a" : [ 1, 2 ] }, { "b" : [ 3, 4 ] } ]
    Utils.check(typeof(complex_strong) == TYPE_ARRAY)
    Utils.check(complex_strong[1]["b"][1] == 4)
    var complex_dict_elem = complex_strong[0]
    Utils.check(typeof(complex_dict_elem) == TYPE_DICTIONARY)
    var complex_array_elem = complex_dict_elem["a"]
    Utils.check(typeof(complex_array_elem) == TYPE_ARRAY)

    # Test 7: Assignment chains
    var chain1 = [[1]]
    var chain2 = chain1 # Should be weak
    chain2 = "different type"
    Utils.check(typeof(chain2) == TYPE_STRING)

    # Test 8: Function parameter inference
    test_param_inference([[ 1, 2 ]])

    print("ok")

func test_param_inference(param = [[0]]):
    # param should be weakly typed since default uses =
    param = "can change type"
    Utils.check(typeof(param) == TYPE_STRING)
