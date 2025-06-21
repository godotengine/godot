# Test all variations of nested container assignments including @ export, weak / strong refs, and explicit typing

extends Node

# Export variations with nested types
@export var export_weak_nested_array = [ [ 1, 2 ], [ 3, 4 ] ]
@export var export_weak_nested_dict = { "a" : { "x" : 1 }, "b" : { "y" : 2 } }
@export var export_strong_nested_array : Array[Array[int]] = [ [ 1, 2 ], [ 3, 4 ] ]
@export var export_strong_nested_dict : Dictionary[String, Dictionary[String, int]] = { "a" : { "x" : 1 }, "b" : { "y" : 2 } }

# Regular variable variations
var weak_nested_array = [ [ 5, 6 ], [ 7, 8 ] ]
var weak_nested_dict = { "c" : { "z" : 3 }, "d" : { "w" : 4 } }

var strong_infer_nested_array : = [ [ 9, 10 ], [ 11, 12 ] ]
var strong_infer_nested_dict : = { "e" : { "v" : 5 }, "f" : { "u" : 6 } }

var explicit_nested_array : Array[Array[int]] = [ [ 13, 14 ], [ 15, 16 ] ]
var explicit_nested_dict : Dictionary[String, Dictionary[String, int]] = { "g" : { "t" : 7 }, "h" : { "s" : 8 } }

# Mixed explicit typing
var mixed_explicit_1 : Array[Dictionary[String, int]] = [ { "a" : 1 }, { "b" : 2 } ]
var mixed_explicit_2 : Dictionary[String, Array[int]] = { "nums" : [ 1, 2, 3 ] }

# Deep nesting variations
var deep_weak = [ [[1]], [[2]] ]
var deep_strong : = [ [[3]], [[4]] ]
var deep_explicit : Array[Array[Array[int]]] = [ [[5]], [[6]] ]

# @onready variations
@onready var onready_weak_nested = [ [ 17, 18 ], [ 19, 20 ] ]
@onready var onready_strong_nested : = [ [ 21, 22 ], [ 23, 24 ] ]
@onready var onready_explicit_nested : Array[Array[int]] = [ [ 25, 26 ], [ 27, 28 ] ]

func test():
    # Test that weak assignments can be reassigned to different types
    weak_nested_array = [ [ "now", "strings" ], [ "are", "allowed" ] ]
    weak_nested_dict = { "now" : { "different" : "types" } }
    Utils.check(weak_nested_array[0][0] == "now")

    # Test that strong assignments maintain their types
    strong_infer_nested_array[0][0] = 99
    Utils.check(strong_infer_nested_array[0][0] == 99)
    explicit_nested_array[0] = [ 100, 101 ]
    Utils.check(explicit_nested_array[0][0] == 100)

    # Test export behavior
    Utils.check(typeof(export_weak_nested_array) == TYPE_ARRAY)
    Utils.check(export_strong_nested_array.is_typed())

    # Test nested access
    mixed_explicit_1[0]["c"] = 3
    Utils.check(mixed_explicit_1[0]["c"] == 3)
    mixed_explicit_2["more"] = [ 4, 5, 6 ]
    Utils.check(mixed_explicit_2["more"][0] == 4)

    # Test deep nesting behavior
    deep_weak = "can become anything"
    Utils.check(typeof(deep_weak) == TYPE_STRING)
    deep_explicit[0][0][0] = 999
    Utils.check(deep_explicit[0][0][0] == 999)

    # Function call with nested types
    var result = process_nested(explicit_nested_array)
    Utils.check(result == 100)
    var created = create_nested_dict("test", 42)
    Utils.check(created["test"]["value"] == 42)

    print("ok")

func process_nested(data : Array[Array[int]]) -> int:
    return data[0][0] if data.size() > 0 and data[0].size() > 0 else 0

func create_nested_dict(key : String, value : int) -> Dictionary[String, Dictionary[String, int]]:
    return {
        key: { "value" : value, "count" : 1 }
    }
