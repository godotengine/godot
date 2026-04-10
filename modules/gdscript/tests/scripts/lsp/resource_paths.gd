extends Node

var uid_invalid = "uid://zzz_not_a_real_uid"
var uid_valid = "uid://a1b2c3"
var res_path = "res://lsp/local_variables.gd"
var normal_string = "hello world"
var empty_string = ""
var number = 42
var multi = "first" + "second" + "third"
var escaped = "hello \"world\" bye"
var single_quoted = 'single quotes'
var mixed_quotes = "it's a test"
var multiline_str = """I'm also "a" String"""
var uid_in_call = str("uid://a1b2c3")
# var uid_commented = "uid://a1b2c3"

func use_path(_p: String) -> void:
	pass

func caller():
	use_path("uid://a1b2c3")
