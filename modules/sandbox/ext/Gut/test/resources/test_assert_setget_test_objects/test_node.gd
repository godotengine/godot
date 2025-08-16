extends Node
var _priv_var = 'asdf'

# -------------------------------------------------
var no_setget = 1

# -------------------------------------------------
var has_setter = 2 :
	set(val):
		has_setter = val

# -------------------------------------------------
var has_getter = 3 :
	get:
		return has_getter
	set(val):
		has_getter = val

# -------------------------------------------------
var has_both = 4 :
	get:
		return has_both
	set(val):
		has_both = val

# -------------------------------------------------
var non_default_both = 5 :
	get:
		return non_default_both
	set(val):
		non_default_both = val

# -------------------------------------------------
var non_default_setter = 6 :
	get:
		return non_default_setter
	set(val):
		non_default_setter = val

# -------------------------------------------------
var non_default_getter = 7 :
	get:
		return non_default_getter
	set(val):
		non_default_getter = val

# -------------------------------------------------
# dnu = "does not use"
var has_both_dnu_setget = 8
func get_has_both_dnu_setget():
	return has_both_dnu_setget

func set_has_both_dnu_setget(val):
	has_both_dnu_setget = val

# -------------------------------------------------
var typed_setter:int = 9 :
	get:
		return typed_setter
	set(val):
		typed_setter = val

var _backed_property = 10
var backed_property = 10 :
	get: return _backed_property
	set(val): _backed_property = val

var _backed_get_broke = 11
var backed_get_broke = 11 :
	get: return backed_get_broke
	set(val): _backed_get_broke = val


var _backed_set_broke = 12
var backed_set_broke = 12 :
	get: return _backed_set_broke
	set(val): backed_set_broke = val
