class UserDef:
	static var ItemStatic = preload("../../utils.notest.gd")

static var ItemStatic = preload("../../utils.notest.gd")
var Item = preload("../../utils.notest.gd")

func get_conf():
	return UserDef.ItemStatic.new()

func test() -> void:
	var local1: UserDef.ItemStatic = get_conf()
	var local2: Item = get_conf()
	var local3: ItemStatic = get_conf()

var member1: UserDef.ItemStatic = get_conf()
var member2: Item = get_conf()
var member3: ItemStatic = get_conf()
