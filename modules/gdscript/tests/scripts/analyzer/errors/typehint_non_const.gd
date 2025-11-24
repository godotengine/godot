class UserDef:
	static var ItemStatic = preload("../../utils.notest.gd")

static var ItemStatic = preload("../../utils.notest.gd")
var Item = preload("../../utils.notest.gd")

var member1: UserDef.ItemStatic
var member2: Item
var member3: ItemStatic

func test() -> void:
	var local1: UserDef.ItemStatic
	var local2: Item
	var local3: ItemStatic
