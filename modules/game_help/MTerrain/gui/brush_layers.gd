@tool
extends ItemList


var no_image = preload("res://addons/m_terrain/icons/no_images.png")

var index:int
var uniform:String
var brush_name:String

func create_layers(input:Array):
	var index = -1
	for i in input:
		index +=1
		var iname:String= i["name"]
		if iname.is_empty():
			iname = "layer "+str(index)
		var icon:Texture = i["icon"]
		if not icon:
			icon = no_image
		var id = add_item(iname,icon)
