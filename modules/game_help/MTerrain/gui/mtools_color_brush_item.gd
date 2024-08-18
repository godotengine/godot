extends Control

var no_image = preload("res://addons/m_terrain/icons/no_images.png")

var index:int
var uniform:String
var brush_name:String

#func create(brush_name="", icon=no_image):
	#if brush_name.is_empty():
	#	brush_name = "layer "+str(index)				
	#var id = add_item(iname,icon)
	#set_item_custom_bg_color(index,i["icon-color"])
