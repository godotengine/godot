@tool
extends HBoxContainer

func toggle_visible(toggled_on):
	visible = not toggled_on

func set_height_label(input:float):
	$h_label.text = ("H: "+str(input)+"m").pad_decimals(2)

func disable_height_label():
	$h_label.text = "H: UNKOWN"

func set_distance_label(input:float):
	$d_label.text = ("D: "+str(input)+"m").pad_decimals(2)

func disable_distance_label():
	$d_label.text = "D: UNKOWN"
	
func set_grass_label(input:int):
	$g_label.text = " G: " + str(input)

func disable_grass_label():
	$g_label.text = ""
	
func set_region_label(input:int):
	$r_label.text = "R:"+str(input)

func disable_region_label():
	$r_label.text = ""


func _on_hide_status_toggled(toggled_on: bool) -> void:
	pass # Replace with function body.
