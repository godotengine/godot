@tool
extends HBoxContainer

@onready var height_lable := $h_lable
@onready var distance_lable := $d_lable
@onready var grass_lable := $g_lable
@onready var region_lable :=$r_lable
@onready var human_male_button:= $human_male
@onready var save_button:= $save

signal toggle_paint_mode
signal save_request
signal info_window_open_request
var active_paint_mode := false
var human_male_active:=false



func _on_paint_mode_toggled(button_pressed):
	active_paint_mode = button_pressed
	emit_signal("toggle_paint_mode",button_pressed)


func set_height_lable(input:float):
	height_lable.text = ("H: "+str(input)+"m").pad_decimals(2)

func disable_height_lable():
	height_lable.text = "H: UNKOWN"

func set_distance_lable(input:float):
	distance_lable.text = ("D: "+str(input)+"m").pad_decimals(2)

func disable_distance_lable():
	distance_lable.text = "D: UNKOWN"
	
func set_grass_label(input:int):
	grass_lable.text = " G: " + str(input)

func disable_grass_lable():
	grass_lable.text = ""
	
func set_region_lable(input:int):
	region_lable.text = "R:"+str(input)

func disable_region_lable():
	region_lable.text = ""

func _on_human_male_toggled(button_pressed):
	human_male_active = button_pressed


func set_save_button_disabled(input:bool):
	save_button.disabled = input

func _on_save_pressed():
	emit_signal("save_request")


func _on_info_btn_pressed():
	emit_signal("info_window_open_request")
