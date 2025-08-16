extends Node2D

var PanelControls = load('res://addons/gut/gui/panel_controls.gd')

@onready var _ctrls = {
	pc_vbox = $PanelControls/VBox
}

var _save_load = PanelControls.SaveLoadControl.new('whatever', 1, 'hint')
var _res_dir = PanelControls.DirectoryControl.new('some dir', 'res://', 'hint')
var _res_dir_enabled = PanelControls.DirectoryControl.new('other dir', 'res://', 'hint')


func _ready():
	#_save_load.dlg_load.show_diretory_types = false
	#_save_load.dlg_load.show_user = false
	#_save_load.dlg_save.show_os = false
	
	_res_dir_enabled.enabled_button.visible = true
	
	_ctrls.pc_vbox.add_child(_save_load)
	_ctrls.pc_vbox.add_child(_res_dir)
	_ctrls.pc_vbox.add_child(_res_dir_enabled)

	
