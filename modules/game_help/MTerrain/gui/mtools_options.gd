@tool
extends Button

func _ready():
	var panel = get_child(0)
	panel.visible = false
	#panel.size.y = 
	panel.position.y = -panel.size.y - 4


func _on_panel_container_resized():
	$PanelContainer.position.y = -$PanelContainer.size.y - 2
