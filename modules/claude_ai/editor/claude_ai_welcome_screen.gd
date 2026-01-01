@tool
extends Control

## Welcome screen for Godot AI Studio
## Shows on first launch or when no project is open

signal create_new_project_requested
signal open_project_requested

@onready var welcome_label: Label = $VBoxContainer/WelcomeLabel
@onready var subtitle_label: Label = $VBoxContainer/SubtitleLabel
@onready var new_project_button: Button = $VBoxContainer/ButtonContainer/NewProjectButton
@onready var open_project_button: Button = $VBoxContainer/ButtonContainer/OpenProjectButton
@onready var ai_info_panel: Panel = $VBoxContainer/AIInfoPanel

func _ready():
	setup_ui()

func setup_ui():
	welcome_label.text = "Welcome to Godot AI Studio"
	subtitle_label.text = "AI-Powered Game Development Environment"
	
	new_project_button.text = "Create New Project"
	open_project_button.text = "Open Project"
	
	new_project_button.pressed.connect(_on_new_project_pressed)
	open_project_button.pressed.connect(_on_open_project_pressed)
	
	# Show AI features info
	update_ai_info()

func _on_new_project_pressed():
	create_new_project_requested.emit()

func _on_open_project_pressed():
	open_project_requested.emit()

func update_ai_info():
	# Display information about AI features
	pass

