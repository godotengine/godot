# Godot Chat Interface Example
# This script demonstrates how to use the Godot Chat Assistant
extends Control

@onready var ai_agent = $AIAgent
@onready var chat_input = $VBoxContainer/ChatInput
@onready var chat_output = $VBoxContainer/ScrollContainer/ChatOutput
@onready var send_button = $VBoxContainer/HBoxContainer/SendButton
@onready var clear_button = $VBoxContainer/HBoxContainer/ClearButton

var chat_assistant_loaded = false

func _ready():
	# Setup UI connections
	send_button.pressed.connect(_on_send_pressed)
	clear_button.pressed.connect(_on_clear_pressed)
	chat_input.text_submitted.connect(_on_text_submitted)
	
	# Setup AI Agent
	setup_ai_agent()
	
	# Load the chat assistant TypeScript
	load_chat_assistant()

func setup_ai_agent():
	"""Configure the AI Agent with basic settings"""
	# Set your Gemini API key here
	ai_agent.set_gemini_api_key("YOUR_GEMINI_API_KEY_HERE")
	ai_agent.set_agent_name("Godot Chat Assistant")
	ai_agent.set_agent_role("Interactive Godot development guide and helper")
	
	# Connect signals
	ai_agent.connect("response_received", _on_ai_response)
	ai_agent.connect("error_occurred", _on_ai_error)
	ai_agent.connect("thinking_started", _on_thinking_started)
	ai_agent.connect("thinking_finished", _on_thinking_finished)

func load_chat_assistant():
	"""Load the Godot Chat Assistant TypeScript module"""
	var script_path = "res://modules/ai_agent/examples/godot_chat_assistant.ts"
	
	if ai_agent.load_typescript_file(script_path):
		chat_assistant_loaded = true
		add_chat_message("🐾 Godot Chat Assistant loaded! Type 'hello' to start!", "system")
	else:
		add_chat_message("❌ Failed to load chat assistant. Check TypeScript setup.", "error")

func _on_send_pressed():
	"""Handle send button press"""
	send_message()

func _on_text_submitted(text: String):
	"""Handle Enter key in chat input"""
	send_message()

func send_message():
	"""Send user message to the chat assistant"""
	var message = chat_input.text.strip_edges()
	if message.is_empty():
		return
	
	# Add user message to chat
	add_chat_message(message, "user")
	chat_input.clear()
	
	if chat_assistant_loaded:
		# Use the chat assistant for contextual responses
		var ts_code = """
		// Create chat assistant instance
		const context = {
			agent: globalThis.ai_agent,
			node: globalThis.current_node,
			scene: globalThis.current_scene,
			project: globalThis.project_data,
			editor: globalThis.editor_data
		};
		
		const assistant = new GodotChatAssistant(context);
		
		// Process the message
		assistant.processMessage("%s").then(response => {
			console.log("Chat Assistant Response: " + response);
			// The response will be handled by the response_received signal
		});
		""" % [message.replace('"', '\\"')]
		
		ai_agent.execute_typescript_code(ts_code)
	else:
		# Fallback to simple AI agent
		ai_agent.send_message(message)

func _on_ai_response(response: String):
	"""Handle AI response"""
	add_chat_message(response, "assistant")

func _on_ai_error(error: String):
	"""Handle AI error"""
	add_chat_message("❌ Error: " + error, "error")

func _on_thinking_started():
	"""Show thinking indicator"""
	add_chat_message("🤔 Thinking...", "thinking")

func _on_thinking_finished():
	"""Remove thinking indicator"""
	# Remove last message if it's a thinking message
	var children = chat_output.get_children()
	if children.size() > 0:
		var last_message = children[-1]
		if last_message.has_method("get_message_type") and last_message.get_message_type() == "thinking":
			last_message.queue_free()

func add_chat_message(text: String, type: String = ""):
	"""Add a message to the chat output"""
	var message_container = create_message_container(text, type)
	chat_output.add_child(message_container)
	
	# Auto-scroll to bottom
	call_deferred("scroll_to_bottom")

func create_message_container(text: String, type: String) -> Control:
	"""Create a styled message container"""
	var container = HBoxContainer.new()
	container.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	
	# Add custom method to track message type
	container.set_script(preload("res://modules/ai_agent/examples/message_container.gd"))
	container.set_message_type(type)
	
	# Create icon
	var icon = Label.new()
	match type:
		"user":
			icon.text = "👤"
			container.modulate = Color(0.9, 0.95, 1.0)  # Light blue tint
		"assistant":
			icon.text = "🐾"
			container.modulate = Color(0.95, 1.0, 0.95)  # Light green tint
		"system":
			icon.text = "⚙️"
			container.modulate = Color(1.0, 1.0, 0.9)   # Light yellow tint
		"error":
			icon.text = "❌"
			container.modulate = Color(1.0, 0.9, 0.9)   # Light red tint
		"thinking":
			icon.text = "💭"
			container.modulate = Color(0.95, 0.95, 0.95)  # Light gray tint
		_:
			icon.text = "💬"
	
	icon.custom_minimum_size = Vector2(30, 0)
	container.add_child(icon)
	
	# Create message label
	var label = RichTextLabel.new()
	label.bbcode_enabled = true
	label.fit_content = true
	label.scroll_active = false
	label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	label.custom_minimum_size = Vector2(0, 30)
	
	# Format text with markdown-like styling
	var formatted_text = format_message_text(text)
	label.text = formatted_text
	
	container.add_child(label)
	
	# Add margin
	var margin = MarginContainer.new()
	margin.add_theme_constant_override("margin_bottom", 10)
	margin.add_child(container)
	
	return margin

func format_message_text(text: String) -> String:
	"""Format message text with BBCode styling"""
	# Convert markdown-style formatting to BBCode
	text = text.replace("**", "[b]").replace("**", "[/b]")  # Bold
	text = text.replace("*", "[i]").replace("*", "[/i]")    # Italic
	text = text.replace("`", "[code]").replace("`", "[/code]")  # Code
	
	# Handle code blocks
	text = text.replace("```gdscript", "[code]")
	text = text.replace("```bash", "[code]")
	text = text.replace("```", "[/code]")
	
	return text

func scroll_to_bottom():
	"""Scroll chat to bottom"""
	var scroll_container = $VBoxContainer/ScrollContainer
	scroll_container.scroll_vertical = scroll_container.get_v_scroll_bar().max_value

func _on_clear_pressed():
	"""Clear chat history"""
	for child in chat_output.get_children():
		child.queue_free()
	
	add_chat_message("Chat cleared. Type 'hello' to restart!", "system")

# Example helper functions that can be called from the chat
func get_current_scene_info() -> Dictionary:
	"""Get information about the current scene"""
	var scene = get_tree().current_scene
	if scene:
		return {
			"name": scene.name,
			"type": scene.get_class(),
			"children_count": scene.get_child_count(),
			"script_attached": scene.get_script() != null
		}
	return {}

func get_project_info() -> Dictionary:
	"""Get basic project information"""
	return {
		"name": ProjectSettings.get_setting("application/config/name", "Unknown"),
		"version": ProjectSettings.get_setting("application/config/version", "1.0"),
		"main_scene": ProjectSettings.get_setting("application/run/main_scene", ""),
		"godot_version": Engine.get_version_info()
	}

# Example commands that users can try:
# - "hello" or "hi"
# - "create project"
# - "add node" 
# - "help"
# - "project settings"
# - "how do I make a button?"
# - "show me player movement code"
# - "what is a RigidBody2D?"