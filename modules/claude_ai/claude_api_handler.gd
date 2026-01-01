@tool
extends Node

## Claude AI API Handler (Direct API Mode)
## Handles HTTP requests directly to the Claude API
## For SaaS mode, use saas_api_handler.gd instead

signal request_complete(response_text: String)
signal request_error(error_message: String)
signal ai_question(question: String)  # AI is asking a question
signal conversation_updated()  # Conversation history updated

const CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
const MODEL = "claude-3-5-sonnet-20240620"  # Using latest Sonnet model for best quality
const MAX_TOKENS = 32768  # Increased for complete game generation (multiple files, scenes, scripts)

# SaaS mode detection
const SAAS_API_URL = "https://api.dotai.dev"  # DotAI SaaS API URL
var use_saas_mode = false  # Set to true to use SaaS backend
var saas_handler = null  # Reference to SaaS handler if available

var http_request: HTTPRequest
var current_response_text: String = ""

# Cache loaded scripts
var _codebase_scanner_script = null
var _file_writer_script = null

# AI-Native Engine Components
var conversation_manager: ConversationManager = null

func _ready():
	# Check if SaaS handler is available and should be used
	_check_saas_mode()
	
	if not use_saas_mode:
		http_request = HTTPRequest.new()
		add_child(http_request)
		http_request.request_completed.connect(_on_request_completed)
	
	# Try to load scripts (with fallback paths)
	_codebase_scanner_script = _load_script("res://addons/claude_ai/codebase_scanner.gd", "res://codebase_scanner.gd")
	_file_writer_script = _load_script("res://addons/claude_ai/file_writer.gd", "res://file_writer.gd")
	
	# Initialize AI-Native Engine components
	_initialize_ai_native_components()

func _check_saas_mode():
	# SaaS mode DISABLED FOR TESTING - always use direct API mode
	use_saas_mode = false
	
	# Original SaaS mode check (commented out for testing)
	# var saas_script = _load_script("res://addons/claude_ai/saas_api_handler.gd", "res://saas_api_handler.gd")
	# if saas_script != null:
	# 	# Check if SaaS mode is enabled in config
	# 	var config = ConfigFile.new()
	# 	var config_path = "user://godot_ai_studio.cfg"
	# 	if config.load(config_path) == OK:
	# 		use_saas_mode = config.get_value("settings", "use_saas_mode", true)
	# 	else:
	# 		use_saas_mode = true  # Default to SaaS mode
	# 	
	# 	if use_saas_mode:
	# 		# Create SaaS handler instance
	# 		saas_handler = Node.new()
	# 		saas_handler.set_script(saas_script)
	# 		add_child(saas_handler)
	# 		
	# 		# Connect SaaS handler signals
	# 		if saas_handler.has_signal("request_complete"):
	# 			saas_handler.connect("request_complete", _on_saas_request_complete)
	# 		if saas_handler.has_signal("request_error"):
	# 			saas_handler.connect("request_error", _on_saas_request_error)
	# 		if saas_handler.has_signal("auth_status_changed"):
	# 			saas_handler.connect("auth_status_changed", _on_auth_status_changed)
	# 		if saas_handler.has_signal("usage_updated"):
	# 			saas_handler.connect("usage_updated", _on_usage_updated)

func _on_saas_request_complete(response_text: String):
	request_complete.emit(response_text)

func _on_saas_request_error(error_message: String):
	request_error.emit(error_message)

func _on_auth_status_changed(is_authenticated: bool):
	# Emit signal for UI to update
	pass  # Can be handled by UI

func _on_usage_updated(usage_data: Dictionary):
	# Emit signal for UI to update usage display
	pass  # Can be handled by UI

func _load_script(primary_path: String, fallback_path: String):
	var script = load(primary_path)
	if script == null:
		script = load(fallback_path)
	return script

func _initialize_ai_native_components():
	print("DotAI: Initializing AI-Native components...")
	
	# Load conversation manager script (if not already loaded)
	if conversation_manager == null:
		var conv_script = _load_script("res://addons/claude_ai/conversation_manager.gd", "res://conversation_manager.gd")
		if conv_script:
			# Try to instantiate using the script directly (more reliable than class_name)
			var conv_node = Node.new()
			conv_node.set_script(conv_script)
			add_child(conv_node)
			
			# Check if it has the methods we need
			if conv_node.has_method("start_conversation") and conv_node.has_method("add_message"):
				conversation_manager = conv_node
				conversation_manager.start_conversation()
				if conversation_manager.has_signal("conversation_updated"):
					conversation_manager.conversation_updated.connect(_on_conversation_updated)
				if conversation_manager.has_signal("ai_question"):
					conversation_manager.ai_question.connect(_on_ai_question)
				print("DotAI: ConversationManager initialized successfully")
			else:
				push_error("ConversationManager: Script loaded but missing required methods (start_conversation, add_message)")
				conv_node.queue_free()
		else:
			push_warning("DotAI: conversation_manager.gd script not found. Conversation features disabled.")

func send_request(params: Dictionary) -> void:
	print("DotAI: send_request called")
	
	# If SaaS mode is enabled, delegate to SaaS handler
	if use_saas_mode and saas_handler != null:
		if saas_handler.has_method("send_request"):
			saas_handler.call("send_request", params)
			return
		else:
			request_error.emit("SaaS handler not properly initialized")
			return
	
	# Direct API mode (original implementation)
	var api_key: String = params.get("api_key", "")
	var prompt: String = params.get("prompt", "")
	var include_codebase: bool = params.get("include_codebase", true)
	var is_conversation: bool = params.get("is_conversation", true)  # Default to conversation mode
	
	# API key is required
	if api_key.is_empty():
		request_error.emit("API key is required. Please provide your Claude API key in the DotAI panel or use SaaS mode.")
		return
	
	if prompt.is_empty():
		request_error.emit("Prompt is required")
		return
	
	print("DotAI: Prompt received: ", prompt.substr(0, 50), "...")
	print("DotAI: Conversation manager: ", "available" if conversation_manager != null else "null")
	
	# Add user message to conversation
	if conversation_manager != null and is_conversation:
		conversation_manager.add_message("user", prompt)
		print("DotAI: User message added to conversation")
	
	# Build conversation context
	var messages = []
	if conversation_manager != null and is_conversation:
		# Use conversation history (get a copy so we can modify it)
		messages = conversation_manager.get_conversation_context()
		
		# Add codebase context to the last user message if needed
		if include_codebase and messages.size() > 0:
			# Find the last user message
			var last_user_idx = -1
			for i in range(messages.size() - 1, -1, -1):
				if messages[i].role == "user":
					last_user_idx = i
					break
			
			if last_user_idx >= 0:
				var original_content = messages[last_user_idx].content
				var enhanced_content = _build_enhanced_prompt(original_content, include_codebase)
				messages[last_user_idx].content = enhanced_content
	else:
		# Single-shot mode (no conversation)
		var enhanced_prompt = _build_enhanced_prompt(prompt, include_codebase)
		messages = [
			{
				"role": "user",
				"content": enhanced_prompt
			}
		]
	
	# Build the request payload with enhanced settings for game generation
	var payload = {
		"model": MODEL,
		"max_tokens": MAX_TOKENS,  # Maximum tokens for complete game generation
		"messages": messages,
		"temperature": 0.6,  # Lower temperature for more consistent, production-ready code
		"top_p": 0.95  # Focus on high-quality, complete responses
	}
	
	var json_string = JSON.stringify(payload)
	var headers = [
		"Content-Type: application/json",
		"x-api-key: " + api_key,
		"anthropic-version: 2023-06-01"
	]
	
	print("DotAI: Sending HTTP request to Claude API...")
	var error = http_request.request(CLAUDE_API_URL, headers, HTTPClient.METHOD_POST, json_string)
	if error != OK:
		print("DotAI: HTTP request failed with error: ", error)
		request_error.emit("Failed to create HTTP request: " + str(error))
	else:
		print("DotAI: HTTP request sent successfully")

func _build_enhanced_prompt(user_prompt: String, include_codebase: bool) -> String:
	var system_prompt = """═══════════════════════════════════════════════════════════════
🎮 DOTAI - WORLD'S MOST ADVANCED AI GAME ENGINE 🎮
═══════════════════════════════════════════════════════════════

You are DotAI - a COMPLETE GAME DEVELOPMENT SYSTEM, not just a code generator.
Your mission: Transform ANY game idea into a FULLY PLAYABLE, PRODUCTION-READY GAME in MINUTES.

🚀 WHAT YOU DO:
- User says: "Create a 2D side scroller"
- You create: COMPLETE GAME with scenes, scripts, UI, managers, everything
- Result: User can press PLAY and the game works immediately

❌ WHAT YOU DON'T DO:
- Generate code snippets without file markers
- Create incomplete implementations
- Leave TODOs or placeholders
- Split features across multiple responses
- Create games that need manual setup

✅ WHAT YOU ALWAYS DO:
- Create ALL files needed for a complete game
- Use proper file markers (# File: path/to/file.ext)
- Generate production-ready, fully functional code
- Set up proper project structure
- Make games playable immediately

═══════════════════════════════════════════════════════════════

You are a master game developer with expertise in:
- 🎯 Game Design: Understanding player experience, game loops, mechanics, and fun factor
- 🏗️ Architecture: State Machines, Component Systems, ECS, Observer Pattern, Singletons, Managers
- 🎨 Scene Design: Optimal node hierarchies, performance, organization, and maintainability
- ⚡ Performance: Node caching, signal efficiency, memory management, optimization
- 🛡️ Quality: Error handling, edge cases, validation, graceful degradation
- 📚 Best Practices: SOLID principles, clean code, design patterns, maintainability
- 🎪 Game Systems: Movement, combat, UI, inventory, save systems, audio, particles, animations

🚀 WHEN A USER DESCRIBES A GAME, YOU MUST:

1. **UNDERSTAND THE COMPLETE GAME**: What type of game? What mechanics? What's the core loop?
2. **DESIGN THE ARCHITECTURE**: What systems are needed? How do they interact?
3. **CREATE EVERYTHING**: Scripts, scenes, resources, UI, managers, configs - EVERYTHING needed
4. **MAKE IT PLAYABLE**: The game must work immediately after generation - no manual setup needed
5. **FOLLOW BEST PRACTICES**: Production-quality code, proper structure, error handling
6. **EXPLAIN YOUR DESIGN**: Help the user understand why you made certain choices

💡 CRITICAL: You are creating COMPLETE GAMES, not code snippets. Every response should result in a playable game.

🎯 GAME GENERATION REQUIREMENTS (MANDATORY):

For EVERY game request, you MUST create:

1. **MAIN SCENE** (.tscn):
   - Complete scene with all nodes properly configured
   - Player, enemies, UI, world, cameras, etc.
   - All scripts attached and connected
   - Scene is ready to run immediately

2. **ALL SCRIPTS** (.gd):
   - Player controller with complete movement
   - Enemy AI with behavior
   - Game manager for state management
   - UI controllers for menus/HUD
   - Collectibles, power-ups, etc.
   - Every script is complete and functional

3. **PROJECT STRUCTURE**:
   - scripts/player/ - Player-related scripts
   - scripts/enemies/ - Enemy scripts
   - scripts/ui/ - UI scripts
   - scripts/managers/ - Game managers
   - scripts/collectibles/ - Collectible items
   - scenes/ - All scene files
   - resources/ - Any resources needed

4. **COMPLETE SYSTEMS**:
   - Movement: WASD/Arrow keys, jumping, physics
   - Combat: If mentioned, complete combat system
   - UI: Health bars, score, menus, game over screens
   - Game State: Win/lose conditions, restart functionality
   - Audio: Sound effects and music setup (if mentioned)
   - Particles: Visual effects (if mentioned)

5. **PRODUCTION QUALITY**:
   - Type hints everywhere: @export var name: Type
   - Error handling: Null checks, validation
   - Performance: Node caching, efficient signals
   - Documentation: Comments explaining complex logic
   - Edge cases: Handle all boundary conditions
   - Signals: Properly declared, connected, and emitted

📋 EXAMPLE: "Create a 2D side scroller"
You MUST create:
- scenes/main.tscn (complete game scene)
- scripts/player/player.gd (movement, jumping, animations)
- scripts/player/player.tscn (player scene with sprite, collision)
- scripts/enemies/enemy.gd (enemy AI)
- scripts/enemies/enemy.tscn (enemy scene)
- scripts/managers/game_manager.gd (game state, score)
- scripts/ui/hud.gd (UI display)
- scripts/collectibles/coin.gd (collectibles)
- project.godot (if needed for input map)

ALL files must be created in ONE response. The game must be PLAYABLE immediately.

ADVANCED GDScript Requirements:
- Use GDScript syntax exclusively (not C# or other languages)
- Include comprehensive type hints: @export var name: Type, var variable: Type
- Follow Godot naming conventions: snake_case for variables/functions, PascalCase for classes
- Use @tool directive when appropriate (editor scripts, custom resources)
- Declare signals properly: signal signal_name(param: Type) at class level
- Handle ALL edge cases: null checks, boundary conditions, error states
- Use proper access modifiers: private functions with _ prefix
- Document complex logic: add comments explaining WHY, not just WHAT
- Optimize performance: cache nodes, avoid repeated calculations, use appropriate process functions
- Follow SOLID principles: Single Responsibility, Open/Closed, etc.
- Use design patterns appropriately: Singleton for managers, State Machine for complex behaviors
- Write testable code: separate logic from presentation, use dependency injection where helpful

🎬 SCENE CREATION (CRITICAL):

When creating scenes (.tscn files), you MUST:

1. **Complete Node Hierarchy**:
   - Root node (usually Node2D or Control)
   - Player node with CharacterBody2D, Sprite2D, CollisionShape2D
   - Camera2D properly configured
   - UI layer with Control nodes
   - World/environment nodes
   - All nodes properly named and organized

2. **Proper Configuration**:
   - Scripts attached to nodes
   - Collision shapes configured
   - Sprites positioned correctly
   - Camera limits set (if needed)
   - UI anchors and margins set
   - Signals connected in scene

3. **Scene Format**:
   Use proper .tscn format:
   ```
   [gd_scene load_steps=2 format=3 uid="uid://..."]
   
   [ext_resource type="Script" path="res://scripts/player/player.gd" id="1"]
   
   [node name="Player" type="CharacterBody2D"]
   script = ExtResource("1")
   
   [node name="Sprite2D" type="Sprite2D" parent="."]
   position = Vector2(0, 0)
   
   [node name="CollisionShape2D" type="CollisionShape2D" parent="."]
   ```
   
4. **Main Scene Setup**:
   - Create scenes/main.tscn as the main game scene
   - Include all game elements
   - Set up proper scene tree
   - Make it runnable immediately

User Request: """ + user_prompt

	if include_codebase:
		# Get project context with enhanced scanning
		# Call static methods directly on the class name (class_name registration makes it available globally)
		if _codebase_scanner_script != null:
			var project_summary = ""
			var relevant_files = []
			
			# Try to call static methods on the class name
			# The class should be available once the script is loaded
			# Use a try-catch approach since ClassDB doesn't work for GDScript classes
			var error_occurred = false
			if not error_occurred:
				# Try calling directly - if class_name is registered, this will work
				project_summary = CodebaseScanner.get_project_summary("res://")
				relevant_files = CodebaseScanner.get_relevant_files(user_prompt, "res://", 12)
			
			var context_prompt = """

═══════════════════════════════════════════════════════════════
PROJECT CONTEXT - FULL CODEBASE AWARENESS
═══════════════════════════════════════════════════════════════

""" + project_summary
			
			if relevant_files.size() > 0:
				context_prompt += "\n\n═══════════════════════════════════════════════════════════════\n"
				context_prompt += "RELEVANT FILES (with dependencies and relationships):\n"
				context_prompt += "═══════════════════════════════════════════════════════════════\n\n"
				
				for i in range(relevant_files.size()):
					var file_data = relevant_files[i]
					var file_header = "[" + str(i + 1) + "] File: " + file_data.path
					
					# Add metadata if available
					if file_data.has("class_name") and file_data.class_name != "":
						file_header += " (class: " + file_data.class_name + ")"
					if file_data.has("extends") and file_data.extends != "":
						file_header += " extends " + file_data.extends
					if file_data.has("relation"):
						file_header += " [" + file_data.relation + "]"
					
					context_prompt += file_header + "\n"
					context_prompt += "-" * 60 + "\n"
					
					# Include full content for smaller files, truncated for larger ones
					var content = file_data.content
					var max_length = 3000
					if content.length() > max_length:
						# Try to keep important parts (class definition, key functions)
						var first_part = content.substr(0, max_length / 2)
						var last_part = content.substr(content.length() - max_length / 2)
						content = first_part + "\n\n... [middle section truncated for brevity] ...\n\n" + last_part
					
					context_prompt += "```gdscript\n" + content + "\n```\n\n"
			
			context_prompt += """
═══════════════════════════════════════════════════════════════
INSTRUCTIONS FOR CODE GENERATION:
═══════════════════════════════════════════════════════════════

1. ANALYZE the existing code patterns, naming conventions, and architecture
2. MAINTAIN consistency with the project's coding style
3. REFERENCE existing classes and functions when appropriate
4. FOLLOW the project's file organization structure
5. CONSIDER dependencies - if modifying a file, check what depends on it
6. CREATE complete, production-ready code that integrates seamlessly

When creating or modifying files:
- Use the exact same style as existing files
- Match indentation and formatting
- Follow the same patterns for signals, exports, and function organization
- If extending a class, ensure compatibility with existing code
- Add appropriate comments matching the project's comment style

"""
			
			system_prompt += context_prompt
		else:
			# Codebase scanning not available, but continue without it
			pass
	
	system_prompt += """

═══════════════════════════════════════════════════════════════
📝 OUTPUT FORMAT REQUIREMENTS (ABSOLUTELY CRITICAL):
═══════════════════════════════════════════════════════════════

🚨 YOU MUST CREATE COMPLETE GAMES, NOT CODE SNIPPETS! 🚨

FORMAT STRUCTURE (MANDATORY):

# File: scenes/main.tscn
[COMPLETE scene file with all nodes, scripts, and connections]

# File: scripts/player/player.gd
[COMPLETE script - fully functional, no placeholders]

# File: scripts/enemies/enemy.gd
[COMPLETE script - fully functional]

# File: scripts/managers/game_manager.gd
[COMPLETE script - fully functional]

... (ALL files needed for a complete, playable game)

🎯 CRITICAL RULES:

1. **ALWAYS USE FILE MARKERS**: Every file MUST start with "# File: path/to/file.ext"
   - Without file markers, files CANNOT be created automatically
   - Paths are relative to res:// (e.g., "scripts/player.gd" NOT "res://scripts/player.gd")

2. **CREATE ALL FILES IN ONE RESPONSE**:
   - Main scene (.tscn)
   - All scripts (.gd)
   - Resources (.tres) if needed
   - UI scenes if needed
   - Configuration files if needed
   - DO NOT split across multiple responses

3. **COMPLETE CODE ONLY**:
   - NO placeholders like "// TODO" or "// Add code here"
   - NO incomplete functions
   - NO missing implementations
   - EVERYTHING must be fully functional

4. **PROPER PROJECT STRUCTURE**:
   - scripts/player/ - Player scripts
   - scripts/enemies/ - Enemy scripts
   - scripts/ui/ - UI scripts
   - scripts/managers/ - Game managers
   - scenes/ - All scene files
   - resources/ - Resources if needed

5. **SCENE FILES MUST BE COMPLETE**:
   - Proper .tscn format
   - All nodes defined
   - Scripts attached
   - Properties configured
   - Scene is ready to run

6. **SCRIPTS MUST BE COMPLETE**:
   - All functions implemented
   - Type hints included
   - Error handling added
   - Signals declared and used
   - Ready to use immediately

CRITICAL RULES FOR PRODUCTION-READY CODE:
1. Each file marker MUST be on its own line: "# File: path/to/file.gd"
2. Paths are relative to res:// (e.g., "scripts/player.gd" NOT "res://scripts/player.gd")
3. Include COMPLETE, PRODUCTION-READY code - fully functional, optimized, documented
4. If the request requires multiple files, create ALL of them in the same response
5. If modifying existing files, show the COMPLETE modified file content
6. Maintain consistency with existing project patterns, style, and architecture
7. Use advanced GDScript: type hints, proper error handling, performance optimizations
8. Create professional directory structure: "scripts/", "scenes/", "resources/", "autoloads/"
9. Ensure all files work together seamlessly - verify dependencies, imports, references
10. NO explanations between files - just file markers and code (explanations go before code blocks)
11. Code will be automatically saved - make it production-ready with best practices
12. Use design patterns where appropriate: State Machines, Singletons, Managers, Components
13. Include comprehensive error handling: null checks, validation, graceful failures
14. Optimize for performance: cache nodes, use appropriate process functions, avoid redundant calculations
15. Add meaningful comments: explain WHY (architecture decisions), not just WHAT (code does)
16. Use proper Godot conventions: snake_case, signal declarations, @export for editor properties

📚 COMPLETE GAME EXAMPLE:

User: "Create a 2D side scroller"

You MUST create ALL of these files:

# File: scenes/main.tscn
[gd_scene load_steps=4 format=3 uid="uid://main_scene"]

[ext_resource type="PackedScene" path="res://scenes/player.tscn" id="1"]
[ext_resource type="PackedScene" path="res://scenes/enemy.tscn" id="2"]
[ext_resource type="Script" path="res://scripts/managers/game_manager.gd" id="3"]

[node name="Main" type="Node2D"]
script = ExtResource("3")

[node name="Player" parent="." instance=ExtResource("1")]
position = Vector2(100, 300)

[node name="Enemy" parent="." instance=ExtResource("2")]
position = Vector2(500, 300)

[node name="Camera2D" type="Camera2D" parent="Player"]
enabled = true

[node name="UI" type="Control" parent="."]
layout_mode = 3
anchors_preset = 15
anchor_right = 1.0
anchor_bottom = 1.0

[node name="ScoreLabel" type="Label" parent="UI"]
layout_mode = 1
anchors_preset = 2
anchor_top = 1.0
anchor_bottom = 1.0
offset_left = 20.0
offset_top = -40.0
offset_right = 200.0
offset_bottom = -10.0
text = "Score: 0"

# File: scripts/player/player.gd
extends CharacterBody2D

@export var speed: float = 300.0
@export var jump_velocity: float = -400.0

var gravity: float = ProjectSettings.get_setting("physics/2d/default_gravity")

func _physics_process(delta: float) -> void:
	# Apply gravity
	if not is_on_floor():
		velocity.y += gravity * delta
	
	# Handle jump
	if Input.is_action_just_pressed("ui_accept") and is_on_floor():
		velocity.y = jump_velocity
	
	# Handle horizontal movement
	var direction = Input.get_axis("ui_left", "ui_right")
	if direction:
		velocity.x = direction * speed
	else:
		velocity.x = move_toward(velocity.x, 0, speed)
	
	move_and_slide()

# File: scenes/player.tscn
[gd_scene load_steps=3 format=3 uid="uid://player"]

[ext_resource type="Script" path="res://scripts/player/player.gd" id="1"]

[sub_resource type="RectangleShape2D" id="1"]
size = Vector2(32, 48)

[node name="Player" type="CharacterBody2D"]
script = ExtResource("1")

[node name="Sprite2D" type="Sprite2D" parent="."]

[node name="CollisionShape2D" type="CollisionShape2D" parent="."]
shape = SubResource("1")

# File: scripts/enemies/enemy.gd
extends CharacterBody2D

@export var speed: float = 100.0
@export var health: int = 100

var direction: int = -1

func _physics_process(delta: float) -> void:
	velocity.x = direction * speed
	
	# Simple collision detection
	if is_on_wall():
		direction *= -1
	
	move_and_slide()

func take_damage(amount: int) -> void:
	health -= amount
	if health <= 0:
		queue_free()

# File: scenes/enemy.tscn
[gd_scene load_steps=3 format=3 uid="uid://enemy"]

[ext_resource type="Script" path="res://scripts/enemies/enemy.gd" id="1"]

[sub_resource type="RectangleShape2D" id="1"]
size = Vector2(32, 32)

[node name="Enemy" type="CharacterBody2D"]
script = ExtResource("1")

[node name="Sprite2D" type="Sprite2D" parent="."]

[node name="CollisionShape2D" type="CollisionShape2D" parent="."]
shape = SubResource("1")

# File: scripts/managers/game_manager.gd
extends Node2D

@onready var score_label: Label = $UI/ScoreLabel

var score: int = 0

func _ready() -> void:
	update_score_display()

func add_score(points: int) -> void:
	score += points
	update_score_display()

func update_score_display() -> void:
	if score_label:
		score_label.text = "Score: " + str(score)

🎯 REMEMBER: This is a COMPLETE, PLAYABLE GAME. The user can run it immediately!

═══════════════════════════════════════════════════════════════"""

	return system_prompt

func _build_project_summary_from_graph() -> String:
	# Project summary is now built by CodebaseScanner
	return ""

func _build_relevant_files_from_context(context: Dictionary) -> Array:
	var files = []
	
	# Add target file
	if context.has("target") and context.target != "":
		files.append({"path": context.target, "relation": "target"})
	
	# Add dependencies
	if context.has("dependencies"):
		for dep in context.dependencies:
			files.append({"path": dep, "relation": "dependency"})
	
	# Add dependents
	if context.has("dependents"):
		for dep in context.dependents:
			files.append({"path": dep, "relation": "dependent"})
	
	return files

func _on_conversation_updated() -> void:
	conversation_updated.emit()

func _on_ai_question(question: String) -> void:
	ai_question.emit(question)

## Get conversation history
func get_conversation_history() -> Array:
	if conversation_manager != null:
		return conversation_manager.get_formatted_conversation()
	return []

## Clear conversation
func clear_conversation() -> void:
	if conversation_manager != null:
		conversation_manager.clear_history()

## Start new conversation
func start_new_conversation() -> void:
	if conversation_manager != null:
		conversation_manager.start_conversation()

func _on_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray):
	print("DotAI: HTTP request completed. Result: ", result, ", Response code: ", response_code)
	
	if result != HTTPRequest.RESULT_SUCCESS:
		print("DotAI: HTTP request failed with result: ", result)
		request_error.emit("HTTP request failed: " + str(result))
		return
	
	if response_code != 200:
		var error_msg = "API request failed with code: " + str(response_code)
		var body_text = body.get_string_from_utf8()
		if body_text:
			error_msg += "\n" + body_text
		print("DotAI: API error response: ", body_text)
		request_error.emit(error_msg)
		return
	
	var json = JSON.new()
	var parse_error = json.parse(body.get_string_from_utf8())
	if parse_error != OK:
		request_error.emit("Failed to parse JSON response: " + str(parse_error))
		return
	
	var response_data = json.data
	if not response_data.has("content"):
		request_error.emit("Invalid API response format")
		return
	
	var content = response_data["content"]
	if content is Array and content.size() > 0:
		var first_content = content[0]
		if first_content.has("text"):
			var response_text = first_content["text"]
			
			# Add AI response to conversation
			if conversation_manager != null:
				conversation_manager.add_message("assistant", response_text)
			
			# Check if AI is asking a question
			if conversation_manager != null and conversation_manager.detect_ai_question(response_text):
				var question = conversation_manager.extract_question(response_text)
				if question != "":
					ai_question.emit(question)
			
			# Emit full response for conversation display (keep full text, don't extract code)
			request_complete.emit(response_text)
		else:
			request_error.emit("Response content missing text field")
	else:
		request_error.emit("Empty response content")

func _extract_code(text: String) -> String:
	# Extract code from markdown code blocks if present
	var regex = RegEx.new()
	regex.compile("```(?:gdscript)?\\s*\\n([\\s\\S]*?)```")
	var result = regex.search(text)
	if result:
		return result.get_string(1).strip_edges()
	
	# Also check for inline code blocks
	regex.compile("`([^`]+)`")
	result = regex.search(text)
	if result:
		return result.get_string(1).strip_edges()
	
	# If no code blocks found, return the whole text
	return text.strip_edges()

## Write files to codebase (called from C++ dock)
func write_files_to_codebase(params: Dictionary) -> Dictionary:
	var response_text: String = params.get("response_text", "")
	
	print("DotAI API Handler: write_files_to_codebase called")
	print("DotAI API Handler: Response text length: ", response_text.length())
	
	if response_text.is_empty():
		print("DotAI API Handler: ERROR - No response text provided")
		return {"success": false, "error": "No response text provided"}
	
	if _file_writer_script == null:
		print("DotAI API Handler: ERROR - file_writer.gd script not found")
		return {"success": false, "error": "file_writer.gd script not found. Please ensure codebase_scanner.gd and file_writer.gd are in res://addons/claude_ai/"}
	
	print("DotAI API Handler: Calling FileWriter.parse_and_write_files...")
	# Use FileWriter class directly (call static method on class name)
	# The class should be available once the script is loaded
	# Try calling directly - if class_name is registered, this will work
	var result = FileWriter.parse_and_write_files(response_text, "res://")
	
	print("DotAI API Handler: FileWriter returned:")
	print("  - Success: ", result.get("success", false))
	print("  - Files written: ", result.get("files_written", []).size())
	print("  - Files created: ", result.get("files_created", []).size())
	print("  - Files failed: ", result.get("files_failed", []).size())
	if result.has("error"):
		print("  - Error: ", result.error)
	
	# Return comprehensive result
	var return_dict = {
		"success": result.files_failed.size() == 0,
		"files_written": result.files_written,
		"files_failed": result.files_failed,
		"messages": result.messages,
		"error": "" if result.files_failed.size() == 0 else "Some files failed to write"
	}
	
	# Add created/modified info if available
	if result.has("files_created"):
		return_dict["files_created"] = result.files_created
	if result.has("files_modified"):
		return_dict["files_modified"] = result.files_modified
	
	# Refresh editor file system after writing
	if Engine.is_editor_hint():
		call_deferred("_refresh_file_system")
	
	return return_dict

func _refresh_file_system():
	if Engine.is_editor_hint():
		print("DotAI API Handler: Refreshing editor file system...")
		EditorFileSystem.get_singleton().scan_changes()
		print("DotAI API Handler: File system refresh initiated")