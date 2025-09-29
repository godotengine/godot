# AI Agent Module for Godot Engine

A comprehensive AI agent system for Godot Engine with Google Gemini API integration and TypeScript support.

## Features

- 🤖 **AI Agent Node**: Fully integrated Godot node with AI capabilities
- 🌟 **Google Gemini API**: Direct integration with Google's Gemini AI models
- 📝 **TypeScript Support**: Execute TypeScript code for dynamic AI behaviors  
- 💬 **Conversation Management**: Persistent conversation history and context
- 🧠 **Memory System**: AI agent memory for context retention
- 🎯 **Action System**: Extensible action framework for AI interactions
- 🔧 **Editor Integration**: Built-in editor tools for AI agent development
- 📚 **TypeScript Definitions**: Full type definitions for Godot API

## Quick Start

### 1. Setup API Key

The AI Agent requires a Google Gemini API key. You can obtain one from the Google AI Studio.

```gdscript
# In your scene script
extends Node

@onready var ai_agent = $AIAgent

func _ready():
    # Set your Gemini API key (hardcoded as requested)
    ai_agent.set_gemini_api_key("YOUR_GEMINI_API_KEY_HERE")
    ai_agent.set_agent_name("My Assistant")
    ai_agent.set_agent_role("Helpful game development assistant")
    
    # Connect to AI responses
    ai_agent.connect("response_received", _on_ai_response)
    ai_agent.connect("error_occurred", _on_ai_error)

func _on_ai_response(response: String):
    print("AI said: ", response)

func _on_ai_error(error: String):
    print("AI error: ", error)

# Send a message to the AI
func ask_ai_something():
    ai_agent.send_message("Help me create a player character script")
```

### 2. Using TypeScript AI Scripts

The AI Agent can execute TypeScript code for advanced behaviors:

```typescript
// example_agent.ts
class GameAssistant {
    processUserQuery(query: string, context: any): string {
        if (query.includes("player")) {
            return this.generatePlayerScript();
        }
        if (query.includes("enemy")) {
            return this.generateEnemyScript();
        }
        return "I can help you with player scripts, enemy AI, and more!";
    }
    
    private generatePlayerScript(): string {
        return `
extends CharacterBody2D

@export var speed = 300.0
@export var jump_velocity = -400.0

func _physics_process(delta):
    if not is_on_floor():
        velocity.y += get_gravity() * delta
    
    if Input.is_action_just_pressed("ui_accept") and is_on_floor():
        velocity.y = jump_velocity
    
    var direction = Input.get_axis("ui_left", "ui_right")
    velocity.x = direction * speed
    
    move_and_slide()
        `;
    }
}
```

Load and execute TypeScript:

```gdscript
# Load TypeScript AI behavior
ai_agent.load_typescript_file("res://ai_scripts/game_assistant.ts")

# Or execute TypeScript code directly
var ts_code = """
console.log("AI Agent is thinking...");
// TypeScript logic here
"""
ai_agent.execute_typescript_code(ts_code)
```

### 3. Memory and Actions

The AI Agent has built-in memory and action systems:

```gdscript
# Store information in AI memory
ai_agent.remember("player_level", 5)
ai_agent.remember("game_mode", "adventure")

# Retrieve from memory
var level = ai_agent.recall("player_level")

# Add custom actions
ai_agent.add_action("spawn_enemy", spawn_enemy_callback, "Spawns an enemy at a location")

func spawn_enemy_callback():
    print("Spawning enemy as requested by AI")
    # Your enemy spawning logic here
```

## Module Structure

```
modules/ai_agent/
├── ai_agent.h/cpp              # Main AI Agent node
├── gemini_client.h/cpp         # Google Gemini API client
├── ai_conversation.h/cpp       # Conversation management
├── typescript/
│   └── typescript_runner.h/cpp # TypeScript execution engine
├── editor/
│   └── ai_agent_editor_plugin.h/cpp # Editor integration
├── examples/
│   ├── basic_agent.ts         # Example TypeScript AI
│   └── tsconfig.json          # TypeScript configuration
├── config.py                  # Module configuration
├── SCsub                      # Build script
└── register_types.h/cpp       # Module registration
```

## Classes Overview

### AIAgent
- Main node class that orchestrates all AI functionality
- Handles Gemini API communication, TypeScript execution, and state management
- Provides signals for AI responses and state changes

### GeminiClient  
- HTTP client for Google Gemini API
- Supports all Gemini models (Pro, Pro Vision, Flash)
- Handles conversation history and context management

### AIConversation
- Manages conversation history and message persistence
- Supports role-based messages (user, assistant, system)
- Provides search and filtering capabilities

### TypeScriptRunner
- Compiles and executes TypeScript code
- Provides Godot API type definitions
- Manages execution context and temporary files

## Building

The module is automatically included in Godot builds when present in the modules directory. To build Godot with the AI Agent module:

```bash
scons platform=windows target=editor
```

Make sure you have TypeScript and Node.js installed for full functionality:

```bash
npm install -g typescript
node --version
tsc --version
```

## Configuration

The module supports several build options:

```bash
# Build with TypeScript support (default: yes)
scons builtin_typescript=yes

# Specify custom TypeScript compiler path
scons typescript_path="C:/path/to/tsc.exe"
```

## Examples

See the `examples/` directory for complete TypeScript examples:
- `basic_agent.ts`: Simple AI assistant with code generation
- `tsconfig.json`: TypeScript configuration file

## Requirements

- **Godot Engine 4.x**
- **Google Gemini API key**
- **TypeScript** (optional, for TypeScript features)
- **Node.js** (optional, for TypeScript execution)

## API Reference

### AIAgent Properties

- `agent_name: String` - Name of the AI agent
- `agent_role: String` - Role/personality of the agent
- `auto_execute: bool` - Automatically execute AI-generated code
- `thinking_delay: float` - Delay before processing responses
- `max_conversation_turns: int` - Maximum conversation history

### AIAgent Methods

- `send_message(message: String)` - Send message to AI
- `execute_typescript_code(code: String)` - Execute TypeScript code
- `remember(key: String, value: Variant)` - Store in memory
- `recall(key: String) -> Variant` - Retrieve from memory
- `add_action(name: String, callback: Callable)` - Add custom action

### AIAgent Signals

- `response_received(response: String)` - AI response ready
- `error_occurred(error: String)` - Error in AI processing
- `state_changed(new_state: int)` - Agent state changed
- `thinking_started()` - AI started processing
- `thinking_finished()` - AI finished processing

## License

This module follows the same MIT license as Godot Engine.

## Contributing

Contributions are welcome! Please ensure your code follows Godot's coding standards and includes appropriate tests.