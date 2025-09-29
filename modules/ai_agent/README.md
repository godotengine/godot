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

## 💬 Interactive Chat Interface Setup

For the ultimate Godot development assistant experience:

### 1. Create Chat UI Scene

```gdscript
# Create a scene with this structure:
ChatInterface (Control)
└─ AIAgent (AIAgent node)
└─ VBoxContainer
   ├─ ScrollContainer
   │  └─ ChatOutput (VBoxContainer)
   ├─ ChatInput (LineEdit)
   └─ HBoxContainer
      ├─ SendButton (Button)
      └─ ClearButton (Button)
```

### 2. Attach Chat Script

1. **Select ChatInterface node**
2. **Attach script** → Use `chat_interface_example.gd`
3. **Set API key** → Replace `"YOUR_GEMINI_API_KEY_HERE"`
4. **Run scene** → Start chatting!

### 3. Try These Commands

- 👄 **"hello"** → Get started
- 📁 **"create project"** → Project setup guide
- 🎭 **"add CharacterBody2D"** → Add player node
- ⚙️ **"project settings"** → Configuration help
- 📝 **"attach script"** → Script creation
- 𫒫 **"help"** → All available commands

**The assistant adapts to your skill level and remembers conversation context!** 😺

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

See the `examples/` directory for complete examples:

### 🎮 Interactive Chat Interface
- `godot_chat_assistant.ts`: **Advanced chat AI** for Godot development
- `chat_interface_example.gd`: **Complete GDScript implementation**
- `message_container.gd`: **UI helper for chat messages**

**Features:**
- 🐾 **Friendly cat-themed assistant** with personality
- 💬 **Natural conversation** with context awareness  
- 🎯 **Smart topic detection** (projects, nodes, scripts, settings)
- 📚 **Step-by-step guides** for common Godot tasks
- 🧠 **Skill level adaptation** (beginner/intermediate/advanced)
- ⚡ **Interactive help system** with commands

**Chat Commands:**
- "create project" → Project setup guide
- "add node" → Node creation tutorial
- "project settings" → Configuration help
- "help" → Full command list

### 🤖 Basic Examples
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

## Troubleshooting

### 🔑 API Key Issues

**Problem**: "Invalid API key" or "Authentication failed"
- ✅ **Solution**: Verify your Gemini API key is correct
- ✅ **Check**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey) to generate/verify key
- ✅ **Test**: Try the key in a simple HTTP request first

```gdscript
# Test your API key
ai_agent.set_gemini_api_key("your-actual-api-key-here")
ai_agent.send_message("Hello, test connection")
```

### 🔧 TypeScript Compilation Errors

**Problem**: "TypeScript compiler not found" or compilation fails
- ✅ **Install TypeScript**: `npm install -g typescript`
- ✅ **Verify installation**: `tsc --version`
- ✅ **Check path**: Ensure TypeScript is in your system PATH

**Problem**: "Cannot find module" errors in TypeScript
- ✅ **Solution**: Check `tsconfig.json` configuration
- ✅ **Verify**: Ensure all imports are correct

### 🌐 Network Connection Issues

**Problem**: "Connection timeout" or "Network unreachable"
- ✅ **Check internet**: Verify internet connection works
- ✅ **Firewall**: Check if firewall blocks Godot
- ✅ **Proxy**: Configure proxy settings if needed

```gdscript
# Enable debug output for network issues
ai_agent.set_debug_mode(true)
```

### 🏗️ Build Configuration Issues

**Problem**: Module not found during build
- ✅ **Location**: Ensure module is in `modules/ai_agent/`
- ✅ **SCsub**: Verify `SCsub` file exists and is correct
- ✅ **Clean build**: `scons --clean` then rebuild

**Problem**: Compilation errors during build
- ✅ **Dependencies**: Check C++ compiler version
- ✅ **Headers**: Verify all header files are present
- ✅ **Platform**: Use correct platform flags for your OS

```bash
# Clean and rebuild with verbose output
scons --clean
scons platform=windows target=editor verbose=yes
```

### 🐛 Runtime Errors

**Problem**: AI Agent node crashes or freezes
- ✅ **Memory**: Check available system memory
- ✅ **API limits**: Verify you haven't exceeded API quotas
- ✅ **Logs**: Check Godot output and error logs

**Problem**: TypeScript execution fails
- ✅ **Syntax**: Verify TypeScript syntax is correct
- ✅ **Context**: Ensure proper context is passed to scripts
- ✅ **Permissions**: Check file system permissions

### 📊 Performance Issues

**Problem**: Slow AI responses
- ✅ **Model**: Try different Gemini models (Flash for speed)
- ✅ **Context**: Reduce conversation history length
- ✅ **Threading**: Ensure requests aren't blocking main thread

```gdscript
# Optimize for speed
ai_agent.set_model("gemini-1.5-flash")
ai_agent.set_max_conversation_turns(5)  # Limit history
```

### ⚡ Quick Diagnostic

Run this diagnostic script to check common issues:

```gdscript
func diagnostic_check():
    print("=== AI Agent Diagnostics ===")
    
    # Check if module is loaded
    if Engine.has_singleton("AIAgent"):
        print("✅ AI Agent module loaded")
    else:
        print("❌ AI Agent module not found")
        return
    
    # Check API key
    var agent = AIAgent.new()
    if agent.has_api_key():
        print("✅ API key configured")
    else:
        print("❌ API key missing")
    
    # Check TypeScript
    if agent.is_typescript_available():
        print("✅ TypeScript available")
    else:
        print("⚠️ TypeScript not available")
    
    # Check network
    agent.connect("response_received", func(response): print("✅ Network connection working"))
    agent.connect("error_occurred", func(error): print("❌ Network error: " + error))
    agent.send_message("Test connection")
```

### 🆘 Getting Help

If you're still having issues:

1. **Check logs**: Look at Godot's output for error messages
2. **Minimal test**: Create a simple scene with just an AIAgent node
3. **Report bugs**: Include error messages, OS, and Godot version
4. **Community**: Ask on Godot forums or Discord

## License

This module follows the same MIT license as Godot Engine.

## Contributing

Contributions are welcome! Please ensure your code follows Godot's coding standards and includes appropriate tests.
