# DotAI - AI-Powered Game Development Platform

**DotAI** is an AI-native game engine built on Godot Engine, enabling developers to create complete games through natural language conversations with Claude AI. Think of it as "Cursor for Game Engines" - an intelligent development environment where AI is a first-class collaborator.

## 🎮 What is DotAI?

DotAI transforms Godot Engine into an AI-powered game development platform where you can:

- **Build games by describing them** - "Create a 2D platformer with jumping and enemies"
- **Generate complete, production-ready code** - Not just snippets, but full game systems
- **Automatic file creation** - AI generates and saves files directly to your project
- **Codebase awareness** - AI understands your entire project structure and context
- **Conversational development** - Have a dialogue with AI about your game design

## ✨ Key Features

### 🤖 AI-Powered Code Generation
- Natural language to GDScript conversion
- Multi-file generation (scripts, scenes, resources)
- Production-ready code with proper structure
- Automatic file saving and project integration

### 📚 Codebase Awareness
- Full project scanning and indexing
- Context-aware code generation
- Dependency tracking
- Style consistency with existing code

### 💬 Conversational Interface
- Multi-turn conversations with AI
- Context memory across requests
- AI-initiated questions for clarification
- Cursor-like chat interface

### 🚀 Automatic Project Setup
- Zero-configuration installation
- Automatic file copying to new projects
- Seamless integration with Godot Editor

## 📋 Requirements

- **Godot Engine 4.x** (source code)
- **Claude API Key** from Anthropic
- **Python 3.x** (for SCons build system)
- **C++ Compiler** (MSVC on Windows, GCC/Clang on Linux/macOS)
- **SCons** build system (`pip install scons`)

## 🛠️ Building DotAI

### Windows

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/dotai.git
   cd dotai
   ```

2. **Install dependencies:**
   ```bash
   pip install scons
   ```

3. **Build the editor:**
   ```bash
   scons platform=windows target=editor -j8
   ```
   
   The `-j8` flag uses 8 parallel jobs (adjust based on your CPU cores).

4. **Run the editor:**
   ```bash
   bin\godot.windows.editor.x86_64.exe
   ```

### Linux

```bash
scons platform=linuxbsd target=editor -j8
bin/godot.linuxbsd.editor.x86_64
```

### macOS

```bash
scons platform=macos target=editor -j8
bin/godot.macos.editor.universal
```

## 🚀 Quick Start

### 1. Build DotAI

Follow the build instructions above for your platform.

### 2. Open/Create a Godot Project

- Open an existing project or create a new one
- DotAI will automatically set up required files on first launch

### 3. Access DotAI Panel

- In the Godot Editor, look for the **"DotAI"** dock panel (usually on the right side)
- If not visible, go to **Editor → Editor Layout → Default** or check **View → Docks**

### 4. Configure API Key

**API key is required!** Enter your Claude API key in the DotAI panel:
- Open the DotAI dock panel
- Enter your API key in the "API Key" field
- Get your API key from: https://console.anthropic.com/

### 5. Start Building!

Type a prompt like:
```
Create a 2D platformer game with a player that can jump and move left/right
```

The AI will:
1. Generate complete, production-ready code
2. Create all necessary files (scripts, scenes, resources)
3. Automatically save them to your project
4. Display the generated code in the conversation

## 📁 Project Structure

```
modules/claude_ai/
├── editor/                          # C++ Editor Plugin
│   ├── claude_ai_editor_plugin.cpp # Main plugin implementation
│   ├── claude_ai_editor_plugin.h   # Plugin header
│   └── ai_studio_main_ui.*         # UI components
├── claude_api_handler.gd           # API communication handler
├── codebase_scanner.gd              # Project scanning and indexing
├── file_writer.gd                   # File creation and writing
├── conversation_manager.gd          # Conversation history management
├── register_types.cpp               # Module registration
├── register_types.h
├── application_config.h             # Product configuration
├── SCsub                            # SCons build script
└── README.md                        # This file
```

## 🎯 How It Works

### Architecture Overview

1. **C++ Editor Plugin** (`claude_ai_editor_plugin.cpp`)
   - Integrates DotAI into Godot Editor
   - Provides the UI dock panel
   - Handles file system operations
   - Manages GDScript script loading

2. **API Handler** (`claude_api_handler.gd`)
   - Communicates with Claude AI API
   - Builds prompts with codebase context
   - Processes AI responses
   - Manages conversation history

3. **Codebase Scanner** (`codebase_scanner.gd`)
   - Scans project files
   - Extracts code structure and dependencies
   - Provides context to AI prompts

4. **File Writer** (`file_writer.gd`)
   - Parses AI responses for code blocks
   - Extracts file paths and content
   - Creates directories as needed
   - Writes files to project

5. **Conversation Manager** (`conversation_manager.gd`)
   - Maintains conversation history
   - Manages context across multiple turns
   - Detects AI questions

### Workflow

```
User Prompt → API Handler → Claude AI API
                ↓
         AI Response (code + explanation)
                ↓
         File Writer (parse & extract)
                ↓
         File System (create files)
                ↓
         Editor Refresh (show in project)
```

## 🔧 Configuration

### API Key

**API key is required!** You must enter your Claude API key in the DotAI panel UI. The API key is not stored in code for security reasons.

To get your API key:
1. Visit https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy and paste it into the DotAI panel's "API Key" field

### Model Selection

Change the model in `claude_api_handler.gd`:

```gdscript
const MODEL = "claude-3-5-sonnet-20240620"  # or another Claude model
```

### Auto-Save

Auto-save is enabled by default. To disable:
- Uncheck the "Auto-save" checkbox in the DotAI panel
- Or modify `auto_save_enabled = true` in `claude_ai_editor_plugin.cpp`

## 📝 Usage Examples

### Example 1: Create a Player Character

**Prompt:**
```
Create a 2D player character that can move left/right with arrow keys and jump with spacebar
```

**Result:**
- `scripts/player.gd` - Player movement script
- `scenes/player.tscn` - Player scene with sprite and collision
- Complete, production-ready code with proper physics

### Example 2: Build a Complete Game System

**Prompt:**
```
Create a complete inventory system with items, UI display, and drag-and-drop functionality
```

**Result:**
- `scripts/inventory/inventory.gd` - Core inventory logic
- `scripts/inventory/item.gd` - Item data structure
- `scripts/ui/inventory_ui.gd` - UI controller
- `scenes/ui/inventory_panel.tscn` - UI scene
- All files properly connected and ready to use

### Example 3: Add Features to Existing Code

**Prompt:**
```
Add a health system to the player with a health bar UI
```

**Result:**
- AI analyzes existing player code
- Generates health system that integrates seamlessly
- Creates health bar UI component
- Updates existing files appropriately

## 🐛 Troubleshooting

### "API handler script not found"

**Solution:** The GDScript files need to be in your project. DotAI automatically copies them to `res://addons/claude_ai/` on first launch. If this fails:

1. Manually create `res://addons/claude_ai/` directory
2. Copy these files from `modules/claude_ai/`:
   - `claude_api_handler.gd`
   - `codebase_scanner.gd`
   - `file_writer.gd`
   - `conversation_manager.gd`

### "No code to save" error

**Solution:** 
1. Check the Output panel (View → Output) for debug messages
2. Ensure your prompt includes code generation requests
3. Try being more specific: "Create a script that..." instead of just "How do I..."

### Build Errors

**Common issues:**
- **Missing SCons:** `pip install scons`
- **Wrong directory:** Run build from Godot source root (where `SConstruct` is)
- **Compiler not found:** Install Visual Studio Build Tools (Windows) or build-essential (Linux)

### Files Not Appearing

**Solution:**
1. Check Output panel for file writing errors
2. Ensure project directory is writable
3. Try manually refreshing: **File → Reimport** or **File → Scan**

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is built on Godot Engine, which is licensed under the MIT License. DotAI-specific code follows the same license.

## 🙏 Acknowledgments

- **Godot Engine** - The amazing open-source game engine
- **Anthropic** - Claude AI API
- **Godot Community** - For inspiration and support

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/dotai/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/dotai/discussions)

## 🗺️ Roadmap

- [ ] Enhanced codebase understanding with dependency graphs
- [ ] Scene file generation (.tscn) support
- [ ] Resource file generation (.tres) support
- [ ] Multi-model support (GPT-4, etc.)
- [ ] Offline mode with local models
- [ ] Code refactoring and optimization suggestions
- [ ] Visual scripting integration
- [ ] Template library for common game patterns

---

**Built with ❤️ for game developers who want to focus on creativity, not syntax.**
