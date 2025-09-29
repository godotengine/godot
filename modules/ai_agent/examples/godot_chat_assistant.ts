// Godot Development Chat Assistant
// An interactive AI assistant that guides users through Godot development

interface GodotContext {
    agent: any;
    node: any;
    scene: any;
    project: any;
    editor: any;
}

interface ChatMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
}

class GodotChatAssistant {
    private context: GodotContext;
    private name: string;
    private conversationHistory: ChatMessage[];
    private currentTopic: string | null;
    private userSkillLevel: 'beginner' | 'intermediate' | 'advanced';

    constructor(context: GodotContext) {
        this.context = context;
        this.name = "🐾 Godot Cat Assistant";
        this.conversationHistory = [];
        this.currentTopic = null;
        this.userSkillLevel = 'beginner';
        
        this.addSystemMessage("Hello! I'm your friendly Godot development assistant! 🎮 I'm here to help you with everything from creating projects to advanced scripting. What would you like to learn today?");
    }

    async processMessage(message: string): Promise<string> {
        this.addUserMessage(message);
        
        const lowerMessage = message.toLowerCase();
        let response = "";

        // Detect user skill level from conversation
        this.detectSkillLevel(lowerMessage);

        // Route message to appropriate handler
        if (this.isGreeting(lowerMessage)) {
            response = this.handleGreeting();
        }
        else if (this.isProjectRelated(lowerMessage)) {
            response = await this.handleProjectQuestions(lowerMessage);
        }
        else if (this.isSceneRelated(lowerMessage)) {
            response = await this.handleSceneQuestions(lowerMessage);
        }
        else if (this.isNodeRelated(lowerMessage)) {
            response = await this.handleNodeQuestions(lowerMessage);
        }
        else if (this.isScriptRelated(lowerMessage)) {
            response = await this.handleScriptingQuestions(lowerMessage);
        }
        else if (this.isSettingsRelated(lowerMessage)) {
            response = await this.handleSettingsQuestions(lowerMessage);
        }
        else if (this.isUIRelated(lowerMessage)) {
            response = await this.handleUIQuestions(lowerMessage);
        }
        else if (this.isPhysicsRelated(lowerMessage)) {
            response = await this.handlePhysicsQuestions(lowerMessage);
        }
        else if (this.isAnimationRelated(lowerMessage)) {
            response = await this.handleAnimationQuestions(lowerMessage);
        }
        else if (this.isAudioRelated(lowerMessage)) {
            response = await this.handleAudioQuestions(lowerMessage);
        }
        else if (this.isDebuggingRelated(lowerMessage)) {
            response = await this.handleDebuggingQuestions(lowerMessage);
        }
        else if (this.isHelpRequest(lowerMessage)) {
            response = this.getHelpMenu();
        }
        else {
            response = await this.handleGeneralQuestion(lowerMessage);
        }

        this.addAssistantMessage(response);
        return response;
    }

    // Detection methods
    private isGreeting(message: string): boolean {
        return /\b(hello|hi|hey|good morning|good afternoon|good evening)\b/.test(message);
    }

    private isProjectRelated(message: string): boolean {
        return /\b(project|create|new|setup|organize|folder|import|export)\b/.test(message);
    }

    private isSceneRelated(message: string): boolean {
        return /\b(scene|level|world|environment|background|camera|lighting)\b/.test(message);
    }

    private isNodeRelated(message: string): boolean {
        return /\b(node|add|create|delete|parent|child|tree|hierarchy)\b/.test(message);
    }

    private isScriptRelated(message: string): boolean {
        return /\b(script|code|gdscript|function|variable|class|method|programming)\b/.test(message);
    }

    private isSettingsRelated(message: string): boolean {
        return /\b(setting|settings|config|configuration|project settings|editor settings|preferences)\b/.test(message);
    }

    private isUIRelated(message: string): boolean {
        return /\b(ui|interface|button|label|menu|hud|gui|control|canvas)\b/.test(message);
    }

    private isPhysicsRelated(message: string): boolean {
        return /\b(physics|collision|rigidbody|kinematic|area|gravity|movement|player)\b/.test(message);
    }

    private isAnimationRelated(message: string): boolean {
        return /\b(animation|animate|tween|timeline|keyframe|sprite|move|rotate)\b/.test(message);
    }

    private isAudioRelated(message: string): boolean {
        return /\b(audio|sound|music|sfx|volume|3d audio|stream)\b/.test(message);
    }

    private isDebuggingRelated(message: string): boolean {
        return /\b(debug|error|bug|fix|crash|problem|issue|console|print)\b/.test(message);
    }

    private isHelpRequest(message: string): boolean {
        return /\b(help|what can you do|commands|guide|tutorial|learn)\b/.test(message);
    }

    // Handler methods
    private handleGreeting(): string {
        const greetings = [
            "Hello there! 🎮 Ready to dive into some Godot development?",
            "Hey! 😸 I'm excited to help you build amazing games in Godot!",
            "Hi! 🚀 What Godot adventure shall we embark on today?",
            "Greetings, game developer! 🎯 Let's create something awesome!"
        ];
        
        const greeting = greetings[Math.floor(Math.random() * greetings.length)];
        return `${greeting}\n\n${this.getQuickStartMenu()}`;
    }

    private async handleProjectQuestions(message: string): Promise<string> {
        this.currentTopic = 'project';

        if (message.includes('create') || message.includes('new')) {
            return this.getProjectCreationGuide();
        }
        else if (message.includes('setup') || message.includes('organize')) {
            return this.getProjectSetupGuide();
        }
        else if (message.includes('import') || message.includes('asset')) {
            return this.getAssetImportGuide();
        }
        else if (message.includes('export') || message.includes('build')) {
            return this.getExportGuide();
        }
        else if (message.includes('setting')) {
            return this.getProjectSettingsGuide();
        }
        
        return this.getGeneralProjectHelp();
    }

    private async handleNodeQuestions(message: string): Promise<string> {
        this.currentTopic = 'nodes';

        if (message.includes('add') || message.includes('create')) {
            return this.getNodeCreationGuide();
        }
        else if (message.includes('delete') || message.includes('remove')) {
            return this.getNodeDeletionGuide();
        }
        else if (message.includes('parent') || message.includes('child') || message.includes('hierarchy')) {
            return this.getNodeHierarchyGuide();
        }
        else if (message.includes('type') || message.includes('kind')) {
            return this.getNodeTypesGuide();
        }
        
        return this.getGeneralNodeHelp();
    }

    private async handleScriptingQuestions(message: string): Promise<string> {
        this.currentTopic = 'scripting';

        if (message.includes('attach') || message.includes('create script')) {
            return this.getScriptAttachmentGuide();
        }
        else if (message.includes('variable') || message.includes('property')) {
            return this.getVariableGuide();
        }
        else if (message.includes('function') || message.includes('method')) {
            return this.getFunctionGuide();
        }
        else if (message.includes('signal') || message.includes('connect')) {
            return this.getSignalGuide();
        }
        else if (message.includes('input') || message.includes('keyboard') || message.includes('mouse')) {
            return this.getInputGuide();
        }
        
        return this.getGeneralScriptingHelp();
    }

    private async handleSettingsQuestions(message: string): Promise<string> {
        this.currentTopic = 'settings';

        if (message.includes('project settings') || message.includes('project config')) {
            return this.getProjectSettingsGuide();
        }
        else if (message.includes('input map') || message.includes('controls')) {
            return this.getInputMapGuide();
        }
        else if (message.includes('layer') || message.includes('collision')) {
            return this.getLayerSettingsGuide();
        }
        else if (message.includes('resolution') || message.includes('display')) {
            return this.getDisplaySettingsGuide();
        }
        
        return this.getGeneralSettingsHelp();
    }

    private async handleGeneralQuestion(message: string): Promise<string> {
        // Use context-aware responses based on conversation history
        const recentTopics = this.getRecentTopics();
        
        if (recentTopics.includes('project')) {
            return `Based on our project discussion, here's what I think might help:\n\n${this.getContextualProjectAdvice(message)}`;
        }
        else if (recentTopics.includes('nodes')) {
            return `Since we were talking about nodes, let me suggest:\n\n${this.getContextualNodeAdvice(message)}`;
        }
        
        return this.getGeneralAdvice(message);
    }

    // Guide methods
    private getProjectCreationGuide(): string {
        return `🎮 **Creating a New Project** 🎮

**Step-by-step:**
1. **Open Godot** → Click "New Project" 
2. **Choose Location** → Browse to your desired folder
3. **Project Name** → Give it a meaningful name
4. **Create Folder** → Check this to create a project folder
5. **Click "Create & Edit"**

**📁 Recommended Folder Structure:**
\`\`\`
MyGame/
├── scenes/          # .tscn files
├── scripts/         # .gd files  
├── assets/          # Images, sounds, etc.
│   ├── textures/
│   ├── audio/
│   └── fonts/
├── autoloads/       # Singletons
└── ui/              # UI scenes
\`\`\`

**💡 Pro Tips:**
- Use descriptive names (avoid "Game1", "Test", etc.)
- Choose a dedicated games folder on your system
- Consider version control (Git) from the start

Would you like help with any specific part of project setup? 😸`;
    }

    private getNodeCreationGuide(): string {
        const skillBasedGuide = this.userSkillLevel === 'beginner' 
            ? "Don't worry, it's easier than it looks! 😊"
            : "You've got this! 💪";

        return `🎭 **Adding Nodes Like a Pro** 🎭

${skillBasedGuide}

**Method 1: Scene Panel**
1. **Right-click** in Scene panel → "Add Node" (or press Ctrl+A)
2. **Search** for the node type you want
3. **Click** on it → "Create"

**Method 2: Plus Icon**
1. **Click the "+"** icon in Scene panel
2. Browse or search node types
3. **Double-click** to add

**🎯 Common Node Types:**
- **Node2D** → Basic 2D object
- **CharacterBody2D** → Player characters
- **RigidBody2D** → Physics objects  
- **Area2D** → Trigger zones
- **Sprite2D** → Display images
- **Control** → UI elements
- **Button** → Clickable UI
- **Label** → Text display

**⚡ Quick Tips:**
- **F1** → Open help for selected node
- **Ctrl+D** → Duplicate node
- **Del** → Delete selected node

What type of node are you trying to add? I can give specific guidance! 🐾`;
    }

    private getScriptAttachmentGuide(): string {
        return `📝 **Attaching Scripts** 📝

**Method 1: Right-click Approach**
1. **Select your node** in the scene
2. **Right-click** → "Attach Script"
3. **Choose language** (GDScript recommended)
4. **Set path** (usually same name as node)
5. **Click "Create"**

**Method 2: Script Icon**
1. **Select node** → Click the scroll icon next to the node name
2. **"Attach Script"** → Configure and create

**🎮 Basic Script Template:**
\`\`\`gdscript
extends Node2D  # Change to match your node type

# Called when the node enters the scene tree
func _ready():
    print("Hello, Godot world!")

# Called every frame
func _process(delta):
    # Your update logic here
    pass

# Called for physics processing (60fps)
func _physics_process(delta):
    # Physics logic here
    pass
\`\`\`

**💡 Script Best Practices:**
- **One script per node** (usually)
- **Meaningful file names** (player.gd, enemy_ai.gd)
- **Use @export** for public variables
- **Comment your code** liberally

Need help with specific script functionality? Ask away! 😸`;
    }

    private getProjectSettingsGuide(): string {
        return `⚙️ **Project Settings Made Easy** ⚙️

**Access Settings:**
- **Menu**: Project → Project Settings (or F5)
- **Tabs**: General, Input Map, Layer Names, etc.

**🎯 Essential Settings to Configure:**

**1. Application Settings:**
- **Name**: Your game's display name
- **Icon**: Game icon (.ico, .png)
- **Main Scene**: Starting scene of your game

**2. Display Settings:**
- **Width/Height**: Default 1920x1080 or 1280x720
- **Resizable**: Allow window resizing
- **Fullscreen**: Default fullscreen mode
- **Aspect**: How to handle different screen ratios

**3. Input Map:**
- **Add actions**: "move_left", "jump", "attack"
- **Assign keys**: Map keyboard/controller inputs
- **Test inputs**: Use Input.is_action_pressed()

**4. Physics Settings:**
- **Gravity**: Default gravity strength
- **Time Scale**: Game speed multiplier
- **Collision Layers**: Organize collision detection

**5. Rendering:**
- **Renderer**: Forward+ (default) vs Mobile
- **VSync**: Screen tearing prevention
- **MSAA**: Anti-aliasing quality

**🔧 Quick Setup Example:**
\`\`\`gdscript
# In your script - check if input map is working
func _process(delta):
    if Input.is_action_pressed("move_left"):
        position.x -= 100 * delta
\`\`\`

Which settings would you like to configure first? 🎮`;
    }

    // Utility methods
    private detectSkillLevel(message: string): void {
        const beginnerKeywords = ['how to', 'what is', 'help', 'basic', 'simple', 'beginner', 'new to'];
        const advancedKeywords = ['optimize', 'performance', 'shader', 'plugin', 'advanced', 'gdnative', 'threading'];

        if (beginnerKeywords.some(keyword => message.includes(keyword))) {
            this.userSkillLevel = 'beginner';
        } else if (advancedKeywords.some(keyword => message.includes(keyword))) {
            this.userSkillLevel = 'advanced';
        } else {
            this.userSkillLevel = 'intermediate';
        }
    }

    private getQuickStartMenu(): string {
        return `**🚀 Quick Start Menu:**
• **"create project"** - Set up a new game
• **"add node"** - Learn about adding nodes
• **"write script"** - Scripting guidance
• **"project settings"** - Configure your project
• **"help"** - See all available commands

What interests you most? 😸`;
    }

    private getHelpMenu(): string {
        return `🆘 **Godot Chat Assistant Commands** 🆘

**📁 Project Management:**
- "create project" - Project setup guide
- "organize project" - Folder structure tips
- "project settings" - Configuration help
- "import assets" - Asset management
- "export game" - Build and deployment

**🎭 Scene & Node Help:**
- "add node" - Node creation guide
- "node types" - Available node explanations
- "delete node" - Node management
- "scene hierarchy" - Parent/child relationships

**📝 Scripting Assistance:**
- "attach script" - Script creation
- "variables" - Variable usage
- "functions" - Method creation
- "signals" - Event system
- "input handling" - User input

**🎮 Game Development:**
- "player movement" - Character controls
- "collision detection" - Physics setup
- "ui creation" - Interface building
- "animation" - Animation systems
- "audio setup" - Sound integration

**🔧 Advanced Topics:**
- "debugging" - Error fixing
- "optimization" - Performance tips
- "plugins" - Extending Godot
- "custom resources" - Data management

**💬 Chat Features:**
- "skill level beginner/advanced" - Adjust my responses
- "current topic" - What we're discussing
- "history" - Recent conversation

Just type naturally! I understand context and follow-up questions. 🐾`;
    }

    // Message management
    private addUserMessage(content: string): void {
        this.conversationHistory.push({
            role: 'user',
            content,
            timestamp: new Date()
        });
    }

    private addAssistantMessage(content: string): void {
        this.conversationHistory.push({
            role: 'assistant', 
            content,
            timestamp: new Date()
        });
    }

    private addSystemMessage(content: string): void {
        this.conversationHistory.push({
            role: 'system',
            content,
            timestamp: new Date()
        });
    }

    private getRecentTopics(): string[] {
        return this.conversationHistory
            .slice(-5)
            .map(msg => this.currentTopic)
            .filter(topic => topic !== null) as string[];
    }

    // Placeholder methods for additional functionality
    private getNodeTypesGuide(): string { return "Node types guide coming soon! 🎭"; }
    private getInputGuide(): string { return "Input handling guide! 🎮"; }
    private getGeneralProjectHelp(): string { return "General project help! 📁"; }
    private getGeneralNodeHelp(): string { return "General node help! 🎭"; }
    private getGeneralScriptingHelp(): string { return "General scripting help! 📝"; }
    private getGeneralSettingsHelp(): string { return "General settings help! ⚙️"; }
    private getContextualProjectAdvice(message: string): string { return "Contextual project advice! 💡"; }
    private getContextualNodeAdvice(message: string): string { return "Contextual node advice! 🎯"; }
    private getGeneralAdvice(message: string): string { return "I'm here to help with any Godot questions! 😸"; }
    
    // Additional guide methods
    private getProjectSetupGuide(): string { return "Project setup guide! 🏗️"; }
    private getAssetImportGuide(): string { return "Asset import guide! 📦"; }
    private getExportGuide(): string { return "Export guide! 🚀"; }
    private getNodeDeletionGuide(): string { return "Node deletion guide! 🗑️"; }
    private getNodeHierarchyGuide(): string { return "Node hierarchy guide! 🌳"; }
    private getVariableGuide(): string { return "Variable guide! 📊"; }
    private getFunctionGuide(): string { return "Function guide! ⚙️"; }
    private getSignalGuide(): string { return "Signal guide! 📡"; }
    private getInputMapGuide(): string { return "Input map guide! 🎮"; }
    private getLayerSettingsGuide(): string { return "Layer settings guide! 🎭"; }
    private getDisplaySettingsGuide(): string { return "Display settings guide! 🖥️"; }
    private handleSceneQuestions(message: string): Promise<string> { return Promise.resolve("Scene help coming soon! 🎬"); }
    private handleUIQuestions(message: string): Promise<string> { return Promise.resolve("UI help coming soon! 🎨"); }
    private handlePhysicsQuestions(message: string): Promise<string> { return Promise.resolve("Physics help coming soon! ⚽"); }
    private handleAnimationQuestions(message: string): Promise<string> { return Promise.resolve("Animation help coming soon! 🎭"); }
    private handleAudioQuestions(message: string): Promise<string> { return Promise.resolve("Audio help coming soon! 🔊"); }
    private handleDebuggingQuestions(message: string): Promise<string> { return Promise.resolve("Debug help coming soon! 🐛"); }
}

// Export for use in Godot
if (typeof exports !== 'undefined') {
    exports.GodotChatAssistant = GodotChatAssistant;
} else if (typeof globalThis !== 'undefined') {
    (globalThis as any).GodotChatAssistant = GodotChatAssistant;
}