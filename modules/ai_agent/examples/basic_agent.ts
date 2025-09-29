// Basic AI Agent Example
// This TypeScript file demonstrates how to create a simple AI agent

interface AIContext {
    agent: any;
    node: any;
    scene: any;
}

class BasicAIAgent {
    private context: AIContext;
    private name: string;
    private personality: string;

    constructor(context: AIContext, name: string = "Assistant") {
        this.context = context;
        this.name = name;
        this.personality = "I am a helpful AI assistant built into Godot Engine.";
    }

    // Main agent logic
    async processMessage(message: string): Promise<string> {
        console.log(`${this.name} received message: ${message}`);
        
        // Simple keyword-based responses
        if (message.toLowerCase().includes("hello")) {
            return `Hello! I'm ${this.name}, your AI assistant. How can I help you today?`;
        }
        
        if (message.toLowerCase().includes("help")) {
            return this.getHelpMessage();
        }
        
        if (message.toLowerCase().includes("scene")) {
            return this.analyzeScene();
        }
        
        if (message.toLowerCase().includes("node")) {
            return this.analyzeCurrentNode();
        }
        
        if (message.toLowerCase().includes("code")) {
            return this.generateCode(message);
        }
        
        // Default response
        return `I understand you said: "${message}". I'm still learning how to respond to that. Try asking for help!`;
    }

    private getHelpMessage(): string {
        return `
        I can help you with:
        • Scene analysis - ask about the current scene
        • Node information - ask about nodes
        • Code generation - ask me to generate code
        • General Godot questions
        
        Just type your question naturally!
        `;
    }

    private analyzeScene(): string {
        try {
            if (this.context.scene) {
                const nodeCount = this.context.scene.get_child_count();
                return `Current scene has ${nodeCount} child nodes. The scene tree looks organized!`;
            }
            return "I can't access the current scene right now.";
        } catch (error) {
            return `Error analyzing scene: ${error}`;
        }
    }

    private analyzeCurrentNode(): string {
        try {
            if (this.context.node) {
                const nodeName = this.context.node.name || "Unknown";
                const nodeType = this.context.node.get_class() || "Node";
                return `Current node: "${nodeName}" of type ${nodeType}`;
            }
            return "No current node available.";
        } catch (error) {
            return `Error analyzing node: ${error}`;
        }
    }

    private generateCode(request: string): string {
        const lowerRequest = request.toLowerCase();
        
        if (lowerRequest.includes("button")) {
            return this.generateButtonCode();
        }
        
        if (lowerRequest.includes("player") || lowerRequest.includes("character")) {
            return this.generatePlayerCode();
        }
        
        if (lowerRequest.includes("ui") || lowerRequest.includes("interface")) {
            return this.generateUICode();
        }
        
        return `
        Here's a basic GDScript template:
        
        extends Node
        
        func _ready():
            print("Node is ready!")
        
        func _process(delta):
            # Update logic here
            pass
        `;
    }

    private generateButtonCode(): string {
        return `
        # Button interaction example
        extends Button
        
        func _ready():
            connect("pressed", _on_button_pressed)
        
        func _on_button_pressed():
            print("Button was pressed!")
            # Add your button logic here
        `;
    }

    private generatePlayerCode(): string {
        return `
        # Basic player movement example
        extends CharacterBody2D
        
        @export var speed = 300.0
        @export var jump_velocity = -400.0
        
        var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")
        
        func _physics_process(delta):
            # Add gravity
            if not is_on_floor():
                velocity.y += gravity * delta
            
            # Handle jump
            if Input.is_action_just_pressed("ui_accept") and is_on_floor():
                velocity.y = jump_velocity
            
            # Handle movement
            var direction = Input.get_axis("ui_left", "ui_right")
            if direction != 0:
                velocity.x = direction * speed
            else:
                velocity.x = move_toward(velocity.x, 0, speed)
            
            move_and_slide()
        `;
    }

    private generateUICode(): string {
        return `
        # UI Manager example
        extends Control
        
        @onready var health_bar = $HealthBar
        @onready var score_label = $ScoreLabel
        
        var current_health = 100
        var current_score = 0
        
        func _ready():
            update_ui()
        
        func update_health(new_health):
            current_health = clamp(new_health, 0, 100)
            update_ui()
        
        func update_score(points):
            current_score += points
            update_ui()
        
        func update_ui():
            health_bar.value = current_health
            score_label.text = "Score: " + str(current_score)
        `;
    }

    // Memory functions
    remember(key: string, value: any): void {
        if (this.context.agent) {
            this.context.agent.remember(key, value);
        }
    }

    recall(key: string): any {
        if (this.context.agent) {
            return this.context.agent.recall(key);
        }
        return null;
    }

    // Agent actions
    addAction(name: string, callback: Function, description: string = ""): void {
        if (this.context.agent) {
            this.context.agent.add_action(name, callback, description);
        }
    }
}

// Export the agent class for use in Godot
if (typeof exports !== 'undefined') {
    exports.BasicAIAgent = BasicAIAgent;
} else if (typeof globalThis !== 'undefined') {
    (globalThis as any).BasicAIAgent = BasicAIAgent;
}