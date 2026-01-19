# GDScript Structs - Cookbook

Real-world recipes and patterns for using structs effectively.

---

## 🎮 Game Development Patterns

### Pattern 1: Entity Component System (ECS-lite)

```gdscript
extends Node2D

# Define component structs
struct Transform:
    var position: Vector2
    var rotation: float = 0.0
    var scale: Vector2 = Vector2.ONE

struct Velocity:
    var linear: Vector2 = Vector2.ZERO
    var angular: float = 0.0

struct Health:
    var current: int
    var maximum: int

struct Sprite:
    var texture: Texture2D
    var modulate: Color = Color.WHITE

# Entity is just an ID with components
struct Entity:
    var id: int
    var transform: Transform
    var velocity: Velocity
    var health: Health
    var sprite: Sprite

var entities: Array = []

func create_enemy(pos: Vector2) -> Entity:
    return Entity(
        entities.size(),
        Transform(pos),
        Velocity(Vector2(randf_range(-50, 50), randf_range(-50, 50))),
        Health(100, 100),
        Sprite(preload("res://enemy.png"))
    )

func _process(delta):
    # Movement system
    for e in entities:
        e.transform.position += e.velocity.linear * delta
        e.transform.rotation += e.velocity.angular * delta
    
    # Render system
    queue_redraw()

func _draw():
    for e in entities:
        draw_set_transform(e.transform.position, e.transform.rotation, e.transform.scale)
        draw_texture(e.sprite.texture, Vector2.ZERO, e.sprite.modulate)
```

### Pattern 2: Event Queue

```gdscript
extends Node

# Define event types
struct InputEvent:
    var type: String  # "move", "attack", "jump"
    var player_id: int
    var data: Dictionary  # Event-specific data

struct DamageEvent:
    var attacker_id: int
    var target_id: int
    var amount: float
    var type: String = "physical"  # "physical", "magical", "true"

struct SpawnEvent:
    var entity_type: String
    var position: Vector2
    var team: int

# Event queue
var event_queue: Array = []

func queue_event(event):
    event_queue.append(event)

func process_events():
    for event in event_queue:
        if event is InputEvent:
            handle_input(event)
        elif event is DamageEvent:
            handle_damage(event)
        elif event is SpawnEvent:
            handle_spawn(event)
    
    event_queue.clear()

func handle_damage(event: DamageEvent):
    var target = get_entity(event.target_id)
    if target:
        target.health -= event.amount
        if target.health <= 0:
            queue_event(DeathEvent.new(event.target_id))
```

### Pattern 3: State Machines

```gdscript
extends CharacterBody2D

# Define states as structs
struct IdleState:
    var time_elapsed: float = 0.0

struct PatrolState:
    var waypoints: Array = []
    var current_waypoint: int = 0
    var move_speed: float = 100.0

struct ChaseState:
    var target: Node2D
    var chase_speed: float = 200.0

struct AttackState:
    var target: Node2D
    var cooldown: float = 0.0
    var damage: float = 10.0

# Current state (only one active)
var current_state = null

func _ready():
    enter_idle()

func enter_idle():
    current_state = IdleState()

func enter_patrol(waypoints: Array):
    current_state = PatrolState()
    current_state.waypoints = waypoints

func enter_chase(target: Node2D):
    current_state = ChaseState()
    current_state.target = target

func _physics_process(delta):
    if current_state is IdleState:
        update_idle(delta)
    elif current_state is PatrolState:
        update_patrol(delta)
    elif current_state is ChaseState:
        update_chase(delta)
    elif current_state is AttackState:
        update_attack(delta)

func update_chase(delta):
    var state = current_state as ChaseState
    if state.target and is_instance_valid(state.target):
        var dir = (state.target.position - position).normalized()
        velocity = dir * state.chase_speed
        move_and_slide()
        
        # Switch to attack if close
        if position.distance_to(state.target.position) < 50:
            enter_attack(state.target)
    else:
        enter_patrol(get_patrol_waypoints())
```

### Pattern 4: Configuration Management

```gdscript
# config.gd
extends Node

struct AudioConfig:
    var master_volume: float = 1.0
    var music_volume: float = 0.7
    var sfx_volume: float = 0.8
    var muted: bool = false

struct GraphicsConfig:
    var fullscreen: bool = false
    var vsync: bool = true
    var resolution: Vector2i = Vector2i(1920, 1080)
    var quality: String = "high"  # "low", "medium", "high", "ultra"

struct GameplayConfig:
    var difficulty: String = "normal"
    var aim_assist: bool = false
    var auto_save: bool = true
    var language: String = "en"

struct GameConfig:
    var audio: AudioConfig = AudioConfig()
    var graphics: GraphicsConfig = GraphicsConfig()
    var gameplay: GameplayConfig = GameplayConfig()

var config: GameConfig = GameConfig()

func load_config():
    var file = FileAccess.open("user://config.json", FileAccess.READ)
    if file:
        var json = JSON.parse_string(file.get_as_text())
        if json:
            apply_json_to_config(json)
        file.close()

func save_config():
    var json = config_to_json()
    var file = FileAccess.open("user://config.json", FileAccess.WRITE)
    if file:
        file.store_string(JSON.stringify(json, "\t"))
        file.close()

func apply_audio_settings():
    AudioServer.set_bus_volume_db(0, linear_to_db(config.audio.master_volume))
    # ... more settings
```

---

## 🏗️ Data Structure Patterns

### Pattern 5: Spatial Grid

```gdscript
extends Node2D

struct GridCell:
    var x: int
    var y: int
    var entities: Array = []  # Entity IDs in this cell

struct SpatialGrid:
    var cell_size: float = 64.0
    var cells: Dictionary = {}  # Key: "x,y" -> GridCell

func grid_key(x: int, y: int) -> String:
    return "%d,%d" % [x, y]

func world_to_grid(pos: Vector2) -> Vector2i:
    return Vector2i(
        int(pos.x / grid.cell_size),
        int(pos.y / grid.cell_size)
    )

func add_entity_to_grid(entity_id: int, pos: Vector2):
    var grid_pos = world_to_grid(pos)
    var key = grid_key(grid_pos.x, grid_pos.y)
    
    if not grid.cells.has(key):
        grid.cells[key] = GridCell(grid_pos.x, grid_pos.y)
    
    grid.cells[key].entities.append(entity_id)

func get_nearby_entities(pos: Vector2, radius: float) -> Array:
    var results = []
    var grid_pos = world_to_grid(pos)
    var grid_radius = int(radius / grid.cell_size) + 1
    
    for dx in range(-grid_radius, grid_radius + 1):
        for dy in range(-grid_radius, grid_radius + 1):
            var key = grid_key(grid_pos.x + dx, grid_pos.y + dy)
            if grid.cells.has(key):
                results.append_array(grid.cells[key].entities)
    
    return results
```

### Pattern 6: Weighted Random Selection

```gdscript
struct LootEntry:
    var item_id: String
    var weight: float
    var min_count: int = 1
    var max_count: int = 1

struct LootTable:
    var entries: Array = []  # Array of LootEntry
    var total_weight: float = 0.0

func create_loot_table(entries: Array) -> LootTable:
    var table = LootTable()
    table.entries = entries
    for entry in entries:
        table.total_weight += entry.weight
    return table

func roll_loot(table: LootTable) -> Array:
    var results = []
    var roll = randf() * table.total_weight
    var accumulated = 0.0
    
    for entry in table.entries:
        accumulated += entry.weight
        if roll <= accumulated:
            var count = randi_range(entry.min_count, entry.max_count)
            for i in count:
                results.append(entry.item_id)
            break
    
    return results

# Example usage
var common_loot = create_loot_table([
    LootEntry("gold", 50.0, 1, 10),
    LootEntry("potion", 30.0, 1, 3),
    LootEntry("sword", 15.0, 1, 1),
    LootEntry("rare_gem", 5.0, 1, 1)
])
```

### Pattern 7: Command Pattern

```gdscript
# Define command structs
struct MoveCommand:
    var entity_id: int
    var from: Vector2
    var to: Vector2

struct AttackCommand:
    var attacker_id: int
    var target_id: int
    var damage: float

struct SpawnCommand:
    var entity_type: String
    var position: Vector2

# Command history for undo/redo
var command_history: Array = []
var history_index: int = 0

func execute_command(cmd):
    if cmd is MoveCommand:
        execute_move(cmd)
    elif cmd is AttackCommand:
        execute_attack(cmd)
    elif cmd is SpawnCommand:
        execute_spawn(cmd)
    
    command_history.append(cmd)
    history_index = command_history.size()

func undo():
    if history_index > 0:
        history_index -= 1
        var cmd = command_history[history_index]
        undo_command(cmd)

func redo():
    if history_index < command_history.size():
        var cmd = command_history[history_index]
        execute_command(cmd)
        history_index += 1
```

---

## 🎨 UI Patterns

### Pattern 8: Menu System

```gdscript
struct MenuItem:
    var id: String
    var label: String
    var icon: Texture2D = null
    var enabled: bool = true
    var submenu: Array = []  # Array of MenuItem

struct MenuState:
    var current_menu: Array = []  # Current menu items
    var selected_index: int = 0
    var menu_stack: Array = []  # For back navigation

func create_main_menu() -> Array:
    return [
        MenuItem("new_game", "New Game", preload("res://icons/play.png")),
        MenuItem("continue", "Continue", preload("res://icons/continue.png")),
        MenuItem("settings", "Settings", preload("res://icons/settings.png"), true, create_settings_menu()),
        MenuItem("quit", "Quit", preload("res://icons/exit.png"))
    ]

func create_settings_menu() -> Array:
    return [
        MenuItem("audio", "Audio", null, true, create_audio_menu()),
        MenuItem("graphics", "Graphics", null, true, create_graphics_menu()),
        MenuItem("back", "Back")
    ]
```

### Pattern 9: Dialogue System

```gdscript
struct DialogueLine:
    var speaker: String
    var text: String
    var emotion: String = "neutral"  # "happy", "sad", "angry", etc.
    var choices: Array = []  # Array of DialogueChoice

struct DialogueChoice:
    var text: String
    var next_dialogue: String  # ID of next dialogue
    var condition: String = ""  # Optional condition

struct Dialogue:
    var id: String
    var lines: Array = []  # Array of DialogueLine

# Example
var greet_dialogue = Dialogue("greet", [
    DialogueLine("NPC", "Hello, traveler!", "happy", [
        DialogueChoice("Hello!", "ask_quest"),
        DialogueChoice("...", "ignore")
    ])
])
```

---

## 🔧 Utility Patterns

### Pattern 10: Object Pool

```gdscript
struct PooledObject:
    var instance: Node
    var in_use: bool = false

struct ObjectPool:
    var scene: PackedScene
    var pool: Array = []  # Array of PooledObject
    var initial_size: int = 10

func create_pool(scene: PackedScene, size: int) -> ObjectPool:
    var pool = ObjectPool()
    pool.scene = scene
    pool.initial_size = size
    
    for i in size:
        var instance = scene.instantiate()
        instance.hide()
        add_child(instance)
        pool.pool.append(PooledObject(instance, false))
    
    return pool

func get_from_pool(pool: ObjectPool) -> Node:
    for obj in pool.pool:
        if not obj.in_use:
            obj.in_use = true
            obj.instance.show()
            return obj.instance
    
    # Expand pool if needed
    var instance = pool.scene.instantiate()
    add_child(instance)
    pool.pool.append(PooledObject(instance, true))
    return instance

func return_to_pool(pool: ObjectPool, instance: Node):
    for obj in pool.pool:
        if obj.instance == instance:
            obj.in_use = false
            obj.instance.hide()
            break
```

---

## 📊 Performance Patterns

### Pattern 11: Batch Processing

```gdscript
struct UpdateBatch:
    var entities: Array = []
    var batch_size: int = 100

func process_in_batches(all_entities: Array, batch_size: int = 100):
    var batch = UpdateBatch()
    batch.batch_size = batch_size
    
    for i in all_entities.size():
        batch.entities.append(all_entities[i])
        
        if batch.entities.size() >= batch_size:
            process_batch(batch)
            batch.entities.clear()
    
    # Process remaining
    if batch.entities.size() > 0:
        process_batch(batch)

func process_batch(batch: UpdateBatch):
    for entity in batch.entities:
        update_entity(entity)
```

### Pattern 12: Dirty Flag System

```gdscript
struct DirtyFlags:
    var position: bool = false
    var rotation: bool = false
    var scale: bool = false
    var color: bool = false

struct CachedTransform:
    var transform: Transform2D
    var dirty: DirtyFlags = DirtyFlags()

func mark_dirty(cache: CachedTransform, flag: String):
    match flag:
        "position": cache.dirty.position = true
        "rotation": cache.dirty.rotation = true
        "scale": cache.dirty.scale = true
        "color": cache.dirty.color = true

func update_if_dirty(cache: CachedTransform):
    if cache.dirty.position or cache.dirty.rotation or cache.dirty.scale:
        recalculate_transform(cache)
        cache.dirty.position = false
        cache.dirty.rotation = false
        cache.dirty.scale = false
```

---

## 🎓 Advanced Patterns

### Pattern 13: Behavior Trees

```gdscript
struct BTNode:
    var type: String  # "sequence", "selector", "condition", "action"
    var children: Array = []
    var condition_func: Callable = Callable()
    var action_func: Callable = Callable()

struct BTContext:
    var entity: Node
    var blackboard: Dictionary = {}

func create_enemy_bt() -> BTNode:
    return BTNode("selector", [
        BTNode("sequence", [
            BTNode("condition", [], func(ctx): return can_see_player(ctx.entity)),
            BTNode("action", [], func(ctx): chase_player(ctx.entity))
        ]),
        BTNode("action", [], func(ctx): patrol(ctx.entity))
    ])

func evaluate_bt(node: BTNode, context: BTContext) -> bool:
    match node.type:
        "sequence":
            for child in node.children:
                if not evaluate_bt(child, context):
                    return false
            return true
        "selector":
            for child in node.children:
                if evaluate_bt(child, context):
                    return true
            return false
        "condition":
            return node.condition_func.call(context)
        "action":
            node.action_func.call(context)
            return true
    return false
```

---

## 📝 Tips & Tricks

1. **Keep It Simple**: Structs are for data, not complex logic
2. **Use Defaults**: Make optional fields have sensible defaults
3. **Type Everything**: Use type annotations for better errors
4. **Document Fields**: Add comments for complex structs
5. **Test Early**: Compile-time errors are your friend!

---

## 🚀 Ready for More?

- See **STRUCTS_QUICK_START.md** for basics
- See **GDSCRIPT_STRUCTS_USAGE.md** for full documentation
- See **PERFORMANCE_ANALYSIS.md** for optimization details

Happy coding! 🎮
