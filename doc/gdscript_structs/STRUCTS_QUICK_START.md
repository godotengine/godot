# GDScript Structs - Quick Start Guide

## 🚀 What Are Structs?

Structs are lightweight data containers with compile-time type safety. Perfect for:
- Game entities (bullets, enemies, particles)
- Configuration data
- Events and messages
- Data-oriented design

**Key Benefits:**
- ✅ Catch typos at compile time
- ✅ IDE auto-completion works perfectly
- ✅ Self-documenting code
- ✅ Zero performance cost (same as dictionaries)

---

## 📝 Basic Usage

### Declaring a Struct

```gdscript
# Simple struct
struct Point:
    var x: float
    var y: float

# Struct with default values
struct Enemy:
    var id: int = 0
    var health: int = 100
    var damage: float = 15.5

# Struct with multiple types
struct PlayerData:
    var name: String = ""
    var level: int = 1
    var position: Vector2 = Vector2.ZERO
    var inventory: Array = []
```

### Creating Instances

```gdscript
# All fields required
var p1 = Point(10.5, 20.3)

# With defaults
var e1 = Enemy()              # All defaults: id=0, health=100, damage=15.5
var e2 = Enemy(1)             # id=1, health=100, damage=15.5
var e3 = Enemy(2, 150)        # id=2, health=150, damage=15.5
var e4 = Enemy(3, 200, 25.0)  # id=3, health=200, damage=25.0
```

### Accessing Members

```gdscript
# Dot notation (recommended)
var p = Point(10, 20)
print(p.x)        # 10
p.x += 5
print(p.x)        # 15

# Dictionary notation (also works)
print(p["y"])     # 20
p["y"] = 30

# Both work the same - structs are dictionaries internally
```

---

## 🎮 Real-World Examples

### Example 1: Bullet Hell Game

```gdscript
extends Node2D

struct Bullet:
    var x: float
    var y: float
    var vx: float
    var vy: float
    var damage: int = 10
    var alive: bool = true

var bullets: Array = []

func _ready():
    # Spawn 1000 bullets
    for i in 1000:
        var angle = randf() * TAU
        var speed = 200.0
        bullets.append(Bullet(
            400, 300,                    # x, y (center of screen)
            cos(angle) * speed,          # vx
            sin(angle) * speed,          # vy
            10                           # damage
        ))

func _process(delta):
    # Update all bullets - type-safe!
    for b in bullets:
        if not b.alive:
            continue
        
        b.x += b.vx * delta
        b.y += b.vy * delta
        
        # Remove off-screen bullets
        if b.x < 0 or b.x > 800 or b.y < 0 or b.y > 600:
            b.alive = false
    
    # Clean up dead bullets
    bullets = bullets.filter(func(b): return b.alive)
```

**Why This Is Better:**
```gdscript
# Without structs (error-prone):
b.x += b.vx * delta  # Typo: could be b["vx"] or b["velocity_x"]?
b.helth -= damage    # TYPO! Runtime error later

# With structs (caught immediately):
b.x += b.vx * delta  # IDE knows about "vx"
b.helth -= damage    # ERROR at compile time: "helth" doesn't exist
```

### Example 2: RTS Unit Management

```gdscript
extends Node

struct Unit:
    var id: int
    var team: int
    var unit_type: String = "soldier"
    var position: Vector2 = Vector2.ZERO
    var health: int = 100
    var target_id: int = -1

var units: Array = []

func spawn_unit(team: int, pos: Vector2, type: String = "soldier"):
    var unit = Unit(
        units.size(),  # id
        team,
        type,
        pos,
        100,           # health
        -1             # no target
    )
    units.append(unit)
    return unit

func _process(_delta):
    # Update all units
    for unit in units:
        if unit.health <= 0:
            continue
        
        # Find enemy targets
        if unit.target_id == -1:
            for other in units:
                if other.team != unit.team and other.health > 0:
                    unit.target_id = other.id
                    break
        
        # Attack target
        if unit.target_id != -1:
            var target = units[unit.target_id]
            if target.health > 0:
                target.health -= 1
            else:
                unit.target_id = -1
```

### Example 3: Particle System

```gdscript
extends Node2D

struct Particle:
    var pos: Vector2
    var vel: Vector2
    var life: float = 1.0
    var color: Color = Color.WHITE

var particles: Array = []

func emit_particle(position: Vector2):
    particles.append(Particle(
        position,
        Vector2(randf_range(-100, 100), randf_range(-200, -50)),
        1.0,
        Color(randf(), randf(), randf())
    ))

func _process(delta):
    for p in particles:
        p.pos += p.vel * delta
        p.vel.y += 98.0 * delta  # Gravity
        p.life -= delta
    
    # Remove dead particles
    particles = particles.filter(func(p): return p.life > 0)
    queue_redraw()

func _draw():
    for p in particles:
        var alpha = p.life
        draw_circle(p.pos, 3.0, Color(p.color, alpha))
```

---

## 💡 Best Practices

### DO: Use Structs For Data

```gdscript
# ✅ GOOD: Lightweight data structure
struct Inventory:
    var gold: int = 0
    var items: Array = []
    var capacity: int = 20

# ✅ GOOD: Event messages
struct DamageEvent:
    var attacker_id: int
    var victim_id: int
    var damage: float
    var is_critical: bool = false
```

### DON'T: Use Structs For Complex Logic

```gdscript
# ❌ BAD: Structs can't have methods
struct Enemy:
    var health: int
    # Can't do this:
    # func take_damage(amount): ...

# ✅ GOOD: Use classes for behavior
class_name Enemy
extends Node2D

var health: int = 100

func take_damage(amount: int):
    health -= amount
    if health <= 0:
        queue_free()
```

### DO: Use Type Annotations

```gdscript
# ✅ GOOD: Explicit types
func process_unit(unit: Unit):
    unit.health -= 10  # Auto-complete works!

var entities: Array = []  # Will hold Entity structs

# ⚠️ OKAY: Without types (still works)
func process_thing(thing):
    thing.health -= 10  # No auto-complete
```

### DO: Keep Structs Simple

```gdscript
# ✅ GOOD: Simple data
struct Point:
    var x: float
    var y: float

# ✅ GOOD: Reasonable size (5-10 fields)
struct Character:
    var name: String
    var level: int
    var health: int
    var mana: int
    var position: Vector2

# ⚠️ AVOID: Too many fields (use class instead)
struct ComplexSystem:
    var field1: int
    var field2: int
    # ... 50 fields ...
    # This should probably be a class!
```

---

## 🐛 Common Mistakes

### Mistake 1: Modifying Struct in Function

```gdscript
# This modifies the original!
func boost_health(unit: Unit):
    unit.health += 50  # ✅ Works - modifies original

var u = Unit(1, 100)
boost_health(u)
print(u.health)  # 150 - original was modified

# Structs are dictionaries - passed by reference!
```

### Mistake 2: Expecting Inheritance

```gdscript
# ❌ ERROR: Structs can't extend
struct Base:
    var x: int

struct Derived extends Base:  # COMPILE ERROR!
    var y: int

# ✅ SOLUTION: Use composition
struct Derived:
    var base: Base
    var y: int
```

### Mistake 3: Missing Required Fields

```gdscript
struct Point:
    var x: float
    var y: float  # No default!

# ❌ ERROR: Missing field
var p = Point(10)  # Compile error: y is required

# ✅ FIX: Provide all fields
var p = Point(10, 20)  # OK

# Or add defaults:
struct Point:
    var x: float = 0.0
    var y: float = 0.0

var p = Point()  # OK with defaults
```

---

## 🔄 Migration from Dictionaries

### Before (Dictionary)

```gdscript
# Error-prone, no type safety
var entity = {
    "id": 1,
    "health": 100,
    "damage": 15.5
}

func process(e):
    e["helth"] -= 10  # TYPO! Runtime error
    e["damage"] = "high"  # TYPE ERROR! Runtime crash
```

### After (Struct)

```gdscript
# Type-safe, auto-complete
struct Entity:
    var id: int
    var health: int
    var damage: float

var entity = Entity(1, 100, 15.5)

func process(e: Entity):
    e.helth -= 10  # COMPILE ERROR: "helth" doesn't exist
    e.damage = "high"  # COMPILE ERROR: wrong type
```

**Migration is Easy:**
1. Define struct with same fields as dictionary
2. Replace `{...}` with `StructName(...)`
3. Replace `["field"]` with `.field`
4. Add type annotations to functions
5. Fix errors found by compiler!

---

## ⚡ Performance Tips

### Current Performance (Phase 1-2)

```gdscript
# Struct = Dictionary (same performance)
var entities: Array = []
for i in 10000:
    entities.append(Entity(i, 100))

# This is as fast as manual dictionaries
# No performance penalty!
```

### Future Performance (Phase 3+)

```gdscript
# When FlatArray is implemented:
var entities: Array = []  # Regular array
for i in 10000:
    entities.append(Entity(i, 100))

# Compiler will automatically optimize to flat layout
# for typed struct arrays - 10x faster iteration!
```

**What To Do Now:**
- ✅ Use structs for data (get type safety)
- ✅ Write clean, type-safe code
- ✅ Wait for FlatArray optimization (automatic!)

---

## 📚 More Examples

### Config Data

```gdscript
struct GameConfig:
    var difficulty: String = "normal"
    var music_volume: float = 0.7
    var sfx_volume: float = 0.8
    var fullscreen: bool = false

var config = GameConfig("hard", 0.5, 0.6, true)
```

### Network Messages

```gdscript
struct PlayerJoinMessage:
    var player_name: String
    var player_id: int
    var team: int

struct ChatMessage:
    var sender: String
    var text: String
    var timestamp: float
```

### Save Data

```gdscript
struct SaveData:
    var player_name: String
    var level: int
    var gold: int
    var position: Vector2
    var inventory: Array
    var timestamp: float

func save_game() -> SaveData:
    return SaveData(
        player_name,
        current_level,
        gold,
        player.position,
        inventory.duplicate(),
        Time.get_unix_time_from_system()
    )
```

---

## 🎯 Summary

**What Structs Are:**
- Lightweight data containers
- Type-safe at compile time
- Dictionaries at runtime (currently)

**When to Use:**
- Game entities (bullets, particles, units)
- Configuration data
- Events and messages
- Any data-focused code

**When NOT to Use:**
- Need methods → Use classes
- Need inheritance → Use classes
- Very complex data → Use classes

**Current Status:**
- ✅ Fully implemented (Phase 1-2)
- ✅ Type safety working
- ✅ No performance cost
- 🔜 10x speedup coming (Phase 3)

---

## 📖 See Also

- **GDSCRIPT_STRUCTS_USAGE.md** - Complete technical documentation
- **PERFORMANCE_ANALYSIS.md** - Performance details and roadmap
- **IMPLEMENTATION_COMPLETE.md** - Implementation summary

Happy coding! 🚀
