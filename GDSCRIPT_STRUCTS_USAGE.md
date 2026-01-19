# GDScript Structs - User Documentation

## Overview

GDScript now supports lightweight struct types for improved performance when working with high-entity-count games. Structs provide compile-time type safety while being represented as dictionaries at runtime.

## Basic Syntax

### Declaring a Struct

```gdscript
struct Point:
    var x: float
    var y: float
```

### With Default Values

```gdscript
struct Enemy:
    var health: int = 100
    var damage: float = 15.5
    var name: String = "Goblin"
```

### With Type Annotations

```gdscript
struct Transform:
    var position: Vector3
    var rotation: Quaternion
    var scale: Vector3 = Vector3.ONE
```

## Creating Struct Instances

### Basic Instantiation

```gdscript
# Create with all values
var p = Point(10.5, 20.3)

# Create with some values (rest use defaults)
var e = Enemy(150)  # health=150, damage=15.5, name="Goblin"

# Create with no arguments (all defaults)
var e2 = Enemy()
```

### Accessing Members

Struct instances are dictionaries at runtime, so you access members with dictionary syntax:

```gdscript
var p = Point(10, 20)
print(p["x"])  # Access members
p["x"] = 30   # Modify members
```

## Type Safety

### Compile-Time Checking

```gdscript
struct Point:
    var x: int
    var y: int

func test():
    var p = Point(10, 20)  # ✅ OK
    var p2 = Point("hello", 20)  # ❌ Error: Type mismatch
```

### Struct as Parameter Types

```gdscript
struct Position:
    var x: float
    var y: float

func move_entity(pos: Position):
    print("Moving to:", pos["x"], pos["y"])

# Usage
var p = Position(100.0, 200.0)
move_entity(p)
```

## Performance Benefits

Structs are designed for scenarios where you have many instances:

```gdscript
struct Particle:
    var position: Vector3
    var velocity: Vector3
    var lifetime: float

# Create thousands of particles efficiently
var particles = []
for i in range(10000):
    particles.append(Particle(
        Vector3(randf(), randf(), randf()),
        Vector3(randf(), randf(), randf()),
        randf() * 5.0
    ))
```

## Best Practices

### ✅ DO

- Use structs for data-only types (no methods)
- Use structs for high-frequency allocations
- Use structs when you need simple data aggregation
- Keep structs lightweight (few members)

```gdscript
struct BulletData:
    var position: Vector3
    var direction: Vector3
    var speed: float
    var damage: int
```

### ❌ DON'T

- Don't use structs when you need methods or inheritance
- Don't use structs for complex logic
- Don't use structs as replacements for classes

```gdscript
# ❌ BAD - Use a class instead
struct Enemy:
    var health: int
    # Can't add methods like take_damage()
```

## Limitations

### No Inheritance

Structs cannot inherit from other structs or classes:

```gdscript
struct Base:
    var x: int

# ❌ Error: Structs don't support extends
struct Derived extends Base:
    var y: int
```

### No Methods

Structs are data-only:

```gdscript
struct Point:
    var x: int
    var y: int
    # ❌ Cannot add functions to structs
```

### Runtime Representation

Structs are dictionaries at runtime, so:
- Member access uses dictionary syntax: `point["x"]`
- No performance benefit over dictionaries at runtime
- Performance benefits come from compile-time optimization opportunities

## Advanced Usage

### Nested Structs

```gdscript
struct Position:
    var x: float
    var y: float

struct Entity:
    var id: int
    var pos: Position
    var health: int

# Create nested structs
var entity = Entity(1, Position(10.0, 20.0), 100)
print(entity["pos"]["x"])  # Access nested members
```

### Arrays of Structs

```gdscript
struct Point:
    var x: int
    var y: int

# Create array of struct instances
var points = [
    Point(0, 0),
    Point(10, 20),
    Point(30, 40)
]

# Iterate over them
for p in points:
    print("Point:", p["x"], p["y"])
```

### Optional Members

```gdscript
struct Config:
    var width: int = 800
    var height: int = 600
    var fullscreen: bool = false

# Create with some defaults
var cfg = Config()  # All defaults
var cfg2 = Config(1920, 1080)  # Custom resolution, fullscreen=false
```

## Migration Guide

### From Dictionary

Before (using dictionaries):
```gdscript
func create_point(x, y):
    return {"x": x, "y": y}

var p = create_point(10, 20)
print(p["x"])
```

After (using structs):
```gdscript
struct Point:
    var x: int
    var y: int

var p = Point(10, 20)
print(p["x"])  # Same runtime access
```

Benefits:
- ✅ Compile-time type checking
- ✅ Auto-completion in editor
- ✅ Cleaner syntax
- ✅ Self-documenting code

### From Class (When Appropriate)

Before (using class):
```gdscript
class_name Particle

var position: Vector3
var velocity: Vector3
var lifetime: float

func _init(pos, vel, life):
    position = pos
    velocity = vel
    lifetime = life
```

After (using struct, when no methods needed):
```gdscript
struct Particle:
    var position: Vector3
    var velocity: Vector3
    var lifetime: float

# Simpler instantiation
var p = Particle(Vector3.ZERO, Vector3.UP, 5.0)
```

## Examples

### Game Entities

```gdscript
struct EnemyData:
    var health: int
    var max_health: int
    var damage: int
    var speed: float
    var position: Vector3

func spawn_enemy(type: String) -> EnemyData:
    match type:
        "goblin":
            return EnemyData(50, 50, 10, 3.0, Vector3.ZERO)
        "orc":
            return EnemyData(150, 150, 25, 2.0, Vector3.ZERO)
        _:
            return EnemyData(100, 100, 15, 2.5, Vector3.ZERO)
```

### Configuration Data

```gdscript
struct GraphicsSettings:
    var resolution: Vector2i = Vector2i(1920, 1080)
    var vsync: bool = true
    var msaa: int = 2
    var shadows: bool = true
    var shadow_quality: int = 2

func apply_graphics(settings: GraphicsSettings):
    get_window().size = settings.resolution
    DisplayServer.window_set_vsync_mode(
        DisplayServer.VSYNC_ENABLED if settings.vsync else DisplayServer.VSYNC_DISABLED
    )
    # Apply other settings...
```

### Particle Systems

```gdscript
struct Particle:
    var position: Vector3
    var velocity: Vector3
    var color: Color
    var size: float
    var lifetime: float
    var age: float = 0.0

var particles: Array[Particle] = []

func emit_particle(pos: Vector3, vel: Vector3):
    particles.append(Particle(
        pos,
        vel,
        Color.WHITE,
        1.0,
        2.0,
        0.0
    ))

func _process(delta):
    for i in range(particles.size() - 1, -1, -1):
        var p = particles[i]
        p["age"] += delta
        if p["age"] >= p["lifetime"]:
            particles.remove_at(i)
        else:
            p["position"] += p["velocity"] * delta
```

## FAQ

**Q: When should I use a struct vs a class?**
A: Use structs for simple data containers with no methods. Use classes when you need inheritance, methods, or signals.

**Q: Are structs passed by value or reference?**
A: Structs are dictionaries at runtime, so they're passed by reference (like all objects in GDScript).

**Q: Can I add methods to structs?**
A: No, structs are data-only. Use a class if you need methods.

**Q: Do structs improve performance?**
A: Structs provide compile-time optimization opportunities and cleaner code. At runtime, they're dictionaries.

**Q: Can I serialize structs?**
A: Yes, since they're dictionaries at runtime, they can be serialized like any other dictionary.

## Known Limitations

### Current Implementation

- Type annotations in nested structs may show empty type names in error messages (cosmetic only)
- Structs are dictionaries at runtime (no special runtime representation yet)
- No automatic conversion between compatible struct types

### Future Enhancements

Planned improvements for structs:
- FlatArray optimization for arrays of structs (10x performance)
- Value semantics (copy-on-write)
- Structural type compatibility
- Better error messages for nested types
- Native runtime representation

## Contributing

This feature is under active development. If you encounter issues or have suggestions:

1. Check the roadmap in `GDSCRIPT_STRUCTS_ROADMAP.md`
2. Review implementation details in `GODOT_DEVELOPMENT_GUIDE.md`
3. Report issues on GitHub with the `gdscript` label

## Version History

- **Godot 4.6**: Initial struct support
  - Basic struct declarations
  - Type-safe instantiation
  - Dictionary runtime representation
  - Compile-time type checking
