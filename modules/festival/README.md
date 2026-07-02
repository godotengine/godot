# Festival module — a Godot engine tailored for *Festival of Disguises*

This module turns stock Godot into a purpose-built engine for **Festival of
Disguises**, the 2D narrative puzzle / social simulation described in
[`Festival of Disguises — Game Design.md`](../../Festival%20of%20Disguises%20%E2%80%94%20Game%20Design.md).

Instead of rebuilding the game's systems in GDScript on top of a generic engine,
the game's *nouns and verbs are first-class engine types*. Costumes, knowledge,
NPC profiles, the day/phase clock, the weather roll, the per-run world state and
the cross-run notebook are all native classes registered in `ClassDB`. They show
up in the editor inspector, are usable from GDScript and C#, and are documented
in the built-in class reference.

The whole engine is organized around one rule from the design spec:

```
NPC_Reaction = Outfit + Held_Items + World_State_Flags + Available_Knowledge_Options
```

and one structural insight: **almost everything resets each run; only knowledge
persists.**

## Class catalogue

### Authoring resources (`Resource`, edit as `.tres`)

| Class | Purpose | Design section |
|-------|---------|----------------|
| `FestivalOutfit` | A costume identity with a perceived `role` and `authority`. | §3.1 Costume-Driven Identity |
| `FestivalItem` | An inventory item; can be held or *presented*. | §3.1, §4 |
| `FestivalKnowledge` | One discoverable fact (secret / dark secret / rumor / false rumor / password / schedule / route / fact). | §6, §7 |
| `FestivalNPCProfile` | The complete NPC schema: species, costume, per-phase schedules, weather variants, the four guarded facts, and data-driven interactions. | §8 |

### Runtime singletons (globals in GDScript)

| Singleton | Responsibility | Reset each run? | Design section |
|-----------|----------------|-----------------|----------------|
| `FestivalClock` | Irreversible Morning → Afternoon → Night → Ended phases + milestones. | **Yes** | §5.1 |
| `FestivalWeather` | Sun/Rain, rolled at run start. | **Yes** (re-rolled) | §5.2 |
| `FestivalWorld` | Per-run flags, inventory, current outfit, presented items. | **Yes** | §6 (Reset table) |
| `FestivalNotebook` | Everything Alex has ever learned; saved to disk. | **No — persists** | §6, §7 |
| `FestivalRegistry` | Content database: id → resource lookup. | No | — |
| `Festival` (`FestivalDirector`) | Run lifecycle + the perception/interaction resolver. | orchestrator | §3.1, §3.2 |

### Scene node

| Node | Purpose |
|------|---------|
| `FestivalNPC` (`Node2D`) | Places a `FestivalNPCProfile` in the world, auto-syncs its schedule state to the clock/weather, and offers a one-call `interact()`. |

## The run loop

```gdscript
func _ready() -> void:
    Festival.load_notebook()          # persistent knowledge from previous runs
    FestivalRegistry.scan_directory("res://content")  # load all authored .tres
    Festival.begin_run()              # reset world, reset clock, roll weather

    # ... play the day: change outfits, present items, talk to NPCs ...

    Festival.end_run()                # fast-forward to Night end + save notebook
```

`begin_run()` wipes `FestivalWorld`, resets `FestivalClock` to Morning, and
re-rolls `FestivalWeather` — but never touches `FestivalNotebook`. That single
asymmetry is the entire knowledge-loop design.

## Perception in one call

```gdscript
FestivalWorld.set_outfit("constable")     # islanders now believe Alex IS a constable
FestivalWorld.add_item("warrant")
FestivalWorld.present_item("warrant")

var reaction := $Mayor.interact()
# reaction = {
#   "npc": &"mayor",
#   "perceived_role": &"constable",
#   "authority": 3,
#   "surface_personality": "...",
#   "state": { "location": "plaza", "activity": "greeting guests" },
#   "available_interactions": [ { "id": &"confess_bribe", "dialogue": "..." } ],
# }

if reaction.available_interactions.size() > 0:
    Festival.apply_interaction($Mayor.profile, &"confess_bribe")
    # -> learns knowledge, sets flags, moves items, maybe advances the phase.
```

An NPC interaction is authored as a plain `Dictionary` on the profile, so writers
never touch code. Requirement keys gate it and outcome keys fire when it is
applied — see the `FestivalNPCProfile` class docs for the full schema.

## The knowledge loop, concretely (design §6.2)

```
Run 1, Afternoon: apply an interaction whose outcome grants &"gate_password".
Run 1, Night:     too late to use it.
Run 2, Morning:   Festival.knows(&"gate_password") is already true (notebook
                  persisted), so an interaction gated on requires_knowledge:
                  [&"gate_password"] is available immediately -> new area/flag.
```

The runnable [`demo/festival_demo.gd`](demo/festival_demo.gd) plays exactly this
two-run scenario headlessly and prints each step.

## Coverage of the implementation priorities (design §15)

- **P0** — outfit system, inventory, NPC interaction, phase progression: `FestivalWorld`,
  `FestivalOutfit`, `FestivalItem`, `FestivalNPC`, `FestivalDirector`, `FestivalClock`.
  (Movement stays generic Godot 2D — `FestivalNPC` is a `Node2D`.)
- **P1** — notebook persistence, quest/interaction chains, weather, knowledge-gated
  dialogue: `FestivalNotebook`, data-driven interactions, `FestivalWeather`,
  `requires_knowledge` gating.
- **P2** — knowledge-gated areas/items, richer schedules, consequence propagation:
  `requires_flags` + `sets_flags`, per-phase schedules with weather variants,
  `trigger_milestone`/`advance_phase` outcomes.

## Building

This is a standard in-tree Godot module. From the repository root:

```sh
scons platform=linuxbsd target=editor
# run the tests that ship with the module:
scons platform=linuxbsd target=editor tests=yes
bin/godot.linuxbsd.editor.x86_64 --test --test-suite=Festival
```

The module is self-contained under `modules/festival/` with no third-party
dependencies, so it also compiles into template/export builds unchanged.

## Files

```
modules/festival/
  festival_outfit.*        FestivalItem  FestivalKnowledge  FestivalNPCProfile
  festival_clock.*         FestivalWeather
  festival_world.*         FestivalNotebook  FestivalRegistry
  festival_director.*      the Festival singleton + perception resolver
  festival_npc.*           the in-scene NPC node
  register_types.*  config.py  SCsub
  doc_classes/             built-in class reference for every class above
  tests/test_festival.h    doctest unit tests for the pure resource logic
  demo/festival_demo.gd    a runnable two-run knowledge-loop demo
```
