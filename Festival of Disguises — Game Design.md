# Festival of Disguises — Game Design Specification (AI-Optimized)

Source: :contentReference[oaicite:0]{index=0}

---

## 1. High-Level Concept

**Genre:** 2D Narrative Puzzle / Social Simulation  
**Setting:** Kraed Maas (fictional island)

**Core Premise:**  
During the annual Festival of Disguises, all inhabitants believe costumes are real identities.

**Player Objective:**  
Navigate social systems, uncover secrets, and influence outcomes across a single in-game day.

**Design Philosophy:**
- The player is expected to fail on the first run
- Knowledge, not traditional progression, is the primary form of advancement
- Replay loops are required to apply previously learned information in earlier phases of later runs

---

## 2. Player Character

### Alex (Protagonist)
- Non-binary teenager
- Lifelong resident of Kraed Maas
- Only human-presenting character

### Visual Design
- Alex uses more semi-realistic proportions
- NPCs are anthropomorphic animals
- NPCs have large heads and very small faces
- Overall style should feel cute, slightly eerie, and subtly unsettling

---

## 3. Core Gameplay System

### 3.1 Costume-Driven Identity System

NPC perception is determined by the combination of:
- Alex's current outfit
- Alex's currently held or visibly presented items
- World state flags created by previous actions and quest outcomes
- Knowledge-based dialogue options unlocked by information learned in past runs

### System Rule
NPCs do not pretend Alex is the role implied by the costume.  
They believe the costume identity is real.

### Functional Model
NPC_Reaction = Outfit + Held_Items + World_State_Flags + Available_Knowledge_Options

### Example Effects
| Outfit | Possible NPC Reaction |
|------|------------------------|
| Constable | Confessions, fear, avoidance, requests for help |
| Merchant | Trade access, pricing changes, inventory changes |
| Civilian | Neutral dialogue, lower authority, reduced access |

---

### 3.2 State Propagation

Actions can:
- Unlock future interactions
- Lock alternative paths
- Affect unrelated NPCs indirectly
- Change what information becomes available later in the day
- Create opportunities that only become usable in a future run

This creates:
- Cascading consequence chains
- Systemic social gameplay
- Cross-run planning

---

## 4. Quest System

### Quest Types

| Type | Description |
|------|-------------|
| Fetch | Retrieve an item for an NPC |
| Trade Chain | Multi-step item exchanges |
| Dialogue-Gated | Requires specific outfit, knowledge, or timing |
| Minigame | Context-specific interaction or challenge |
| Timed/Conditional | Depends on phase, weather, or prior actions |
| Secret Discovery | Requires probing NPC systems and contradictions |

### Constraints
- No combat system
- All conflict is social, informational, moral, or logistical

---

## 5. Time System

### 5.1 Phase Structure

| Phase | Description |
|------|-------------|
| Morning | Festival setup, lighter tone, weather determined, early opportunities available |
| Afternoon | Escalation, secrets emerge, social complexity increases |
| Night | Climax, consequences resolve, run ends regardless of success |

### Rules
- Phases are irreversible
- Progression is triggered by hidden or semi-hidden milestone events
- Some content is permanently lost once a phase ends
- A full run always ends after Night

---

### 5.2 Weather System

Weather state at run start:
- Rain
- Sun

### Weather Effects
Weather may affect:
- NPC schedules
- NPC mood/state variants
- Quest availability
- Certain items or locations
- Which interactions are safe, visible, or accessible

### Weather Rule
Morning weather is randomized at the start of each run to ensure that repeated runs are not perfectly identical.

---

## 6. Replay & Knowledge Loop System

### Core Principle
Progression is based on applying knowledge across runs, not on carrying forward standard inventory or stats.

### Reset Per Run

| System | Reset |
|------|--------|
| Inventory | Yes |
| Outfits | Yes |
| Temporary world flags | Yes |
| Current phase progress | Yes |
| World state | Yes |

### Persistent Across Runs
- Player memory
- Notebook entries
- Knowledge of schedules, secrets, rumors, passwords, routes, and conditions

---

### 6.1 Knowledge-Based Progression Rules

Replay does not:
- Automatically rewrite NPC dialogue trees
- Produce wholly different character personalities on later runs
- Create alternate dialogue scripts solely because a new run started

Replay does:
- Let the player use learned information at earlier points in the day
- Let the player access areas or items that were previously unreachable because the information was learned too late
- Unlock new dialogue options when the player has learned relevant information such as a rumor, secret, password, or fact
- Enable new quest paths based on player-applied knowledge

---

### 6.2 Example Pattern

Example:
- Run 1, Afternoon: player learns a password
- Run 1, Night: it is too late to use the password meaningfully
- Run 2, Morning: player uses the password immediately
- Result: new area, item, or quest branch becomes accessible

### Meaning of Replay
Replay is not mainly about seeing altered writing.  
Replay is about exploiting timing with previously discovered information.

---

### 6.3 Unlock Types

Knowledge learned in one run may unlock the following in later runs:
- New areas
- New items
- New routes
- New quest branches
- New dialogue options
- Earlier access to content that otherwise appears too late

### Dialogue Constraint
Knowledge may unlock new dialogue options in later runs, but the base dialogue content itself should not be considered a separate replay-specific rewrite.

---

## 7. Notebook System

### Purpose
The notebook is a persistent cross-run record of what the player has learned.

### NPC Knowledge Categories

Each NPC has at least the following discoverable information:
- Secret: a personal truth they guard carefully
- Dark Secret: something disturbing, dangerous, or morally significant
- Rumor: something true that others say about them
- False Rumor: something believed about them that is not true

### Notebook Behavior
- Entries persist across runs
- Entries support planning and route optimization
- Entries may unlock new dialogue options if Alex now knows relevant information
- Entries may help the player identify contradictions, leverage timing, or access hidden systems

### NPC Knowledge Schema (Conceptual)
- secret
- dark_secret
- rumor
- false_rumor

---

## 8. NPC System Design

### NPC Requirements

Each NPC must be designed with the following:
- Name
- Species
- Costume
- Surface personality
- Morning schedule/state
- Afternoon schedule/state
- Night schedule/state
- Rain variant or weather-sensitive behavior
- Secret
- Dark secret
- Rumor
- False rumor
- At least one outfit-dependent interaction
- At least one item-dependent interaction
- At least one knowledge-dependent interaction
- At least one way their behavior shifts based on phase or world state

### NPC Schema (Conceptual)
- name
- species
- costume
- surface_personality
- schedule_morning
- schedule_afternoon
- schedule_night
- weather_variant_rain
- weather_variant_sun
- secret
- dark_secret
- rumor
- false_rumor
- outfit_interactions
- item_interactions
- knowledge_interactions
- world_state_dependencies

---

## 9. Dialogue System Rules

### Dialogue Structure
Dialogue is contextual and branching based on:
- Outfit
- Items
- Current phase
- Weather
- World state flags
- Available learned knowledge

### Replay Constraint
- Dialogue does not broadly change just because this is a later run
- Instead, later runs may provide additional dialogue options if the player has learned relevant information previously
- This means replay expands player agency, not authorially rewritten scenes

---

## 10. Progression Model

### Progression Type
**Knowledge-Based Progression**

The game does not primarily progress through:
- XP
- Levels
- Stats
- Permanent combat upgrades

The game does progress through:
- Information
- Better timing
- Route planning
- Understanding NPC logic
- Using learned facts in earlier or more effective contexts

---

## 11. Tone & Narrative

### Tone
- Dark humor
- Slight meta awareness
- Light surrealism
- Emotional unease emerging naturally from comedy

### Emotional Flow
Humor leads into discomfort, then realization.

### Setting Tone
Kraed Maas should feel like a world with its own internal logic.  
It should not feel alien so much as wrong in a familiar way.

---

## 12. Inspirations

### Mechanical Inspiration
- *Majora's Mask*
  - Time loop structure
  - NPC schedules
  - Single-day pressure
  - Meaningful repetition

### Narrative/System Inspiration
- *Detroit: Become Human*
  - Branching consequence structures
  - Outcome sensitivity
  - Social-choice pressure

---

## 13. Art Direction

### Style
- 2D
- Cute + creepy hybrid
- Rounded silhouettes
- Muted but saturated palettes

### NPC Visual Design
- Anthropomorphic animal people
- Large rounded heads
- Very small expressive faces
- Charming at first glance, subtly unsettling when observed closely

### Alex Visual Role
- Visually distinct from the islanders
- Slightly more realistic
- Should feel like Alex belongs in the setting narratively but not fully visually

---

## 14. Design Pillars

1. Identity is systemic, not cosmetic
2. Failure is required, especially at first
3. Knowledge is progression
4. NPCs are layered systems, not one-note quest dispensers
5. Player agency is indirect and informational
6. Replay is about better use of learned information, not simple repetition

---

## 15. Implementation Priorities

### P0 (MVP)
- Movement system
- Inventory system
- Outfit system
- NPC interaction system
- Phase progression system

### P1
- Notebook persistence
- Basic quest chains
- Weather system
- Knowledge-gated dialogue options

### P2
- Knowledge-gated areas and items
- More complex NPC schedules
- Multi-layer route planning
- Broader consequence propagation

---

## 16. Key Risks

- Over-complex state interactions
- Player confusion if knowledge loops are not legible
- Replay feeling repetitive if newly usable knowledge does not create meaningful changes in access
- Too much hidden logic without enough feedback

---

## 17. Open Questions

- How visible should knowledge gating be to the player?
- Should the notebook auto-fill, partially auto-fill, or require manual confirmation?
- How strongly should the game communicate that failure and reset are expected?
- How much should the player know about phase-ending conditions?
- Should some knowledge remain purely player-memory based rather than notebook-recorded?

---