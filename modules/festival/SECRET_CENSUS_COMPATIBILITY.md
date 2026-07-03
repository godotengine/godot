# Secret Census ↔ Festival Engine — Compatibility Contract

This document is the **binding contract** between the Secret Census world-building
app (`jxburros/Secret-Census`) and the Festival of Disguises Godot fork
(`jxburros/godot-costume`, `modules/festival`). An identical copy lives in both
repositories:

- Secret Census: `GODOT_COMPATIBILITY.md` (repo root)
- Godot fork: `modules/festival/SECRET_CENSUS_COMPATIBILITY.md`

**If you change anything listed as LOCKED below in either app, the export/import
pipeline breaks.** Keep both copies of this document in sync when the contract
is ever extended.

---

## 1. The interchange file

Secret Census exports one **JSON** file (UTF-8) per game, recommended extension
`.census.json`. The Godot fork imports it with the `FestivalCensusImporter`
singleton:

```gdscript
# One-time import, producing editable .tres resources:
FestivalCensusImporter.import_file("res://my_world.census.json", "res://content")
# Later runs just load the (possibly hand-edited) resources:
FestivalRegistry.scan_directory("res://content")
```

### Top-level shape — LOCKED

```json
{
  "format": "secret-census-world",
  "formatVersion": 1,
  "exportedAt": 1751500000000,
  "game":      { "id": "...", "name": "...", "description": "...", "createdAt": 0 },
  "settings":  { ...subset of AppSettings, see §3... },
  "npcs":      [ ...NPC records... ],
  "locations": [ ...Location records... ],
  "items":     [ ...Item records... ],
  "outfits":   [ ...Outfit records... ],
  "plotHooks": [ ...PlotHook records... ],
  "events":    [ ...GameEvent records... ]
}
```

- `format` MUST be the string `secret-census-world`. LOCKED.
- `formatVersion` MUST be `1` for this contract. The importer rejects files
  with a higher major version. Only bump it for a breaking change, and only
  together with a matching importer update.
- Every array may be empty, and **any field inside a record may be missing,
  empty, or null** — partially filled worlds are valid. The importer applies
  neutral defaults. "Not all fields have to be filled out to be exported" is a
  feature of the format.

## 2. Entity records — field names are LOCKED

Records are exported **verbatim** from Secret Census's `src/types.ts`, with the
exact camelCase field names below. These interfaces are the single source of
truth; the fields listed here are **locked in** — they must never be renamed,
removed, or change type/meaning in either app. *Adding* new optional fields is
always allowed (see §7).

| Package key | Secret Census type | Locked fields (current contract) |
|---|---|---|
| `npcs` | `NPC` | `id, npcNumber, name, species, personality, backstory, secret, darkSecret, workplace, occupation, residence, notes, rumors, relationships, customFields, locations {morningSun, morningRain, afternoon, night}, locationOptions, dialogue, givesPasswords, linkedPlotHookIds, missingDialogue, createdAt, lastUpdated, ownerId` |
| `locations` | `Location` | `id, name, type, parentId, description, notes, requiredPassword, isResidence, residentNpcIds, linkedPlotHookIds, linkedItemIds, connectedLocationIds, isProgression, isTemplate, lastUpdated, ownerId` |
| `items` | `Item` | `id, name, description, locationType, locationId, grantsExtraPower, powerDescription, powerCategory, linkedPlotHookId, givesPassword, stageAvailability, isProgression, lastUpdated, ownerId` |
| `outfits` | `Outfit` | `id, name, description, grantsExtraPower, powers, stageAvailability, linkedDialogueIds, lastUpdated, ownerId` |
| `plotHooks` | `PlotHook` | `id, title, description, status, linkedNpcIds, linkedLocationIds, linkedItemIds, linkedRumorIds, linkedEventIds, linkedNpcSecrets, linkedOutfits, isProgression, lastUpdated, ownerId` |
| `events` | `GameEvent` | `id, name, description, script, locationId, triggerDialogueId, triggerNpcId, characterIds, gameStates, isProgression, lastUpdated, ownerId` |

Locked **nested** structures (embedded, exported verbatim):

- `Rumor`: `id, text, sourceType, sourceId, isTrue, verification, linkedDialogueId, linkedItemId`
- `Relationship`: `targetNpcId, type, reciprocalType`
- `CustomField`: `id, label, value`
- `DialogueEntry`: `id, gameState, outfitId, outfit, requiredPassword, type, hasSpecialItem, specialItemName, text, linkedRumorId, triggersEventIds, givesItemId, givesPassword`
- `LocationOption`: `locationId, requiredPassword`
- `StageAvailability`: `gameState, locationType, locationId, availableByNormalMeans`

Locked **enum-like string values**:

- `Location.type`: `overworld | building | sub-location | sub-sub-location`
- `Item.locationType`, `StageAvailability.locationType`, `Rumor.sourceType`
  (`npc | item | external`)
- `DialogueEntry.type`: `standard | rumor | secret | darkSecret`
- `PlotHook.status`: `active | completed | failed`
- `PlotHook.linkedNpcSecrets[].secretType`: `secret | darkSecret`

### IDs — LOCKED semantics

- Every entity `id` is an opaque, unique string, stable for the lifetime of the
  entity. Re-exporting the same world must reuse the same ids.
- Cross-references (`linked*Ids`, `parentId`, `locationId`, `residentNpcIds`,
  `targetNpcId`, `sourceId`, …) always refer to those ids. Dangling references
  are tolerated by the importer (logged, not fatal), so partial exports work.
- **Never recycle an id for a different entity** — the Godot notebook persists
  derived knowledge ids (§5) across runs and save files.

## 3. Exported settings subset — LOCKED

Only world/game data is exported from `AppSettings` — never API keys, sync
tokens, AI provider config, theme, or accessibility settings:

```
gameName, gameDescription, fantasyLevel, scifiLevel, generateDarkSecrets,
species, workplaces, occupations, outfits, gameStates, itemCategories,
plotHookStatuses, connectionTypes, worldRules, notebook
```

`settings.gameStates` is the ordered list of game-state names that
`DialogueEntry.gameState`, `StageAvailability.gameState` and
`GameEvent.gameStates` refer to. Game-state **names are referenced by value**,
so renaming a game state in Secret Census renames content references with it —
export again after renaming.

## 4. Deliberately excluded (for now)

- **Visual content** (`imageUrl` on NPC/Item/Outfit) and **audio content** are a
  future step and are NOT part of formatVersion 1. Exporters must omit
  `imageUrl`; the importer ignores it if present.
- Secrets/credentials: `googleApiKey, anthropicApiKey, openaiApiKey,
  customApiKey, googleDriveAccessToken, oneDriveAccessToken`, endpoints and
  model names must never be exported.

## 5. Godot-side mapping — LOCKED

The importer maps records onto festival resources. Resource **class names and
property names below are locked** because saved `.tres` files and user edits
depend on them.

| Package data | Godot resource |
|---|---|
| `npcs[]` | `FestivalNPCProfile` |
| `locations[]` | `FestivalLocation` |
| `items[]` | `FestivalItem` |
| `outfits[]` | `FestivalOutfit` |
| `plotHooks[]` | `FestivalPlotHook` |
| `events[]` | `FestivalEvent` |
| NPC secrets / rumors / passwords | generated `FestivalKnowledge` |
| `game` + `settings` | `FestivalRegistry.world_info` Dictionary |

Every imported resource carries a `census_data` Dictionary holding the **raw
source record verbatim** (including any fields this contract doesn't know yet),
so no Secret Census detail is ever lost, and everything remains inspectable and
editable in the Godot editor.

### Derived knowledge ids — LOCKED (notebook saves depend on these)

| Source | Generated `FestivalKnowledge` id | Category |
|---|---|---|
| `NPC.secret` (non-empty) | `npc.<npcId>.secret` | `CATEGORY_SECRET` |
| `NPC.darkSecret` (non-empty) | `npc.<npcId>.dark_secret` | `CATEGORY_DARK_SECRET` |
| `NPC.rumors[i]` | `rumor.<rumorId>` | `CATEGORY_RUMOR` if `isTrue`, else `CATEGORY_FALSE_RUMOR` (`veracity = isTrue`) |
| any password string | `password.<slug>` | `CATEGORY_PASSWORD` |

`slug(password)` = lowercase, every run of non-alphanumeric characters replaced
by `_`, leading/trailing `_` trimmed. Example: `"Say Friend & Enter"` →
`password.say_friend_enter`. Password strings are collected from
`DialogueEntry.requiredPassword` / `givesPassword`, `NPC.givesPasswords`,
`Item.givesPassword`, `Location.requiredPassword` and
`LocationOption.requiredPassword`. **Changing a password's text changes its
knowledge id** — treat password text as an identifier.

The NPC profile's `secret`, `dark_secret`, `rumor`, `false_rumor` StringName
fields are set to the derived ids above (`rumor`/`false_rumor` = the NPC's
first true/untrue rumor; all further rumors still exist as knowledge).

### Schedule mapping — LOCKED

Secret Census per-phase locations map to schedule Dictionaries:

- `locations.morningSun` → `schedule_morning = { "location": <id>, "options": [...] }`
- `locations.morningRain` → `weather_variant_rain = { "morning": { "location": <id>, "options": [...] } }`
- `locations.afternoon` → `schedule_afternoon = { "location": <id>, ... }`
- `locations.night` → `schedule_night = { "location": <id>, ... }`

`FestivalNPCProfile.get_state_for(phase, weather)` treats the weather-variant
keys `"morning"`, `"afternoon"`, `"night"` (whose values are Dictionaries) as
**phase-scoped**: they merge only when that phase is active. All other variant
keys merge for every phase, as before. `"options"` entries are the raw
`LocationOption` dictionaries (camelCase keys).

### DialogueEntry → interaction mapping — LOCKED

Each `DialogueEntry` becomes one entry of `FestivalNPCProfile.interactions`:

| Interaction key | Source |
|---|---|
| `id` | `DialogueEntry.id` |
| `dialogue` | `text` |
| `game_state` | `gameState` (always preserved verbatim) |
| `dialogue_type` | `type` |
| `requires_phase` | `[0/1/2]` when the game-state name contains `morning`/`afternoon`/`night` (case-insensitive); otherwise omitted |
| `requires_outfit` | `outfitId`, falling back to legacy `outfit` |
| `requires_knowledge` | `password.<slug>` of `requiredPassword` (if any) |
| `special_item_name` | `specialItemName` when `hasSpecialItem`; also `requires_items: [<itemId>]` when an exported item's name matches exactly (case-insensitive) |
| `grants_knowledge` | `rumor.<linkedRumorId>`; `npc.<id>.secret` when `type == "secret"`; `npc.<id>.dark_secret` when `type == "darkSecret"`; `password.<slug>` of `givesPassword` |
| `gives_items` | `[givesItemId]` (if any) |
| `triggers_events` | `triggersEventIds` |
| `census` | the raw `DialogueEntry` record verbatim |

The `FestivalDirector` ignores interaction keys it doesn't evaluate, so extra
keys are safe; they are preserved through editing.

### Nested pass-through data

`relationships`, `customFields`, `stageAvailability`, `locationOptions`,
`linkedNpcSecrets`, and interaction `census` payloads are stored in Godot as
raw Arrays/Dictionaries **with their original camelCase keys**. Those key names
are part of this contract.

## 6. What must NOT change — summary checklist

**In Secret Census:**
- Do not rename/remove/retype any field listed in §2–§3 (all current fields are
  locked in). Adding new optional fields is fine.
- Do not change id generation to recycle or destabilize existing ids.
- Do not change the meaning of the four NPC schedule slots or of
  `gameStates` string matching.
- Keep the export excluding credentials and (for v1) visual/audio content.
- Keep `format` / `formatVersion` handling exactly as in §1.

**In the Godot fork (`modules/festival`):**
- Do not rename the resource classes `FestivalNPCProfile, FestivalLocation,
  FestivalItem, FestivalOutfit, FestivalKnowledge, FestivalPlotHook,
  FestivalEvent`, nor any of their exported properties — saved `.tres` content
  and this pipeline depend on them.
- Do not change the derived knowledge-id scheme (§5) — `FestivalNotebook`
  save files persist those ids across runs.
- Do not change the phase order `Morning(0) → Afternoon(1) → Night(2)` or the
  weather values `Sun(0) / Rain(1)`.
- Do not remove the phase-scoped weather-variant merge, the interaction keys in
  §5, or `census_data` round-tripping.
- `FestivalCensusImporter` must keep accepting every valid formatVersion-1
  package (including sparse ones) without error.

## 7. Evolving the contract

1. **Additive changes** (new optional fields, new entity arrays): allowed any
   time. The importer must ignore unknown top-level arrays and preserve unknown
   record fields inside `census_data`. No version bump needed.
2. **Breaking changes** (rename/remove/retype anything locked): bump
   `formatVersion`, update the importer to accept both old and new versions,
   and update **both copies** of this document in the same change.
3. Visual/audio content will be added as a future additive step (planned:
   optional asset manifest referencing files shipped alongside the JSON).
