extends Node2D

const Simple := preload("res://demo/agents/01_agent_melee_simple.tscn")
const Charger := preload("res://demo/agents/02_agent_charger.tscn")
const Imp := preload("res://demo/agents/03_agent_imp.tscn")
const Skirmisher := preload("res://demo/agents/04_agent_skirmisher.tscn")
const Ranged := preload("res://demo/agents/05_agent_ranged.tscn")
const Combo := preload("res://demo/agents/06_agent_melee_combo.tscn")
const Nuanced := preload("res://demo/agents/07_agent_melee_nuanced.tscn")
const Demon := preload("res://demo/agents/08_agent_demon.tscn")
const Summoner := preload("res://demo/agents/09_agent_summoner.tscn")

const WAVES: Array = [
	[Simple, Simple, Nuanced],
	[Simple, Nuanced, Charger],
	[Simple, Simple, Simple, Ranged, Nuanced],
	[Simple, Simple, Summoner],
	[Ranged, Skirmisher, Nuanced, Simple, Simple],
	[Nuanced, Nuanced, Combo, Ranged, Simple],
	[Demon, Charger, Simple, Simple, Simple, Skirmisher],
	[Demon, Demon, Nuanced, Combo],
	[Summoner, Ranged, Nuanced, Nuanced, Ranged, Skirmisher, Simple],
	[Demon, Demon, Summoner, Skirmisher, Nuanced, Nuanced, Combo],
]

@export var wave_index: int = -1
@export var agents_alive: int = 0

@onready var gong: StaticBody2D = $Gong
@onready var player: CharacterBody2D = $Player
@onready var spawn_points: Node2D = $SpawnPoints
@onready var hp_bar: TextureProgressBar = %HPBar
@onready var round_counter: Label = %RoundCounter


func _ready() -> void:
	hp_bar.max_value = player.get_health().max_health
	player.get_health().damaged.connect(func(_a,_b): hp_bar.value = player.get_health().get_current())
	player.death.connect(_on_player_death)


func _update_round_counter() -> void:
	round_counter.text = "Round %s/%s" % [wave_index + 1, WAVES.size()]


func _on_gong_gong_struck() -> void:
	_start_round()


func _start_round() -> void:
	wave_index += 1
	if wave_index >= WAVES.size():
		player.set_victorious()
		round_counter.text = "Victorious!"
		return

	await get_tree().create_timer(3.0).timeout
	_update_round_counter()

	var spawns: Array = spawn_points.get_children()
	spawns.shuffle()
	for i in WAVES[wave_index].size():
		var agent_resource: PackedScene = WAVES[wave_index][i]
		var agent: CharacterBody2D = agent_resource.instantiate()
		add_child(agent)
		agent.global_position = spawns[i].global_position
		agent.death.connect(_on_agent_death)
		agent.play_summoning_effect()
		agents_alive += 1


func _on_agent_death() -> void:
	agents_alive -= 1
	if agents_alive == 0:
		_start_round()


func _on_player_death() -> void:
	await get_tree().create_timer(3.0).timeout
	get_tree().reload_current_scene()


func _on_switch_to_showcase_pressed() -> void:
	get_tree().change_scene_to_file("res://demo/scenes/showcase.tscn")
