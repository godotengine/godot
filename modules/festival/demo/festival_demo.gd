# Festival of Disguises — engine smoke test / demonstration.
#
# Run it headlessly once the engine is built:
#
#     bin/godot.linuxbsd.editor.x86_64 --headless --script modules/festival/demo/festival_demo.gd
#
# It builds a tiny slice of content in code (no .tres needed), then plays two
# runs to show the central mechanic: knowledge learned too late in run 1 is
# already known at Morning of run 2, which unlocks new access. Every Festival*
# class used here is a native engine type, not a GDScript helper.
extends SceneTree


func _make_outfit(id: StringName, role: StringName, authority: int) -> FestivalOutfit:
	var o := FestivalOutfit.new()
	o.id = id
	o.display_name = String(id).capitalize()
	o.role = role
	o.authority = authority
	return o


func _make_gatekeeper() -> FestivalNPCProfile:
	var npc := FestivalNPCProfile.new()
	npc.id = &"gatekeeper"
	npc.display_name = "Bram the Gatekeeper"
	npc.species = &"badger"
	npc.costume = &"constable"
	npc.surface_personality = "Officious and bored, secretly lonely."
	npc.schedule_morning = {"location": "north_gate", "activity": "checking passes"}
	npc.schedule_afternoon = {"location": "north_gate", "activity": "slouching"}
	npc.schedule_night = {"location": "tavern", "activity": "drinking alone"}
	npc.weather_variant_rain = {"activity": "sheltering under the arch"}
	npc.secret = &"gate_password"
	npc.interactions = [
		{
			# Only offered in the Afternoon, and only to a plain civilian: Bram
			# lets his guard down and reveals the pass phrase.
			"id": &"idle_chat",
			"requires_phase": [FestivalClock.PHASE_AFTERNOON],
			"requires_role": &"civilian",
			"dialogue": "Long shift... between us, the phrase is 'moonlit heron'.",
			"grants_knowledge": [&"gate_password"],
		},
		{
			# Available the instant Alex knows the password — which, after run 1,
			# is already true at Morning of run 2.
			"id": &"open_north_gate",
			"requires_knowledge": [&"gate_password"],
			"dialogue": "Moonlit heron? Go right through.",
			"sets_flags": {"north_gate_open": true},
			"trigger_milestone": &"reached_north_district",
		},
	]
	return npc


func _report(label: String, npc: FestivalNPCProfile) -> void:
	var reaction := Festival.resolve_reaction(npc)
	var ids: Array = []
	for entry in reaction.available_interactions:
		ids.append(entry.id)
	print("    [%s] phase=%s weather=%s perceived_role=%s -> offers %s" % [
		label,
		FestivalClock.get_phase_name(),
		FestivalWeather.get_weather_name(),
		reaction.perceived_role,
		ids,
	])


func _initialize() -> void:
	# Start from a clean slate so the demo is deterministic across invocations.
	FestivalNotebook.clear()
	FestivalRegistry.clear()

	FestivalRegistry.register_outfit(_make_outfit(&"civilian", &"civilian", 0))
	FestivalRegistry.register_outfit(_make_outfit(&"constable", &"constable", 3))
	var gatekeeper := _make_gatekeeper()
	FestivalRegistry.register_npc(gatekeeper)

	# ---- RUN 1 -----------------------------------------------------------
	print("=== RUN 1 ===")
	Festival.begin_run(1) # deterministic weather seed
	FestivalWorld.set_outfit(&"civilian")

	_report("Morning", gatekeeper)
	print("    know password? ", Festival.knows(&"gate_password"))
	print("    can open gate? ", Festival.can_interact(gatekeeper, &"open_north_gate"))

	FestivalClock.advance_phase() # -> Afternoon
	_report("Afternoon", gatekeeper)
	print("    Bram chats: ", Festival.apply_interaction(gatekeeper, &"idle_chat"))
	print("    know password now? ", Festival.knows(&"gate_password"))

	FestivalClock.advance_phase() # -> Night (too late to reach the gate meaningfully)
	Festival.end_run() # persists the notebook to disk
	print("    run ended; notebook holds ", FestivalNotebook.get_known())

	# ---- RUN 2 -----------------------------------------------------------
	print("=== RUN 2 (a fresh day) ===")
	Festival.begin_run(2)
	FestivalWorld.set_outfit(&"civilian")

	# World, inventory, flags and the clock all reset -- but knowledge did not.
	print("    Morning: know password? ", Festival.knows(&"gate_password"))
	_report("Morning", gatekeeper)
	print("    can open gate at Morning? ", Festival.can_interact(gatekeeper, &"open_north_gate"))

	var opened := Festival.apply_interaction(gatekeeper, &"open_north_gate")
	print("    opened the north gate at Morning: ", opened)
	print("    north_gate_open flag: ", FestivalWorld.get_flag("north_gate_open"))
	print("    milestone reached: ", FestivalClock.has_milestone(&"reached_north_district"))
	print("")
	print("Knowledge from run 1 unlocked new access at Morning of run 2. That is")
	print("the whole game in miniature.")

	quit()
