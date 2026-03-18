extends CanvasLayer

@onready var title_label: Label = $PanelContainer/MarginContainer/VBoxContainer/TitleLabel
@onready var summary_label: RichTextLabel = $PanelContainer/MarginContainer/VBoxContainer/SummaryLabel
@onready var phases_label: RichTextLabel = $PanelContainer/MarginContainer/VBoxContainer/PhasesLabel
@onready var recommendations_label: RichTextLabel = $PanelContainer/MarginContainer/VBoxContainer/RecommendationsLabel
@onready var footer_label: Label = $PanelContainer/MarginContainer/VBoxContainer/FooterLabel

func _ready() -> void:
	visible = false

func show_report(report: Dictionary) -> void:
	var overall: Dictionary = report.get("overall", {})
	var score := float(report.get("score", 0.0))
	var duration_s := float(report.get("duration_s", 0.0))
	var output_path := str(report.get("output_path", ""))
	var phases: Array = report.get("phase_summaries", [])
	var recommendations: Array = report.get("recommendations", [])

	title_label.text = "Unified Benchmark Complete"
	summary_label.text = "Score: %.1f/100\\nDuration: %.1fs\\nAvg FPS: %.1f\\nP1 FPS: %.1f\\nP99 Frame: %.2f ms\\nStability: %.2f" % [
		score,
		duration_s,
		float(overall.get("avg_fps", 0.0)),
		float(overall.get("p1_fps", 0.0)),
		float(overall.get("p99_frame_ms", 0.0)),
		float(overall.get("stability", 0.0)),
	]

	var phase_lines: PackedStringArray = []
	phase_lines.append("Per-Phase Snapshot")
	for phase_variant in phases:
		if not (phase_variant is Dictionary):
			continue
		var phase: Dictionary = phase_variant
		phase_lines.append("- %s: avg %.1f fps | p1 %.1f | p99 %.2f ms" % [
			str(phase.get("name", "phase")),
			float(phase.get("avg_fps", 0.0)),
			float(phase.get("p1_fps", 0.0)),
			float(phase.get("p99_frame_ms", 0.0)),
		])
	phases_label.text = "\\n".join(phase_lines)

	var recommendation_lines: PackedStringArray = []
	recommendation_lines.append("Suggested Setting Adjustments")
	for rec_variant in recommendations:
		if not (rec_variant is Dictionary):
			continue
		var rec: Dictionary = rec_variant
		recommendation_lines.append("- %s: %s -> %s" % [
			str(rec.get("setting", "setting")),
			str(rec.get("current", "n/a")),
			str(rec.get("suggested", "n/a")),
		])
		recommendation_lines.append("  Why: %s" % str(rec.get("reason", "")))
		recommendation_lines.append("  Tradeoff: %s" % str(rec.get("tradeoff", "")))
	recommendations_label.text = "\\n".join(recommendation_lines)

	footer_label.text = "Results written to: %s\\nPress Esc to quit, R to rerun." % output_path
	visible = true
