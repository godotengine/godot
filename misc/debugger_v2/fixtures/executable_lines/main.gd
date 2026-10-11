extends SceneTree

var total = 0

func _init():
    call_deferred("_exercise")

func _exercise():
    total += 1

    # The debugger must skip both comment-only lines.
    # This second comment must not consume another step.

    total += 2 # An inline comment remains an executable line.
    if total == 3:

        # Blank and comment-only lines inside a branch are skipped too.

        total += 3
    total += 4; total += 5
    quit()
