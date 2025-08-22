# ------------------------------------------------------------------------------
# Description
# -----------
# Entry point for the command line interface.  The actual logic for GUT's CLI
# is in addons/gut/cli/gut_cli.gd.
#
# This script should conform to, or ignore, the strictest warning settings.
# ------------------------------------------------------------------------------
extends SceneTree

var VersionConversion = load("res://addons/gut/version_conversion.gd")

@warning_ignore("unsafe_method_access")
@warning_ignore("inferred_declaration")
func _init() -> void:
	if(VersionConversion.error_if_not_all_classes_imported()):
		quit(0)
		return

	var max_iter := 20
	var iter := 0

	var Loader : Object = load("res://addons/gut/gut_loader.gd")

	# Not seen this wait more than 1.
	while(Engine.get_main_loop() == null and iter < max_iter):
		await create_timer(.01).timeout
		iter += 1

	if(Engine.get_main_loop() == null):
		push_error('Main loop did not start in time.')
		quit(0)
		return

	var cli : Node = load('res://addons/gut/cli/gut_cli.gd').new()
	get_root().add_child(cli)

	Loader.restore_ignore_addons()
	cli.main()




# ##############################################################################
#(G)odot (U)nit (T)est class
#
# ##############################################################################
# The MIT License (MIT)
# =====================
#
# Copyright (c) 2025 Tom "Butch" Wesley
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ##############################################################################
