# ------------------------------------------------------------------------------
# This script should be loaded as soon as possible when running tests.  This
# will disable warnings and then load all scripts that are registered with the
# LazyLoader.
#
# Once you are ready to run tests, restore_ignore_addons should be called so
# that it has the expected value.  This should be done after whatever loaded
# this is done loading and doing setup stuff.
#
# This was created after a first attempt to suppress all GUT warnings did not
# work for the strictest warning settings.  This has turned the LazyLoader into
# just a Loader...so maybe all that should be reworked or renamed.  A problem
# for a time when we are absolutely sure that all warnings are being correctly
# suppressed I suppose.
#
# You can use the cli script test/resources/change_project_warnings.gd to
# quickly alter project warning levels for testing purposes.
# 	gdscript test/resources/change_project_warnings.gd --headless ++ -h
#
# You can set project warning settings from the command line with:
#	godot -s addons/gut/cli/change_project_warnings.gd ++ -h
#
# This script should conform to, or ignore, the strictest warning settings.
# ------------------------------------------------------------------------------
const WARNING_PATH : String = 'debug/gdscript/warnings/'


static var were_addons_disabled : bool = true


@warning_ignore("unsafe_method_access")
@warning_ignore("unsafe_property_access")
@warning_ignore("untyped_declaration")
static func _static_init() -> void:
	were_addons_disabled = ProjectSettings.get(str(WARNING_PATH, 'exclude_addons'))
	ProjectSettings.set(str(WARNING_PATH, 'exclude_addons'), true)

	var WarningsManager = load('res://addons/gut/warnings_manager.gd')

	# Turn everything back on (if it originally was on) if the warnings manager
	# is disabled.  This makes sure we see all the warnings for all the scripts
	# in the LazyLoader (except WarningsManager, but that's not a big deal).
	#
	# With the warnings manager disabled and all_warn warnings:
	#	test_warnings_manager.gd 	-> 5471 errors
	#	full run 				 	-> 131,742 errors
	#
	# With the warnings manager disabled and gut_default warnings:
	#	test_warnings_manager.gd 	-> 46 errors
	#	full run 					-> 165 errors.
	if(WarningsManager.disabled):
		ProjectSettings.set(str(WARNING_PATH, 'exclude_addons'), were_addons_disabled)

	# Force a reference to utils.gd by path.  Using the class_name would cause
	# utils.gd to load when this script loads, before we could turn off the
	# warnings.
	var _utils : Object = load('res://addons/gut/utils.gd')

	# Since load_all exists on the LazyLoader, it should be done now so nothing
	# sneaks in later...This essentially defeats the "lazy" part of the
	# LazyLoader, but not the "loader" part of LazyLoader.
	_utils.LazyLoader.load_all()

	# Make sure that the values set in WarningsManager's static_init actually
	# reflect the project settings and not whatever we do here to make things
	# not warn.
	WarningsManager._project_warnings.exclude_addons = were_addons_disabled


# this can be called before tests are run to reinstate whatever exclude_addons
# was set to before this script disabled it.
static func restore_ignore_addons() -> void:
	ProjectSettings.set(str(WARNING_PATH, 'exclude_addons'), were_addons_disabled)




# ##############################################################################
# (G)odot (U)nit (T)est class
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
