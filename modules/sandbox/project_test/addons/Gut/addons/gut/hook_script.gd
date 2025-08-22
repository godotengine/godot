class_name GutHookScript
## This script is the base for custom scripts to be used in pre and post
## run hooks.
##
## GUT Wiki:  [url=https://gut.readthedocs.io]https://gut.readthedocs.io[/url]
## [br][br]
## Creating a hook script requires that you:[br]
##  - Inherit [code skip-lint]GutHookScript[/code][br]
##  - Implement a [code skip-lint]run()[/code] method[br]
##  - Configure the path in GUT (gutconfig aand/or editor) as the approparite hook (pre or post).[br]
##
## See [wiki]Hooks[/wiki]


## Class responsible for generating xml.  You could use this to generate XML
## yourself instead of using the built in GUT xml generation options.  See
## [addons/gut/junit_xml_export.gd]
var JunitXmlExport = load('res://addons/gut/junit_xml_export.gd')

## This is the instance of [GutMain] that is running the tests.  You can get
## information about the run from this object.  This is set by GUT when the
## script is instantiated.
var gut  = null

# the exit code to be used by gut_cmdln.  See set method.
var _exit_code = null

var _should_abort =  false

## Virtual method that will be called by GUT after instantiating this script.
## This is where you put all of your logic.
func run():
	gut.logger.error("Run method not overloaded.  Create a 'run()' method in your hook script to run your code.")


## Set the exit code when running from the command line.  If not set then the
## default exit code will be returned (0 when no tests fail, 1 when any tests
## fail).
func set_exit_code(code : int):
	_exit_code  = code

## Returns the exit code set with [code skip-lint]set_exit_code[/code]
func get_exit_code():
	return _exit_code

## Usable by pre-run script to cause the run to end AFTER the run() method
## finishes.  GUT will quit and post-run script will not be ran.
func abort():
	_should_abort = true

## Returns if [code skip-lint]abort[/code] was called.
func should_abort():
	return _should_abort
