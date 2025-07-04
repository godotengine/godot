@warning_ignore_start("unused_variable", "unused_local_constant", "unused_parameter")

func test():
	print("warn")

# These cases are tested:
# * types
# * native classes (except global classes)
# * @GlobalScope functions
# * @GDScript functions
# * @GlobalScope enum classes
# * @GlobalScope enum members
# * @GDScript constants

# Some built-in @GDScript functions cannot be shadowed,
# because they count as keywords,
# i.e. assert, preload.

# NamedEnum.MEMBER does not shadow anything.

# --- Variables ---

# Member variable declarations.

#var int = 1 # TODO: Inconsistent behavior.
#var Node = 2 # TODO: Inconsistent behavior.
var abs = 3
var range = 4
var Error = 5
var OK = 6
var PI = 7

# Local variable declarations.

func shadowed_local_variables():
	#var int = 11 # TODO: Inconsistent behavior.
	#var Node = 12 # TODO: Inconsistent behavior.
	var abs = 13
	var range = 14
	var Error = 15
	var OK = 16
	var PI = 17

# --- Constants ---

const min = 22
# The rest behave similarly to member variable declarations.

func shadowed_local_constants():
	const min = 32
	# The rest behave similarly to local variable declarations.

# --- Named Enum Classes ---

enum max { A = 42 }
# The rest behave similarly to member variable declarations.

# --- Named Enum Members ---

# No warnings here.

enum NamedEnum {
	abs = 52,
	# The rest behave similarly to local variable declarations.
}

# --- Unnamed Enum Members ---

enum {
	round = 62,
	# The rest behave similarly to member variable declarations.
}

# --- Functions ---

#func float(): # TODO: Inconsistent behavior.
#	print(71)

#func Node2D(): # TODO: Inconsistent behavior.
#	print(72)

# TODO: This should also produce NATIVE_METHOD_OVERRIDE, see Issue #106840.
func ceil():
	print(73)

# TODO: This should also produce NATIVE_METHOD_OVERRIDE, see Issue #106840.
func char():
	print(74)

func Corner():
	print(75)

func CORNER_TOP_LEFT():
	print(76)

func TAU():
	print(77)

# --- Functions Parameters ---

#func function_parameter_1(int): # TODO: Inconsistent behavior.
#	print(int)

#func function_parameter_2(Node): # TODO: Inconsistent behavior.
#	print(Node)

func function_parameter_3(abs):
	print(abs)

func function_parameter_4(range):
	print(range)

func function_parameter_5(Error):
	print(Error)

func function_parameter_6(OK):
	print(OK)

func function_parameter_7(PI):
	print(PI)

# --- Unnamed Lambda Parameters ---

#var lambda_parameter_1 = func (int): # TODO: Inconsistent behavior.
#	print(int)

#var lambda_parameter_2 = func (Node): # TODO: Inconsistent behavior.
#	print(Node)

var lambda_parameter_3 = func (abs):
	print(abs)

var lambda_parameter_4 = func (range):
	print(range)

var lambda_parameter_5 = func (Error):
	print(Error)

var lambda_parameter_6 = func (OK):
	print(OK)

var lambda_parameter_7 = func (PI):
	print(PI)

# --- Named Lambdas ---

# No warnings here.

#var named_lambda1 = func int(): # TODO: Inconsistent behavior.
#	print(101)

#var named_lambda2 = func Node(): # TODO: Inconsistent behavior.
#	print(102)

var named_lambda3 = func abs():
	print(103)

var named_lambda4 = func range():
	print(104)

var named_lambda5 = func Error():
	print(105)

var named_lambda6 = func OK():
	print(106)

var named_lambda7 = func PI():
	print(107)
