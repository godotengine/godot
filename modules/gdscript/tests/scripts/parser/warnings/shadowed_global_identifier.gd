@warning_ignore_start("unused_variable", "unused_local_constant", "unused_parameter")

# Note: The following cases are tested:
# * built-in type
# * native (or global) class
# * built-in @GlobalScope function
# * built-in @GDScript function
# * built-in @GlobalScope enum class
# * built-in @GlobalScope enum member
# * built-in @GDScript constant

# Note: Some built-in @GDScript functions cannot be shadowed,
# e.g. assert, preload

# --- Variables --- #

# Member var declarations.
#var int = "" # Parse Error, skipped.
#var Node = "" # Parse Error, skipped.
var abs = ""
var range = ""
var Error = ""
var OK = ""
var PI = ""

func shadowed_local_variables():
	# Local var declarations.
	#var int = "" # Memory leak?
	#var Node = "" # Memory leak?
	var abs = ""
	var range = ""
	var Error = ""
	var OK = ""
	var PI = ""

# --- Constants --- #

# The others behave similarly to member var declarations.
const min = ""

func shadowed_local_constants():
	# The others behave similarly to local var declarations.
	const min = ""

# --- Named Enum Classes --- #

# The others behave similarly to member var declarations.
enum max { }

# --- Named Enum Members --- #

# Note: NamedEnum.MEMBER does not shadow anything.

enum NamedEnum {
	# No warnings here.
	#int = 0, # Memory leak?
	#Node = 1, # Memory leak?
	abs = 2,
	range = 3,
	Error = 4,
	OK = 5,
	PI = 6,
}

# --- Unnamed Enum Members --- #

enum {
	# The others behave similarly to member var declarations.
	round = 2,
}

# --- Functions --- #

# Note: If NATIVE_METHOD_OVERRIDE exists, SHADOWED_GLOBAL_IDENTIFIER is not produced.

#func float(): # Memory leak?
#	pass

#func Node2D(): # Memory leak?
#	pass

# TODO: This should produce NATIVE_METHOD_OVERRIDE, see Issue #106840.
func ceil():
	pass

# TODO: This should produce NATIVE_METHOD_OVERRIDE, see Issue #106840.
func char():
	pass

func Corner():
	pass

func CORNER_TOP_LEFT():
	pass

func TAU():
	pass

# --- Functions Parameters --- #

#func function_parameter_1(int): # Memory leak?
#	pass

#func function_parameter_2(Node): # Memory leak?
#	pass

func function_parameter_3(abs):
	pass

func function_parameter_4(range):
	pass

func function_parameter_5(Error):
	pass

func function_parameter_6(OK):
	pass

func function_parameter_7(PI):
	pass

# --- Unnamed Lambdas --- #

# No warnings here.
#var lambda1 = func (): print(int) # Parse Error, skipped.
#var lambda2 = func (): print(Node) # Memory leak?
#var lambda3 = func (): print(abs) # Memory leak?
#var lambda4 = func (): print(range) # Memory leak?
#var lambda5 = func (): print(Error) # Memory leak?
#var lambda6 = func (): print(OK) # Memory leak?
#var lambda7 = func (): print(PI) # Memory leak?

# --- Unnamed Lambda Parameters --- #

#var lambda_parameter1 = func (int): print(int) # Memory leak?
#var lambda_parameter2 = func (Node): print(Node) # Memory leak?
#var lambda_parameter3 = func (abs): print(abs) # Memory leak?
#var lambda_parameter4 = func (range): print(range) # Memory leak?
#var lambda_parameter5 = func (Error): print(Error) # Memory leak?
#var lambda_parameter6 = func (OK): print(OK) # Memory leak?
#var lambda_parameter7 = func (PI): print(PI) # Memory leak?

# --- Named Lambdas --- #

#var named_lambda1 = func int(): print("") # Memory leak?
#var named_lambda2 = func Node(): print("") # Memory leak?
#var named_lambda3 = func abs(): print("") # Memory leak?
#var named_lambda4 = func range(): print("") # Memory leak?
#var named_lambda5 = func Error(): print("") # Memory leak?
#var named_lambda6 = func OK(): print("") # Memory leak?
#var named_lambda7 = func PI(): print("") # Memory leak?

# --- Run Test --- #

func test():
	print("warn")
