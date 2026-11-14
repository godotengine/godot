@warning_ignore_start("standalone_expression")
@warning_ignore_start("unused_signal")
@warning_ignore_start("unused_variable")
@warning_ignore_start("unused_local_constant")

## This is a deprecated property.
## @deprecated
var deprecated_property: float = 1.0

## This is a non-deprecated property.
var normal_property: int = 3

## This is a deprecated property with a getter and setter.
## @deprecated
var deprecated_property_2: float = 2.0:
    get:
        return sqrt(5)
    set(value):
        deprecated_property_2 = value

## This is a deprecated enum.
## @deprecated
enum DeprecatedEnum { ONE, TWO, THREE }

## This is a deprecated constant.
## @deprecated
const DEPRECATED_CONST = "hello"

const TWO = 9

# NOT WORKING
enum NamedEnum {
    ONE,
    ## This is a deprecated constant in an enum.
    ## @deprecated
    TWO
}

enum {
    ANONYMOUS_ONE,
    ## This one is deprecated
    ## @deprecated
    ANONYMOUS_TWO
}

## This is an inner deprecated class.
## @deprecated
class InnerDeprecatedClass:
    ## @deprecated
    enum InnerDeprecatedEnum { ONE, TWO }

class InnerClass:
    enum  {
        ## @deprecated
        ONE,
        TWO
    }

    ## @deprecated
    const INNER_DEPRECATED_CONST = 3

    ## @deprecated
    enum InnerDeprecatedEnum { ONE, TWO }

    enum InnerDeprecatedEnum2 {
        ## @deprecated
        DEPRECATED_ONE,
        NON_DEPRECATED_ONE
    }

    ## @deprecated
    var inner_deprecated_property = 3

    ## @deprecated
    signal inner_deprecated_signal

    ## @deprecated
    func inner_dep_function():
        pass

## This is a deprecated signal.
## @deprecated
signal deprecated_signal

func test():
    # User-defined method marked as deprecated
    deprecated_function()
    InnerClass.new().inner_dep_function()

    # Native method marked as deprecated
    # NOTE: This causes issues with the test runner, so it will be commented
    # out for now.
    # AnimationPlayer.new().get_method_call_mode()

    # Inner class of a file marked as deprecated
    var i3: InnerDeprecatedClass
    var i4 = InnerDeprecatedClass.new()

    # Native class
    var native_class: VisualShaderNodeComment
    var native_class_2 = VisualShaderNodeComment.new()

    # User-defined constant marked as deprecated
    print(DEPRECATED_CONST)
    print(InnerClass.INNER_DEPRECATED_CONST)

    # User-defined local constant marked as deprecated
    ## @deprecated
    const DEPRECATED_LOCAL_CONST = 3
    print(DEPRECATED_LOCAL_CONST)

    # User-defined named enum marked as deprecated
    var dep_enum: DeprecatedEnum
    var dep_enum_2 = DeprecatedEnum.ONE
    var inner_dep_enum: InnerClass.InnerDeprecatedEnum
    var inner_dep_enum_2 = InnerClass.InnerDeprecatedEnum.ONE

    # Individual value in a user-defined anonymous enum marked as deprecated
    var dep_enum_value_anon = ANONYMOUS_TWO
    var dep_enum_value_anon_2 = InnerClass.ONE

    # Individual value in a user-defined named enum marked as deprecated
    var dep_enum_value_named = NamedEnum.TWO
    var dep_enum_value_named_2 = InnerClass.InnerDeprecatedEnum2.DEPRECATED_ONE

    # Individual value in an enum from a native class marked as deprecated
    var dep_native_enum_value = AnimationPlayer.ANIMATION_PROCESS_PHYSICS
    var dep_native_enum_value_2 = AnimationPlayer.AnimationProcessCallback.ANIMATION_PROCESS_PHYSICS

    # Constant in GlobalScope
    var global_const = PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE

    # Local variable marked as deprecated
    ## @deprecated
    var dep_local = 3
    print(dep_local)

    # User-defined property marked as deprecated
    print(deprecated_property)
    print(deprecated_property_2)
    print(InnerClass.new().inner_deprecated_property)

    # Native property marked as deprecated
    print(AStarGrid2D.new().size)

    # User-defined signal marked as deprecated
    deprecated_signal.connect(hello)
    InnerClass.new().inner_deprecated_signal.connect(hello)

    # Native signal marked as deprecated
    Resource.new().setup_local_to_scene_requested.connect(hello)


## This is a deprecated function.
## @deprecated
func deprecated_function():
    print("I'm deprecated")


## This is a non-deprecated function.
func non_deprecated_function():
    print("I'm not deprecated")

## @deprecated
func example():
    pass

func hello():
    pass
