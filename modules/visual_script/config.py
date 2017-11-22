def can_build(platform):
    return True

def configure(env):
    pass

def get_doc_classes():
    return [
        "VisualScriptBasicTypeConstant",
        "VisualScriptBuiltinFunc",
        "VisualScriptClassConstant",
        "VisualScriptComment",
        "VisualScriptCondition",
        "VisualScriptConstant",
        "VisualScriptConstructor",
        "VisualScriptCustomNode",
        "VisualScriptDeconstruct",
        "VisualScriptEditor",
        "VisualScriptEmitSignal",
        "VisualScriptEngineSingleton",
        "VisualScriptExpression",
        "VisualScriptFunctionCall",
        "VisualScriptFunctionState",
        "VisualScriptFunction",
        "VisualScriptGlobalConstant",
        "VisualScriptIndexGet",
        "VisualScriptIndexSet",
        "VisualScriptInputAction",
        "VisualScriptIterator",
        "VisualScriptLocalVarSet",
        "VisualScriptLocalVar",
        "VisualScriptMathConstant",
        "VisualScriptNode",
        "VisualScriptOperator",
        "VisualScriptPreload",
        "VisualScriptPropertyGet",
        "VisualScriptPropertySet",
        "VisualScriptResourcePath",
        "VisualScriptReturn",
        "VisualScriptSceneNode",
        "VisualScriptSceneTree",
        "VisualScriptSelect",
        "VisualScriptSelf",
        "VisualScriptSequence",
        "VisualScriptSubCall",
        "VisualScriptSwitch",
        "VisualScriptTypeCast",
        "VisualScriptVariableGet",
        "VisualScriptVariableSet",
        "VisualScriptWhile",
        "VisualScript",
        "VisualScriptYieldSignal",
        "VisualScriptYield",
    ]

def get_doc_path():
    return "doc_classes"
