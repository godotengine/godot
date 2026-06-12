using Godot;

[NoScriptFileAssociation]
[{|GD0011:ScriptPath("res://NoScriptFileAssociationAnalyzer.GD0011.cs")|}]
public class A_NoScriptFileAssociationAnalyzerGD0011 : Godot.GodotObject
{
}

[NoScriptFileAssociation]
public class B_NoScriptFileAssociationAnalyzerGD0011 : Godot.GodotObject
{
}

[ScriptPath("res://NoScriptFileAssociationAnalyzer.GD0011.cs")]
public class C_NoScriptFileAssociationAnalyzerGD0011 : Godot.GodotObject
{
}

[NoScriptFileAssociation]
public class D_NoScriptFileAssociationAnalyzerGD0011 : C_NoScriptFileAssociationAnalyzerGD0011
{
}

[ScriptPath("res://NoScriptFileAssociationAnalyzer.GD0011.cs")]
public class E_NoScriptFileAssociationAnalyzerGD0011 : B_NoScriptFileAssociationAnalyzerGD0011
{
}
