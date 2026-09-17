using Godot;

// Assembly-qualified name missing the assembly suffix (invalid).
[{|GD0007:GenericScriptTypeMetaProvider("A_GenericScriptTypeMetaProviderGD0007`1+MetaProvider")|}]
public class A_GenericScriptTypeMetaProviderGD0007<T> : Godot.GodotObject
{
    private class MetaProvider : IScriptTypeMetaProvider
    {
        public static Godot.Bridge.ScriptTypeMeta GetGodotClassScriptMeta() =>
            new(Type: typeof(A_GenericScriptTypeMetaProviderGD0007<T>),
                NativeType: Godot.GodotObject.CachedType,
                NativeName: Godot.GodotObject.NativeName);
    }
}

// Assembly-qualified name includes the assembly suffix (correct).
[GenericScriptTypeMetaProvider("B_GenericScriptTypeMetaProviderGD0007`1+MetaProvider, GodotAnalyzersTestProject")]
public class B_GenericScriptTypeMetaProviderGD0007<T> : Godot.GodotObject
{
    private class MetaProvider : IScriptTypeMetaProvider
    {
        public static Godot.Bridge.ScriptTypeMeta GetGodotClassScriptMeta() =>
            new(Type: typeof(B_GenericScriptTypeMetaProviderGD0007<T>),
                NativeType: Godot.GodotObject.CachedType,
                NativeName: Godot.GodotObject.NativeName);
    }
}
