using Godot;

// The provider type cannot be found.
[{|GD0005:GenericScriptTypeMetaProvider("NonExistentType, GodotAnalyzersTestProject")|}]
public class A_GenericScriptTypeMetaProviderGD0005<T> : Godot.GodotObject
{
    private class MetaProvider : IScriptTypeMetaProvider
    {
        public static Godot.Bridge.ScriptTypeMeta GetGodotClassScriptMeta() =>
            new(Type: typeof(A_GenericScriptTypeMetaProviderGD0005<T>),
                NativeType: Godot.GodotObject.CachedType,
                NativeName: Godot.GodotObject.NativeName);
    }
}

// Should be able to find this provider type.
[GenericScriptTypeMetaProvider("B_GenericScriptTypeMetaProviderGD0005`1+MetaProvider, GodotAnalyzersTestProject")]
public class B_GenericScriptTypeMetaProviderGD0005<T> : Godot.GodotObject
{
    private class MetaProvider : IScriptTypeMetaProvider
    {
        public static Godot.Bridge.ScriptTypeMeta GetGodotClassScriptMeta() =>
            new(Type: typeof(B_GenericScriptTypeMetaProviderGD0005<T>),
                NativeType: Godot.GodotObject.CachedType,
                NativeName: Godot.GodotObject.NativeName);
    }
}
