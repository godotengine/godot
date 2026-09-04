using Godot;

// The provider type doesn't implement IScriptTypeMetaProvider (invalid).
[{|GD0008:GenericScriptTypeMetaProvider("A_GenericScriptTypeMetaProviderGD0008`1+MetaProvider, GodotAnalyzersTestProject")|}]
public class A_GenericScriptTypeMetaProviderGD0008<T> : Godot.GodotObject
{
    private class MetaProvider
    {
    }
}

// The provider type declares GetGodotClassScriptMeta correctly, but doesn't implement IScriptTypeMetaProvider (invalid).
[{|GD0008:GenericScriptTypeMetaProvider("B_GenericScriptTypeMetaProviderGD0008`1+MetaProvider, GodotAnalyzersTestProject")|}]
public class B_GenericScriptTypeMetaProviderGD0008<T> : Godot.GodotObject
{
    private class MetaProvider
    {
        public static Godot.Bridge.ScriptTypeMeta GetGodotClassScriptMeta() =>
            new(Type: typeof(B_GenericScriptTypeMetaProviderGD0008<T>),
                NativeType: Godot.GodotObject.CachedType,
                NativeName: Godot.GodotObject.NativeName);
    }
}

// The provider type implements IScriptTypeMetaProvider (correct).
[GenericScriptTypeMetaProvider("C_GenericScriptTypeMetaProviderGD0008`1+MetaProvider, GodotAnalyzersTestProject")]
public class C_GenericScriptTypeMetaProviderGD0008<T> : Godot.GodotObject
{
    private class MetaProvider : IScriptTypeMetaProvider
    {
        public static Godot.Bridge.ScriptTypeMeta GetGodotClassScriptMeta() =>
            new(Type: typeof(C_GenericScriptTypeMetaProviderGD0008<T>),
                NativeType: Godot.GodotObject.CachedType,
                NativeName: Godot.GodotObject.NativeName);
    }
}
