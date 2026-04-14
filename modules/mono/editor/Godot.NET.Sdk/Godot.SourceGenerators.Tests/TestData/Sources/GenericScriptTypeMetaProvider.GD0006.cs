using Godot;

class MetaProvider : IScriptTypeMetaProvider
{
    public static Godot.Bridge.ScriptTypeMeta GetGodotClassScriptMeta() =>
        throw new System.NotImplementedException();
}

// The provider type is not a nested type (invalid).
[{|GD0006:GenericScriptTypeMetaProvider("MetaProvider, GodotAnalyzersTestProject")|}]
public class A_GenericScriptTypeMetaProviderGD0006<T> : Godot.GodotObject
{
}

// The provider type is directly nested type (valid).
[GenericScriptTypeMetaProvider("B_GenericScriptTypeMetaProviderGD0006`1+MetaProvider, GodotAnalyzersTestProject")]
public class B_GenericScriptTypeMetaProviderGD0006<T> : Godot.GodotObject
{
    private class MetaProvider : IScriptTypeMetaProvider
    {
        public static Godot.Bridge.ScriptTypeMeta GetGodotClassScriptMeta() =>
            new(Type: typeof(B_GenericScriptTypeMetaProviderGD0006<T>),
                NativeType: Godot.GodotObject.CachedType,
                NativeName: Godot.GodotObject.NativeName);
    }
}

// The provider type is nested by multiple levels (valid).
[GenericScriptTypeMetaProvider("C_GenericScriptTypeMetaProviderGD0006`1+GodotInternal+MetaProvider, GodotAnalyzersTestProject")]
public class C_GenericScriptTypeMetaProviderGD0006<T> : Godot.GodotObject
{
    private static class GodotInternal
    {
        private class MetaProvider : IScriptTypeMetaProvider
        {
            public static Godot.Bridge.ScriptTypeMeta GetGodotClassScriptMeta() =>
                new(Type: typeof(C_GenericScriptTypeMetaProviderGD0006<T>),
                    NativeType: Godot.GodotObject.CachedType,
                    NativeName: Godot.GodotObject.NativeName);
        }
    }
}
