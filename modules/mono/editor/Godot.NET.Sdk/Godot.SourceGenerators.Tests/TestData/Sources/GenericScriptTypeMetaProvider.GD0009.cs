using Godot;

// The provider type has additional generic type parameters.
[{|GD0009:GenericScriptTypeMetaProvider("A_GenericScriptTypeMetaProviderGD0009`1+MetaProviderWithTypeParams`1, GodotAnalyzersTestProject")|}]
public class A_GenericScriptTypeMetaProviderGD0009<T> : Godot.GodotObject
{
    private class MetaProviderWithTypeParams<T2> : IScriptTypeMetaProvider
    {
        public static Godot.Bridge.ScriptTypeMeta GetGodotClassScriptMeta() =>
            new(Type: typeof(A_GenericScriptTypeMetaProviderGD0009<T>),
                NativeType: Godot.GodotObject.CachedType,
                NativeName: Godot.GodotObject.NativeName);
    }
}

// The provider type has no additional generic type parameters,
// but is nested within another nested type that has additional generic type parameters.
[{|GD0009:GenericScriptTypeMetaProvider("B_GenericScriptTypeMetaProviderGD0009`1+NestedWithTypeParams`1+MetaProvider, GodotAnalyzersTestProject")|}]
public class B_GenericScriptTypeMetaProviderGD0009<T> : Godot.GodotObject
{
    private class NestedWithTypeParams<T2>
    {
        private class MetaProvider : IScriptTypeMetaProvider
        {
            public static Godot.Bridge.ScriptTypeMeta GetGodotClassScriptMeta() =>
                new(Type: typeof(B_GenericScriptTypeMetaProviderGD0009<T>),
                    NativeType: Godot.GodotObject.CachedType,
                    NativeName: Godot.GodotObject.NativeName);
        }
    }
}

// The provider type doesn't have additional generic type parameters,
// and all its containing types beyond the script type don't have additional generic type parameters either.
[GenericScriptTypeMetaProvider("C_GenericScriptTypeMetaProviderGD0009`1+GodotInternal+MetaProvider, GodotAnalyzersTestProject")]
public class C_GenericScriptTypeMetaProviderGD0009<T> : Godot.GodotObject
{
    private static class GodotInternal
    {
        private class MetaProvider : IScriptTypeMetaProvider
        {
            public static Godot.Bridge.ScriptTypeMeta GetGodotClassScriptMeta() =>
                new(Type: typeof(C_GenericScriptTypeMetaProviderGD0009<T>),
                    NativeType: Godot.GodotObject.CachedType,
                    NativeName: Godot.GodotObject.NativeName);
        }
    }
}
