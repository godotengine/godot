using Godot;

// B_GenericScriptTypeMetaProviderGD0004 is not generic.
[{|GD0004:GenericScriptTypeMetaProvider("A_GenericScriptTypeMetaProviderGD0004+MetaProvider, GodotAnalyzersTestProject")|}]
public class A_GenericScriptTypeMetaProviderGD0004 : Godot.GodotObject
{
    private class MetaProvider : IScriptTypeMetaProvider
    {
        public static Godot.Bridge.ScriptTypeMeta GetGodotClassScriptMeta() =>
            new(Type: typeof(A_GenericScriptTypeMetaProviderGD0004),
                NativeType: Godot.GodotObject.CachedType,
                NativeName: Godot.GodotObject.NativeName);
    }
}

// B_GenericScriptTypeMetaProviderGD0004<T> is generic.
[GenericScriptTypeMetaProvider("B_GenericScriptTypeMetaProviderGD0004`1+MetaProvider, GodotAnalyzersTestProject")]
public class B_GenericScriptTypeMetaProviderGD0004<T> : Godot.GodotObject
{
    private class MetaProvider : IScriptTypeMetaProvider
    {
        public static Godot.Bridge.ScriptTypeMeta GetGodotClassScriptMeta() =>
            new(Type: typeof(B_GenericScriptTypeMetaProviderGD0004<T>),
                NativeType: Godot.GodotObject.CachedType,
                NativeName: Godot.GodotObject.NativeName);
    }
}

public class GenericNestingClass<T>
{
    // C_GenericScriptTypeMetaProviderGD0004 has no type parameters,
    // but it's still generic as it's a nested class within GenericNestingClass<T>.
    [GenericScriptTypeMetaProvider("GenericNestingClass`1+C_GenericScriptTypeMetaProviderGD0004+MetaProvider, GodotAnalyzersTestProject")]
    public class C_GenericScriptTypeMetaProviderGD0004 : Godot.GodotObject
    {
        private class MetaProvider : IScriptTypeMetaProvider
        {
            public static Godot.Bridge.ScriptTypeMeta GetGodotClassScriptMeta() =>
                new(Type: typeof(C_GenericScriptTypeMetaProviderGD0004),
                    NativeType: Godot.GodotObject.CachedType,
                    NativeName: Godot.GodotObject.NativeName);
        }
    }
}
