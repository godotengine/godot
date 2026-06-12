namespace Godot.Bridge
{
    public static class ScriptRegistryExtensions
    {
        // This is an extension method because C# does not allow additional type constraints for an already existing T
        public static ScriptMethodRegistry<T> Register<T, TBase>(this ScriptMethodRegistry<T> registry, ScriptMethodRegistry<TBase> baseTypeRegistry)
            where TBase : GodotObject
            where T : GodotObject, TBase
        {
            foreach (var (methodKey, alias) in baseTypeRegistry.Aliases)
            {
                registry.AddAlias(methodKey.Name, methodKey.ArgCount, alias);
            }

            foreach (var (methodKey, value) in baseTypeRegistry.MethodsByNameAndArgc)
            {
                registry.Register(methodKey.Name, methodKey.ArgCount, value);
            }

            return registry;
        }

        // This is an extension method because C# does not allow additional type constraints for an already existing T
        public static ScriptSignalRegistry<T> Register<T, TBase>(this ScriptSignalRegistry<T> registry, ScriptSignalRegistry<TBase> baseTypeRegistry)
            where TBase : GodotObject
            where T : GodotObject, TBase
        {
            foreach (var (methodKey, alias) in baseTypeRegistry.Aliases)
            {
                registry.AddAlias(methodKey.Name, methodKey.ArgCount, alias);
            }

            foreach (var (methodKey, value) in baseTypeRegistry.MethodsByNameAndArgc)
            {
                registry.Register(methodKey.Name, methodKey.ArgCount, value);
            }

            return registry;
        }

        // This is an extension method because C# does not allow additional type constraints for an already existing T
        public static ScriptPropertyRegistry<T> Register<T, TBase>(this ScriptPropertyRegistry<T> registry, ScriptPropertyRegistry<TBase> baseTypeRegistry)
            where TBase : GodotObject
            where T : GodotObject, TBase
        {
            foreach (var (methodKey, alias) in baseTypeRegistry.Aliases)
            {
                registry.AddAlias(methodKey.Name, methodKey.ArgCount, alias);
            }

            foreach (var (methodKey, value) in baseTypeRegistry.MethodsByNameAndArgc)
            {
                registry.Register(methodKey.Name, methodKey.ArgCount, value);
            }

            return registry;
        }
    }
}
