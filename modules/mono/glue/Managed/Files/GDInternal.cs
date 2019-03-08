using System;
using System.Reflection;

namespace Godot
{
    internal static class GDInternal
    {

        /// <summary>
        /// For the project assembly, try finding the script instance factory to use and
        /// return it. Otherwise return null.
        /// </summary>
        public static IScriptInstanceFactory FindScriptInstanceFactory(Assembly projectAssembly)
        {

            var factoryAttribute = projectAssembly
                .GetCustomAttribute<ScriptInstanceFactoryAttribute>();

            if (factoryAttribute == null)
            {
                return null;
            }

            var factoryType = factoryAttribute.FactoryType;

            if (!typeof(IScriptInstanceFactory).IsAssignableFrom(factoryType))
            {
                GD.PushError($"The type {factoryType} does not implement IScriptInstanceFactory.");
                return null;
            }

            return (IScriptInstanceFactory) Activator.CreateInstance(factoryType);

        }

    }
}
