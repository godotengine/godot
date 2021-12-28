using System.Reflection;
using Godot.Collections;
using Godot.NativeInterop;

namespace Godot
{
    public partial class SceneTree
    {
        /// <summary>
        /// Returns a list of all nodes assigned to the given <paramref name="group"/>.
        /// </summary>
        /// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Node"/>.</typeparam>
        public Array<T> GetNodesInGroup<T>(StringName group) where T : class
        {
            var array = GetNodesInGroup(group);

            if (array.Count == 0)
                return new Array<T>(array);

            var typeOfT = typeof(T);
            bool nativeBase = InternalIsClassNativeBase(typeOfT);

            if (nativeBase)
            {
                // Native type
                var field = typeOfT.GetField("NativeName",
                    BindingFlags.DeclaredOnly | BindingFlags.Static |
                    BindingFlags.Public | BindingFlags.NonPublic);

                var nativeName = (StringName)field!.GetValue(null);
                var nativeNameSelf = (godot_string_name)nativeName!.NativeValue;
                var inputSelf = (godot_array)array.NativeValue;
                NativeFuncs.godotsharp_array_filter_godot_objects_by_native(nativeNameSelf, inputSelf,
                    out godot_array filteredArray);
                return Array<T>.CreateTakingOwnershipOfDisposableValue(filteredArray);
            }
            else
            {
                // Custom derived type
                var inputSelf = (godot_array)array.NativeValue;
                NativeFuncs.godotsharp_array_filter_godot_objects_by_non_native(inputSelf,
                    out godot_array filteredArray);

                var filteredArrayWrapped = Array.CreateTakingOwnershipOfDisposableValue(filteredArray);

                // Re-use first array as its size is the same or greater than the filtered one
                var resWrapped = new Array<T>(array);

                int j = 0;
                for (int i = 0; i < filteredArrayWrapped.Count; i++)
                {
                    if (filteredArrayWrapped[i] is T t)
                    {
                        resWrapped[j] = t;
                        j++;
                    }
                }

                // Remove trailing elements, since this was re-used
                resWrapped.Resize(j);

                return resWrapped;
            }
        }
    }
}
