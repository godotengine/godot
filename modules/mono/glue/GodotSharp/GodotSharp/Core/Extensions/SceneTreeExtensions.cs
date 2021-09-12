using System.Reflection;
using System.Runtime.CompilerServices;
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
        public unsafe Array<T> GetNodesInGroup<T>(StringName group) where T : class
        {
            var array = GetNodesInGroup(group);

            if (array.Count == 0)
                return new Array<T>(array);

            var typeOfT = typeof(T);
            bool nativeBase = InternalIsClassNativeBase(typeOfT);

            if (nativeBase)
            {
                // Native type
                var field = typeOfT.GetField("NativeName", BindingFlags.DeclaredOnly | BindingFlags.Static |
                                                           BindingFlags.Public | BindingFlags.NonPublic);

                var nativeName = (StringName)field!.GetValue(null);
                godot_string_name nativeNameAux = nativeName.NativeValue;
                godot_array inputAux = array.NativeValue;
                godot_array filteredArray;
                godotsharp_array_filter_godot_objects_by_native(&nativeNameAux, &inputAux, &filteredArray);
                return Array<T>.CreateTakingOwnershipOfDisposableValue(filteredArray);
            }
            else
            {
                // Custom derived type
                godot_array inputAux = array.NativeValue;
                godot_array filteredArray;
                godotsharp_array_filter_godot_objects_by_non_native(&inputAux, &filteredArray);

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

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern unsafe void godotsharp_array_filter_godot_objects_by_native(godot_string_name* p_native_name,
            godot_array* p_input, godot_array* r_output);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern unsafe void godotsharp_array_filter_godot_objects_by_non_native(godot_array* p_input,
            godot_array* r_output);
    }
}
