using System;
using System.Runtime.CompilerServices;

// ReSharper disable InconsistentNaming

namespace Godot.NativeInterop
{
    internal static unsafe partial class NativeFuncs
    {
        public static godot_string_name godotsharp_string_name_new_copy(godot_string_name* src)
        {
            godot_string_name ret;
            godotsharp_string_name_new_copy(&ret, src);
            return ret;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_string_name godotsharp_string_name_new_copy(godot_string_name src)
            => godotsharp_string_name_new_copy(&src);

        public static godot_node_path godotsharp_node_path_new_copy(godot_node_path* src)
        {
            godot_node_path ret;
            godotsharp_node_path_new_copy(&ret, src);
            return ret;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_node_path godotsharp_node_path_new_copy(godot_node_path src)
            => godotsharp_node_path_new_copy(&src);

        public static godot_array godotsharp_array_new_copy(godot_array* src)
        {
            godot_array ret;
            godotsharp_array_new_copy(&ret, src);
            return ret;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_array godotsharp_array_new_copy(godot_array src)
            => godotsharp_array_new_copy(&src);

        public static godot_dictionary godotsharp_dictionary_new_copy(godot_dictionary* src)
        {
            godot_dictionary ret;
            godotsharp_dictionary_new_copy(&ret, src);
            return ret;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_dictionary godotsharp_dictionary_new_copy(godot_dictionary src)
            => godotsharp_dictionary_new_copy(&src);

        public static godot_string_name godotsharp_string_name_new_from_string(string name)
        {
            godot_string_name ret;
            using godot_string src = Marshaling.mono_string_to_godot(name);
            godotsharp_string_name_new_from_string(&ret, &src);
            return ret;
        }

        public static godot_node_path godotsharp_node_path_new_from_string(string name)
        {
            godot_node_path ret;
            using godot_string src = Marshaling.mono_string_to_godot(name);
            godotsharp_node_path_new_from_string(&ret, &src);
            return ret;
        }
    }
}
