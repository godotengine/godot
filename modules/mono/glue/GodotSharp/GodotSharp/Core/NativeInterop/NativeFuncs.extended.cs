// ReSharper disable InconsistentNaming

namespace Godot.NativeInterop
{
    public static partial class NativeFuncs
    {
        public static godot_string_name godotsharp_string_name_new_copy(in godot_string_name src)
        {
            if (src.IsEmpty)
                return default;
            godotsharp_string_name_new_copy(out godot_string_name ret, src);
            return ret;
        }

        public static godot_node_path godotsharp_node_path_new_copy(in godot_node_path src)
        {
            if (src.IsEmpty)
                return default;
            godotsharp_node_path_new_copy(out godot_node_path ret, src);
            return ret;
        }

        public static godot_array godotsharp_array_new()
        {
            godotsharp_array_new(out godot_array ret);
            return ret;
        }

        public static godot_array godotsharp_array_new_copy(in godot_array src)
        {
            godotsharp_array_new_copy(out godot_array ret, src);
            return ret;
        }

        public static godot_dictionary godotsharp_dictionary_new()
        {
            godotsharp_dictionary_new(out godot_dictionary ret);
            return ret;
        }

        public static godot_dictionary godotsharp_dictionary_new_copy(in godot_dictionary src)
        {
            godotsharp_dictionary_new_copy(out godot_dictionary ret, src);
            return ret;
        }

        public static godot_string_name godotsharp_string_name_new_from_string(string name)
        {
            using godot_string src = Marshaling.ConvertStringToNative(name);
            godotsharp_string_name_new_from_string(out godot_string_name ret, src);
            return ret;
        }

        public static godot_node_path godotsharp_node_path_new_from_string(string name)
        {
            using godot_string src = Marshaling.ConvertStringToNative(name);
            godotsharp_node_path_new_from_string(out godot_node_path ret, src);
            return ret;
        }
    }
}
