// ReSharper disable InconsistentNaming

namespace Godot.NativeInterop
{
    public static partial class NativeFuncs
    {
        public static godot_variant godotsharp_variant_new_copy(in godot_variant src)
        {
            switch (src.Type)
            {
                case VariantType.Nil:
                    return default;
                case VariantType.Bool:
                    return new godot_variant() { Bool = src.Bool, Type = VariantType.Bool };
                case VariantType.Int:
                    return new godot_variant() { Int = src.Int, Type = VariantType.Int };
                case VariantType.Float:
                    return new godot_variant() { Float = src.Float, Type = VariantType.Float };
                case VariantType.Vector2:
                    return new godot_variant() { Vector2 = src.Vector2, Type = VariantType.Vector2 };
                case VariantType.Vector2I:
                    return new godot_variant() { Vector2I = src.Vector2I, Type = VariantType.Vector2I };
                case VariantType.Rect2:
                    return new godot_variant() { Rect2 = src.Rect2, Type = VariantType.Rect2 };
                case VariantType.Rect2I:
                    return new godot_variant() { Rect2I = src.Rect2I, Type = VariantType.Rect2I };
                case VariantType.Vector3:
                    return new godot_variant() { Vector3 = src.Vector3, Type = VariantType.Vector3 };
                case VariantType.Vector3I:
                    return new godot_variant() { Vector3I = src.Vector3I, Type = VariantType.Vector3I };
                case VariantType.Vector4:
                    return new godot_variant() { Vector4 = src.Vector4, Type = VariantType.Vector4 };
                case VariantType.Vector4I:
                    return new godot_variant() { Vector4I = src.Vector4I, Type = VariantType.Vector4I };
                case VariantType.Plane:
                    return new godot_variant() { Plane = src.Plane, Type = VariantType.Plane };
                case VariantType.Quaternion:
                    return new godot_variant() { Quaternion = src.Quaternion, Type = VariantType.Quaternion };
                case VariantType.Color:
                    return new godot_variant() { Color = src.Color, Type = VariantType.Color };
                case VariantType.Rid:
                    return new godot_variant() { Rid = src.Rid, Type = VariantType.Rid };
            }

            godotsharp_variant_new_copy(out godot_variant ret, src);
            return ret;
        }

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
