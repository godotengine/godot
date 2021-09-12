using System.Runtime.CompilerServices;
using Godot.NativeInterop;

namespace GodotTools.Internals
{
    public static class GodotSharpDirs
    {
        public static string ResMetadataDir
        {
            get
            {
                internal_ResMetadataDir(out godot_string dest);
                using (dest)
                    return Marshaling.mono_string_from_godot(dest);
            }
        }

        public static string ResTempAssembliesBaseDir
        {
            get
            {
                internal_ResTempAssembliesBaseDir(out godot_string dest);
                using (dest)
                    return Marshaling.mono_string_from_godot(dest);
            }
        }

        public static string MonoUserDir
        {
            get
            {
                internal_MonoUserDir(out godot_string dest);
                using (dest)
                    return Marshaling.mono_string_from_godot(dest);
            }
        }

        public static string BuildLogsDirs
        {
            get
            {
                internal_BuildLogsDirs(out godot_string dest);
                using (dest)
                    return Marshaling.mono_string_from_godot(dest);
            }
        }

        public static string ProjectSlnPath
        {
            get
            {
                internal_ProjectSlnPath(out godot_string dest);
                using (dest)
                    return Marshaling.mono_string_from_godot(dest);
            }
        }

        public static string ProjectCsProjPath
        {
            get
            {
                internal_ProjectCsProjPath(out godot_string dest);
                using (dest)
                    return Marshaling.mono_string_from_godot(dest);
            }
        }

        public static string DataEditorToolsDir
        {
            get
            {
                internal_DataEditorToolsDir(out godot_string dest);
                using (dest)
                    return Marshaling.mono_string_from_godot(dest);
            }
        }

        #region Internal

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_ResMetadataDir(out godot_string r_dest);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_ResTempAssembliesBaseDir(out godot_string r_dest);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_MonoUserDir(out godot_string r_dest);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_BuildLogsDirs(out godot_string r_dest);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_ProjectSlnPath(out godot_string r_dest);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_ProjectCsProjPath(out godot_string r_dest);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_DataEditorToolsDir(out godot_string r_dest);

        #endregion
    }
}
