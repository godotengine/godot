using System.Runtime.CompilerServices;

namespace GodotTools.Internals
{
    public static class GodotSharpDirs
    {
        public static string ResDataDir => internal_ResDataDir();
        public static string ResMetadataDir => internal_ResMetadataDir();
        public static string ResAssembliesBaseDir => internal_ResAssembliesBaseDir();
        public static string ResAssembliesDir => internal_ResAssembliesDir();
        public static string ResConfigDir => internal_ResConfigDir();
        public static string ResTempDir => internal_ResTempDir();
        public static string ResTempAssembliesBaseDir => internal_ResTempAssembliesBaseDir();
        public static string ResTempAssembliesDir => internal_ResTempAssembliesDir();

        public static string MonoUserDir => internal_MonoUserDir();
        public static string MonoLogsDir => internal_MonoLogsDir();

        #region Tools-only
        public static string MonoSolutionsDir => internal_MonoSolutionsDir();
        public static string BuildLogsDirs => internal_BuildLogsDirs();

        public static string ProjectAssemblyName => internal_ProjectAssemblyName();
        public static string ProjectSlnPath => internal_ProjectSlnPath();
        public static string ProjectCsProjPath => internal_ProjectCsProjPath();

        public static string DataEditorToolsDir => internal_DataEditorToolsDir();
        public static string DataEditorPrebuiltApiDir => internal_DataEditorPrebuiltApiDir();
        #endregion

        public static string DataMonoEtcDir => internal_DataMonoEtcDir();
        public static string DataMonoLibDir => internal_DataMonoLibDir();

        #region Windows-only
        public static string DataMonoBinDir => internal_DataMonoBinDir();
        #endregion


        #region Internal

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_ResDataDir();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_ResMetadataDir();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_ResAssembliesBaseDir();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_ResAssembliesDir();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_ResConfigDir();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_ResTempDir();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_ResTempAssembliesBaseDir();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_ResTempAssembliesDir();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_MonoUserDir();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_MonoLogsDir();

        #region Tools-only
        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_MonoSolutionsDir();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_BuildLogsDirs();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_ProjectAssemblyName();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_ProjectSlnPath();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_ProjectCsProjPath();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_DataEditorToolsDir();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_DataEditorPrebuiltApiDir();
        #endregion

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_DataMonoEtcDir();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_DataMonoLibDir();

        #region Windows-only
        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_DataMonoBinDir();
        #endregion

        #endregion
    }
}
