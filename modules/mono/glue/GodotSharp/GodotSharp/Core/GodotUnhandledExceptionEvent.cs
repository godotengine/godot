using System;

namespace Godot
{
    public static partial class GD
    {
        /// <summary>
        /// 当发生未处理的异常时触发，与项目设置无关.
        /// </summary>
        public static event EventHandler<UnhandledExceptionArgs> UnhandledException;

        private static void OnUnhandledException(Exception e)
        {
            UnhandledException?.Invoke(null, new UnhandledExceptionArgs(e));
        }
    }
}
