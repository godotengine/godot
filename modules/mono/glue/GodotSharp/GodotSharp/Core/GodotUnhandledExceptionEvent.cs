using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Godot
{
    public static partial class GD
    {
        /// <summary>
        /// Fires when an unhandled exception occurs, regardless of project settings.
        /// </summary>
        public static event EventHandler<UnhandledExceptionArgs> UnhandledException;

        private static void OnUnhandledException(Exception e)
        {
            UnhandledException?.Invoke(null, new UnhandledExceptionArgs(e));
        }
    }
}
