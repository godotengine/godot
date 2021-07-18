;; copy to .dir-locals.el for configuring godot specifc debug templates
;;
;; GODOT_HOME is an environmental variable  pointing to Godot home directory
;; eg in ~/.zshenv (for ZSH) : export GODOT_HOME="${HOME}"/bin/thirdparty/godot
;;
;; :target may vary between releases
;;    eg godot.linuxbsd.tools.64,godot.x11.tools.64
;;
;; dap-mode llvm/cpp:  https://emacs-lsp.github.io/dap-mode/page/configuration
((c++-mode . ((dap-debug-template-configurations . (
                                                    ("Godot GDB"
                                                     :type "gdb"
                                                     :request "launch"
                                                     :target (concat (getenv "GODOT_HOME") "/bin/godot.linuxbsd.tools.64"))
                                                    ("Godot LLDB"
                                                     :type "lldb"
                                                     :request "launch"
                                                     :target (concat (getenv "GODOT_HOME") "/bin/godot.linuxbsd.tools.64"))
                                                    )))))
