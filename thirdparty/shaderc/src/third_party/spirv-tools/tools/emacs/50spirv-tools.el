;; Copyright (c) 2016 LunarG Inc.
;;
;; Licensed under the Apache License, Version 2.0 (the "License");
;; you may not use this file except in compliance with the License.
;; You may obtain a copy of the License at
;;
;;     http://www.apache.org/licenses/LICENSE-2.0
;;
;; Unless required by applicable law or agreed to in writing, software
;; distributed under the License is distributed on an "AS IS" BASIS,
;; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
;; See the License for the specific language governing permissions and
;; limitations under the License.

;; Upon loading a file with the .spv extension into emacs, the file
;; will be disassembled using spirv-dis, and the result colorized with
;; asm-mode in emacs.  The file may be edited within the constraints
;; of validity, and when re-saved will be re-assembled using spirv-as.

;; Note that symbol IDs are not preserved through a load/edit/save operation.
;; This may change if the ability is added to spirv-as.

;; It is required that those tools be in your PATH.  If that is not the case
;; when starting emacs, the path can be modified as in this example:
;; (setenv "PATH" (concat (getenv "PATH") ":/path/to/spirv/tools"))
;;
;; See https://github.com/KhronosGroup/SPIRV-Tools/issues/359

(require 'jka-compr)
(require 'asm-mode)

(add-to-list 'jka-compr-compression-info-list
             '["\\.spv\\'"
               "Assembling SPIRV" "spirv-as" ("-o" "-")
               "Disassembling SPIRV" "spirv-dis" ("--no-color" "--raw-id")
               t nil "\003\002\043\007"])

(add-to-list 'auto-mode-alist '("\\.spv\\'" . asm-mode))

(jka-compr-update)
