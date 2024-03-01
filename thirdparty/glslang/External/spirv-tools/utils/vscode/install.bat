@REM Copyright (c) 2019 Google Inc.
@REM
@REM Licensed under the Apache License, Version 2.0 (the "License");
@REM you may not use this file except in compliance with the License.
@REM You may obtain a copy of the License at
@REM
@REM     http://www.apache.org/licenses/LICENSE-2.0
@REM
@REM Unless required by applicable law or agreed to in writing, software
@REM distributed under the License is distributed on an "AS IS" BASIS,
@REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@REM See the License for the specific language governing permissions and
@REM limitations under the License.

@set EXT_PATH=%userprofile%\.vscode\extensions\google.spirvls-0.0.1
@set ROOT_PATH=%~dp0

go run %ROOT_PATH%\src\tools\gen-grammar.go --cache %ROOT_PATH%\cache --template %ROOT_PATH%\spirv.json.tmpl --out %ROOT_PATH%\spirv.json
go run %ROOT_PATH%\src\tools\gen-grammar.go --cache %ROOT_PATH%\cache --template %ROOT_PATH%\src\schema\schema.go.tmpl --out %ROOT_PATH%\src\schema\schema.go

if not exist %EXT_PATH% mkdir -p %EXT_PATH%
copy %ROOT_PATH%\extension.js %EXT_PATH%
copy %ROOT_PATH%\package.json %EXT_PATH%
copy %ROOT_PATH%\spirv.json %EXT_PATH%

go build -o %EXT_PATH%\langsvr.exe %ROOT_PATH%\src\langsvr.go

@pushd %EXT_PATH%
call npm install
@popd
