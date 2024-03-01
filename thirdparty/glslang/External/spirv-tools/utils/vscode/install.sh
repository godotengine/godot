#!/usr/bin/env bash
# Copyright (c) 2019 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e # Fail on any error.

EXT_PATH=~/.vscode/extensions/google.spirvls-0.0.1
ROOT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pushd ${ROOT_PATH}
    go run ./src/tools/gen-grammar.go --cache ./cache --template ./spirv.json.tmpl --out ./spirv.json
    go run ./src/tools/gen-grammar.go --cache ./cache --template ./src/schema/schema.go.tmpl --out ./src/schema/schema.go

    mkdir -p ${EXT_PATH}
    cp ./extension.js ${EXT_PATH}
    cp ./package.json ${EXT_PATH}
    cp ./spirv.json ${EXT_PATH}

    go build -o ${EXT_PATH}/langsvr ./src/langsvr.go
popd

cd ${EXT_PATH}
npm install
