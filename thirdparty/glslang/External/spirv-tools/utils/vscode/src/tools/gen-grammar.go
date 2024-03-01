// Copyright (C) 2019 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// gen-grammar generates the spirv.json grammar file from the official SPIR-V
// grammar JSON file.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"text/template"

	"github.com/pkg/errors"

	"github.com/KhronosGroup/SPIRV-Tools/utils/vscode/src/grammar"
)

type grammarDefinition struct {
	name string
	url  string
}

var (
	spirvGrammar = grammarDefinition{
		name: "SPIR-V",
		url:  "https://raw.githubusercontent.com/KhronosGroup/SPIRV-Headers/master/include/spirv/unified1/spirv.core.grammar.json",
	}

	extensionGrammars = []grammarDefinition{
		{
			name: "GLSL.std.450",
			url:  "https://raw.githubusercontent.com/KhronosGroup/SPIRV-Headers/master/include/spirv/unified1/extinst.glsl.std.450.grammar.json",
		}, {
			name: "OpenCL.std",
			url:  "https://raw.githubusercontent.com/KhronosGroup/SPIRV-Headers/master/include/spirv/unified1/extinst.opencl.std.100.grammar.json",
		}, {
			name: "OpenCL.DebugInfo.100",
			url:  "https://raw.githubusercontent.com/KhronosGroup/SPIRV-Headers/master/include/spirv/unified1/extinst.opencl.debuginfo.100.grammar.json",
		},
	}

	templatePath = flag.String("template", "", "Path to input template file (required)")
	outputPath   = flag.String("out", "", "Path to output generated file (required)")
	cachePath    = flag.String("cache", "", "Cache directory for downloaded files (optional)")

	thisDir = func() string {
		_, file, _, _ := runtime.Caller(1)
		return filepath.Dir(file)
	}()
)

func main() {
	flag.Parse()
	if *templatePath == "" || *outputPath == "" {
		flag.Usage()
		os.Exit(1)
	}
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	tf, err := ioutil.ReadFile(*templatePath)
	if err != nil {
		return errors.Wrap(err, "Could not open template file")
	}

	type extension struct {
		grammar.Root
		Name string
	}

	args := struct {
		SPIRV      grammar.Root
		Extensions []extension
		All        grammar.Root // Combination of SPIRV + Extensions
	}{}

	if args.SPIRV, err = parseGrammar(spirvGrammar); err != nil {
		return errors.Wrap(err, "Failed to parse SPIR-V grammar file")
	}
	args.All.Instructions = append(args.All.Instructions, args.SPIRV.Instructions...)
	args.All.OperandKinds = append(args.All.OperandKinds, args.SPIRV.OperandKinds...)

	for _, ext := range extensionGrammars {
		root, err := parseGrammar(ext)
		if err != nil {
			return errors.Wrap(err, "Failed to parse extension grammar file: "+ext.name)
		}
		args.Extensions = append(args.Extensions, extension{Root: root, Name: ext.name})
		args.All.Instructions = append(args.All.Instructions, root.Instructions...)
		args.All.OperandKinds = append(args.All.OperandKinds, root.OperandKinds...)
	}

	t, err := template.New("tmpl").
		Funcs(template.FuncMap{
			"GenerateArguments": func() string {
				relPath := func(path string) string {
					rel, err := filepath.Rel(thisDir, path)
					if err != nil {
						return path
					}
					return rel
				}
				escape := func(str string) string {
					return strings.ReplaceAll(str, `\`, `/`)
				}
				args := []string{
					"--template=" + escape(relPath(*templatePath)),
					"--out=" + escape(relPath(*outputPath)),
				}
				return "gen-grammar.go " + strings.Join(args, " ")
			},
			"OperandKindsMatch": func(k grammar.OperandKind) string {
				sb := strings.Builder{}
				for i, e := range k.Enumerants {
					if i > 0 {
						sb.WriteString("|")
					}
					sb.WriteString(e.Enumerant)
				}
				return sb.String()
			},
			"AllExtOpcodes": func() string {
				sb := strings.Builder{}
				for _, ext := range args.Extensions {
					for _, inst := range ext.Root.Instructions {
						if sb.Len() > 0 {
							sb.WriteString("|")
						}
						sb.WriteString(inst.Opname)
					}
				}
				return sb.String()
			},
			"Title":   strings.Title,
			"Replace": strings.ReplaceAll,
			"Global": func(s string) string {
				return strings.ReplaceAll(strings.Title(s), ".", "")
			},
		}).Parse(string(tf))
	if err != nil {
		return errors.Wrap(err, "Failed to parse template")
	}

	buf := bytes.Buffer{}
	if err := t.Execute(&buf, args); err != nil {
		return errors.Wrap(err, "Failed to execute template")
	}

	out := buf.String()
	out = strings.ReplaceAll(out, "•", "")

	if err := ioutil.WriteFile(*outputPath, []byte(out), 0777); err != nil {
		return errors.Wrap(err, "Failed to write output file")
	}

	return nil
}

// parseGrammar downloads (or loads from the cache) the grammar file and returns
// the parsed grammar.Root.
func parseGrammar(def grammarDefinition) (grammar.Root, error) {
	file, err := getOrDownload(def.name, def.url)
	if err != nil {
		return grammar.Root{}, errors.Wrap(err, "Failed to load grammar file")
	}

	g := grammar.Root{}
	if err := json.NewDecoder(bytes.NewReader(file)).Decode(&g); err != nil {
		return grammar.Root{}, errors.Wrap(err, "Failed to parse grammar file")
	}

	return g, nil
}

// getOrDownload loads the specific file from the cache, or downloads the file
// from the given url.
func getOrDownload(name, url string) ([]byte, error) {
	if *cachePath != "" {
		if err := os.MkdirAll(*cachePath, 0777); err == nil {
			path := filepath.Join(*cachePath, name)
			if isFile(path) {
				return ioutil.ReadFile(path)
			}
		}
	}
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if *cachePath != "" {
		ioutil.WriteFile(filepath.Join(*cachePath, name), data, 0777)
	}
	return data, nil
}

// isFile returns true if path is a file.
func isFile(path string) bool {
	s, err := os.Stat(path)
	if err != nil {
		return false
	}
	return !s.IsDir()
}

// isDir returns true if path is a directory.
func isDir(path string) bool {
	s, err := os.Stat(path)
	if err != nil {
		return false
	}
	return s.IsDir()
}
