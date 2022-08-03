---
parent: Contributing
nav_order: 1
---

# Release Process

## Create a Changelog Entry

Every release must have a changelog entry.
The changelog entry should include:
* A summary of the major accomplishments of the release.
* A list of all the changes in the release.
* A list of all the bugs fixed by the release.

Contributions from new collaborators should be acknowledged in the changelog.

## Create Git Annotated Tags and GitHub Releases

Each release needs to have a Git annotated tag and a GitHub release for that tag.
The changelog for the release should be used for the text of the GitHub release.

## Update Compiler Explorer

Thrust and CUB are bundled together on
[Compiler Explorer](https://www.godbolt.org/) (CE) as libraries for the CUDA
language. When releasing a new version of these projects, CE will need to be
updated.

There are two files in two repos that need to be updated:

### libraries.yaml

- Repo: https://github.com/compiler-explorer/infra
- Path: bin/yaml/libraries.yaml

This file tells CE how to pull in library files and defines which versions to
fetch. Look for the `thrustcub:` section:

```yaml
    thrustcub:
      type: github
      method: clone_branch
      repo: NVIDIA/thrust
      check_file: dependencies/cub/cub/cub.cuh
      targets:
        - 1.9.9
        - 1.9.10
        - 1.9.10-1
        - 1.10.0
```

Simply add the new version tag to list of `targets:`. This will check out the
specified tag to `/opt/compiler-explorer/libs/thrustcub/<tag>/`.

### cuda.amazon.properties

- Repo: https://github.com/compiler-explorer/compiler-explorer
- File: etc/config/cuda.amazon.properties

This file defines the library versions displayed in the CE UI and maps them
to a set of include directories. Look for the `libs.thrustcub` section:

```yaml
libs.thrustcub.name=Thrust+CUB
libs.thrustcub.description=CUDA collective and parallel algorithms
libs.thrustcub.versions=trunk:109090:109100:109101:110000
libs.thrustcub.url=http://www.github.com/NVIDIA/thrust
libs.thrustcub.versions.109090.version=1.9.9
libs.thrustcub.versions.109090.path=/opt/compiler-explorer/libs/thrustcub/1.9.9:/opt/compiler-explorer/libs/thrustcub/1.9.9/dependencies/cub
libs.thrustcub.versions.109100.version=1.9.10
libs.thrustcub.versions.109100.path=/opt/compiler-explorer/libs/thrustcub/1.9.10:/opt/compiler-explorer/libs/thrustcub/1.9.10/dependencies/cub
libs.thrustcub.versions.109101.version=1.9.10-1
libs.thrustcub.versions.109101.path=/opt/compiler-explorer/libs/thrustcub/1.9.10-1:/opt/compiler-explorer/libs/thrustcub/1.9.10-1/dependencies/cub
libs.thrustcub.versions.110000.version=1.10.0
libs.thrustcub.versions.110000.path=/opt/compiler-explorer/libs/thrustcub/1.10.0:/opt/compiler-explorer/libs/thrustcub/1.10.0/dependencies/cub
libs.thrustcub.versions.trunk.version=trunk
libs.thrustcub.versions.trunk.path=/opt/compiler-explorer/libs/thrustcub/trunk:/opt/compiler-explorer/libs/thrustcub/trunk/dependencies/cub
```

Add a new version identifier to the `libs.thrustcub.versions` key, using the
convention `X.Y.Z-W -> XXYYZZWW`. Then add a corresponding UI label (the
`version` key) and set of colon-separated include paths for Thrust and CUB
(`path`). The version used in the `path` entries must exactly match the tag
specified in `libraries.yaml`.
