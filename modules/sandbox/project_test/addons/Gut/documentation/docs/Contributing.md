# Contributing

## Contributing Documentation
The wiki is hosted at https://gut.readthedocs.io.  The source for documentation is in the `documentation` directory.  There's info about the wiki structure and local documentation generation in `documentation/README.md` in the repo.


## Contributing Code

### Checklist for PRs
* Open PRs against `main` for Godot 4 issues, or the `godot_3x` branc for Godot 3 issues.
* PRs __must have unit tests__.  See sections below.
* Include any wiki text in the PR.
  * Info about documentation changes can be found in `documentation/README.md`.
* CHANGES.md
  * I will take care of making any changes to CHANGES.md.
  * I will credit you in the CHANGES.md.  If you have a handle you would like me to use (other than your GitHub username) then let me know in the PR

### Creating Tests for GUT

#### All GUT tests are found in
* `res://test/unit`
* `res://test/integration`

Edit existing scripts or add new ones there.

#### Any resources needed by tests should be placed in:
* `res://test/resources`

If you don't see an existing directory that matches your needs then you can create a new directory or place them directly in `res://test/resources`.


### Running GUT Tests
Due to the nature of using GUT to test GUT, some tests may not work as intended.  These are being cleaned up, but there may still be some left in the codebase.  You should run all the tests once before developing to see which tests are currently in use.

Here's some common errors:

#### The GUT Panel doesn't do anything.
Sometimes when you edit GUT files, the plugin doesn't like it.  Reload the plugin.

#### If you see this in the IDE Output
```
res://addons/gut/gui/GutBottomPanel.gd:### - Invalid get index '<whatever>' (on base: 'Nil').
```
Then reload the plugin.
