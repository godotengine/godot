#!/bin/bash

mkdir "${HOME}/.npm-packages"
npm config set prefix "~/.npm-packages"

cat >> ~/.bashrc <<EOL
export PATH="$PATH:~/.npm-packages/bin"
EOL
