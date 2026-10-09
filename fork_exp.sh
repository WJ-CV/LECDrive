#!/usr/bin/env bash

EXP_PATH=${1:-""}

DATETIME=$(date +"%Y-%m-%d_%H%M")
EXP_NAME=$(basename "${EXP_PATH}")
EXP_ROOT=$(dirname "${EXP_PATH}")
NEW_EXP_PATH="${EXP_ROOT}/${DATETIME}_${EXP_NAME}"

function prompt_yes_or_no() {
    # Prompt the user for yes or no input
    while true; do
        read -r -p "$1 [y/n]: " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer [y]es or [n]o.";;
        esac
    done
}


function fork_experiment() {
  # Backup the code
  local EXP=${1}
  if git status --porcelain | grep -q '??'; then
    { echo >&2 "Attention! There are untracked files!!!"; git status --porcelain | grep '??'; }
    # Prompt the user for input
    if ! prompt_yes_or_no "Do you want to continue?"; then
      exit 1
    fi
  fi
  if [ ! -d "${EXP}" ]; then
    mkdir -p "${EXP}"
    cp -r --parents .git $(git ls-files) "${EXP}"
    echo "The code has been backed up to ${EXP} ... "
  fi
}


fork_experiment "${NEW_EXP_PATH}"