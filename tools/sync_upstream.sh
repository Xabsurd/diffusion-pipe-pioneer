#!/usr/bin/env bash
# Merge upstream/main into the current branch.
# Stops on conflicts. A ComfyUI gitlink conflict is left for a person:
# keep this branch when upstream is older or has no flux3 / moss_image;
# take upstream when its pin is newer and those models are still present, then retest.
set -euo pipefail
cd "$(dirname "$0")/.."
git fetch upstream
git merge upstream/main
git -c safe.directory="$(pwd)/submodules/ComfyUI" submodule update --init submodules/ComfyUI
