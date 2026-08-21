#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

apply_patch_once() {
  local patch="$1"
  local marker="$2"
  local target="$3"
  if grep -Fq "$marker" "$target"; then
    echo "$patch is already applied."
    return
  fi
  if git apply --check "$patch"; then
    git apply "$patch"
    echo "Applied $patch."
  else
    echo "DeepSeek vendor files do not match the expected baseline for $patch." >&2
    exit 1
  fi
}

apply_patch_once \
  patches/deepseek-tool-recovery.patch \
  "maximumRetries = 2" \
  vendor/deepseek-web-api/dist/deepseek/client.js
apply_patch_once \
  patches/deepseek-tool-inference.patch \
  "function inferredToolName" \
  vendor/deepseek-web-api/dist/deepseek/toolCalls.js
apply_patch_once \
  patches/deepseek-malformed-opening-tag.patch \
  "function normalizeMalformedOpeningTags" \
  vendor/deepseek-web-api/dist/deepseek/toolCalls.js
apply_patch_once \
  patches/deepseek-flat-arguments.patch \
  'Object.prototype.hasOwnProperty.call(nested, "arguments")' \
  vendor/deepseek-web-api/dist/deepseek/toolCalls.js
apply_patch_once \
  patches/deepseek-json-tool-variants.patch \
  "const embeddedName = !explicitName" \
  vendor/deepseek-web-api/dist/deepseek/toolCalls.js
apply_patch_once \
  patches/deepseek-current-task-mutation-evidence.patch \
  "function hasVerifiedMutationEvidence" \
  vendor/deepseek-web-api/dist/deepseek/client.js
apply_patch_once \
  patches/deepseek-chinese-deferred-tool-intent.patch \
  "const finalSentence = compact" \
  vendor/deepseek-web-api/dist/deepseek/toolOutcome.js
apply_patch_once \
  patches/deepseek-mismatched-command-tool.patch \
  "function normalizeMismatchedCommandTags" \
  vendor/deepseek-web-api/dist/deepseek/toolCalls.js
