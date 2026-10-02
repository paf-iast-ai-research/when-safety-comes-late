#!/usr/bin/env bash
# Create the annotated tag "registration" on the registration commit (Part 9, Table 9.0;
# First Tasks, Role 1 step 2a). A tag is added to an existing commit without changing it.
#
#   bash scripts/tag_registration.sh          # verify and create the tag locally
#   git push origin registration              # then publish it (the pilot owner does this)
set -euo pipefail

COMMIT="735b394d18d1bb046c7a74f900af00a83f1fa316"
TIMESTAMP="2026-09-18T23:22:45+05:00"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
git cat-file -e "${COMMIT}^{commit}" || { echo "error: registration commit $COMMIT not found" >&2; exit 1; }

# diff-tree --root lists a root commit's files whatever the user's log.showRoot (git show may list none)
files="$(git diff-tree --root --no-commit-id --name-only -r "$COMMIT" | sed '/^$/d' | sort | tr '\n' ' ')"
if [ "$files" != "prereg/Preregistration.docx prereg/Preregistration.pdf " ]; then
  echo "error: $COMMIT does not contain exactly the two registration files: $files" >&2; exit 1
fi
if [ "$(git show -s --format=%cI "$COMMIT")" != "$TIMESTAMP" ]; then
  echo "error: commit timestamp differs from $TIMESTAMP" >&2; exit 1
fi
if [ -n "$(git diff "$COMMIT" HEAD -- prereg/)" ]; then
  echo "warning: prereg/ has changed since the registration commit" \
    "(expected after Table 9.0 is filled, First Tasks, Role 1 step 3, and after an amendment)" >&2
fi
if git rev-parse -q --verify "refs/tags/registration" >/dev/null; then
  existing="$(git rev-parse "registration^{commit}")"
  if [ "$(git cat-file -t refs/tags/registration)" != "tag" ]; then
    echo "error: tag 'registration' (on $existing) is a lightweight tag; X-registration names an annotated one" \
      "(if it is not pushed yet: git tag -d registration, then run this script again)" >&2; exit 1
  fi
  if [ "$existing" = "$COMMIT" ]; then echo "tag 'registration' already points at $COMMIT"; exit 0; fi
  echo "error: tag 'registration' exists and points at $existing" >&2; exit 1
fi

git tag -a registration "$COMMIT" -m "Registration commit"
echo "created tag 'registration' -> $COMMIT (committed $TIMESTAMP)"
echo "publish it with: git push origin registration"
