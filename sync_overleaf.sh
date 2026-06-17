#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PAPER_DIR="PoleSight----WACV-2027"
OVERLEAF_REMOTE="overleaf"
OVERLEAF_BRANCH="master"

cd "$REPO_ROOT"

usage() {
    cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  pull    Pull latest changes from Overleaf into $PAPER_DIR/
  push    Push local $PAPER_DIR/ changes to Overleaf
  status  Show diff between local paper dir and Overleaf

Overleaf project: https://www.overleaf.com/project/6a32624961683d3174b7b418
EOF
    exit 1
}

[[ $# -lt 1 ]] && usage

ensure_remote() {
    if ! git remote get-url "$OVERLEAF_REMOTE" &>/dev/null; then
        echo "Adding overleaf remote..."
        git remote add "$OVERLEAF_REMOTE" "https://git.overleaf.com/6a32624961683d3174b7b418"
    fi
}

pull_from_overleaf() {
    ensure_remote
    echo "Fetching from Overleaf..."
    git fetch "$OVERLEAF_REMOTE" "$OVERLEAF_BRANCH"

    echo "Pulling Overleaf changes into $PAPER_DIR/..."
    git subtree pull \
        --prefix="$PAPER_DIR" \
        "$OVERLEAF_REMOTE" "$OVERLEAF_BRANCH" \
        --squash \
        -m "Sync: pull latest from Overleaf"

    echo "Done. Overleaf changes merged into $PAPER_DIR/."
}

push_to_overleaf() {
    ensure_remote

    if ! git diff --quiet HEAD -- "$PAPER_DIR/"; then
        echo "You have uncommitted changes in $PAPER_DIR/. Commit them first."
        git status -- "$PAPER_DIR/"
        exit 1
    fi

    echo "Pushing $PAPER_DIR/ to Overleaf..."
    git subtree push \
        --prefix="$PAPER_DIR" \
        "$OVERLEAF_REMOTE" "$OVERLEAF_BRANCH"

    echo "Done. Local paper pushed to Overleaf."
}

show_status() {
    ensure_remote
    echo "Fetching from Overleaf..."
    git fetch "$OVERLEAF_REMOTE" "$OVERLEAF_BRANCH"

    echo ""
    echo "=== Local uncommitted changes in $PAPER_DIR/ ==="
    git status -- "$PAPER_DIR/"

    echo ""
    echo "=== Diff between local $PAPER_DIR/ and Overleaf ==="
    git diff HEAD "$OVERLEAF_REMOTE/$OVERLEAF_BRANCH" -- . 2>/dev/null \
        || echo "(Cannot diff directly — use pull/push to sync.)"
}

case "$1" in
    pull)   pull_from_overleaf ;;
    push)   push_to_overleaf ;;
    status) show_status ;;
    *)      usage ;;
esac
