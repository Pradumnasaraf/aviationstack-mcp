// Pre-commit hook for TriPSs/conventional-changelog-action.
//
// The action bumps the version in pyproject.toml, then calls this hook, then
// runs `git add .` and creates the release commit and tag. Refreshing uv.lock
// here means the lock file version rides along in the same commit and tag as
// the version bump, instead of drifting behind it.
//
// .cjs so it is loaded as CommonJS regardless of any package.json added later.

const { execFileSync } = require('node:child_process')

module.exports = {
  preCommit: ({ version }) => {
    console.log(`Refreshing uv.lock for version ${version}`)
    execFileSync('uv', ['lock'], { stdio: 'inherit' })
  },
}
