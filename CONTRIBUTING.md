> IMPORTANT **Note**
>
> **Pull Requests having no issue associated with them will not be accepted. Firstly get an issue assigned, whether it's already opened or raised by you, and then create a Pull Request.**

## Prerequisite

#### Documentation

- [Git](https://git-scm.com/)
- [Markdown](https://www.markdownguide.org/basic-syntax/)

#### Code

- [Python](https://www.python.org/)
- [AviationStack API](https://aviationstack.com/)

## How to Contribute

- Look at the existing [**Issues**](https://github.com/pradumnasaraf/aviationstack-mcp/issues) or [**create a new issue**](https://github.com/pradumnasaraf/aviationstack-mcp/issues/new/choose)!
- [**Fork the Repo**](https://github.com/pradumnasaraf/aviationstack-mcp/fork). Then, create a branch for any issue that you are working on. Finally, commit your work.
- Create a **[Pull Request](https://github.com/pradumnasaraf/aviationstack-mcp/compare)** (_PR_), which will be promptly reviewed and given suggestions for improvements by the community.
- Add screenshots or screen captures to your Pull Request to help us understand the effects of the changes proposed in your PR.

## Before you open a pull request

Run the same checks CI runs. All three must pass.

```bash
uv sync --all-groups

uv run python -m unittest discover -s tests -v
uv run pylint $(git ls-files '*.py')
uv run python scripts/generate_server_card.py --check
```

A few conventions worth knowing:

- **Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).**
  This is not cosmetic. Releases are cut automatically, and only `feat:` and `fix:`
  trigger one. A commit with no prefix ships nothing.
- **Every tool returns the shared envelope**, never a bare list or a plain string.
  Success is `{"ok": true, "count": N, "data": [...]}`, failure is
  `{"ok": false, "context": "...", "error": "..."}`.
- **Tools never raise.** Catch with `TOOL_ERRORS` and return `_error_response`.
- **Never put the API key in an error.** Pass exception text through `_scrub_secrets`.
- **Adding or changing a tool means regenerating the server card.**
- **Add a test.** New tools need coverage of the happy path, the error path, and any
  validation they add.

Please note that the above instructions assume familiarity with Git, Markdown, Python. If you need any assistance or have further questions, feel free to ask. Happy contributing!
