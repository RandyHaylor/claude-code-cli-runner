"""The runner keeps its own files (live log, control channel, run status, session store)
in the directory the agent works in. A repository the agent creates there with
`git init` + `git add .` must not pick them up."""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_code_cli_runner.runner import (
    keep_runner_bookkeeping_files_out_of_any_git_repository_the_agent_makes,
)


def test_git_add_all_in_the_working_directory_skips_the_runners_files(tmp_path):
    workspace = str(tmp_path)
    for runner_file in ("task_live_log.jsonl", "task_control_channel.jsonl", "task_run_status.json"):
        (tmp_path / runner_file).write_text("runner data\n", encoding="utf-8")
    (tmp_path / "pi_sessions").mkdir()
    (tmp_path / "pi_sessions" / "session.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "game.py").write_text("print('the agent wrote this')\n", encoding="utf-8")

    harness_environment = dict(os.environ)
    harness_environment.pop("GIT_CONFIG_COUNT", None)
    keep_runner_bookkeeping_files_out_of_any_git_repository_the_agent_makes(
        workspace, harness_environment
    )

    def git(*arguments):
        return subprocess.run(
            ["git", *arguments], cwd=workspace, env=harness_environment,
            capture_output=True, text=True, check=True,
        ).stdout

    git("init", "-q")
    git("add", ".")
    staged = set(git("diff", "--cached", "--name-only").split())
    assert staged == {"game.py"}


def test_an_environment_that_already_drives_git_config_is_left_alone(tmp_path):
    harness_environment = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "user.name",
                           "GIT_CONFIG_VALUE_0": "someone"}
    keep_runner_bookkeeping_files_out_of_any_git_repository_the_agent_makes(
        str(tmp_path), harness_environment
    )
    assert harness_environment["GIT_CONFIG_KEY_0"] == "user.name"
    assert not (tmp_path / ".runner_bookkeeping_git_excludes").exists()
