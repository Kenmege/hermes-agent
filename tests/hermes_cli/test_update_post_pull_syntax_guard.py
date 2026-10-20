"""Tests for the post-pull syntax guard in ``hermes update``.

When a bad commit lands on ``main`` with a syntax error in a critical file
(e.g. orphan merge-conflict markers in ``hermes_cli/config.py``), the CLI
becomes unbootable — every ``hermes`` invocation imports those files at
startup. The guard validates them after ``git pull`` and rolls back to the
pre-pull SHA on failure so the user's install stays runnable.

Reference incident: PR #28452 (May 18, 2026) shipped unresolved conflict
markers in ``hermes_cli/config.py``; users who ran ``hermes update`` in
the 7-minute window before #28458 landed could not run any ``hermes``
command afterward.
"""

from __future__ import annotations

from hermes_cli import update_cmd
from hermes_cli import main
import pytest
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# _validate_critical_files_syntax
# ---------------------------------------------------------------------------

def test_validate_critical_files_syntax_tolerates_missing_files(tmp_path):
    """A refactor may legitimately remove one of the critical files — the
    guard should skip missing files, not falsely flag the install as broken."""
    # Populate everything except hermes_constants.py
    for relpath in update_cmd._UPDATE_CRITICAL_FILES:
        if relpath == "hermes_constants.py":
            continue
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stub\n", encoding="utf-8")

    ok, failing_path, error = update_cmd._validate_critical_files_syntax(tmp_path)

    assert ok is True
    assert failing_path is None
    assert error is None


def test_pull_rolls_back_broken_critical_file_and_accepts_corrected_retry(tmp_path, monkeypatch, capsys):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()

    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    source = tmp_path / "hermes_constants.py"
    source.write_text("print('runnable')\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "working")
    previous = git("rev-parse", "HEAD")
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    assert update_cmd._capture_head_sha(["git"], tmp_path) == previous
    # Missing critical files are legitimate after refactors, not syntax failures.
    assert update_cmd._validate_critical_files_syntax(tmp_path) == (True, None, None)

    source.write_text("<<<<<<< HEAD\n", encoding="utf-8")
    git("commit", "-am", "broken upstream")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    git("reset", "--hard", previous)

    def pull():
        return update_cmd._pull_updates(
            ["git"], "main", None, prompt_for_restore=False, gw_input_fn=None,
            discard_local_changes=False, keep_stash=False,
        )

    with pytest.raises(SystemExit) as failure:
        pull()
    assert failure.value.code == 1
    assert "syntax error" in capsys.readouterr().out
    assert git("rev-parse", "HEAD") == previous
    assert subprocess.run([sys.executable, str(source)], capture_output=True, text=True, encoding="utf-8", check=True).stdout == "runnable\n"

    git("reset", "--hard", "origin/main")
    source.write_text("print('corrected')\n", encoding="utf-8")
    git("commit", "-am", "corrected upstream")
    corrected = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/origin/main", corrected)
    git("reset", "--hard", previous)
    assert pull() == previous
    assert git("rev-parse", "HEAD") == corrected
    assert subprocess.run([sys.executable, str(source)], capture_output=True, text=True, encoding="utf-8", check=True).stdout == "corrected\n"


def _python_bump_repo(tmp_path, broken_source: str):
    """Commits: ``newer`` (requires-python >=3.99, ``hermes_constants.py`` = broken_source) and ``same``
    (the same file, a requires-python this interpreter satisfies)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()

    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (tmp_path / "hermes_constants.py").write_text(broken_source, encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\nrequires-python = ">=3.99"\n', encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "requires a future python")
    newer = git("rev-parse", "HEAD")
    pyproject.write_text('[project]\nname = "x"\nrequires-python = ">=3.8"\n', encoding="utf-8")
    git("commit", "-am", "same file, this python")
    return git, newer, git("rev-parse", "HEAD")


def test_a_python_bump_with_no_admitted_interpreter_is_still_syntax_checked(tmp_path, monkeypatch):
    """A release that raises requires-python past this interpreter used to be held only to a
    conflict-marker scan, so ``def broken(:`` (broken under every Python) passed preflight and the
    post-pull guard (review N15). With no installed interpreter the target admits, this one judges,
    and the refusal names the Python to install (real git object store, real worktree)."""
    from hermes_cli import update_cmd_commit as commit

    monkeypatch.setattr(commit, "_admitted_python", lambda spec: None)
    git, newer, same = _python_bump_repo(tmp_path, "def broken(:\n")
    critical = ["hermes_constants.py"]
    refused = commit.target_syntax_error(["git"], tmp_path, newer, critical)
    assert refused is not None and refused[0] == "hermes_constants.py"
    assert "uv python install '>=3.99'" in refused[1]
    assert commit.target_syntax_error(["git"], tmp_path, same, critical)[0] == "hermes_constants.py"

    # The post-pull backstop judges the worktree the same way: rolled back, exit 1.
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    git("checkout", "-q", newer)
    with pytest.raises(SystemExit):
        update_cmd._rollback_if_pulled_syntax_error(["git"], same)
    assert git("rev-parse", "HEAD") == same


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("verdict", ["null", '["hermes_constants.py", "SyntaxError: judged by the target"]'])
def test_a_python_bump_is_judged_by_an_interpreter_the_target_admits(tmp_path, monkeypatch, verdict):
    """Newer grammar this interpreter rejects is admitted when the target's own Python accepts it,
    and refused when that Python rejects it: its verdict, not this compile(), decides (N15)."""
    from hermes_cli import update_cmd_commit as commit

    target_python = tmp_path / "python3.99"
    target_python.write_text(f"#!/bin/sh\ncat > /dev/null\necho '{verdict}'\n", encoding="utf-8")
    target_python.chmod(0o755)
    monkeypatch.setattr(commit, "_admitted_python", lambda spec: str(target_python))
    repo = tmp_path / "repo"
    repo.mkdir()
    _git, newer, _same = _python_bump_repo(repo, "def newer_syntax(:\n")
    judged = commit.target_syntax_error(["git"], repo, newer, ["hermes_constants.py"])
    if verdict == "null":
        assert judged is None
    else:
        assert judged == ("hermes_constants.py", f"SyntaxError: judged by the target (judged by {target_python})")


def test_the_critical_inventory_covers_every_module_the_entry_paths_import_first():
    """``hermes_bootstrap.py`` was absent from the inventory, so a target that broke only the
    bootstrap passed preflight and the post-pull guard and failed at the first import (review G3).
    Derived from the real launcher text, ``hermes_bootstrap``'s imports and the recovery closure."""
    import ast

    from hermes_cli import _launchers
    from hermes_cli._early_recovery import RECOVERY_CLOSURE

    root = Path(update_cmd.__file__).resolve().parents[1]

    def files(source: str) -> set[str]:
        names = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                names |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names |= {node.module, *(f"{node.module}.{alias.name}" for alias in node.names)}
        found = set()
        for name in names:
            parts = name.split(".")
            for i in range(1, len(parts) + 1):
                stem = "/".join(parts[:i])
                found |= {rel for rel in (f"{stem}.py", f"{stem}/__init__.py") if (root / rel).is_file()}
        return found

    launcher = _launchers._launcher_script("hermes", root, None)
    entry = files(launcher) | files((root / "hermes_bootstrap.py").read_text(encoding="utf-8"))
    entry |= {"hermes_bootstrap.py", *RECOVERY_CLOSURE}
    assert "hermes_bootstrap.py" in entry and "hermes_cli/main.py" in entry
    assert entry <= set(update_cmd._UPDATE_CRITICAL_FILES), sorted(entry - set(update_cmd._UPDATE_CRITICAL_FILES))


_BOOT_DUMP = "HERMES-BOOT-MODULES="


def _disposable_checkout(source: Path, destination: Path) -> Path:
    """A disposable copy of the real boot sources, with no inherited install state.

    Copies the checkout's root modules and its ``hermes_cli``, ``pm`` and
    ``hermes_platform`` packages, plus ``pyproject.toml`` (and PM's lockfile when
    present), from ``source`` -- never the checkout the suite runs from as a boot
    target. Only ``.py`` sources are copied, so no install stamp, update or
    recovery marker, launch artifact or bytecode can be inherited; the empty
    ``.git`` directory is the controlled checkout metadata the boot checks for.
    """
    import shutil

    destination.mkdir(parents=True)
    sources = list(source.glob("*.py"))
    for package in ("hermes_cli", "pm", "hermes_platform"):
        sources.extend(
            path for path in (source / package).rglob("*.py")
            if "__pycache__" not in path.parts)
    for path in sources:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(source / "pyproject.toml", destination / "pyproject.toml")
    lockfile = source / "pm" / "lock.json"
    if lockfile.is_file():
        shutil.copy2(lockfile, destination / "pm" / "lock.json")
    (destination / ".git").mkdir()
    assert not (destination / "install-stamp.json").exists()
    assert not (destination / ".hermes-update-zip-swap").exists()
    assert not (destination / ".git" / "hermes-update-pull").exists()
    return destination


def _booted_checkout_files(root: Path, home: Path) -> set[str]:
    """The disposable checkout's files loaded through completion of the bootstrap import.

    Runs the published launcher's own text, generated for the disposable ``root``,
    up to the end of its ``import hermes_bootstrap`` block, on this interpreter,
    with nothing of the caller's environment: a clean data root, a home under it
    and the platform locations under it, so every run boots the same way.
    The dump is every loaded module's file, kept when it lies in that disposable
    checkout and outside any ``site-packages``/``dist-packages`` and the
    interpreter's stdlib -- by path, not by name. ``-B`` is passed explicitly
    alongside ``-I`` (which ignores ``PYTHONDONTWRITEBYTECODE``), and UTF-8 mode
    is requested with the explicit ``-X utf8`` interpreter option, so the child
    cannot write bytecode into any checkout.
    """
    import json
    import os
    import sysconfig

    from hermes_cli import _launchers

    lines = _launchers._launcher_script("hermes", root, None).splitlines(keepends=True)
    start = next(n for n, line in enumerate(lines) if line.strip() == "import hermes_bootstrap")
    end = start + 1
    while end < len(lines) and (lines[end][:1].isspace() or lines[end].startswith(("except", "finally"))):
        end += 1
    probe = "".join(lines[:end]) + (
        "import json as _boot_json\n"
        "_boot_files = {m.__file__ for m in list(sys.modules.values())"
        " if isinstance(getattr(m, '__file__', None), str)}\n"
        f"print({_BOOT_DUMP!r} + _boot_json.dumps(sorted(_boot_files)))\n")
    user = home.parent
    env = {"HERMES_HOME": str(home), "HOME": str(user), "PATH": os.defpath}
    if os.name == "nt":
        env.update(SYSTEMROOT=os.environ.get("SYSTEMROOT", r"C:\Windows"), USERPROFILE=str(user),
                   LOCALAPPDATA=str(user / "AppData" / "Local"), APPDATA=str(user / "AppData" / "Roaming"),
                   TEMP=str(user), TMP=str(user))
    # Run from a file, as a published launcher is run. ``-I`` keeps the file's directory off
    # sys.path, exactly as it keeps it off for ``-c``; ``-B`` (not an environment
    # variable, which ``-I`` ignores) keeps bytecode out of the disposable tree.
    script = user / "launcher-boot-probe.py"
    script.write_text(probe, encoding="utf-8")
    result = subprocess.run([sys.executable, "-I", "-B", "-X", "utf8", str(script)], cwd=user, env=env, capture_output=True,
                            text=True, encoding="utf-8", timeout=120, check=False)
    dumps = [line for line in result.stdout.splitlines() if line.startswith(_BOOT_DUMP)]
    assert result.returncode == 0 and dumps, result.stdout + result.stderr
    outside = [Path(sysconfig.get_paths()[key]).resolve() for key in ("stdlib", "platstdlib", "purelib", "platlib")]
    booted = set()
    for name in json.loads(dumps[-1][len(_BOOT_DUMP):]):
        path = Path(name).resolve()
        if not path.is_relative_to(root) or any(path.is_relative_to(other) for other in outside):
            continue
        relative = path.relative_to(root)
        if not {"site-packages", "dist-packages"} & set(relative.parts):
            booted.add(relative.as_posix())
    return booted


@pytest.mark.platforms("any")
@pytest.mark.parametrize("state", ["fresh", "committed"])
def test_the_critical_inventory_covers_every_module_a_real_boot_imports(tmp_path, monkeypatch, state):
    """The existing inventory test walks imports in the launcher and bootstrap source, but
    does not follow transitive imports inside the modules they load: for example
    ``pm.paths`` is reached through ``prepare_launch`` -> ``read_install_stamp`` ->
    ``pm.paths``, and the inventory missed it, so a target that broke only that
    file passed the pre-move syntax check. This test collects modules loaded
    through completion of the bootstrap import in clean fresh/committed fixtures
    on the host OS, booting a disposable copy of the checkout under ``tmp_path``
    so the probe can never mutate the checkout or installation it runs from.
    The disposable copy carries no install stamp, so ``prepare_launch`` still
    takes its unstamped-source branch (``hermes_cli/post_update.py`` is loaded)
    without any mock or disable flag.
    """
    import json

    from pm.environments import install_state_dir, site_packages

    source = Path(update_cmd.__file__).resolve().parents[1]
    root = _disposable_checkout(source, tmp_path / "checkout")
    home = tmp_path / "user" / ".hermes"
    home.mkdir(parents=True)
    if state == "committed":
        monkeypatch.setenv("HERMES_HOME", str(home))
        selected = install_state_dir(root) / "environments" / "selected" / "venv"
        site_packages(selected).mkdir(parents=True)
        (selected / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
        (install_state_dir(root) / "facts.json").write_text(
            json.dumps({"packages": {"venv": {"environment": str(selected)}}}), encoding="utf-8")

    booted = _booted_checkout_files(root, home)

    assert {"hermes_bootstrap.py", "pm/environments.py"} <= booted, sorted(booted)
    assert "hermes_cli/post_update.py" in booted, sorted(booted)
    assert ("hermes_cli/runtime_state.py" in booted) is (state == "committed"), sorted(booted)
    assert booted <= set(update_cmd._UPDATE_CRITICAL_FILES), sorted(booted - set(update_cmd._UPDATE_CRITICAL_FILES))


def test_a_malformed_bootstrap_is_refused_before_the_move(tmp_path):
    """The preflight reads the real inventory from the target commit: a broken bootstrap refuses."""
    from hermes_cli import update_cmd_commit as commit

    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()

    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (tmp_path / "hermes_bootstrap.py").write_text("def broken(:\n", encoding="utf-8")
    (tmp_path / "hermes_constants.py").write_text("ok = 1\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "broken bootstrap")
    refused = commit.target_syntax_error(["git"], tmp_path, "HEAD", update_cmd._UPDATE_CRITICAL_FILES)
    assert refused is not None and refused[0] == "hermes_bootstrap.py"


def test_the_preflight_reads_every_critical_file_in_one_git_spawn(tmp_path, monkeypatch):
    """One ``git cat-file --batch`` for pyproject + the whole inventory, not a ``git show`` each."""
    from hermes_cli import update_cmd_commit as commit

    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()

    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    for rel in ("hermes_constants.py", "cli.py"):
        (tmp_path / rel).write_text("ok = 1\n", encoding="utf-8")
    (tmp_path / "hermes_cli").mkdir()
    (tmp_path / "hermes_cli" / "main.py").write_text("x = (\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "c")
    spawns = []
    real = commit.run_git
    monkeypatch.setattr(commit, "run_git", lambda git_cmd, args, **kw: spawns.append(args) or real(git_cmd, args, **kw))
    refused = commit.target_syntax_error(["git"], tmp_path, "HEAD", update_cmd._UPDATE_CRITICAL_FILES)
    assert refused is not None and refused[0] == "hermes_cli/main.py"
    assert spawns == [["cat-file", "--batch"]]


def test_the_tree_move_marker_records_the_absolute_git_the_repair_reruns(tmp_path, monkeypatch):
    """The next launch's repair cannot ask ``pm`` for the store git when the move tore a module ``pm``
    imports, so the marker carries the absolute git this run resolved (PATH or ``expose_pm_git``)."""
    from hermes_cli import _early_recovery
    from hermes_cli import update_cmd_commit as commit

    (tmp_path / ".git").mkdir()
    fake = tmp_path / "store" / "git" / "cmd" / ("git.exe" if sys.platform == "win32" else "git")
    fake.parent.mkdir(parents=True)
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr("shutil.which", lambda name: str(fake) if name == "git" else None)
    marker = commit.arm_tree_move(["git", "-c", "gc.autoDetach=false"], tmp_path, pre="a" * 40,
                                  target="b" * 40, stash=None)
    fields = dict(line.partition("=")[::2] for line in marker.read_text(encoding="utf-8").splitlines())
    assert fields["git"] == str(fake)
    # ...and the repair prefers it to PATH (empty here) and to the pm lookup, once it answers as a
    # git (m5: this fake is an empty file; the version probe is what decides).
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(_early_recovery, "_is_git", lambda path: path == str(fake))
    assert _early_recovery._git_executable(fields["git"]) == str(fake)
    assert _early_recovery._git_executable(str(tmp_path / "gone")) != str(tmp_path / "gone")


def test_a_python_bump_never_admits_a_conflict_marker(tmp_path, monkeypatch):
    """A release that raises requires-python past this interpreter is excused only syntax a newer
    Python may parse: a merge-conflict marker breaks every Python, before and after the move (N15)."""
    from hermes_cli import update_cmd_commit as commit

    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()

    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (tmp_path / "hermes_constants.py").write_text("good = True\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nrequires-python = ">=3.8"\n', encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "pre")
    pre = git("rev-parse", "HEAD")
    (tmp_path / "hermes_constants.py").write_text("<<<<<<< HEAD\na = 1\n=======\na = 2\n>>>>>>> topic\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nrequires-python = ">=3.99"\n', encoding="utf-8")
    git("commit", "-am", "conflicted, and bumps python")
    broken = git("rev-parse", "HEAD")

    assert commit.target_syntax_error(["git"], tmp_path, broken, ["hermes_constants.py"]) is not None
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    with pytest.raises(SystemExit):
        update_cmd._rollback_if_pulled_syntax_error(["git"], pre)
    assert git("rev-parse", "HEAD") == pre


def test_a_marker_arm_that_times_out_still_rolls_the_broken_pull_back(tmp_path, monkeypatch, capsys):
    """arm_tree_move's ``symbolic-ref`` can raise subprocess.TimeoutExpired; the rollback caught only
    OSError, so a timeout aborted it before any reset and left the broken release checked out
    (review C10). It must roll back unmarked, as for an unwritable marker."""
    from hermes_cli import update_cmd_commit

    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True,
                              encoding="utf-8").stdout.strip()

    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    source = tmp_path / "hermes_constants.py"
    source.write_text("print('runnable')\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "working")
    previous = git("rev-parse", "HEAD")
    source.write_text("<<<<<<< HEAD\n", encoding="utf-8")
    git("commit", "-am", "broken upstream")
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    def times_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["git", "symbolic-ref", "-q", "HEAD"], 60)

    monkeypatch.setattr(update_cmd_commit, "arm_tree_move", times_out)
    with pytest.raises(SystemExit) as failure:
        update_cmd._rollback_if_pulled_syntax_error(["git"], previous)

    assert failure.value.code == 1
    assert "Rollback complete" in capsys.readouterr().out
    assert git("rev-parse", "HEAD") == previous
    assert source.read_text(encoding="utf-8") == "print('runnable')\n"
