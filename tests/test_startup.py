"""Run the launcher against fake host commands without touching Docker/services."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path):
    project = tmp_path / "studio"
    project.mkdir()
    shutil.copyfile(ROOT / "start.sh", project / "start.sh")
    commands = tmp_path / "commands"
    commands.mkdir()
    log = tmp_path / "commands.jsonl"
    fake_command = f"""#!{sys.executable}
import json
import os
from pathlib import Path
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['TEST_COMMAND_LOG'], 'a') as output:
    output.write(json.dumps([name, *args]) + '\\n')
if name == 'df':
    print('Filesystem 1024-blocks Used Available Capacity Mounted on')
    print('/dev/test 200000000 0 ' + os.environ.get('TEST_FREE_KIB', '104857600') + ' 0% /')
elif name == 'docker':
    if args[:2] == ['image', 'inspect']:
        loaded = Path(os.environ['TEST_COMMAND_LOG'] + '.loaded').exists()
        sys.exit(0 if loaded or os.environ.get('TEST_IMAGE_EXISTS', '1') == '1' else 1)
    if args[:2] == ['image', 'load']:
        Path(os.environ['TEST_COMMAND_LOG'] + '.loaded').touch()
    if 'build' in args:
        sys.exit(int(os.environ.get('TEST_BUILD_EXIT', '0')))
    if 'port' in args:
        print('127.0.0.1:7861')
"""
    for command in ("docker", "systemctl", "df"):
        executable = commands / command
        executable.write_text(fake_command)
        executable.chmod(0o755)

    def run(*args, archive=False, **overrides):
        if archive:
            (project / ".cache").mkdir(exist_ok=True)
            (project / ".cache/t2000-gpu.tar").touch()
        env = {**os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"],
               "TEST_COMMAND_LOG": str(log), **{key: str(value) for key, value in overrides.items()}}
        result = subprocess.run(["bash", str(project / "start.sh"), *args],
                                capture_output=True, text=True, env=env)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return run


def test_default_launcher_starts_shared_image_and_releases_native_studios(launcher):
    result, calls = launcher()
    assert result.returncode == 0, result.stderr
    assert ["docker", "image", "inspect", "t2000:gpu"] in calls
    assert ["systemctl", "--user", "stop", "t2000.service"] in calls
    assert ["systemctl", "--user", "stop", "t1000-3050.service"] in calls
    assert ["docker", "compose", "up", "-d", "--no-build", "--pull", "never", "--wait",
            "--wait-timeout", "180", "studio"] in calls
    assert "http://127.0.0.1:7861" in result.stdout


def test_high_vram_launcher_uses_its_own_compose_and_does_not_stop_native_studios(launcher):
    result, calls = launcher("--high-vram")
    assert result.returncode == 0, result.stderr
    assert ["docker", "image", "inspect", "t2000:high-vram"] in calls
    assert ["docker", "compose", "-p", "t2000-high-vram", "-f", "compose.high-vram.yaml",
            "up", "-d", "--no-build", "--pull", "never", "--wait", "--wait-timeout",
            "3600", "studio"] in calls
    assert not any(call[0] == "systemctl" for call in calls)
    for call in calls:
        if call[:2] == ["docker", "compose"]:
            assert call[2:6] == ["-p", "t2000-high-vram", "-f", "compose.high-vram.yaml"]


@pytest.mark.parametrize("options", [(), ("--high-vram",)])
def test_build_failure_never_starts_a_stale_image_or_stops_native_services(launcher, options):
    result, calls = launcher("--build", *options, TEST_BUILD_EXIT=7)
    assert result.returncode != 0
    assert any("build" in call for call in calls)
    assert not any("up" in call or call[0] == "systemctl" for call in calls)


@pytest.mark.parametrize("options,expected_exit,minimum", [((), 1, "60"), (("--high-vram",), 0, "40")])
def test_disk_budget_follows_selected_deployment(launcher, options, expected_exit, minimum):
    result, calls = launcher(*options, "--build", TEST_FREE_KIB=50 * 1024 * 1024)
    assert result.returncode == expected_exit, result.stderr
    if expected_exit:
        assert f"{minimum} GiB" in result.stderr
        assert not any("build" in call or "up" in call for call in calls)
    else:
        assert any("build" in call for call in calls)


def test_shared_launcher_restores_recovery_archive_before_stopping_services(launcher):
    result, calls = launcher(archive=True, TEST_IMAGE_EXISTS=0)
    assert result.returncode == 0, result.stderr
    load_index = next(index for index, call in enumerate(calls) if call[:3] == ["docker", "image", "load"])
    stop_index = next(index for index, call in enumerate(calls) if call[:3] == ["systemctl", "--user", "stop"])
    assert load_index < stop_index
    assert not any("build" in call for call in calls)


def test_high_vram_never_loads_the_shared_recovery_archive(launcher):
    result, calls = launcher("--high-vram", archive=True, TEST_IMAGE_EXISTS=0)
    assert result.returncode == 0, result.stderr
    assert any("build" in call for call in calls)
    assert not any(call[:3] == ["docker", "image", "load"] for call in calls)


def test_shared_image_finishes_building_before_native_studios_are_stopped(launcher):
    result, calls = launcher("--build")
    assert result.returncode == 0, result.stderr
    build_index = next(index for index, call in enumerate(calls) if "build" in call)
    stop_index = next(index for index, call in enumerate(calls) if call[:3] == ["systemctl", "--user", "stop"])
    assert build_index < stop_index


def test_invalid_option_does_not_touch_docker_or_host_services(launcher):
    result, calls = launcher("--unknown")
    assert result.returncode == 2
    assert calls == []
