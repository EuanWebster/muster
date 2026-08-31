"""Regression: spawn_logged must not deadlock a chatty child (the old PIPE bug)."""
import time
import app


def test_chatty_child_survives():
    # 1 MB of output, far past the 64K pipe buffer that used to wedge the child.
    err = app.spawn_logged(
        "python3 -c \"import sys,time\n"
        "[sys.stdout.write('x'*1000+chr(10)) for _ in range(1000)]\nsys.stdout.flush()\ntime.sleep(30)\"",
        "pytest-chatty", shell=True)
    assert err is None, err
    time.sleep(1)
    log = app.LOG_DIR / "pytest-chatty.log"
    assert log.stat().st_size > 500_000, f"child blocked at {log.stat().st_size} bytes"


def test_immediate_exit_is_reported():
    err = app.spawn_logged("echo boom >&2; exit 3", "pytest-fail", shell=True)
    assert err and "code 3" in err and "boom" in err


if __name__ == "__main__":
    test_chatty_child_survives()
    test_immediate_exit_is_reported()
    print("both pass")
