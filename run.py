"""Robust launcher for HoopTracker.

Handles two real-world launch hazards seen on this machine:

1. Double-clicking can run Python with the per-user site-packages effectively
   hidden, giving 'ModuleNotFoundError: torch'. We fix our OWN sys.path from
   inside Python (where the locations can be computed reliably).

2. Antivirus (Bitdefender) false-positives: it flagged a helper PowerShell as a
   "malicious command line" and can block torch's native DLLs as they load,
   making 'import torch' fail on its compiled core. So we DON'T spawn any
   PowerShell — the browser is opened from inside this Python process — and the
   preflight reports the REAL error (a blocked DLL looks different from a
   genuinely missing package) so we can tell the two apart.
"""
import os
import sys
import site
import time
import socket
import threading
import webbrowser


def _add(path):
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


# --- 1) Make sure the per-user site-packages is importable. ---
try:
    _add(site.getusersitepackages())
except Exception:
    pass

_ver = "Python%d%d" % (sys.version_info.major, sys.version_info.minor)
for _root in (os.environ.get("APPDATA"), os.path.join(os.path.expanduser("~"), "AppData", "Roaming")):
    if _root:
        _add(os.path.join(_root, "Python", _ver, "site-packages"))

_add(os.path.dirname(os.path.abspath(__file__)))  # project dir (config/stats/detection)

# --- Preflight: report the REAL error, not just "missing". ---
_problems = []
for _mod in ("torch", "ultralytics", "cv2", "flask"):
    try:
        __import__(_mod)
    except Exception as e:  # ModuleNotFoundError OR ImportError/OSError (blocked DLL)
        _first = (str(e).splitlines() or [""])[0]
        _problems.append((_mod, type(e).__name__, _first))

if _problems:
    print("=" * 66)
    print(" HoopTracker could not load these packages:")
    for _m, _t, _msg in _problems:
        print("   - %-12s %s: %s" % (_m, _t, _msg))
    print("=" * 66)
    print(" Python exe :", sys.executable)
    print(" Version    :", sys.version.split()[0])
    print(" APPDATA    :", os.environ.get("APPDATA") or "(blank)")
    print(" sys.path   :")
    for _p in sys.path:
        print("     ", _p)
    print("-" * 66)
    _looks_blocked = any(("_C" in _msg) or ("DLL" in _msg.upper()) or ("dynamic link" in _msg.lower())
                         for _m, _t, _msg in _problems)
    if _looks_blocked:
        print(" This looks like the package is INSTALLED but its native library was")
        print(" BLOCKED while loading -- almost always an ANTIVIRUS false positive.")
        print(" Fix: add a Bitdefender exception for this folder and the Python")
        print(" packages folder (see the chat for exact steps), then relaunch.")
    else:
        print(" If a package truly isn't installed, run:")
        print('   "%s" -m pip install --user torch ultralytics opencv-python flask' % sys.executable)

    # ----------------------------- DEEP PROBE ----------------------------- #
    # Reports WHY the finder can't see the package, from inside THIS process.
    print("-" * 66)
    print(" DEEP PROBE")
    import importlib.util as _ilu
    import traceback as _tb

    _site_dirs = [p for p in sys.path if p.lower().endswith("site-packages")]
    print(" site-packages dirs on sys.path:")
    for _sd in _site_dirs:
        try:
            _names = os.listdir(_sd)
            print("   [listable, %d entries, 'torch' present=%s] %s"
                  % (len(_names), ("torch" in _names), _sd))
        except Exception as _e:
            print("   [CANNOT LIST: %s: %s] %s" % (type(_e).__name__, _e, _sd))

    print(" find_spec() resolution (where each package is actually found):")
    for _m in ("torch", "ultralytics", "cv2", "flask"):
        try:
            _spec = _ilu.find_spec(_m)
            print("   %-11s -> %s" % (_m, (_spec.origin if _spec else "None (NOT FOUND)")))
        except Exception as _e:
            print("   %-11s -> RAISED %s: %s" % (_m, type(_e).__name__, _e))

    print(" torch on disk, as seen by THIS process:")
    for _sd in _site_dirs:
        _td = os.path.join(_sd, "torch")
        _ti = os.path.join(_td, "__init__.py")
        print("   in %s :" % _sd)
        print("       isdir(torch)=%s  isfile(__init__)=%s  readable=%s"
              % (os.path.isdir(_td), os.path.isfile(_ti), os.access(_ti, os.R_OK)))
        if os.path.isfile(_ti):
            try:
                with open(_ti, "rb") as _f:
                    _f.read(16)
                print("       open(torch/__init__.py) OK")
            except Exception as _e:
                print("       open(torch/__init__.py) -> %s: %s" % (type(_e).__name__, _e))

    print(" full traceback of 'import torch':")
    try:
        import torch  # noqa: F401
    except Exception:
        for _line in _tb.format_exc().splitlines():
            print("     " + _line)
    print("-" * 66)

    print(" Copy everything above and send it to Claude.")
    print("=" * 66)
    try:
        input("Press Enter to close...")
    except EOFError:
        pass
    raise SystemExit(1)

# --- 2) Open the browser from inside Python once the server is listening. ---
import app as _app  # noqa: E402  (import after sys.path is fixed)

_HOST = _app.config.HOST
_PORT = _app.config.PORT
_CONNECT_HOST = "127.0.0.1" if _HOST in ("0.0.0.0", "") else _HOST


def _open_when_ready():
    url = "http://%s:%s" % (_CONNECT_HOST, _PORT)
    for _ in range(120):  # up to ~60s
        try:
            with socket.create_connection((_CONNECT_HOST, _PORT), timeout=1):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.5)


if not os.environ.get("HOOP_NO_BROWSER"):
    threading.Thread(target=_open_when_ready, daemon=True).start()

print("HoopTracker -> http://%s:%s" % (_CONNECT_HOST, _PORT))
_app.app.run(host=_HOST, port=_PORT, threaded=True, debug=False)
