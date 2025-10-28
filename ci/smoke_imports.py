"""Simple smoke test to ensure required runtime imports are available in CI.
Exits with non-zero status on first import failure.
"""
import sys

modules = [
    ("packaging", "packaging.version"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("praw", "praw"),
    ("requests", "requests"),
    ("yfinance", "yfinance"),
    ("googleapiclient", "googleapiclient.discovery"),
    ("google.auth", "google.auth"),
    ("google_auth_oauthlib", "google_auth_oauthlib.flow"),
]

failed = False
for name, mod in modules:
    try:
        __import__(mod)
        print(f"OK: imported {mod}")
    except Exception as e:
        print(f"FAIL: cannot import {mod}: {e}")
        failed = True

if failed:
    print("One or more imports failed. Exiting with error.")
    sys.exit(1)

print("All smoke imports succeeded.")
