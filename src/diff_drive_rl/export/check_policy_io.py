from __future__ import annotations

import argparse

from diff_drive_rl.export.policy_io import load_policy_io, validate_policy_io


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate diff-drive policy_io.json")
    parser.add_argument("--policy-io", required=True, help="Path to policy_io.json")
    args = parser.parse_args()
    payload = load_policy_io(args.policy_io)
    validate_policy_io(payload)
    print("policy_io OK")


if __name__ == "__main__":
    main()
