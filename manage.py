"""
Small management CLI for the staff database.

Usage:
    uv run manage.py list
    uv run manage.py deactivate "Jane Doe"
    uv run manage.py badge "Jane Doe" 40123      # link a badge (use "" to remove)
"""

import argparse

import config
import database


def main():
    parser = argparse.ArgumentParser(description="Manage enrolled staff.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all enrolled staff")

    deact = sub.add_parser("deactivate", help="Revoke a staff member's access")
    deact.add_argument("name", help="Full name as used at enrollment")

    badge = sub.add_parser("badge", help="Set or clear a staff member's badge ID")
    badge.add_argument("name", help="Full name as used at enrollment")
    badge.add_argument("badge_id", help='Badge number, or "" to remove it')

    args = parser.parse_args()
    database.init_db()

    if args.command == "list":
        rows = database.list_staff(config.RECOGNIZER)
        if not rows:
            print("No staff enrolled yet.")
            return
        print(f"{'ID':<4} {'Name':<25} {'Employee ID':<15} {'Badge':<10} {'Active':<7} {'Photos':<7} {'Zones'}")
        for s in rows:
            print(f"{s['id']:<4} {s['name']:<25} {s['employee_id'] or '-':<15} {s['badge_id'] or '-':<10} "
                  f"{'yes' if s['active'] else 'no':<7} {s['num_embeddings']:<7} {', '.join(s['zones']) or '-'}")
        print(f"\nPhotos counts embeddings for the current recognizer ({config.RECOGNIZER}).")

    elif args.command == "deactivate":
        row = database.get_staff_by_name(args.name)
        if row is None:
            print(f"No staff member named '{args.name}' found.")
            return
        database.set_staff_active(row["id"], False)
        print(f"Deactivated '{args.name}'. They will no longer match as authorized.")

    elif args.command == "badge":
        row = database.get_staff_by_name(args.name)
        if row is None:
            print(f"No staff member named '{args.name}' found.")
            return
        try:
            database.set_staff_badge(row["id"], args.badge_id.strip() or None)
        except ValueError as e:
            parser.error(str(e))
        print(f"Badge for '{args.name}' set to {args.badge_id.strip() or '(none)'}.")


if __name__ == "__main__":
    main()
