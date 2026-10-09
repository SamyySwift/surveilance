"""
Enroll a staff member's face(s) into the recognition database.

Usage:
    python enroll.py --name "Jane Doe" --employee-id E1023 --badge-id 40123 \
        --zones entrance,vault photo1.jpg photo2.jpg

Use 2-5 clear, front-facing photos per person if you can: a few different
angles/lighting conditions improve match reliability far more than any
single "perfect" photo does. Each photo is checked first (one person only,
sharp, well lit, close enough, facing the camera, not someone else already
enrolled) and refused with the reason if it doesn't pass. Enrolling from the
door camera in the dashboard usually gives better matches than uploads.

--zones sets which camera zones the person may be in (see config.CAMERAS).
Without it, a new person gets no zone access, and an existing person keeps
whatever zones they already have. --badge-id links the person to the badge
the door reader sends for the badge + face check.
"""

import argparse
import sys
from pathlib import Path

import config
import database
from enrollment import decode_image, enroll_image
from face_engine import load_engine


def main():
    parser = argparse.ArgumentParser(description="Enroll a staff member.")
    parser.add_argument("--name", required=True, help="Full name of the staff member")
    parser.add_argument("--employee-id", default=None, help="Optional employee ID")
    parser.add_argument("--badge-id", default=None, help="Badge number the door reader sends")
    parser.add_argument(
        "--zones",
        default=None,
        help=f"Comma-separated zones this person may access ({','.join(config.CAMERAS)}). "
             "Omit to keep an existing person's zones unchanged.",
    )
    parser.add_argument("photos", nargs="+", help="One or more clear, front-facing photos")
    args = parser.parse_args()

    zones = None
    if args.zones is not None:
        zones, unknown = database.match_zones([z for z in args.zones.split(",") if z.strip()], config.CAMERAS)
        if unknown:
            parser.error(f"unknown zone(s): {', '.join(unknown)} "
                         f"(configured zones: {', '.join(config.CAMERAS)})")

    database.init_db()
    Path(config.ENROLLMENT_DIR).mkdir(parents=True, exist_ok=True)

    existing = database.get_staff_by_name(args.name)
    if args.badge_id:
        holder = database.get_staff_by_badge(args.badge_id)
        if holder is not None and (existing is None or holder["id"] != existing["id"]):
            parser.error(f"badge {args.badge_id} is already assigned to {holder['name']}")
    if existing:
        staff_id = existing["id"]
        if zones is not None:
            database.set_staff_zones(staff_id, zones)
        if args.badge_id:
            database.set_staff_badge(staff_id, args.badge_id)
    else:
        staff_id = database.add_staff(args.name, args.employee_id, zones, args.badge_id)

    engine = load_engine()
    enrolled_count = 0
    for photo_path in args.photos:
        try:
            data = Path(photo_path).read_bytes()
        except OSError as e:
            print(f"  [skip] {photo_path}: {e}", file=sys.stderr)
            continue
        result = enroll_image(engine, staff_id, decode_image(data), photo_path, original_bytes=data)
        if result.ok:
            enrolled_count += 1
            print(f"  [ok] Enrolled face from {photo_path}")
        else:
            print(f"  [skip] {result.message}", file=sys.stderr)

    print(f"\nDone. {enrolled_count}/{len(args.photos)} photo(s) enrolled for '{args.name}'.")
    if enrolled_count == 0 and not existing:
        database.delete_staff(staff_id)
        print("  Nothing was enrolled, so no record was created.")
        sys.exit(1)

    if not database.get_staff_zones(staff_id):
        print(f"  [warn] '{args.name}' has no zone access, so the dashboard will raise an "
              f"'Unauthorized access' alarm whenever they're seen on any camera. Re-run with "
              f"--zones (e.g. --zones {','.join(config.CAMERAS)}) or set zones in the dashboard.")


if __name__ == "__main__":
    main()
