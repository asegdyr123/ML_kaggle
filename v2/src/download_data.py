"""Download authorized Kaggle data, or safely import a user-downloaded ZIP."""
import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

from common import DATA, EXPECTED, find_data, write_json


def import_zip(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        selected = {}
        for member in bundle.infolist():
            name = Path(member.filename.replace("\\", "/")).name
            if name in EXPECTED:
                if name in selected:
                    raise ValueError(f"Duplicate CSV in archive: {name}")
                selected[name] = member
        if set(selected) != set(EXPECTED):
            raise ValueError(f"Archive missing files: {set(EXPECTED) - set(selected)}")
        required = sum(item.file_size for item in selected.values())
        if shutil.disk_usage(destination).free < required + 1024**3:
            raise OSError("Insufficient disk space to extract CSVs with 1 GiB reserve")
        for name, member in selected.items():
            target = destination / name
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite existing data: {target}")
            partial = target.with_suffix(".csv.partial")
            print(f"Extracting {name} ({member.file_size:,} bytes)", flush=True)
            # Extract only the four fixed basenames, never archive-supplied paths.
            with bundle.open(member) as source, partial.open("wb") as out:
                shutil.copyfileobj(source, out, length=8 * 1024 * 1024)
            os.replace(partial, target)


def has_credentials():
    token = Path.home() / ".kaggle" / "access_token"
    legacy = Path(os.getenv("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))) / "kaggle.json"
    return bool(os.getenv("KAGGLE_API_TOKEN")) or (
        bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))
    ) or (token.is_file() and token.stat().st_size > 0) or legacy.is_file()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, help="Import a downloaded competition ZIP")
    parser.add_argument("--output", type=Path, default=DATA)
    parser.add_argument("--login", action="store_true", help="Prompt locally for an API token")
    args = parser.parse_args()
    if args.zip:
        import_zip(args.zip, args.output)
    else:
        import kagglehub
        if args.login:
            kagglehub.login()
        elif not has_credentials():
            print("Kaggle credentials are not configured. No download attempted.")
            print("Either use --zip <downloaded.zip>, or run with --login in your own terminal.")
            print("API settings: https://www.kaggle.com/settings/api")
            return 2
        kagglehub.competition_download("2026ml", output_dir=str(args.output))
    files = find_data(args.output)
    write_json(args.output / "download_manifest.json", {
        "competition": "2026ml",
        "files": {name: {"path": str(path), "bytes": path.stat().st_size} for name, path in files.items()},
    })
    print("All four competition CSV files are available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

