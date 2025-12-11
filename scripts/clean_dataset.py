"""
Script to clean the dataset by removing leagues with too few images.
This improves model performance by ensuring sufficient training samples per class.
"""

import os
import shutil
from pathlib import Path


def count_images_per_league(data_dir):
    """
    Count number of images in each league folder.

    Args:
        data_dir: Path to data directory

    Returns:
        Dictionary with league names as keys and image counts as values
    """
    league_counts = {}

    for league_folder in Path(data_dir).iterdir():
        if league_folder.is_dir():
            # Count image files (jpg, png, jpeg)
            image_count = len(list(league_folder.glob('*.jpg'))) + \
                         len(list(league_folder.glob('*.png'))) + \
                         len(list(league_folder.glob('*.jpeg')))
            league_counts[league_folder.name] = image_count

    return league_counts


def clean_dataset(data_dir, min_images=15, backup=True):
    """
    Remove leagues with fewer than min_images from dataset.

    Args:
        data_dir: Path to data directory
        min_images: Minimum number of images required per league
        backup: If True, move removed leagues to backup folder instead of deleting
    """
    league_counts = count_images_per_league(data_dir)

    # Sort by count to show which will be removed
    sorted_leagues = sorted(league_counts.items(), key=lambda x: x[1])

    print("=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    print(f"Total leagues: {len(league_counts)}")
    print(f"Total images: {sum(league_counts.values())}")
    print(f"Minimum threshold: {min_images} images per league")
    print()

    # Identify leagues to remove
    to_remove = [league for league, count in league_counts.items() if count < min_images]
    to_keep = [league for league, count in league_counts.items() if count >= min_images]

    print(f"Leagues to REMOVE ({len(to_remove)}):")
    for league in sorted_leagues:
        if league[0] in to_remove:
            print(f"  ✗ {league[0]}: {league[1]} images")

    print()
    print(f"Leagues to KEEP ({len(to_keep)}):")
    for league in sorted_leagues:
        if league[0] in to_keep:
            print(f"  ✓ {league[0]}: {league[1]} images")

    print()
    images_removed = sum(count for league, count in league_counts.items() if league in to_remove)
    images_kept = sum(count for league, count in league_counts.items() if league in to_keep)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Leagues kept: {len(to_keep)} / {len(league_counts)}")
    print(f"Images kept: {images_kept} / {sum(league_counts.values())}")
    print(f"Images removed: {images_removed}")
    print("=" * 70)

    # Ask for confirmation
    response = input("\nProceed with cleanup? (yes/no): ").strip().lower()

    if response != 'yes':
        print("Cleanup cancelled.")
        return

    # Create backup folder if needed
    if backup:
        backup_dir = Path(data_dir).parent / 'data_backup_removed'
        backup_dir.mkdir(exist_ok=True)
        print(f"\nBackup folder: {backup_dir}")

    # Remove or backup leagues
    for league in to_remove:
        league_path = Path(data_dir) / league

        if backup:
            # Move to backup
            backup_path = backup_dir / league
            shutil.move(str(league_path), str(backup_path))
            print(f"  Moved: {league} → backup")
        else:
            # Delete permanently
            shutil.rmtree(league_path)
            print(f"  Deleted: {league}")

    print("\n✓ Dataset cleanup completed!")
    print(f"✓ Final dataset: {len(to_keep)} leagues, {images_kept} images")


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "/home/jaime/AF/data"
    MIN_IMAGES = 15  # Minimum images per league
    BACKUP = True    # Move to backup instead of delete

    clean_dataset(DATA_DIR, min_images=MIN_IMAGES, backup=BACKUP)
