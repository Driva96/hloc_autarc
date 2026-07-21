from pathlib import Path
import yaml

# Configuration
SEARCH_PATH = Path("/path/to/search")
CONFIG_DIR = Path("/path/to/config")

CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def main(search_path=SEARCH_PATH, config_dir=CONFIG_DIR):
    # Only iterate over top-level directories
    for directory in search_path.iterdir():
        full_dir = directory / "full"
        if (
            directory.is_dir()
            and full_dir.is_dir()
            and any(full_dir.iterdir())
            and "output" not in directory.name
        ):
            config = {
                "experiment": f"{directory.name}",
                "root": '/data/',
                "raw_images": "${dataset.root}/${dataset.experiment}/full",
            }

            output_file = config_dir / f"{directory.name}.yaml"

            if output_file.exists():
                print(f"Configuration for {directory.name} already exists. Skipping...")
                continue

            with output_file.open("w") as f:
                yaml.safe_dump(config, f, sort_keys=False)

            print(f"Created {output_file}")


if __name__ == "__main__":
    main(search_path=Path("/data"), config_dir=Path(__file__).parent)
    print("Configuration files generated successfully.")
