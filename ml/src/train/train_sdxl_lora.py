import argparse
import shlex
import subprocess
from copy import deepcopy
from pathlib import Path


DEFAULT_TRAIN_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "train" / "sdxl_lora.yaml"
)


def load_train_config(config_path=None):
    config_path = Path(config_path) if config_path else DEFAULT_TRAIN_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Train config not found: {config_path}")

    import yaml

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _normalize_path(project_root, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def build_accelerate_command(config, project_root, accelerate_config_path=None):
    config = deepcopy(config)
    script_path = _normalize_path(project_root, config.pop("script_path"))
    if not script_path.exists():
        raise FileNotFoundError(
            "Diffusers SDXL LoRA training script not found: "
            f"{script_path}. Add the diffusers example script first."
        )

    pretrained_model = config.pop("pretrained_model_name_or_path")
    train_data_dir = _normalize_path(project_root, config.pop("train_data_dir"))
    output_dir = _normalize_path(project_root, config.pop("output_dir"))

    command = ["accelerate", "launch"]
    if accelerate_config_path:
        command.extend(["--config_file", str(_normalize_path(project_root, accelerate_config_path))])

    command.extend(
        [
            str(script_path),
            "--pretrained_model_name_or_path",
            pretrained_model,
            "--train_data_dir",
            str(train_data_dir),
            "--output_dir",
            str(output_dir),
        ]
    )

    for key, value in config.items():
        cli_key = f"--{key}"

        if isinstance(value, bool):
            if value:
                command.append(cli_key)
            continue

        if value is None:
            continue

        if isinstance(value, list):
            for item in value:
                command.extend([cli_key, str(item)])
            continue

        command.extend([cli_key, str(value)])

    return command


def train_sdxl_lora(
    train_config_path=None,
    accelerate_config_path=None,
    overrides=None,
    dry_run=False,
):
    project_root = Path(__file__).resolve().parents[3]
    train_config = deepcopy(load_train_config(train_config_path)["sdxl_lora"])

    if overrides:
        train_config.update(overrides)

    command = build_accelerate_command(
        config=train_config,
        project_root=project_root,
        accelerate_config_path=accelerate_config_path,
    )

    print("=" * 55)
    print("SDXL LoRA 학습 명령 준비 완료")
    print("=" * 55)
    print("  command:")
    print(f"  {' '.join(shlex.quote(part) for part in command)}")

    if dry_run:
        return {"command": command, "returncode": 0}

    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"SDXL LoRA training failed with exit code {result.returncode}")

    return {"command": command, "returncode": result.returncode}


def main():
    parser = argparse.ArgumentParser(description="Launch SDXL LoRA training via diffusers example script.")
    parser.add_argument("--train-config-path")
    parser.add_argument("--accelerate-config-path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_sdxl_lora(
        train_config_path=args.train_config_path,
        accelerate_config_path=args.accelerate_config_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
