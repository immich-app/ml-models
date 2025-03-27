import subprocess
from pathlib import Path
from typing import NamedTuple

import yaml

from exporters.constants import ModelSource

from immich_model_exporter.exporters.constants import SOURCE_TO_TASK

class ModelSpec(NamedTuple):
  name: str
  hfName: str
  source: ModelSource

def export_models(models: list[ModelSpec]) -> None:
    profiling_dir = Path("profiling")
    profiling_dir.mkdir(exist_ok=True)
    for model in models:
        try:
            model_dir = f"models/{model.name}"
            task = SOURCE_TO_TASK[model.source]

            print(f"Processing model {model.name}")
            subprocess.check_call(["python", "-m", "immich_model_exporter", "export", model.hfName, model.source])
            subprocess.check_call(
                [
                    "python",
                    "-m",
                    "immich_model_exporter",
                    "profile",
                    model_dir,
                    task,
                    "--output_path",
                    profiling_dir / f"{model}.json",
                ]
            )
            subprocess.check_call(["python", "-m", "immich_model_exporter", "upload", model_dir])
        except Exception as e:
            print(f"Failed to export model {model}: {e}")

def read_models() -> list[ModelSpec]:
  with open('../models.yaml') as f:
    y = yaml.safe_load(f)
  models = []
  for m in y['models']:
    name = m['name']
    source = ModelSource(m['source'])
    hfName = m['hf-name'] if 'hf-name' in m else name
    model = ModelSpec(name, hfName, source)
    models.append(model)
  return models


if __name__ == "__main__":
    models = read_models()

    export_models(models)

    openclip_names = [m.hfName for m in models if m.source == ModelSource.OPENCLIP]
    openclip_names_cleaned = [name.replace("__", ",") for name in openclip_names]

    Path("results").mkdir(exist_ok=True)
    dataset_root = Path("datasets")
    dataset_root.mkdir(exist_ok=True)

    crossmodal3600_root = dataset_root / "crossmodal3600"
    subprocess.check_call(
        [
            "clip_benchmark",
            "eval",
            "--pretrained_model",
            *openclip_names_cleaned,
            "--task",
            "zeroshot_retrieval",
            "--dataset",
            "crossmodal3600",
            "--dataset_root",
            crossmodal3600_root.as_posix(),
            "--batch_size",
            "64",
            "--language",
            "ar",
            "bn",
            "cs",
            "da",
            "de",
            "el",
            "en",
            "es",
            "fa",
            "fi",
            "fil",
            "fr",
            "he",
            "hi",
            "hr",
            "hu",
            "id",
            "it",
            "ja",
            "ko",
            "mi",
            "nl",
            "no",
            "pl",
            "pt",
            "quz",
            "ro",
            "ru",
            "sv",
            "sw",
            "te",
            "th",
            "tr",
            "uk",
            "vi",
            "zh",
            "--recall_k",
            "1",
            "5",
            "10",
            "--no_amp",
            "--output",
            "results/{dataset}_{language}_{model}_{pretrained}.json",
        ]
    )

    xtd10_root = dataset_root / "xtd10"
    subprocess.check_call(
        [
            "clip_benchmark",
            "eval",
            "--pretrained_model",
            *openclip_names_cleaned,
            "--task",
            "zeroshot_retrieval",
            "--dataset",
            "xtd10",
            "--dataset_root",
            xtd10_root.as_posix(),
            "--batch_size",
            "64",
            "--language",
            "de",
            "en",
            "es",
            "fr",
            "it",
            "jp",
            "ko",
            "pl",
            "ru",
            "tr",
            "zh",
            "--recall_k",
            "1",
            "5",
            "10",
            "--no_amp",
            "--output",
            "results/{dataset}_{language}_{model}_{pretrained}.json",
        ]
    )

    flickr30k_root = dataset_root / "flickr30k"
    # note: need ~/.kaggle/kaggle.json to download the dataset automatically
    subprocess.check_call(
        [
            "clip_benchmark",
            "eval",
            "--pretrained_model",
            *openclip_names_cleaned,
            "--task",
            "zeroshot_retrieval",
            "--dataset",
            "flickr30k",
            "--dataset_root",
            flickr30k_root.as_posix(),
            "--batch_size",
            "64",
            "--language",
            "en",
            "zh",
            "--recall_k",
            "1",
            "5",
            "10",
            "--no_amp",
            "--output",
            "results/{dataset}_{language}_{model}_{pretrained}.json",
        ]
    )
