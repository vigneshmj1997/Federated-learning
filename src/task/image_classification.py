import os
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset
from transformers import logging

logging.set_verbosity_warning()


def dataset_splitter(dataset, num_clients, collate_fn, batch_size, shuflle=True):
    split_boundary = []
    for i in range(0, len(dataset), len(dataset) // num_clients):
        split_boundary.append(i)
    split_boundary.append(len(dataset) - 1)
    boudary_select = [
        [temp for temp in range(split_boundary[i], split_boundary[i + 1])]
        for i in range(0, len(split_boundary) - 1)
    ]
    dataloader = []
    for i in boudary_select:
        dataloader.append(
            DataLoader(
                dataset.select(i),
                shuffle=shuflle,
                collate_fn=collate_fn,
                batch_size=batch_size,
            )
        )
    print("Len of datalaoder", dataloader)
    return dataloader


def load_data(
    model_name,
    train_dir,
    accelerator,
    train_val_split=0.1,
    validation_dir=None,
    batch_size=8,
    num_client_id=None,
):
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = Normalize(
        mean=image_processor.image_mean, std=image_processor.image_std
    )
    train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_train(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    def preprocess_val(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            val_transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    # reading data
    data_files = {}
    if train_dir is not None:
        data_files["train"] = os.path.join(train_dir, "**")
    if validation_dir is not None:
        data_files["validation"] = os.path.join(validation_dir, "**")
    dataset = load_dataset(
        "imagefolder",
        data_files=data_files,
        task="image-classification",
    )
    train_val_split = None if "validation" in dataset.keys() else train_val_split
    if isinstance(train_val_split, float) and train_val_split > 0.0:
        split = dataset["train"].train_test_split(test_size=train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(preprocess_train)
        eval_dataset = dataset["validation"].with_transform(preprocess_val)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = []
    eval_dataloader = []
    print(int(os.environ["CLIENTS"]), "Number of clinets")
    if num_client_id != None:
        train_dataloader = dataset_splitter(
            train_dataset,
            num_clients=int(os.environ["CLIENTS"]),
            collate_fn=collate_fn,
            batch_size=batch_size,
        )
        eval_dataloader = dataset_splitter(
            eval_dataset,
            num_clients=int(os.environ["CLIENTS"]),
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuflle=False,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size
        )
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=collate_fn, batch_size=batch_size
        )
    labels = dataset["train"].features["labels"].names

    print(len(train_dataloader), len(eval_dataloader), "Eval Dataloader")
    return (
        train_dataloader[int(num_client_id)],
        eval_dataloader[int(num_client_id)],
        labels,
    )


def create_model(model_name, labels):
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_name, config=config, ignore_mismatched_sizes=True
    )
    return model
