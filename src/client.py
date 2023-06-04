from evaluate import load as load_metric
from transformers import Trainer, get_scheduler, AdamW
import torch
from tqdm import tqdm
import evaluate
from task.image_classification import load_data, create_model
from collections import OrderedDict
from accelerate import Accelerator
import flwr as fl
import math

MODEL_NAME = "google/vit-base-patch16-224-in21k"
WEIGHT_DECAY = 0.0
LEARNING_RATE = 5e-5
NUM_WARMUP_STEPS = 0
GRADIENT_ACCUMLATION_STEPS = 1
LR_SCHEDULER_TYPE = "linear"


def train(
    model,
    train_dataloader,
    accelerator,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps=GRADIENT_ACCUMLATION_STEPS,
    num_train_epochs=3,
    per_device_train_batch_size=8,
):
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / GRADIENT_ACCUMLATION_STEPS
    )
    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataloader)}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {per_device_train_batch_size}"
    )
    print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0
    progress_bar.update(completed_steps)
    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1


def test(model, eval_dataloader, accelerator, metric=evaluate.load("accuracy")):
    model.eval()
    loss = 0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss += outputs.loss
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather_for_metrics(
            (predictions, batch["labels"])
        )
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    return loss, eval_metric


def main():
    # define accelerator
    accelerator = Accelerator(gradient_accumulation_steps=1)
    # define train_dataloader
    train_dataloader, eval_dataloader, labels = load_data(
        model_name=MODEL_NAME,
        train_dir="/Users/vignesh/Downloads/archive",
        accelerator=accelerator,
    )
    model = create_model(model_name=MODEL_NAME, labels=labels)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
    lr_scheduler = get_scheduler(
        name=LR_SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=NUM_WARMUP_STEPS * GRADIENT_ACCUMLATION_STEPS,
        num_training_steps=NUM_WARMUP_STEPS * GRADIENT_ACCUMLATION_STEPS,
    )

    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    # Flower client
    class IMDBClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in model.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            print("Training Started...")
            train(
                model,
                train_dataloader,
                accelerator=accelerator,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
            print("Training Finished.")
            return self.get_parameters(config={}), len(train_dataloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(model, eval_dataloader, accelerator)
            print(f"Loss {float(loss)}")
            print(f"Accuracy {accuracy}")
            return float(loss), len(eval_dataloader), accuracy

    # Start client
    return IMDBClient()
    #fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=IMDBClient())


if __name__ == "__main__":
    main()
