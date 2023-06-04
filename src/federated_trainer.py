from evaluate import load as load_metric
from transformers import Trainer, get_scheduler, AdamW
import torch
from tqdm import tqdm
import evaluate
from task.image_classification import load_dataset, create_model
from collections import OrderedDict
from accelerate import Accelerator
import flwr as fl



def train(
    model,
    train_dataset,
    accelerator,
    weight_decay=0.0,
    learning_rate=5e-5,
    num_warmup_steps=0,
    gradient_accumulation_steps=1,
    max_train_steps=None,
    lr_scheduler_type="linear",
    num_train_epochs=3,
    per_device_train_batch_size=8,
):

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
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
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
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


def test(accelerator, model, eval_dataloader, metric=evaluate.load("accuracy")):
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather_for_metrics(
            (predictions, batch["labels"])
        )
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    return eval_metric


def main():
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    model = create_model()
    trainloader, testloader = load_dataset()

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
            train(model, trainloader, epochs=1)
            print("Training Finished.")
            return self.get_parameters(config={}), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(model, testloader)
            return float(loss), len(testloader), {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=IMDBClient())


if __name__ == "__main__":
    main()
