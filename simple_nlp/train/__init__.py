
from accelerate import Accelerator
import warnings
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from bitsandbytes.optim import PagedAdam
import os

ACCUMULATION_STEPS = int(os.getenv("ACCUMULATION_STEPS", 128))
TOKEN = os.getenv("HF_TOKEN", None)
EPOCHS = int(os.getenv("EPOCHS", 1))
SAVE_STEPS = int(os.getenv("SAVE_STEPS", 100))
LR= float(os.getenv("LR", 1e-5))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def start_training(
    model,
    processor,
    dloader,
    OUT_MODEL,
    callback=None,
):
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        mixed_precision="fp16",
    )
    accelerator.init_trackers("huggingface")

    model.train()

    optim = PagedAdam(model.parameters(), lr=LR)

    scheduler = ExponentialLR(optim, gamma=0.9995)
    model, optim, dloader, scheduler = accelerator.prepare(
        model, optim, dloader, scheduler
    )

    if callback is not None:
        eval_ = callback()
        if eval_ is not None:
            accelerator.log({"eval_metric": eval_}, step=0)

    def do_save_stuff():
        if callback is not None:
            eval_ = callback()
            if eval_ is not None:
                accelerator.log({"eval_metric": eval_}, step=index - 1)
        model.save_pretrained(
            OUT_MODEL,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
            safe_serialization=False,
        )
        processor.save_pretrained(OUT_MODEL)

        try:
            model.push_to_hub(
                OUT_MODEL,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
                safe_serialization=False,
                token=TOKEN,
            )
            processor.push_to_hub(OUT_MODEL, token=TOKEN)
        except Exception as e:
            warnings.warn(f"Could not push to hub: {e}")

    index = 1
    for epoch in range(EPOCHS):
        for data in (pbar := tqdm(dloader)):
            if index / ACCUMULATION_STEPS % SAVE_STEPS == 0 and index != 0:
                do_save_stuff()

            optim.zero_grad()
            with accelerator.accumulate(model), accelerator.autocast():
                data = {k: v.to(model.device) for k, v in data.items()}
                output = model(return_dict=True, **data)
                loss = output.loss
                accelerator.backward(loss)

                if index % ACCUMULATION_STEPS == 0:
                    pbar.set_description(
                        f"Loss: {loss} LR: {get_lr(optim.optimizer)}",
                        refresh=True,
                    )
                    accelerator.log(
                        values={
                            "training_loss": loss,
                        },
                        step=int(index / ACCUMULATION_STEPS),
                    )
                if accelerator.sync_gradients:
                    accelerator.clip_grad_value_(model.parameters(), 0.7)
                optim.step()
                scheduler.step()

            index += 1
        do_save_stuff()
    do_save_stuff()
