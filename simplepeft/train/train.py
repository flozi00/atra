from accelerate import Accelerator
import warnings
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from bitsandbytes.optim import PagedLion32bit

warnings.simplefilter("ignore")

ACCUMULATION_STEPS = 16


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def start_training(
    model,
    processor,
    dloader,
    PEFT_MODEL,
    LR: float,
    callback=None,
    num_epochs=1,
):
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        mixed_precision="fp16",
    )
    accelerator.init_trackers("huggingface")

    model.train()

    optim = PagedLion32bit(model.parameters(), lr=LR)

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
            PEFT_MODEL,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
            safe_serialization=False,
        )
        model.push_to_hub(
            PEFT_MODEL,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
            safe_serialization=False,
        )
        processor.save_pretrained(PEFT_MODEL)
        processor.push_to_hub(PEFT_MODEL)

    index = 1
    for epoch in range(num_epochs):
        for data in (pbar := tqdm(dloader)):
            if index / ACCUMULATION_STEPS % 1000 == 0:
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
