from torch import optim, nn

from src.dataloader.dataloader import train_dataloader, validation_dataloader
from src.dataloader.tokenizer import VOCABULARY_SIZE
from src.model import Transformer
from src.parameters import EMBEDDING_SIZE, NUM_TOKENS, POSITIONAL_ENCODING_SCALAR, NUM_HEADS, \
    TRAIN_NUM_EPOCHS, NUM_ENCODERS
from src.trainer.trainer import Trainer
from src.utils.device import DEVICE


def lr_rate(step_num, d_model, factor, warmup_steps):
    step_num = max(1, step_num)
    return factor * d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))


if __name__ == '__main__':
    model = Transformer(
        vocabulary_size=VOCABULARY_SIZE,
        embedding_size=EMBEDDING_SIZE,
        num_tokens=NUM_TOKENS,
        positional_encoding_scalar=POSITIONAL_ENCODING_SCALAR,
        num_heads=NUM_HEADS,
        num_encoders=NUM_ENCODERS,
        device=DEVICE
    )

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)

    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step_num: lr_rate(
            step_num, d_model=512, factor=1e5, warmup_steps=4000
        ),
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )

    trainer.fit(num_epochs=TRAIN_NUM_EPOCHS)
