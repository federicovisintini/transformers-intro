from torch import optim

from src.dataloader import train_dataloader, VOCABULARY_SIZE
from src.device import DEVICE
from src.model import init_transformer
from src.parameters import EMBEDDING_SIZE, NUM_TOKENS, POSITIONAL_ENCODING_SCALAR, NUM_HEADS, BATCH_SIZE, \
    TRAIN_NUM_EPOCHS

model = init_transformer(
    vocabulary_size=VOCABULARY_SIZE,
    embedding_size=EMBEDDING_SIZE,
    num_tokens=NUM_TOKENS,
    positional_encoding_scalar=POSITIONAL_ENCODING_SCALAR,
    num_heads=NUM_HEADS,
    batch_size=BATCH_SIZE,
    device=DEVICE
)

optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)


def lr_rate(step_num, d_model, factor, warmup_steps):
    step_num = max(1, step_num)
    return factor * (
        d_model ** (-0.5) * min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))
    )


lr_scheduler = optim.lr_scheduler.LambdaLR(
    optimizer=optimizer,
    lr_lambda=lambda step_num: lr_rate(
        step_num, d_model=512, factor=1, warmup_steps=4000
    ),
)

for epoch in range(TRAIN_NUM_EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, batch in enumerate(train_dataloader):

        # print(model(batch))

        inputs = batch['translation']['de']
        labels = batch['translation']['en']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    print('Finished Training')
