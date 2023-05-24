from torch import nn, optim

from src.dataloader import train_dataloader, VOCABULARY_SIZE
from src.device import DEVICE
from src.model import init_transformer
from src.parameters import EMBEDDING_SIZE, NUM_TOKENS, POSITIONAL_ENCODING_SCALAR, NUM_HEADS, BATCH_SIZE

model = init_transformer(
    vocabulary_size=VOCABULARY_SIZE,
    embedding_size=EMBEDDING_SIZE,
    num_tokens=NUM_TOKENS,
    positional_encoding_scalar=POSITIONAL_ENCODING_SCALAR,
    num_heads=NUM_HEADS,
    batch_size=BATCH_SIZE,
    device=DEVICE
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, batch in enumerate(train_dataloader):

        print(model(batch))

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
