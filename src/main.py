from src.model import Transformer, DEVICE

if __name__ == '__main__':
    model = Transformer()

    model.to(DEVICE)

    print(model)
