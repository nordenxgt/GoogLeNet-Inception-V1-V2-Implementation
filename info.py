from torchinfo import summary
from model import GoogLeNet

def main(): summary(GoogLeNet(), input_size=[1, 3, 224, 224])

if __name__ == "__main__":
    main()