from torchinfo import summary
from model import GoogLeNet

def main(): 
    print("Inception-V1")
    summary(GoogLeNet(), input_size=[1, 3, 224, 224])
    print("\nInception-V2")
    summary(GoogLeNet(bnorm=True), input_size=[1, 3, 224, 224])

if __name__ == "__main__":
    main()