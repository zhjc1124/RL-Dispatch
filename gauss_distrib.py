import torch
# waiting_time = torch.load('dataset/waiting_time.pth')
# running_time = torch.load('dataset/running_time.pth')
delta = torch.load('./dataset/delta.pth')


if __name__ == '__main__':
    print(delta[10, 10, :])