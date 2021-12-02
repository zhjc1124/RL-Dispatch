import torch
waiting_time = torch.load('dataset/waiting_time.pth')
running_time = torch.load('dataset/running_time.pth')


def estimate(day, hour, origin, destination):
    ans = torch.zeros(4)
    ans[:2] = waiting_time[day, hour, origin, destination]
    ans[2:] = running_time[hour, origin, destination]
    return ans


if __name__ == '__main__':
    print(estimate(2, 10, 29, 80))