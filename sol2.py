"""
[Digital Signal Processing] (2020-2)
Assignment #4 - Problem 5
JuYoung Oh / 20161872  / School of Electrical & Electronics Engineering

Solution 2)
    For noise, whose magnitude is similar to that of target voice and has continuity,
    detect it with characteristic of audio noise; 'Repeated & Continuous'.
    Use convolution to detect signal's continuity.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as f

EPSILON = 0.1

NOISE_DELTA = 1.5
TARGET_DELTA = 1.5


def main(total_length, target_num, block_num=10, verbose=False, visual=True):
    # 노이즈 생성
    t = np.linspace(0, 10 * np.pi, total_length)
    # raw_noise = 0.5 * np.sin(t) + NOISE_DELTA
    raw_noise = 0.25 * np.sin(t) + 0.25 * np.sin(2 * t) + NOISE_DELTA
    # raw_noise = 0.25 * np.sin(t) + 0.15 * np.sin(2 * t) + 0.1 * np.sin(3 * t) + NOISE_DELTA
    noisy_voice = raw_noise

    np.random.seed(31)  # 24

    # 타겟 보이스 삽입 위치 설정
    target_loc = np.random.choice(range(total_length), target_num)
    for loc in target_loc:
        raw_noise[loc] = 0

    # 타겟 보이스 생성
    target = np.random.random(target_num) + TARGET_DELTA

    if verbose:
        target_info = sorted(zip(target_loc, target))
        print('Generated Targets')
        for target_info_item in target_info:
            print(target_info_item)

    # 노이즈에 타겟 보이스 추가
    for i, loc in enumerate(target_loc):
        noisy_voice[loc] = target[i]

    # 생성한 신호 plot
    fig, ax = plt.subplots(5, 1, figsize=(10, 7))
    ax[0].bar(range(total_length), raw_noise)
    ax[0].bar(target_loc, target, color='orange')
    ax[0].set_xlim(-1, total_length)
    ax[0].set_xlabel('Blue: Noise / Orange: Target Voice')
    ax[0].set_title('Input Signal')

    # 신호의 최소~최대값 구간을 block_num 개의 구간으로 나눔
    noisy_min = np.min(noisy_voice)
    noisy_max = np.max(noisy_voice)
    block_boundary = np.linspace(noisy_min, noisy_max, block_num + 1)
    feature_map = torch.tensor([[0.0 for _ in range(total_length)] for _ in range(block_num)])

    if verbose:
        print('\nFeature map shape: ', feature_map.shape)

    # 가로축 시간, 세로축 block 인 feature map 생성
    for index, data in enumerate(noisy_voice):
        for block in range(block_num):
            block_low = block_boundary[block]
            block_high = block_boundary[block + 1]
            if block_low <= data <= block_high:
                feature_map[block][index] = data

    # feature map 의 좌, 우는 중간값으로, 상, 하는 0으로 1칸씩 패딩
    median = np.median(noisy_voice).item()
    pad = [1, 1, 0, 0]  # left, right, up, down
    pad_feature_map = f.pad(feature_map, pad, mode='constant', value=median)
    pad = [0, 0, 1, 1]
    pad_feature_map = f.pad(pad_feature_map, pad, mode='constant', value=0)
    pad_feature_map = torch.unsqueeze(torch.unsqueeze(pad_feature_map, 0), 0)

    if verbose:
        print('\nPadded feature map shape: ', pad_feature_map.shape)

    # 자신을 제외한 주변 6칸(윗칸, 아랫칸 제외)의 평균을 구하기 위한 커널
    kernel = torch.tensor([[[
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]]], dtype=torch.float)
    kernel /= 6

    if verbose:
        print('\nKernel shape: ', kernel.shape)

    # feature map 에 대해 자신을 제외한 주변 8칸의 평균을 구함
    conv_result = torch.squeeze(f.conv2d(pad_feature_map, kernel))

    pad_feature_map = torch.squeeze(pad_feature_map)

    if verbose:
        print('\nDenoised feature map shape: ', conv_result.shape)

    # feature map plot
    ax[1].matshow(pad_feature_map, origin='lower', aspect='auto')
    ax[1].set_xlim(-1, total_length + 1)
    ax[1].set_ylim(top=block_num + 1)
    ax[1].set_title('Feature map')
    ax[1].xaxis.set_ticks_position('bottom')

    # conv result plot
    ax[2].matshow(conv_result, origin='lower', aspect='auto')
    ax[2].set_xlim(-1, total_length)
    ax[2].set_ylim(top=block_num)
    ax[2].set_title('Conv result')
    ax[2].xaxis.set_ticks_position('bottom')

    # conv_result 의 값이 기준 이하인 경우, 타겟 보이스로 인식
    # 타겟 보이스로 인식한 지점들만 feature map 에서 남기고 나머지는 삭제
    denoised_feature_map = np.array([[0.0 for _ in range(total_length)] for _ in range(block_num)])
    for row in range(block_num):
        for col in range(total_length):
            if conv_result[row][col] < EPSILON:
                denoised_feature_map[row][col] = feature_map[row][col]

    # denoised feature map plot
    ax[3].matshow(denoised_feature_map, origin='lower', aspect='auto')
    ax[3].set_xlim(-1, total_length)
    ax[3].set_ylim(top=block_num)
    ax[3].set_title('Denoised feature map')
    ax[3].xaxis.set_ticks_position('bottom')

    # denoised feature map 으로부터 신호 복원
    denoised_voice = [0 for _ in range(total_length)]
    for col in range(total_length):
        for row in range(block_num):
            if denoised_feature_map[row][col] > 0:
                denoised_voice[col] = denoised_feature_map[row][col]
                continue

    # denoised signal plot
    ax[4].bar(range(total_length), denoised_voice, color='orange')
    ax[4].set_xlim(-1, total_length)
    ax[4].set_title('Denoised Signal')

    if verbose:
        net_denoised_voice = []
        for index, data in enumerate(denoised_voice):
            if data > 0:
                net_denoised_voice.append((index, data))
        print('\nDenoised Voice')
        for net_denoised_voice_item in net_denoised_voice:
            print(net_denoised_voice_item)

    if visual:
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main(total_length=100, target_num=5)
