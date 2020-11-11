"""
[Digital Signal Processing] (2020-2)
Assignment #4 - Problem 5
JuYoung Oh / 20161872  / School of Electrical & Electronics Engineering

Solution 1)
    For noise, whose magnitude is usually smaller than that of target voice,
    set threshold by equation shown below.
        [{max(total signal) + min{average(total signal), median(total signal)}] / 2
"""
import numpy as np
import matplotlib.pyplot as plt


def main(total_length, target_num, target_delta=1, verbose=False):
    np.random.seed(15)

    noise_num = total_length - target_num

    # 노이즈 및 타겟 보이스 임의 생성
    noise = np.random.random(noise_num)

    # target_delta: 노이즈 대비 타겟 보이스가 큰 정도
    target = np.random.random(target_num) + target_delta
    if verbose:
        print(target)

    total = np.concatenate((noise, target))

    if verbose:
        print(total.shape)

    # 노이즈 및 타겟 보이스 plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].bar(range(noise_num), noise, label='Noise')
    ax[0].bar(range(noise_num, total_length), target, label='Target Voice')

    # 전체 신호의 최대, 평균, 중간값 표시
    ax[0].axhline(np.max(total), color='red', linestyle='dashed', label='Max')
    ax[0].axhline(np.average(total), color='red', label='Average')
    ax[0].axhline(np.median(total), color='red', linestyle='dashdot', label='Median')

    # 평균과 중간값 중 작은 값을 기준으로 설정
    mean_or_med = np.min((np.average(total), np.median(total)))

    # 최대값과 (평균 or 중간값)의 평균으로 threshold 설정
    threshold = (np.max(total) + mean_or_med) / 2
    ax[0].axhline(threshold, color='green', label='Proposing Threshold')

    # 전체 신호에서 타켓 보이스의 비율
    target_ratio = round(target_num / total_length * 100, 1)

    ax[0].set_xlabel("Original Signal\nTarget Voice Ratio: {}%".format(target_ratio))
    ax[0].legend(loc='upper left', framealpha=1)
    ax[0].set_xlim(-1, total_length)

    # Threshold 기반 denoise 된 신호
    denoised_voice = np.array([0 if data < threshold else data for data in total])

    # Denoise 된 신호 중 노이즈, 타겟 보이스의 수
    denoised_voice_num = np.count_nonzero(denoised_voice[-target_num:])
    denoised_noise_num = np.count_nonzero(denoised_voice[:noise_num])
    if verbose:
        print(denoised_voice_num)
        print(denoised_noise_num)

    # 노이즈, 타겟 보이스 분류 정확도
    voice_acc = denoised_voice_num / target_num
    noise_acc = (noise_num - denoised_noise_num) / noise_num

    ax[1].bar(range(total_length), denoised_voice, color='orange')
    ax[1].set_xlabel('Denoised signal\nVoice Accuracy: {}%, Noise Accuracy: {}%'.format(
        round(voice_acc * 100, 1),
        round(noise_acc * 100, 1)
    ))
    ax[1].set_xlim(-1, total_length)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main(total_length=100, target_num=5)
