import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class VKLSVDSessionDataset(Dataset):
    def __init__(self, interactions_df, max_len=20):
        self.max_len = max_len

        # Сортируем по времени и группируем историю просмотров для каждого юзера
        # В VK-LSVD используются колонки user_id, video_id, timestamp
        interactions_df = interactions_df.sort_values('timestamp')
        grouped = interactions_df.groupby('user_id')['video_id'].apply(list)

        self.sequences = []
        self.targets = []

        # Создаем скользящее окно (история -> следующее видео)
        for user_id, history in grouped.items():
            if len(history) < 2:
                continue
            for i in range(1, len(history)):
                # Вырезаем префикс истории фиксированной длины
                seq = history[max(0, i - max_len):i]
                target = history[i]

                # Дополняем нулями (Padding), если история короче max_len
                pad_len = max_len - len(seq)
                padded_seq = [0] * pad_len + seq

                self.sequences.append(padded_seq)
                self.targets.append(target)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'sequence': torch.tensor(self.sequences[idx], dtype=torch.long),
            'target': torch.tensor(self.targets[idx], dtype=torch.long)
        }


# Игрушечный пример инициализации (в реальности загружайте parquet-файлы из VK-LSVD)
mock_data = pd.DataFrame({
    'user_id': [1, 1, 1, 2, 2],
    'video_id': [10, 11, 12, 10, 14],  # 0 зарезервирован под padding
    'timestamp': [1000, 1001, 1002, 1000, 1004]
})

dataset = VKLSVDSessionDataset(mock_data, max_len=5)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
