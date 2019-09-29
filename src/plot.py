import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib
import numpy as np
import os

plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)


def process_folder(folder, plot_columns=("acc", "cer")):
    folder_parts = folder.split("_")
    if len(folder_parts) >= 3:
        preprocessor = folder_parts[0]
        preprocessor = "No additional preprocessing" if preprocessor == "Standard" else f"Preprocessing used: {preprocessor}"
        net_name = folder_parts[1]
        total_epochs = int(folder_parts[2])
        augmentations_text = "No augmentations used"

    if len(folder_parts) == 4:
        augmentations_text = "Augmentations used"

    input_file = "2_period_data.txt"
    names = ["epoch", "train_acc", "train_cer", "train_wer", "test_acc", "test_cer", "test_wer", "name"]
    column_indices = {"epoch": 0, "acc": 1, "cer": 2, "wer": 3}
    column_names_long = {"epoch": "Epoch", "acc": "Accuracy", "cer": "Character Error Rate", "wer": "Word Error Rate"}
    groups = ["train", "test"]
    group_colors = {"train": "#0c619d", "test": "#e48686"}

    with open(os.path.join('..', 'final_plots', folder, input_file), 'r') as fp:
        result = defaultdict(lambda: defaultdict(list))
        for line in fp:
            line = line.strip()
            parts = line.split("\t")
            assert (len(names) == len(parts))
            for group in groups:
                indices = [0] + [i for i, elem in enumerate(names) if elem.startswith(group)]
                result[parts[-1]][group].append(np.asarray(parts)[indices])

    number_of_folds = len(result)

    def extract_column_data(fold_index, group, column, mapper=float):
        data = result[f"Net_00{fold_index}"][group]
        return np.asarray([mapper(entry[column_indices[column]]) for entry in data])

    def plot_avg_folds(col, log=False):
        plt.figure(figsize=(19, 11))
        title = list()
        title.append(f"Used network: {net_name}")
        title.append(f"Mean {column_names_long[col]} of all folds after {total_epochs} Epochs")
        title.append(f"{preprocessor}, {augmentations_text}")
        plt.title("\n".join(title), fontsize=20)
        for group in groups:
            plot_group(group, col, log=log, color=group_colors[group])

        plt.legend(loc="upper right")
        plt.savefig(os.path.join("..", "final_plots", folder, f"{folder}_{col}.svg"))
        # plt.show()

    def plot_group(group, col, log=False, color="#0c619d"):
        epochs = extract_column_data(0, group, "epoch", mapper=int)
        all_values = [extract_column_data(i, group, col) for i in range(number_of_folds)]
        value_mean = np.mean(all_values, axis=0)
        value_std = np.std(all_values, axis=0)
        plt.xlabel("Epochs", fontsize=18)
        plt.ylabel(f"{column_names_long[col]}", fontsize=18)

        if log:
            plt.semilogy(epochs, value_mean, label=group)
        else:
            plt.yticks(list(range(0, 110, 10)))
            plt.xticks(list(range(0, total_epochs + 10, 10)))
            plt.plot(epochs, value_mean, label=group)

        plt.fill_between(epochs,
                         value_mean - value_std,
                         value_mean + value_std,
                         alpha=0.2,
                         edgecolor=color,
                         facecolor=color)

    for plot_column in plot_columns:
        plot_avg_folds(plot_column, log=False)


if __name__ == "__main__":
    preprocessors = ("Deslant", "Standard")
    names = ("big", "small",)
    augmentations = ("", "_augmentations")
    kind_of_stat = {"small": ("acc", "cer"), "big": ("acc", "wer")}

    for preprocessor in preprocessors:
        for name in names:
            for augmentation in augmentations:
                folder = f"{preprocessor}_{name}_150{augmentation}"
                try:
                    process_folder(folder, plot_columns=kind_of_stat[name])
                except FileNotFoundError as e:
                    print(e)

