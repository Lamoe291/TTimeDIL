import os
import numpy as np
import torch
import pandas as pd
import json


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]

    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def split_images_labels_multi_label(imgs, label_csv, list_of_domains=None):
    # split trainset.imgs in ImageFolder
    images = []
    domains = []
    labels = []
    df = pd.read_csv(label_csv, header=None,
                 names=["image_path", "labels", "split"])
    df["labels"] = df["labels"].apply(json.loads)
    for item in imgs:
        if list_of_domains != None:
            if item[0].split('/')[-2] in list_of_domains:
                images.append(item[0])
                domains.append(item[0].split('/')[-2])
                multi_label = df.loc[df.image_path == item[0], "labels"].iat[0]
                #print(multi_label)
                multi_label = np.array(multi_label)
                labels.append(multilabel_to_onehot(multi_label,19))
        else:
            images.append(item[0])
            domains.append(item[0].split('/')[-2])
            multi_label = df.loc[df.image_path == item[0], "labels"].iat[0]
                #print(multi_label)
            multi_label = np.array(multi_label)
        #print(multi_label)
        #print(len(multi_label))
            labels.append(multilabel_to_onehot(multi_label,19))

    return np.array(images), np.array(labels), np.array(domains)


def write_domain_img_file2txt(root_path, domain_name: str, extensions=['jpg', 'png', 'jpeg']):
    """
    Write all image paths and labels to a txt file,
    :param root_path: specific data path, e.g. /home/xxx/data/office-home
    :param domain_name: e.g. 'Art'
    """
    if os.path.exists(os.path.join(root_path, domain_name + '_all.txt')):
        return

    img_paths = []
    domain_path = os.path.join(root_path, domain_name)

    cl_dirs = os.listdir(domain_path)

    for cl_idx in range(len(cl_dirs)):

        cl_name = cl_dirs[cl_idx]
        cl_path = os.path.join(domain_path, cl_name)

        for img_file in os.listdir(cl_path):
            if img_file.split('.')[-1] in extensions:
                img_paths.append(os.path.join(domain_name, cl_name, img_file) + ' ' + str(cl_idx) + '\n')

    with open(os.path.join(root_path, domain_name + '_all.txt'), 'w') as f:
        for img_path in img_paths:
            f.write(img_path)

    # return img_paths


def split_domain_txt2txt(root_path, domain_name: str, train_ratio=0.7, seed=1993):
    """
    Split a txt file to train and test txt files.
    :param root_path: specific data path, e.g. /home/xxx/data/office-home
    :param domain_name: e.g. 'Art'
    :param train_ratio: ratio of train data
    """
    if os.path.exists(os.path.join(root_path, domain_name + '_train.txt')):
        return

    print("Split {} data to train and test txt files.".format(domain_name))
    np.random.seed(seed)
    print("Set numpy random seed to {}.".format(seed))

    with open(os.path.join(root_path, domain_name + '_all.txt'), 'r') as f:
        lines = f.readlines()
        np.random.shuffle(lines)
        train_lines = lines[:int(len(lines) * train_ratio)]
        test_lines = lines[int(len(lines) * train_ratio):]

    with open(os.path.join(root_path, domain_name + '_train.txt'), 'w') as f:
        for line in train_lines:
            f.write(line)

    with open(os.path.join(root_path, domain_name + '_test.txt'), 'w') as f:
        for line in test_lines:
            f.write(line)


def multilabel_to_onehot(class_ids, num_classes):
    """
    Args:
        class_ids: list of ints (e.g. [3, 7, 12])
        num_classes: total number of classes (e.g. 19)

    Returns:
        torch.tensor of shape (num_classes,), dtype=torch.float32
        Example: tensor([0,0,1,0,1,...])
    """
    #### class IDs start at 1
    onehot = np.zeros(num_classes+1)
    if len(class_ids) > 0:
        onehot[class_ids] = 1.0
    return onehot[1:]


def sample_per_domain(
    meta,
    split,
    n_per_domain,
    base_mask,
    random_state=42
):
    split_meta = meta[base_mask & (meta["split"] == split)]
    #print(split_meta)
    #split_meta.info()
    

    sampled = split_meta.groupby("country", group_keys=False).sample(
        n=n_per_domain,
        replace=True,  # if needed
        random_state=random_state
    )
    #sampled = (
        # split_meta
        # .groupby("country", group_keys=True)
        # .apply(
        #     lambda x: x.sample(
        #         n=min(len(x), n_per_domain),
        #         random_state=random_state
        #     )
        # )
    #)
    #print(sampled)
    #sampled.info()
    print("Columns:", sampled.columns)
    print("Index names:", sampled.index.names)
    data = sampled["patch_id"].to_numpy()
    targets = sampled["labels"].to_numpy()
    domains = sampled["country"].to_numpy()

    return data, targets, domains