import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iFlair, iOfficeHome, iDomainNet, iBigEarthNet, iCore50
from configilm.extra.DataSets import BENv2_DataSet


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        self.args = args
        self.dataset_name = dataset_name
        self.seed = seed
        self.dil = args.get("domain_incremental_learning", True)
        #print(self.args['multi-label'])
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)
            
    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    @property
    def nb_classes(self):
        if not self.dil:
            return len(self._class_order)
        else:
            if self.dataset_name == 'flair':
                return 19
            elif self.dataset_name == 'office_home':
                return 65
            elif self.dataset_name == 'core50':
                return 50
            elif self.dataset_name == 'domainnet':
                return 345
            elif self.dataset_name == 'bigearthnet':
                return 19
            

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)
        #sample_domain.append([domain for i in range(len(class_data))])

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)
        
    def get_dataset_ml(
        self, domains, source, mode, appendent=None, ret_data=False, m_rate=None, return_domains=False
    ):
        if source == "train":
            x, y, z = self._train_data, self._train_targets, self._train_domains
        elif source == "validation":
            x, y, z = self._val_data, self._val_targets, self._val_domains
        elif source == "test":
            x, y, z = self._test_data, self._test_targets, self._test_domains
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets, sample_domain = [], [], []
        #val_data, val_targets = [],[]
        print(domains)
        for domain in domains:
            print(domain)
            if m_rate is None:
                class_data, class_targets = self._select_domain(
                    x, y, z, domain=domain
                )
            #else:
            #    class_data, class_targets = self._select_rmm(
            #        x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
            #    )

            # if self.args.get('validation', False):
            #     rng = np.random.default_rng(seed=self.seed)
            #     perm = rng.permutation(len(class_data))
            #     class_data = class_data[perm]
            #     class_targets = class_targets[perm]
            #     num_train = int(len(class_data)*0.8)
            #     num_val = len(class_data) - num_train

            #     val_data.append(class_data[num_train:])
            #     val_targets.append(class_targets[num_train:])

            #     class_data = class_data[:num_train]
            #     class_targets = class_targets[:num_train]

                

            data.append(class_data)
            targets.append(class_targets)
            sample_domain.append([domain for i in range(len(class_data))])

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)
        sample_domain = np.concatenate(sample_domain)
        #if self.args.get('validation', False):
        #    val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)
        #    

        #    val_trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])

            # if ret_data:
            #     return data, targets, DummyDataset(data, targets, trsf, self.use_path), val_data, val_targets, DummyDataset(val_data, val_targets, val_trsf, self.use_path)
            # else:
            #     return DummyDataset(data, targets, trsf, self.use_path), DummyDataset(val_data, val_targets, val_trsf, self.use_path)
        if return_domains:
            if ret_data:
                return data, targets, DummyDataset(data, targets, trsf, domains=sample_domain, use_path=self.use_path)
            else:
                return DummyDataset(data, targets, trsf, domains=sample_domain, use_path=self.use_path)
        else:
            if ret_data:
                return data, targets, DummyDataset(data, targets, trsf, use_path=self.use_path)
            else:
                return DummyDataset(data, targets, trsf, use_path=self.use_path)
        
    
    def get_dataset_ben(
        self, domains, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y, z = self._train_data, self._train_targets, self._train_domains
        elif source == "test":
            x, y, z = self._test_data, self._test_targets, self._test_domains
        elif source == "validation":
            x, y, z = self._val_data, self._val_targets, self._val_domains
            print(x,y,z)
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        print(domains)
        for domain in domains:
            print(domain)
            if m_rate is None:
                class_data, class_targets = self._select_domain(
                    x, y, z, domain=domain
                )
            #else:
            #    class_data, class_targets = self._select_rmm(
            #        x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
            #    )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        data_dir = self.args['data_dir'] # "/scratch/htc/lmoellenbrock/data/BigEarthNet-V2"
        datapath = {
            "images_lmdb": data_dir + "/BENv2.lmdb",
            "metadata_parquet": data_dir + "/metadata.parquet",
            "metadata_snow_cloud_parquet": data_dir + "/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
        }

        dataset = BENv2_DataSet.BENv2DataSet(
            data_dirs=datapath,
            img_size=(3, 120, 120),
            split=source,
            include_snowy=False,
            include_cloudy=False,
            return_extras=True,
            patch_prefilter=lambda pid: pid in data,
            transform=trsf,
            )

        if ret_data:
            return data, targets, dataset
        else:
            return dataset


        # if ret_data:
        #     return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        # else:
        #     return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        # Data
        self._train_data, self._train_targets, self._train_domains = idata.train_data, idata.train_targets, idata.train_domains
        self._test_data, self._test_targets, self._test_domains = idata.test_data, idata.test_targets, idata.test_domains
        self.use_path = idata.use_path
        #if self.dataset_name == 'bigearthnet':
        if self.args.get('validation', False):
            self._val_data, self._val_targets, self._val_domains = idata.val_data, idata.val_targets, idata.val_domains
        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        if not self.dil:
            print('not dil')
            order = [i for i in range(len(np.unique(self._train_targets)))]
            if shuffle:
                np.random.seed(seed)
                order = np.random.permutation(len(order)).tolist()
            else:
                order = idata.class_order
            self._class_order = order
        else:
            if shuffle:
                #order = idata.class_order
                #np.random.seed(seed)
                #order = np.random.permutation(order).tolist()
                order = get_task_order(idata=idata,seed=seed)#.tolist()
            else:
                order = idata.class_order.tolist()
            self._class_order = order
        logging.info(self._class_order)

        # Map indices
        if not self.dil:
            self._train_targets = _map_new_class_index(
                self._train_targets, self._class_order
            )
            self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]
    
    def _select_domain(self, x, y, z, domain):
        #if len(domain)==1:
        idxes = np.where(z == domain)[0]
        #else:
        #    idxes = np.where(np.isin(z,domain))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, domains=None, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path
        self.domains = domains

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        if self.domains is not None:
            return idx, image, label, self.domains[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "cifar224":
        return iCIFAR224(args)
    elif name == "imagenetr":
        return iImageNetR(args)
    elif name == "imageneta":
        return iImageNetA()
    elif name == "cub":
        return CUB()
    elif name == "objectnet":
        return objectnet()
    elif name == "omnibenchmark":
        return omnibenchmark()
    elif name == "vtab":
        return vtab()
    elif name=='resisc45':
        return iRESISC45(args)
    elif name=='aid':
        return iAID(args)
    elif name=='flair':
        return iFlair(args)
    elif name=='office_home':
        return iOfficeHome(args)
    elif name=='domainnet':
        return iDomainNet(args)
    elif name == 'bigearthnet':
        return iBigEarthNet(args)
    elif name == 'core50':
        return iCore50(args)
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
    
def get_task_order(seed, idata):
    if idata.args['dataset']=='office_home':
        if seed == 1995: order = np.array(['Art', "Clipart", "Product", "Real_World"]).tolist()
        elif seed == 1996: order = np.array(["Clipart", "Product", "Real_World", 'Art']).tolist()
        elif seed == 1997: order = np.array(["Product", "Clipart", "Real_World", 'Art']).tolist()
        elif seed == 1998: order = np.array(["Real_World", "Product", "Clipart", 'Art']).tolist()
        elif seed == 1999: order = np.array(['Art',"Real_World", "Product", "Clipart"]).tolist()
        else: 
            order = np.array(['Art', "Clipart", "Product", "Real_World"])
            np.random.seed(seed)
            order = np.random.permutation(order).tolist()
            logging.info('random order')
        return order
    elif idata.args['dataset']=='domainnet':
        if seed == 1995: order = np.array(["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]).tolist()
        elif seed == 1996: order = np.array(["infograph", "painting", "quickdraw", "real", "sketch","clipart" ]).tolist()
        elif seed == 1997: order = np.array(["painting", "quickdraw", "real", "sketch","clipart", "infograph"]).tolist()
        elif seed == 1998: order = np.array(["quickdraw", "real", "sketch","clipart", "infograph", "painting"]).tolist()
        elif seed == 1999: order = np.array(["real", "quickdraw", "painting", "sketch","infograph","clipart"]).tolist()
        else:
            order = np.array(["clipart", "infograph", "painting", "quickdraw", "real", "sketch"])
            np.random.seed(seed)
            order = np.random.permutation(order).tolist()
            logging.info('random order')
        return order
    
    elif idata.args['dataset']=='core50':
        if seed == 1995: order = np.array(["s11", "s4", "s2", "s9", "s1", "s6", "s5", "s8"]).tolist()
        elif seed == 1996: order = np.array(["s2", "s9", "s1", "s6", "s5", "s8", "s11", "s4"]).tolist()
        elif seed == 1997: order = np.array(["s4", "s1", "s9", "s2", "s5", "s6", "s8", "s11"]).tolist()
        elif seed == 1998: order = np.array(["s1", "s9", "s2", "s5", "s6", "s8", "s11", "s4" ]).tolist()
        elif seed == 1999: order = np.array(["s9", "s2", "s5", "s6", "s8", "s11", "s4", "s1" ]).tolist()
        else:
            order = np.array(["s11", "s4", "s2", "s9", "s1", "s6", "s5", "s8"])
            np.random.seed(seed)
            order = np.random.permutation(order).tolist()
            logging.info('random order')
        return order 

    else:
        order = idata.class_order
        np.random.seed(seed)
        order = np.random.permutation(order).tolist()
        logging.info('random order')
        return order
        #raise NotImplementedError("No fixed domain order implemented for the dataset {}.".format(dataset_name))
