import numpy as np
from pathlib import Path
import pandas as pd
import os
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels, split_images_labels_multi_label, write_domain_img_file2txt, split_domain_txt2txt, sample_per_domain
#from dataset import FlairDataset


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


def build_transform_coda_prompt(is_train, args):
    if is_train:        
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
        return transform

    t = []
    if args["dataset"].startswith("imagenet"):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]

    return t

def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t

def build_transform_flair(is_train, args):
    if is_train:
        transform =[
                #transforms.Resize((224, 224)),  # Resize to 224x224
                transforms.RandomHorizontalFlip(),  # Random horizontal flip
                transforms.RandomVerticalFlip(),  # Random vertical flip
                transforms.RandomChoice(
                    [  # Randomly apply one of the rotations
                        transforms.RandomRotation(degrees=(0, 0)),  # No rotation
                        transforms.RandomRotation(degrees=(90, 90)),  # Rotate 90 degrees
                        transforms.RandomRotation(degrees=(180, 180)),  # Rotate 180 degrees
                        transforms.RandomRotation(degrees=(270, 270)),  # Rotate 270 degrees
                    ]
                ),
                transforms.RandomResizedCrop(
                    size=(224, 224), scale=(0.8, 1.0)
                ),  # Random resized crop
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                
            ]   
        
        return transform

    else:
        transform =[
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ]
        return transform

class iFlair(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform_flair(True, args)
            self.test_trsf = build_transform_flair(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.array(['D046', 'D009', 'D091', 'D035', 'D013', 'D041', 'D032', 'D023', 'D078', 'D021'])#['D070', 'D013', 'D017', 'D072', 'D086', 'D044', 'D063', 'D051', 'D080', 'D032'])## np.array(["D034","D041","D007","D044","D021","D023","D006","D009","D032","D035"]) #np.array(['D046', 'D009', 'D091', 'D035', 'D013', 'D041', 'D032', 'D023', 'D078', 'D021'])#np.arange(10).tolist()
        #self.class_order = self.class_order[::-1]

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = self.args["data_dir"] + "/flair/flair_converted_all_domains/train/" #"/scratch/htc/lmoellenbrock/data/flair/flair_converted_all_domains/train/" 
        test_dir = self.args["data_dir"] + "/flair/flair_converted_all_domains/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        label_csv = self.args["data_dir"] + "/flair_converted_all_domains/flair_converted_all_domains.csv" #"/scratch/htc/lmoellenbrock/data/flair/flair_converted/flair_converted.csv"
        
        if self.args.get('validation',False):
            rng = np.random.default_rng(seed=self.args['seed'])
            train_imgs = rng.choice(train_dset.imgs, len(train_dset.imgs),replace=False)  

            val_imgs = train_imgs[int(len(train_imgs)*0.8):]
            train_imgs = train_imgs[:int(len(train_imgs)*0.8)]
            self.val_data, self.val_targets, self.val_domains = split_images_labels_multi_label(val_imgs, label_csv=label_csv, list_of_domains=self.class_order.tolist())
        else:
            train_imgs = train_dset.imgs
        self.train_data, self.train_targets, self.train_domains = split_images_labels_multi_label(train_imgs, label_csv=label_csv, list_of_domains=self.class_order.tolist())
        self.test_data, self.test_targets, self.test_domains = split_images_labels_multi_label(test_dset.imgs, label_csv=label_csv, list_of_domains=self.class_order.tolist())

        #print(self.train_data)
        #print(self.train_targets)


class iOfficeHome(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = np.array(['Art', "Clipart", "Product", "Real_World"])#(np.arange(self.args["init_cls"] * self.args["total_sessions"]).tolist())
        self.class_order = class_order
        self.cl_n_inc = self.args["increment"]
        if "task_name" in args and args["task_name"] is not None:
            self.domain_names = args["task_name"]
        else:
            self.domain_names = ['Art', "Clipart", "Product", "Real_World"]
        #logging.info("Learning sequence of domains: {}".format(self.domain_names))

    def download_data(self):
        self.image_list_root = self.args["data_dir"] + "/OfficeHomeDataset_10072016" #self.args["data_path"]

        # more convinient naming
        realworld_dir = Path(self.image_list_root) / "Real World"
        new_dir = Path(self.image_list_root) / "Real_World"

        if realworld_dir.is_dir() and not new_dir.exists():
            realworld_dir.rename(new_dir)
        
        # realworld_dir = Path(self.image_list_root) / "Real World"
        # if realworld_dir.is_dir():
        #     realworld_dir.rename(Path(self.image_list_root) / "Real_World")

        image_list_paths = []

        for d in self.domain_names:
            write_domain_img_file2txt(self.image_list_root, d)
            split_domain_txt2txt(self.image_list_root, d, train_ratio=0.7, seed=self.args['seed'])

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]

        imgs = []
        imgs_val = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            if self.args.get('validation',False):
                rng = np.random.default_rng(seed=self.args['seed'])
                rng.shuffle(image_list)
                image_list_train = image_list[:int(len(image_list)*0.8)]
                image_list_val = image_list[int(len(image_list)*0.8):]
                imgs_val += [(val.split()[0], int(val.split()[1])) for val in image_list_val]
            # imgs: (relative_path, label)
            #imgs += [(val.split()[0], int(val.split()[1]) + taskid * self.cl_n_inc) for val in image_list]
            else: image_list_train = image_list
            imgs += [(val.split()[0], int(val.split()[1])) for val in image_list_train]


                
        train_x, train_y, train_domains = [], [], []
        val_x, val_y, val_domains = [], [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
            train_domains.append(item[0].split('/')[-3])
        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)
        self.train_domains = np.array(train_domains)
        if self.args.get('validation',False):
            for item in imgs_val:
                val_x.append(os.path.join(self.image_list_root, item[0]))
                val_y.append(item[1])
                val_domains.append(item[0].split('/')[-3])
            self.val_data = np.array(val_x)
            self.val_targets = np.array(val_y)
            self.val_domains = np.array(val_domains)
        


        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            #imgs += [(val.split()[0], int(val.split()[1]) + taskid * self.cl_n_inc) for val in image_list]
            imgs += [(val.split()[0], int(val.split()[1])) for val in image_list]
        test_x, test_y, test_domains = [], [], []
        for item in imgs:
            test_x.append(os.path.join(self.image_list_root, item[0]))
            test_y.append(item[1])
            test_domains.append(item[0].split('/')[-3])
        self.test_data = np.array(test_x)
        self.test_targets = np.array(test_y)
        self.test_domains = np.array(test_domains)



class iCore50(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(brightness=63 / 255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = np.array(['s11', 's4', 's2', 's9', 's1', 's6', 's5', 's8'])#(np.arange(self.args["init_cls"] * self.args["total_sessions"]).tolist())
        self.class_order = class_order
        self.cl_n_inc = self.args["increment"]
        if "task_name" in args and args["task_name"] is not None:
            self.domain_names = args["task_name"]
        else:
            self.domain_names = ['s11', 's4', 's2', 's9', 's1', 's6', 's5', 's8', 's3', 's7', 's10']
        #logging.info("Learning sequence of domains: {}".format(self.domain_names))
        self.train_domain_names = ['s11', 's4', 's2', 's9', 's1', 's6', 's5', 's8']
        self.test_domain_names = ['s3', 's7', 's10']
        self.train_split_ratio = 0.7

    def download_data(self):
        self.image_list_root = self.args["data_dir"] + "/core50_128x128" #self.args["data_path"]

        
        image_list_paths = []
        if self.args.get('test_domain_shift', False):
            for d in self.train_domain_names:
                write_domain_img_file2txt(self.image_list_root, d)
                split_domain_txt2txt(self.image_list_root, d, train_ratio=1, seed=self.args['seed'])
            for d in self.test_domain_names:
                write_domain_img_file2txt(self.image_list_root, d)
                split_domain_txt2txt(self.image_list_root, d, train_ratio=0, seed=self.args['seed'])

            image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.train_domain_names]
            #image_list_paths_train = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]
        else:
            for d in self.domain_names:
                write_domain_img_file2txt(self.image_list_root, d)
                split_domain_txt2txt(self.image_list_root, d, train_ratio=self.train_split_ratio, seed=self.args['seed'])

            image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]

        imgs = []
        imgs_val = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            if self.args.get('validation',False):
                rng = np.random.default_rng(seed=self.args['seed'])
                rng.shuffle(image_list)
                image_list_train = image_list[:int(len(image_list)*self.train_split_ratio)]
                image_list_val = image_list[int(len(image_list)*self.train_split_ratio):]
                imgs_val += [(val.split()[0], int(val.split()[1])) for val in image_list_val]
            # imgs: (relative_path, label)
            #imgs += [(val.split()[0], int(val.split()[1]) + taskid * self.cl_n_inc) for val in image_list]
            else: image_list_train = image_list
            imgs += [(val.split()[0], int(val.split()[1])) for val in image_list_train]


                
        train_x, train_y, train_domains = [], [], []
        val_x, val_y, val_domains = [], [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
            train_domains.append(item[0].split('/')[-3])
        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)
        self.train_domains = np.array(train_domains)
        if self.args.get('validation',False):
            for item in imgs_val:
                val_x.append(os.path.join(self.image_list_root, item[0]))
                val_y.append(item[1])
                val_domains.append(item[0].split('/')[-3])
            self.val_data = np.array(val_x)
            self.val_targets = np.array(val_y)
            self.val_domains = np.array(val_domains)
        

        if self.args.get('test_domain_shift', False):
            image_list_paths = [os.path.join(self.image_list_root, d + "_" + "all" + ".txt") for d in self.test_domain_names] # use all data from test domains for testing
        else:
            image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            #imgs += [(val.split()[0], int(val.split()[1]) + taskid * self.cl_n_inc) for val in image_list]
            imgs += [(val.split()[0], int(val.split()[1])) for val in image_list]
        test_x, test_y, test_domains = [], [], []
        for item in imgs:
            test_x.append(os.path.join(self.image_list_root, item[0]))
            test_y.append(item[1])
            test_domains.append(item[0].split('/')[-3])
        self.test_data = np.array(test_x)
        self.test_targets = np.array(test_y)
        self.test_domains = np.array(test_domains)


class iDomainNet(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = np.array(["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]) #(np.arange(self.args["init_cls"] * self.args["total_sessions"]).tolist())
        self.class_order = class_order
        self.nb_sessions = len(class_order)#args["total_sessions"]
        self.cl_n_inc = self.args["increment"]
        if "task_name" in args and args["task_name"] is not None:
            self.domain_names = args["task_name"]
        else:
            self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        self.class_incremental = True if "class_incremental" in args and args["class_incremental"] else False

        #logging.info("Learning sequence of domains: {}".format(self.domain_names))

    def download_data(self):
        def _read_data(image_list_paths):
            imgs = []
            for taskid, image_list_path in enumerate(image_list_paths):
                print(taskid)
                if taskid >= self.nb_sessions:
                    break
                with open(image_list_path) as f:
                    image_list = f.readlines()
                # 重写 target class := original value + taskid * args["increment"]
                for entry in image_list[:-1]:
                    print(entry)
                    img_label = int(entry.split()[1])
                    #if self.class_incremental:
                    #    if img_label < taskid * self.cl_n_inc or img_label >= (taskid + 1) * self.cl_n_inc:
                    #        continue
                    #elif img_label > self.cl_n_inc:
                    #    raise ValueError("class_incremental is False, but img_label > cl_n_inc")
                    #else:  # correct the label for DIL tasks
                    #img_label #= img_label + taskid * self.cl_n_inc
                    imgs.append((entry.split()[0], img_label))

            img_x, img_y, domains = [], [], []
            for item in imgs:
                img_x.append(os.path.join(self.image_list_root, item[0]))
                img_y.append(item[1])
                domains.append(item[0].split('/')[0])


            return np.array(img_x), np.array(img_y), np.array(domains)

        self.image_list_root = self.args["data_dir"] + "/DomainNet" #self.args["data_path"]

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]
        self.train_data, self.train_targets, self.train_domains = _read_data(image_list_paths)

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        self.test_data, self.test_targets, self.test_domains = _read_data(image_list_paths)




class iBigEarthNet(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(size=224,scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(224),
    ]
    common_trsf = [
        #transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = np.array(['Lithuania', "Finland", "Serbia", "Ireland", "Portugal"])#(np.arange(self.args["init_cls"] * self.args["total_sessions"]).tolist())
        self.class_order = class_order
        self.cl_n_inc = self.args["increment"]
        self.n_train = 5000
        self.n_val = 500
        self.n_test = 500
        if "task_name" in args and args["task_name"] is not None:
            self.domain_names = args["task_name"]
        else:
            self.domain_names = ['Lithuania', "Finland", "Serbia", "Ireland", "Portugal"]
        #logging.info("Learning sequence of domains: {}".format(self.domain_names))

    def download_data(self):
        self.dataset_root = self.args["data_dir"] #self.args["data_path"]
        meta = pd.read_parquet(self.dataset_root + 'BigEarthNet-V2/metadata.parquet')

        base_mask = (meta["country"].isin(self.class_order) & (~meta["contains_seasonal_snow"]) & (~meta["contains_cloud_or_shadow"]))

        self.train_data, self.train_targets, self.train_domains = sample_per_domain(meta, split="train", base_mask=base_mask, n_per_domain=self.n_train)
        self.val_data, self.val_targets, self.val_domains = sample_per_domain(meta, split="validation", base_mask=base_mask, n_per_domain=self.n_val)
        self.test_data, self.test_targets, self.test_domains = sample_per_domain(meta, split="test", base_mask=base_mask, n_per_domain=self.n_test)
        